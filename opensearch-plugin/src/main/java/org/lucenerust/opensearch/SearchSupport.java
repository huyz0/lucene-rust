/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.util.BytesRef;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.search.approximate.ApproximateScoreQuery;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryPhaseExecutionException;
import java.io.IOException;

/**
 * What the native query phase asks of a request and its searcher: the deadline, a query's shape
 * (approximated, post-filtered, a sort by score) and OpenSearch's total-hit-count shortcuts.
 */
final class SearchSupport {
    private static final Logger logger = LogManager.getLogger(SearchSupport.class);

    /**
     * Whether the searcher's cancellation checks -- QueryPhase's timeout, and the task's own with
     * {@code low_level_cancellation} -- say stop, as ContextIndexSearcher asks them before each
     * segment.
     */
    static boolean deadlinePassed(SearchContext ctx) {
        QueryTimeout timeout = ctx.searcher().getTimeout();
        return timeout != null && timeout.shouldExit();
    }

    /**
     * A native search that ran past its timeout, answered as QueryPhase answers one: the task
     * cancelled fails it, as the TaskCancelledException ContextIndexSearcher does not catch would;
     * otherwise the request fails without {@code allow_partial_search_results} and is flagged
     * {@code timed_out} with it. Its results are what it collected by the deadline: the native
     * collection stops there, between windows of documents, as Lucene's under QueryPhase's timeout
     * check does.
     */
    static void timedOut(SearchContext ctx) {
        // Only a registered low_level_cancellation check sees the task; the
        // timeout's alone does not.
        if (ctx.lowLevelCancellation() && ctx.isCancelled()) {
            throw new TaskCancelledException("cancelled task with reason: " + ctx.getTask().getReasonCancelled());
        }
        if (ctx.request().allowPartialSearchResults() == false) {
            throw new QueryPhaseExecutionException(ctx.shardTarget(), "Time exceeded");
        }
        ctx.queryResult().searchTimedOut(true);
    }

    /** {@code +query #filter}: the documents both match, scored by {@code query} alone. */
    static Query postFiltered(Query query, Query filter) {
        return new org.apache.lucene.search.BooleanQuery.Builder().add(query, org.apache.lucene.search.BooleanClause.Occur.MUST)
            .add(filter, org.apache.lucene.search.BooleanClause.Occur.FILTER)
            .build();
    }

    private static final java.lang.reflect.Field RESOLVED_QUERY = resolvedQueryField();

    private static java.lang.reflect.Field resolvedQueryField() {
        try {
            java.lang.reflect.Field f = ApproximateScoreQuery.class.getDeclaredField("resolvedQuery");
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** Whether the searcher reads a view of the writer's buffered documents that holds some. */
    static boolean buffered(ContextIndexSearcher searcher) {
        LiveDirectoryReader live = LiveDirectoryReader.find(searcher.getIndexReader());
        return live != null && live.view().maxDoc() > 0;
    }

    /**
     * Whether {@code query} holds an {@link ApproximateScoreQuery} that OpenSearch resolved to its
     * approximation for this request ({@code setContext}; the choice is package-private, so it is
     * read reflectively). Unreadable counts as approximated: falling back is always correct.
     */
    static boolean approximated(Query query) {
        if (query instanceof ApproximateScoreQuery a) {
            if (RESOLVED_QUERY == null) {
                return true;
            }
            try {
                return RESOLVED_QUERY.get(a) instanceof org.opensearch.search.approximate.ApproximateQuery;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return true;
            }
        }
        if (query instanceof org.apache.lucene.search.BooleanQuery b) {
            for (var c : b.clauses()) {
                if (approximated(c.query())) {
                    return true;
                }
            }
            return false;
        }
        if (query instanceof ConstantScoreQuery c) {
            return approximated(c.getQuery());
        }
        if (query instanceof BoostQuery b) {
            return approximated(b.getQuery());
        }
        return false;
    }

    /** {@code SortField.FIELD_SCORE.equals(sort.getSort()[0])}: the score, descending, leads. */
    static boolean sortByScore(Sort sort) {
        return SortField.FIELD_SCORE.equals(sort.getSort()[0]);
    }

    /**
     * {@code TopDocsCollectorContext.shortcutTotalHitCount} (OpenSearch 3.8.0), for the queries the
     * native engine runs: a match-all's count is the reader's {@code numDocs}, and an exact term's
     * the sum of its {@code docFreq}s when nothing is deleted (from {@code handle}'s segments, the
     * same ones); {@code -1} otherwise.
     */
    static int shortcutTotalHitCount(IndexReader reader, Query query, long handle) throws IOException {
        query = unwrapShortcut(query);
        if (query.getClass() == MatchAllDocsQuery.class) {
            return reader.numDocs();
        }
        Term term = shortcutTerm(reader, query);
        if (term != null) {
            // Summed by the native term dictionaries, which the search reads straight after:
            // Lucene's own, in every segment, cost more than the search itself.
            BytesRef bytes = term.bytes();
            long[] out = new long[1];
            int rc = NativeSearch.docFreq(
                handle,
                term.field().getBytes(java.nio.charset.StandardCharsets.UTF_8),
                java.util.Arrays.copyOfRange(bytes.bytes, bytes.offset, bytes.offset + bytes.length),
                out
            );
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                throw new IOException("native docFreq failed (" + rc + "): " + NativeBridge.lastError());
            }
            return Math.toIntExact(out[0]);
        }
        if (query.getClass() == org.apache.lucene.search.FieldExistsQuery.class && reader.hasDeletions() == false) {
            // The documents with an indexed value: the points' or terms' doc count.
            String field = ((org.apache.lucene.search.FieldExistsQuery) query).getField();
            int count = 0;
            for (LeafReaderContext leaf : reader.leaves()) {
                org.apache.lucene.index.FieldInfo info = leaf.reader().getFieldInfos().fieldInfo(field);
                if (info == null) {
                    continue;
                }
                if (info.getPointIndexDimensionCount() > 0) {
                    org.apache.lucene.index.PointValues points = leaf.reader().getPointValues(field);
                    if (points != null) {
                        count += points.getDocCount();
                    }
                } else if (info.getIndexOptions() != org.apache.lucene.index.IndexOptions.NONE) {
                    org.apache.lucene.index.Terms terms = leaf.reader().terms(field);
                    if (terms != null) {
                        count += terms.getDocCount();
                    }
                } else {
                    // Not indexed: no shortcut.
                    return -1;
                }
            }
            return count;
        }
        return -1;
    }

    /**
     * The term whose document frequency {@link #shortcutTotalHitCount} answers the total with: a
     * plain {@code TermQuery}'s, behind any constant-score or boost wrappers, on a reader without
     * deletions; else null.
     */
    static Term shortcutTerm(IndexReader reader, Query query) {
        query = unwrapShortcut(query);
        if (query.getClass() == TermQuery.class && reader.hasDeletions() == false) {
            return ((TermQuery) query).getTerm();
        }
        return null;
    }

    /** The wrappers {@code shortcutTotalHitCount} looks through. */
    static Query unwrapShortcut(Query query) {
        while (true) {
            if (query instanceof ConstantScoreQuery c) {
                query = c.getQuery();
            } else if (query instanceof BoostQuery b) {
                query = b.getQuery();
            } else if (query instanceof ApproximateScoreQuery a) {
                query = a.getOriginalQuery();
            } else {
                return query;
            }
        }
    }
}
