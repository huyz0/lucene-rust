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
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.common.settings.Setting;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.approximate.ApproximateScoreQuery;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.action.search.SearchType;
import org.opensearch.search.query.QueryCollectorArguments;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryCollectorContextSpecRegistry;
import org.opensearch.search.query.QueryPhaseSearcher;
import org.opensearch.search.query.QueryPhaseSearcherWrapper;

import java.io.IOException;
import java.util.LinkedList;

/**
 * Runs the query phase of a shard search natively when the whole request is inside the supported
 * matrix, and on Lucene otherwise -- per query, transparently (M2 T2.4).
 *
 * <p>This hooks {@link QueryPhaseSearcher}, the extension point OpenSearch gives a plugin for the
 * query phase, rather than an {@code EngineFactory}: the query phase is the only part of a search
 * this milestone moves, and hooking it leaves indexing, refresh, fetch and every unsupported request
 * exactly on OpenSearch's own code. The fallback is OpenSearch's own default ({@link
 * QueryPhaseSearcherWrapper}, which also picks concurrent segment search when the index asks for
 * it), so an unsupported request runs precisely as it would without this plugin.
 *
 * <p>A request runs native when it is a top-hits search -- by score, or by a sort {@link
 * SortEncoder} can encode, with or without {@code search_after} -- with no post filter, min
 * score, terminate_after, scroll, collapse, rescore, timeout or profile, no aggregation beyond the
 * metrics {@link NativeAggregations} plans, and a query {@link QueryEncoder} can encode over fields
 * using the default {@link BM25Similarity}. What it produces is what OpenSearch's own {@code
 * SimpleTopDocsCollectorContext} would: the top {@code from + size} hits (with their sort values),
 * the max score, and total hits under the same {@code track_total_hits} rules; and, with
 * aggregations, the shard results their aggregators would have built, stored before {@code
 * DefaultAggregationProcessor.postProcess} runs so that it keeps them rather than reducing its own
 * (unfed) collectors.
 *
 * <p>A native failure is logged, counted, and the query is re-run on Lucene: it never fails a
 * search that Lucene could answer.
 */
public final class RustQueryPhaseSearcher implements QueryPhaseSearcher {
    private static final Logger logger = LogManager.getLogger(RustQueryPhaseSearcher.class);

    /** Per-index switch, dynamic: {@code PUT idx/_settings {"index.lucene_rust.search.enabled": false}}. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "index.lucene_rust.search.enabled",
        true,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * Which encodable shapes run native: {@code fast} (default) only those measured at least as fast
     * as Lucene ({@link QueryEncoder#isFast}); {@code all} every shape the engine answers correctly.
     * Since read path R1 the two agree -- every encodable shape is measured faster -- and the
     * setting stays so that indices which set it keep opening.
     */
    public static final Setting<String> NATIVE_SHAPES = new Setting<>(
        "index.lucene_rust.search.native_shapes",
        "fast",
        v -> {
            if (v.equals("fast") == false && v.equals("all") == false) {
                throw new IllegalArgumentException("index.lucene_rust.search.native_shapes must be [fast] or [all], got [" + v + "]");
            }
            return v;
        },
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    private final QueryPhaseSearcher fallback = new QueryPhaseSearcherWrapper();
    private final NativeReaders readers;
    private final SearchStats stats;

    public RustQueryPhaseSearcher(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
    }

    @Override
    public AggregationProcessor aggregationProcessor(SearchContext searchContext) {
        return fallback.aggregationProcessor(searchContext);
    }

    @Override
    public boolean searchWith(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        // Aggregations OpenSearch's DefaultAggregationProcessor.preProcess registered (their one
        // collector context is the one in "collectors"), when all of them run natively.
        NativeAggregations.Plan aggs = ctx.queryCollectorManagers().isEmpty() ? null : NativeAggregations.plan(ctx);
        String reason = ineligible(ctx, collectors, hasFilterCollector, hasTimeout, aggs != null);
        byte[] blob = null;
        if (reason == null) {
            QueryEncoder.Encoded enc = QueryEncoder.encode(query, field -> defaultBm25(searcher, field));
            reason = enc.fallbackReason();
            blob = enc.blob();
            if (reason == null && enc.fast() == false && "all".equals(ctx.indexShard().indexSettings().getValue(NATIVE_SHAPES)) == false) {
                reason = "slower_shape";
            }
        }
        // OpenSearch's approximate match_all/range (ApproximateScoreQuery resolved to its
        // ApproximateQuery) collects documents in BKD order and so breaks ties differently from
        // Lucene's exact answer, which is what the native engine gives: stay on OpenSearch's.
        if (reason == null && approximated(query)) {
            reason = "approximate";
        }
        // A sorted search: the sort blob, unless the hits are not asked for at all (size 0 has no
        // order to keep, and runs as the unsorted count it is).
        byte[] sortBlob = null;
        if (reason == null && ctx.sort() != null && ctx.size() > 0) {
            if (indexSorted(searcher.getIndexReader())) {
                // TopFieldCollector stops early on an index sorted by the search's sort; the
                // native collector has no such path.
                reason = "index_sort";
            } else {
                // track_scores behind another key: OpenSearch's MaxScoreCollector over every match.
                boolean trackMaxScore = ctx.trackScores() && sortByScore(ctx.sort().sort) == false;
                // A concurrent search collects each slice separately, as Lucene does.
                int[][] slices = NativeAggregations.slices(ctx);
                SortEncoder.Encoded sorted = slices == null ? new SortEncoder.Encoded(null, "intra_segment")
                    : SortEncoder.encode(ctx.sort().sort, ctx.searchAfter(), trackMaxScore, slices);
                reason = sorted.fallbackReason();
                sortBlob = sorted.blob();
            }
        }
        NativeReaders.Acquired acquired = null;
        if (reason == null) {
            acquired = readers.acquire(searcher.getIndexReader());
            reason = acquired.fallbackReason();
        }
        if (reason == null) {
            reason = searchNative(ctx, acquired.handle(), blob, sortBlob, aggs);
            if (reason == null) {
                stats.nativeQuery();
                return false;
            }
        }
        stats.fallback(reason);
        return fallback.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
    }

    /** Why this request cannot run native, or null when it can. */
    static String ineligible(
        SearchContext ctx,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout,
        boolean nativeAggs
    ) {
        if (ctx.indexShard().indexSettings().getValue(ENABLED) == false) return "disabled";
        // The specific reasons first: each of these also adds a collector, and "collectors" alone
        // would not tell an operator which request feature to look at.
        // Aggregations only when NativeAggregations plans every one of them, and nothing else (a
        // global aggregation, another plugin's manager) added a collector manager.
        if (ctx.queryCollectorManagers().isEmpty() == false
            && (nativeAggs == false
                || ctx.queryCollectorManagers().size() != 1
                || ctx.queryCollectorManagers().containsKey(NonGlobalAggCollectorManager.class) == false)) return "aggregations";
        if (ctx.parsedPostFilter() != null) return "post_filter";
        if (ctx.minimumScore() != null) return "min_score";
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) return "terminate_after";
        if (collectors.size() > (nativeAggs ? 1 : 0) || hasFilterCollector) return "collectors";
        if (ctx.scrollContext() != null) return "scroll";
        // A sort runs native when SortEncoder can encode it (searchWith); search_after only
        // with one. track_scores runs native too: the score leading the sort gives the max
        // score (the first hit), and otherwise the native search tracks it over every match.
        if (ctx.searchAfter() != null && ctx.sort() == null) return "search_after";
        if (ctx.collapse() != null) return "collapse";
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) return "rescore";
        if (ctx.getProfilers() != null) return "profile";
        if (hasTimeout) return "timeout";
        // dfs_query_then_fetch scores with statistics aggregated across shards
        // (ContextIndexSearcher.setAggregatedDfs); the native engine only knows this shard's.
        if (ctx.searchType() == SearchType.DFS_QUERY_THEN_FETCH) return "dfs";
        // Another plugin may replace the top-docs collector (QueryPhase's own first question).
        try {
            if (QueryCollectorContextSpecRegistry.getQueryCollectorContextSpec(
                ctx,
                ctx.query(),
                new QueryCollectorArguments.Builder().hasFilterCollector(hasFilterCollector).build()
            ).isPresent()) return "collector_spec";
        } catch (IOException e) {
            return "collector_spec";
        }
        return null;
    }

    /** True when {@code field} scores with a default-parameter {@link BM25Similarity}. */
    static boolean defaultBm25(ContextIndexSearcher searcher, String field) {
        Similarity sim = searcher.getSimilarity();
        if (sim instanceof PerFieldSimilarityWrapper w) {
            sim = w.get(field);
        }
        return sim != null
            && sim.getClass() == BM25Similarity.class
            && ((BM25Similarity) sim).getK1() == 1.2f
            && ((BM25Similarity) sim).getB() == 0.75f
            && sim.getDiscountOverlaps();
    }

    /**
     * Runs the blob and stores the result; returns null, or the fallback reason when it did not.
     *
     * <p>Cancellation is checked before the call, not during it: a native search runs to completion
     * (see docs/opensearch-native-queries.md, Known limits).
     */
    private String searchNative(SearchContext ctx, long handle, byte[] blob, byte[] sortBlob, NativeAggregations.Plan aggs) {
        int size = ctx.size();
        int numDocs = size == 0 ? 0 : Math.min(ctx.from() + size, Math.max(1, ctx.searcher().getIndexReader().numDocs()));
        int trackUpTo = ctx.trackTotalHitsUpTo();
        // track_total_hits: false -> count nothing; true -> exact; N -> exact below N.
        // TopScoreDocCollectorManager stores max(totalHitsThreshold, numHits).
        long countLimit = trackUpTo == SearchContext.TRACK_TOTAL_HITS_DISABLED ? 0
            : trackUpTo == SearchContext.TRACK_TOTAL_HITS_ACCURATE ? Long.MAX_VALUE
            : Math.max(trackUpTo, numDocs);
        // OpenSearch answers some totals without counting (TopDocsCollectorContext
        // .shortcutTotalHitCount) and then collects with a threshold of 1; so must we, or
        // the total differs ({10000, gte} for its {N, eq}) and the search scores documents
        // Lucene never looks at.
        int shortcut = -1;
        if (trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED) {
            try {
                shortcut = shortcutTotalHitCount(ctx.searcher().getIndexReader(), ctx.query());
            } catch (IOException e) {
                return "native_error";
            }
            if (shortcut >= 0) {
                countLimit = numDocs;
            }
        }
        if (ctx.isCancelled()) {
            return "cancelled";
        }
        // The aggregations first, kept aside until the hits are in too: a failure in either
        // re-runs the whole request on Lucene.
        InternalAggregations aggResult = null;
        if (aggs != null) {
            int[][] slices = NativeAggregations.slices(ctx);
            if (slices == null) {
                return "intra_segment";
            }
            int states = aggs.metrics().size() * Math.max(1, slices.length);
            long[] aggCounts = new long[states];
            double[] aggValues = new double[states * NativeAggregations.VALUES];
            int rc = NativeBridge.aggregate(handle, blob, aggs.blob(slices), aggCounts, aggValues);
            if (rc != NativeBridge.OK) {
                stats.nativeError();
                logger.warn("lucene-rust: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return "native_error";
            }
            aggResult = aggs.build(slices, aggCounts, aggValues, ctx.partialOnShard());
        }
        String reason = sortBlob != null ? searchSorted(ctx, handle, blob, sortBlob, numDocs, countLimit, shortcut)
            : searchUnsorted(ctx, handle, blob, numDocs, countLimit, shortcut);
        if (reason == null && aggResult != null) {
            // DefaultAggregationProcessor.postProcess keeps a result already there (hasAggs).
            ctx.queryResult().aggregations(aggResult);
        }
        return reason;
    }

    /** The unsorted half of {@link #searchNative}. */
    private String searchUnsorted(SearchContext ctx, long handle, byte[] blob, int numDocs, long countLimit, int shortcut) {
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[3];
        int rc = NativeBridge.search(handle, blob, numDocs, countLimit, docs, scores, counts);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return "native_error";
        }
        int n = (int) counts[0];
        ScoreDoc[] hits = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        // Lucene's own shape: with counting off, 0 hits "or more".
        TotalHits total = shortcut >= 0 ? new TotalHits(shortcut, TotalHits.Relation.EQUAL_TO)
            : countLimit == 0 ? new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO)
            : new TotalHits(counts[1], counts[2] != 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO);
        float maxScore = n == 0 ? Float.NaN : scores[0];
        if (ctx.sort() != null) {
            // A sorted search with no hits asked for (size 0): what EmptyTopDocsCollectorContext
            // stores, a TopFieldDocs carrying the sort -- the coordinator reads the class.
            ctx.queryResult().topDocs(
                new TopDocsAndMaxScore(new TopFieldDocs(total, new ScoreDoc[0], ctx.sort().sort.getSort()), Float.NaN),
                null
            );
            return null;
        }
        ctx.queryResult().topDocs(new TopDocsAndMaxScore(new TopDocs(total, hits), maxScore), null);
        return null;
    }

    /**
     * The sorted half of {@link #searchNative}: {@code TopFieldCollectorManager(sort, numDocs,
     * searchAfter, countLimit)}, answered as {@code SimpleTopDocsCollectorContext} answers it -- a
     * {@link TopFieldDocs} of {@link FieldDoc}s with {@code NaN} scores, the max score read off the
     * first hit when the score leads the sort and {@code NaN} otherwise.
     */
    private String searchSorted(SearchContext ctx, long handle, byte[] blob, byte[] sortBlob, int numDocs, long countLimit, int shortcut) {
        SortField[] fields = ctx.sort().sort.getSort();
        int[] docs = new int[numDocs];
        long[] values = new long[numDocs * fields.length];
        long[] counts = new long[4];
        byte[][] terms = new byte[1][];
        int rc = NativeBridge.searchSorted(handle, blob, sortBlob, numDocs, countLimit, docs, values, counts, terms);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native sorted search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return "native_error";
        }
        int n = (int) counts[0];
        FieldDoc[] hits = SortEncoder.hits(fields, n, docs, values, terms[0]);
        TotalHits total = shortcut >= 0 ? new TotalHits(shortcut, TotalHits.Relation.EQUAL_TO)
            : countLimit == 0 ? new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO)
            : new TotalHits(counts[1], counts[2] != 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO);
        // The score leading the sort: its first hit's; a tracked max score (track_scores
        // behind another key): the native MaxScoreCollector's, NaN without matches.
        float maxScore = n > 0 && sortByScore(ctx.sort().sort) ? (float) hits[0].fields[0]
            : ctx.trackScores() ? Float.intBitsToFloat((int) counts[3])
            : Float.NaN;
        ctx.queryResult().topDocs(new TopDocsAndMaxScore(new TopFieldDocs(total, hits, fields), maxScore), ctx.sort().formats);
        return null;
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

    /** Whether any segment carries an index sort ({@code index.sort.*}). */
    static boolean indexSorted(IndexReader reader) {
        for (LeafReaderContext leaf : reader.leaves()) {
            if (leaf.reader().getMetaData().sort() != null) {
                return true;
            }
        }
        return false;
    }

    /**
     * {@code TopDocsCollectorContext.shortcutTotalHitCount} (OpenSearch 3.8.0), for the queries the
     * native engine runs: a match-all's count is the reader's {@code numDocs}, and an exact term's
     * the sum of its {@code docFreq}s when nothing is deleted; {@code -1} otherwise.
     */
    static int shortcutTotalHitCount(IndexReader reader, Query query) throws IOException {
        while (true) {
            if (query instanceof ConstantScoreQuery c) {
                query = c.getQuery();
            } else if (query instanceof BoostQuery b) {
                query = b.getQuery();
            } else if (query instanceof ApproximateScoreQuery a) {
                query = a.getOriginalQuery();
            } else {
                break;
            }
        }
        if (query.getClass() == MatchAllDocsQuery.class) {
            return reader.numDocs();
        } else if (query.getClass() == TermQuery.class && reader.hasDeletions() == false) {
            Term term = ((TermQuery) query).getTerm();
            int count = 0;
            for (LeafReaderContext leaf : reader.leaves()) {
                count += leaf.reader().docFreq(term);
            }
            return count;
        }
        return -1;
    }
}
