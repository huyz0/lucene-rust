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
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.common.settings.Setting;
import org.opensearch.search.aggregations.AggregationProcessor;
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
 * <p>A request runs native only when it is a plain top-hits-by-score search: no sort, aggregation,
 * post filter, min score, terminate_after, scroll, search_after, collapse, rescore, timeout or
 * profile, and a query {@link QueryEncoder} can encode over fields using the default {@link
 * BM25Similarity}. What it produces is what OpenSearch's own {@code SimpleTopDocsCollectorContext}
 * would: the top {@code from + size} hits, the max score, and total hits under the same {@code
 * track_total_hits} rules.
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
        String reason = ineligible(ctx, collectors, hasFilterCollector, hasTimeout);
        byte[] blob = null;
        if (reason == null) {
            QueryEncoder.Encoded enc = QueryEncoder.encode(query, field -> defaultBm25(searcher, field));
            reason = enc.fallbackReason();
            blob = enc.blob();
            if (reason == null && enc.fast() == false && "all".equals(ctx.indexShard().indexSettings().getValue(NATIVE_SHAPES)) == false) {
                reason = "slower_shape";
            }
        }
        NativeReaders.Acquired acquired = null;
        if (reason == null) {
            acquired = readers.acquire(searcher.getIndexReader());
            reason = acquired.fallbackReason();
        }
        if (reason == null) {
            reason = searchNative(ctx, acquired.handle(), blob);
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
        boolean hasTimeout
    ) {
        if (ctx.indexShard().indexSettings().getValue(ENABLED) == false) return "disabled";
        // The specific reasons first: each of these also adds a collector, and "collectors" alone
        // would not tell an operator which request feature to look at.
        if (ctx.queryCollectorManagers().isEmpty() == false) return "aggregations";
        if (ctx.parsedPostFilter() != null) return "post_filter";
        if (ctx.minimumScore() != null) return "min_score";
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) return "terminate_after";
        if (collectors.isEmpty() == false || hasFilterCollector) return "collectors";
        if (ctx.scrollContext() != null) return "scroll";
        if (ctx.sort() != null) return "sort";
        if (ctx.searchAfter() != null) return "search_after";
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
    private String searchNative(SearchContext ctx, long handle, byte[] blob) {
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
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[3];
        if (ctx.isCancelled()) {
            return "cancelled";
        }
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
        ctx.queryResult().topDocs(new TopDocsAndMaxScore(new TopDocs(total, hits), maxScore), null);
        return null;
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
