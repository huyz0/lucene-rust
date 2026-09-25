/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.common.settings.Setting;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
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
        }
        NativeReaders.Acquired acquired = null;
        if (reason == null) {
            acquired = readers.acquire(searcher.getIndexReader());
            reason = acquired.fallbackReason();
        }
        if (reason == null && searchNative(ctx, acquired.handle(), blob)) {
            stats.nativeQuery();
            return false;
        }
        stats.fallback(reason == null ? "native_error" : reason);
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
        if (collectors.isEmpty() == false || hasFilterCollector) return "collectors";
        if (ctx.queryCollectorManagers().isEmpty() == false) return "aggregations";
        if (ctx.scrollContext() != null) return "scroll";
        if (ctx.sort() != null) return "sort";
        if (ctx.searchAfter() != null) return "search_after";
        if (ctx.collapse() != null) return "collapse";
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) return "rescore";
        if (ctx.minimumScore() != null) return "min_score";
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) return "terminate_after";
        if (ctx.getProfilers() != null) return "profile";
        if (hasTimeout) return "timeout";
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

    /** Runs the blob and stores the result; false (after logging) when the native call failed. */
    private boolean searchNative(SearchContext ctx, long handle, byte[] blob) {
        int size = ctx.size();
        int numDocs = size == 0 ? 0 : Math.min(ctx.from() + size, Math.max(1, ctx.searcher().getIndexReader().numDocs()));
        int trackUpTo = ctx.trackTotalHitsUpTo();
        boolean countTotal = trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED;
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[2];
        if (ctx.isCancelled()) {
            return false;
        }
        int rc = NativeBridge.search(handle, blob, numDocs, countTotal, docs, scores, counts);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return false;
        }
        int n = (int) counts[0];
        ScoreDoc[] hits = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        TotalHits total = countTotal
            ? new TotalHits(counts[1], TotalHits.Relation.EQUAL_TO)
            : new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO);
        float maxScore = n == 0 ? Float.NaN : scores[0];
        ctx.queryResult().topDocs(new TopDocsAndMaxScore(new TopDocs(total, hits), maxScore), null);
        return true;
    }
}
