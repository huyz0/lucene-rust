/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.apache.lucene.search.Query;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.function.ToLongFunction;

/** Lays the profile of a native search into the request's profilers. */
final class SearchProfiling {
    private static final Logger logger = LogManager.getLogger(SearchProfiling.class);

    static SimilarityEncoder similarities(ContextIndexSearcher searcher) {
        return new SimilarityEncoder(searcher.getSimilarity());
    }

    /**
     * The profile of a search the native engine answered: the query's scorer tree timed clause by
     * clause ({@link NativeProfile}), the collector tree OpenSearch builds for the request with the
     * time the native calls took, laid into the request's profilers.
     */
    static void profile(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout,
        long handle,
        byte[] blob,
        SimilarityEncoder similarities,
        long totalNanos
    ) throws IOException {
        NativeProfile.Phases phases = NativeProfile.phases();
        phases.totalNanos = totalNanos;
        NativeProfile.aggregated(ctx, System.nanoTime() - totalNanos, phases.aggNanos);
        phases.slices = ctx.shouldUseConcurrentSearch() && searcher.getSlices() != null ? Math.max(1, searcher.getSlices().length) : 1;
        long[] javaClock = new long[1];
        NativeProfile.Report report = NativeProfile.run(handle, blob, phases.topScores, phases.topN, phases.countLimit, javaClock);
        ToLongFunction<Query> weights = q -> {
            long t0 = System.nanoTime();
            QueryEncoder.encode(q, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            return System.nanoTime() - t0;
        };
        int roots = layPostFilter(ctx, searcher, similarities, handle, weights);
        NativeProfile.recordCollectors(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout, phases);
        NativeProfile.dropRootsAfter(ctx, roots);
        if (report != null) {
            try {
                NativeProfile.recordQuery(ctx, searcher, query, report, weights, javaClock[0], phases.topScores, false);
            } catch (RuntimeException e) {
                logger.warn("accelerated-search: laying the native profile into the response failed", e);
            }
        }
        layRescores(ctx, searcher, similarities, handle, weights);
    }

    /** The profile of a native hybrid search: the post_filter's tree, the collectors', and the hybrid query's with a tree per sub-query. */
    static void profileHybrid(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        HybridQuery hybrid,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout,
        long handle,
        List<byte[]> blobs,
        SimilarityEncoder similarities,
        int topN,
        long threshold,
        long totalNanos
    ) throws IOException {
        NativeProfile.Phases phases = NativeProfile.phases();
        if (phases == null) {
            return;
        }
        phases.totalNanos = totalNanos;
        phases.hitsNanos = totalNanos;
        phases.aggNanos = totalNanos / 2;
        NativeProfile.aggregated(ctx, System.nanoTime() - totalNanos, phases.aggNanos);
        phases.slices = ctx.shouldUseConcurrentSearch() && searcher.getSlices() != null ? Math.max(1, searcher.getSlices().length) : 1;
        ToLongFunction<Query> weights = q -> {
            long t0 = System.nanoTime();
            QueryEncoder.encode(q, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            return System.nanoTime() - t0;
        };
        List<Query> subs = new ArrayList<>(hybrid.getSubQueries());
        List<NativeProfile.Report> reports = new ArrayList<>();
        long[] clocks = new long[subs.size()];
        for (int i = 0; i < subs.size(); i++) {
            long[] clock = new long[1];
            NativeProfile.Report r = NativeProfile.run(handle, blobs.get(i), true, topN, threshold, clock);
            if (r == null) {
                break;
            }
            reports.add(r);
            clocks[i] = clock[0];
        }
        int roots = layPostFilter(ctx, searcher, similarities, handle, weights);
        NativeProfile.recordCollectors(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout, phases);
        NativeProfile.dropRootsAfter(ctx, roots);
        if (reports.size() == subs.size()) {
            try {
                NativeProfile.Report merged = NativeProfile.hybridReport(hybrid, subs, reports, clocks);
                NativeProfile.recordQuery(ctx, searcher, hybrid, merged, weights, 0, true, false);
            } catch (RuntimeException e) {
                logger.warn("accelerated-search: laying the native hybrid profile into the response failed", e);
            }
        }
        layRescores(ctx, searcher, similarities, handle, weights);
    }

    /** QueryPhase makes the post_filter's weight before the search makes the query's: its tree, first. */
    static int layPostFilter(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        SimilarityEncoder similarities,
        long handle,
        ToLongFunction<Query> weights
    ) throws IOException {
        if (ctx.parsedPostFilter() == null) {
            return -1;
        }
        Query filter = searcher.rewrite(ctx.parsedPostFilter().query());
        try {
            QueryEncoder.Encoded encoded = QueryEncoder.encode(filter, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            long[] filterClock = new long[1];
            NativeProfile.Report filtered = encoded.blob() == null
                ? null
                : NativeProfile.run(handle, encoded.blob(), false, 0, Long.MAX_VALUE, filterClock);
            if (filtered != null) {
                NativeProfile.recordQuery(ctx, searcher, filter, filtered, weights, filterClock[0], true, true);
            }
        } catch (RuntimeException e) {
            logger.warn("accelerated-search: laying the native post_filter profile into the response failed", e);
        }
        return NativeProfile.rootCount(ctx);
    }

    static void layRescores(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        SimilarityEncoder similarities,
        long handle,
        ToLongFunction<Query> weights
    ) throws IOException {
        // RescorePhase makes each rescore query's weight after the search.
        if (ctx.rescore() != null) {
            for (org.opensearch.search.rescore.RescoreContext r : ctx.rescore()) {
                if (r instanceof org.opensearch.search.rescore.QueryRescorer.QueryRescoreContext q) {
                    try {
                        Query rescoring = searcher.rewrite(q.parsedQuery().query());
                        QueryEncoder.Encoded encoded = QueryEncoder.encode(rescoring, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
                        long[] rescoreClock = new long[1];
                        NativeProfile.Report rescored = encoded.blob() == null
                            ? null
                            : NativeProfile.run(handle, encoded.blob(), false, 0, Long.MAX_VALUE, rescoreClock);
                        if (rescored != null) {
                            NativeProfile.recordQuery(ctx, searcher, rescoring, rescored, weights, rescoreClock[0], true, true);
                        }
                    } catch (RuntimeException e) {
                        logger.warn("accelerated-search: laying the native rescore profile into the response failed", e);
                    }
                }
            }
        }
    }
}
