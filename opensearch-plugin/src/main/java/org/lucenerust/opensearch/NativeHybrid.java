/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.lucenerust.opensearch.neural.search.collector.HybridSearchCollector;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorManager;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorResultsUtilParams;
import org.lucenerust.opensearch.neural.search.query.util.HybridSearchCollectorResultUtil;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * neural-search's hybrid query shard phase run natively: what {@code HybridTopScoreDocCollector}
 * collects (one per slice of a concurrent search), with {@code search_hybrid_blobs} in {@code
 * jvm_reader.rs}, handed to the same {@link HybridSearchCollectorResultUtil} that formats and merges
 * the collectors' results, so the shard answers with stock's delimited top docs.
 */
final class NativeHybrid {
    /** The most sub-queries a hybrid blob holds ({@code MAX_HYBRID_QUERIES} in Rust). */
    static final int MAX_SUB_QUERIES = 16;

    private NativeHybrid() {}

    /**
     * Why {@code query} cannot run natively, or null when it can: only a plain hybrid query (no
     * DLS wrapper, no nesting) collected by score, with nothing but its own collector.
     */
    static String ineligible(SearchContext ctx, Query query, LinkedList<QueryCollectorContext> collectors, boolean hasFilterCollector) {
        if (query instanceof HybridQuery == false) return "hybrid_wrapped";
        if (((HybridQuery) query).getSubQueries().size() > MAX_SUB_QUERIES) return "hybrid_sub_queries";
        if (ctx.queryCollectorManagers().isEmpty() == false) return "hybrid_aggregations";
        if (ctx.sort() != null) return "hybrid_sort";
        if (ctx.collapse() != null) return "hybrid_collapse";
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) return "hybrid_rescore";
        if (ctx.searchAfter() != null || ctx.scrollContext() != null) return "hybrid_search_after";
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) return "hybrid_terminate_after";
        if (ctx.parsedPostFilter() != null) return "hybrid_post_filter";
        if (ctx.getProfilers() != null) return "profile";
        // HybridAggregationProcessor leaves min_score at -Infinity, whose collector keeps everything.
        boolean minScore = ctx.minimumScore() != null;
        if (minScore && ctx.minimumScore() != Float.NEGATIVE_INFINITY) return "min_score";
        if (collectors.size() != (minScore ? 1 : 0) || hasFilterCollector != minScore) return "collectors";
        if (ctx.shouldUseTimeSeriesDescSortOptimization()) return "time_series_order";
        return null;
    }

    /**
     * The request blob: the sub-query count, each sub-query's blob (length first) behind the
     * similarities, the hits each sub-query keeps, and the slices.
     */
    static byte[] blob(List<byte[]> subQueries, int topN, int[][] slices) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(subQueries.size());
        for (byte[] sub : subQueries) {
            NativeAggregations.writeInt(out, sub.length);
            out.writeBytes(sub);
        }
        NativeAggregations.writeInt(out, topN);
        NativeAggregations.writeSlices(out, slices);
        return out.toByteArray();
    }

    /** {@code manager}'s hits per sub-query, as {@link HybridCollectorManager} sizes its collectors. */
    static int numHits(SearchContext ctx, HybridQuery query) {
        return ((HybridCollectorManager) HybridCollectorManager.createHybridCollectorManager(ctx, query)).getNumHits();
    }

    /**
     * Stores the native answer as {@code HybridCollectorManager.reduce} stores its collectors': each
     * slice's results formatted, then merged into the query result in slice order.
     */
    static void store(SearchContext ctx, byte[] answer, int subQueries) throws java.io.IOException {
        ByteBuffer in = ByteBuffer.wrap(answer).order(ByteOrder.LITTLE_ENDIAN);
        int slices = in.getInt();
        HybridCollectorResultsUtilParams params = new HybridCollectorResultsUtilParams.Builder().searchContext(ctx).build();
        for (int s = 0; s < slices; s++) {
            long total = in.getLong();
            List<TopDocs> perQuery = new ArrayList<>(subQueries);
            float maxScore = 0.0f;
            for (int q = 0; q < subQueries; q++) {
                int n = in.getInt();
                ScoreDoc[] hits = new ScoreDoc[n];
                for (int i = 0; i < n; i++) {
                    hits[i] = new ScoreDoc(in.getInt(), Float.intBitsToFloat(in.getInt()));
                    maxScore = Math.max(maxScore, hits[i].score);
                }
                perQuery.add(new TopDocs(new TotalHits(n, TotalHits.Relation.EQUAL_TO), hits));
            }
            // A collector that saw no document has no per-query lists at all.
            Collected collected = new Collected(total == 0 ? List.of() : perQuery, (int) Math.min(total, Integer.MAX_VALUE), maxScore);
            HybridSearchCollectorResultUtil util = new HybridSearchCollectorResultUtil(params, collected);
            util.reduceCollectorResults(ctx.queryResult(), util.getTopDocsAndMaxScore());
        }
    }

    /** One slice's collected results, standing in for its {@code HybridTopScoreDocCollector}. */
    private record Collected(List<TopDocs> topDocs, int getTotalHits, float getMaxScore) implements HybridSearchCollector {
        @Override
        public org.apache.lucene.search.LeafCollector getLeafCollector(org.apache.lucene.index.LeafReaderContext context) {
            throw new UnsupportedOperationException();
        }

        @Override
        public ScoreMode scoreMode() {
            return ScoreMode.COMPLETE;
        }
    }

    /** The blob of each sub-query, or the reason one cannot be encoded. */
    record Encoded(List<byte[]> blobs, String fallbackReason) {}

    static Encoded encode(HybridQuery query, SimilarityEncoder similarities, ContextIndexSearcher searcher, boolean allShapes) {
        List<byte[]> blobs = new ArrayList<>();
        for (Query sub : query.getSubQueries()) {
            QueryEncoder.Encoded enc = QueryEncoder.encode(sub, similarities, searcher.getTopReaderContext());
            if (enc.fallbackReason() != null) {
                return new Encoded(null, enc.fallbackReason());
            }
            if (enc.fast() == false && allShapes == false) {
                return new Encoded(null, "slower_shape");
            }
            blobs.add(enc.blob());
        }
        List<byte[]> prefixed = new ArrayList<>(blobs.size());
        for (byte[] b : blobs) {
            prefixed.add(similarities.prefix(b));
        }
        return new Encoded(prefixed, null);
    }
}
