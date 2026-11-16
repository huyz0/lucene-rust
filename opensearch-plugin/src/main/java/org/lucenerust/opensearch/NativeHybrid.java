/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.grouping.CollapseTopFieldDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.lucenerust.opensearch.neural.search.collector.HybridSearchCollector;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorManager;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorResultsUtilParams;
import org.lucenerust.opensearch.neural.search.query.util.HybridSearchCollectorResultUtil;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
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
 * reader_search.rs}, handed to the same {@link HybridSearchCollectorResultUtil} that formats and merges
 * the collectors' results, so the shard answers with stock's delimited top docs.
 */
final class NativeHybrid {
    /** {@code HYBRID_RETURN_DOCS} in {@code reader_search.rs}. */
    static final int RETURN_DOCS = 1;
    /** {@code HYBRID_POST_FILTER} in {@code reader_search.rs}. */
    static final int POST_FILTER = 2;
    /** {@code HYBRID_SORT} in {@code reader_search.rs}. */
    static final int SORT = 4;
    /** {@code HYBRID_TERMINATE} in {@code reader_search.rs}. */
    static final int TERMINATE = 8;
    /** {@code HYBRID_COLLAPSE} in {@code reader_search.rs}. */
    static final int COLLAPSE = 16;

    private NativeHybrid() {}

    /** Whether the request has a {@code terminate_after} (its collector is one of the chain's). */
    static boolean terminates(SearchContext ctx) {
        return ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER;
    }

    /**
     * Why {@code query} cannot run natively, or null when it can: only a plain hybrid query (no
     * DLS wrapper, no nesting) collected by score, with no collector but its own, a post filter's
     * and the aggregations'. A hybrid query has at most {@link
     * org.lucenerust.opensearch.neural.query.HybridQueryBuilder#MAX_NUMBER_OF_SUB_QUERIES}
     * sub-queries, and HybridAggregationProcessor leaves {@code min_score} at -Infinity, whose
     * collector keeps everything.
     */
    static String ineligible(
        SearchContext ctx,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean nativeAggs
    ) {
        if (ctx.queryCollectorManagers().isEmpty() == false
            && (nativeAggs == false
                || ctx.queryCollectorManagers()
                    .keySet()
                    .stream()
                    .anyMatch(k -> k != NonGlobalAggCollectorManager.class && k != GlobalAggCollectorManager.class))) {
            return NativeAggregationTree.unsupportedReason(ctx);
        }
        if (ctx.scrollContext() != null) {
            // HybridCollectorManager.createHybridCollectorManager's own refusal.
            throw new IllegalArgumentException("Scroll operation is not supported in hybrid query");
        }
        
        boolean minScore = ctx.minimumScore() != null;
        boolean postFilter = ctx.parsedPostFilter() != null;
        if (collectors.size() > (nativeAggs ? 1 : 0) + (postFilter ? 1 : 0) + (minScore ? 1 : 0) + (terminates(ctx) ? 1 : 0)
            || hasFilterCollector != (postFilter || minScore || terminates(ctx))) return "collectors";
        if (ctx.shouldUseTimeSeriesDescSortOptimization()) return "time_series_order";
        return null;
    }

    /**
     * The request blob: the sub-query count, each sub-query's blob (length first) behind the
     * similarities, the hits each sub-query keeps, and the slices.
     */
    static byte[] blob(
        List<byte[]> subQueries,
        int topN,
        int[][] slices,
        boolean returnDocs,
        byte[] postFilter,
        byte[] sort,
        long threshold,
        byte[] collapse,
        long terminateAfter
    ) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(subQueries.size());
        for (byte[] sub : subQueries) {
            NativeAggregations.writeInt(out, sub.length);
            out.writeBytes(sub);
        }
        NativeAggregations.writeInt(out, topN);
        NativeAggregations.writeSlices(out, slices);
        out.write((returnDocs ? RETURN_DOCS : 0) | (postFilter != null ? POST_FILTER : 0) | (sort != null ? SORT : 0) | (collapse != null ? COLLAPSE : 0) | (terminateAfter >= 0 ? TERMINATE : 0));
        if (postFilter != null) {
            NativeAggregations.writeInt(out, postFilter.length);
            out.writeBytes(postFilter);
        }
        if (sort != null) {
            NativeAggregations.writeInt(out, (int) threshold);
            NativeAggregations.writeInt(out, (int) (threshold >>> 32));
            NativeAggregations.writeInt(out, sort.length);
            out.writeBytes(sort);
        }
        if (collapse != null) {
            out.writeBytes(collapse);
        }
        if (terminateAfter >= 0) {
            NativeAggregations.writeInt(out, (int) terminateAfter);
            NativeAggregations.writeInt(out, (int) (terminateAfter >>> 32));
        }
        return out.toByteArray();
    }

    /** {@code manager}'s hits per sub-query, as {@link HybridCollectorManager} sizes its collectors. */
    static int numHits(SearchContext ctx, HybridQuery query) {
        return ((HybridCollectorManager) HybridCollectorManager.createHybridCollectorManager(ctx, query)).getNumHits();
    }

    /** The native answer read: each slice's collected results, and the documents they counted. */
    record Answer(List<Collected> slices, ScoreDoc[] counted, boolean terminated) {}

    /**
     * Reads the native answer. With {@code returnDocs}, the documents the slices counted come with
     * it, each with its sub-queries' scores summed (what an aggregation's scorer reads).
     */
    static Answer read(byte[] answer, int subQueries, boolean returnDocs, SortField[] sort, String collapseField, boolean terminates) {
        ByteBuffer in = ByteBuffer.wrap(answer).order(ByteOrder.LITTLE_ENDIAN);
        List<Collected> collected = new ArrayList<>();
        List<ScoreDoc[]> counted = new ArrayList<>();
        int slices = in.getInt();
        for (int s = 0; s < slices; s++) {
            long total = in.getLong();
            float maxScore = Float.intBitsToFloat(in.getInt());
            boolean thresholdReached = collapseField != null && in.get() != 0;
            List<TopDocs> perQuery = new ArrayList<>(subQueries);
            for (int q = 0; q < subQueries; q++) {
                int n = in.getInt();
                if (collapseField != null) {
                    perQuery.add(readCollapsed(in, n, sort, collapseField, thresholdReached));
                    continue;
                }
                if (sort != null) {
                    perQuery.add(readSorted(in, n, sort));
                    continue;
                }
                ScoreDoc[] hits = new ScoreDoc[n];
                for (int i = 0; i < n; i++) {
                    hits[i] = new ScoreDoc(in.getInt(), Float.intBitsToFloat(in.getInt()));
                }
                perQuery.add(new TopDocs(new TotalHits(n, TotalHits.Relation.EQUAL_TO), hits));
            }
            // A collector that saw no document has no per-query lists at all.
            collected.add(new Collected(total == 0 ? List.of() : perQuery, (int) Math.min(total, Integer.MAX_VALUE), maxScore));
            if (returnDocs) {
                ScoreDoc[] docs = new ScoreDoc[in.getInt()];
                for (int i = 0; i < docs.length; i++) {
                    docs[i] = new ScoreDoc(in.getInt(), Float.intBitsToFloat(in.getInt()));
                }
                counted.add(docs);
            }
        }
        return new Answer(collected, counted.stream().flatMap(java.util.Arrays::stream).toArray(ScoreDoc[]::new), terminates && in.get() != 0);
    }

    /**
     * One sub-query's sorted hits: each {@code FieldDoc} carries the score its slot stored and the
     * sort's values, as {@code HybridTopFieldDocSortCollector.populateResults} builds them; none
     * is that collector's shared empty result.
     */
    private static TopFieldDocs readSorted(ByteBuffer in, int n, SortField[] sort) {
        int[] docs = new int[n];
        float[] scores = new float[n];
        long[] values = new long[n * sort.length];
        for (int i = 0; i < n; i++) {
            docs[i] = in.getInt();
            scores[i] = Float.intBitsToFloat(in.getInt());
            for (int k = 0; k < sort.length; k++) {
                values[i * sort.length + k] = in.getLong();
            }
        }
        byte[] terms = new byte[in.getInt()];
        in.get(terms);
        if (n == 0) {
            return new TopFieldDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0], new SortField[0]);
        }
        FieldDoc[] hits = SortEncoder.hits(sort, n, docs, values, terms);
        for (int i = 0; i < n; i++) {
            hits[i].score = scores[i];
        }
        return new TopFieldDocs(new TotalHits(n, TotalHits.Relation.EQUAL_TO), hits, sort);
    }

    /**
     * One sub-query's hits of the collapsing collector: {@code HybridCollapsingTopDocsCollector.topDocs}'s
     * {@code CollapseTopFieldDocs}, each hit with its group value.
     */
    private static TopDocs readCollapsed(ByteBuffer in, int n, SortField[] sort, String field, boolean thresholdReached) {
        int[] docs = new int[n];
        float[] scores = new float[n];
        long[] values = new long[n * sort.length];
        Object[] groups = new Object[n];
        for (int i = 0; i < n; i++) {
            docs[i] = in.getInt();
            scores[i] = Float.intBitsToFloat(in.getInt());
            for (int k = 0; k < sort.length; k++) {
                values[i * sort.length + k] = in.getLong();
            }
            groups[i] = switch (in.get()) {
                case 0 -> null;
                case 1 -> in.getLong();
                default -> {
                    byte[] term = new byte[in.getInt()];
                    in.get(term);
                    yield new org.apache.lucene.util.BytesRef(term);
                }
            };
        }
        byte[] terms = new byte[in.getInt()];
        in.get(terms);
        int collected = in.getInt();
        TotalHits.Relation relation = thresholdReached ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        if (n == 0 && collected > 0) {
            // No room: the collector's queue has no slot for the first hit (neural-search's own failure).
            throw new ArrayIndexOutOfBoundsException("Index 0 out of bounds for length 0");
        }
        if (n == 0) {
            return new CollapseTopFieldDocs(field, new TotalHits(0, relation), new FieldDoc[0], sort, new Object[0]);
        }
        FieldDoc[] hits = SortEncoder.hits(sort, n, docs, values, terms);
        for (int i = 0; i < n; i++) {
            hits[i].score = scores[i];
        }
        return new CollapseTopFieldDocs(field, new TotalHits(collected, relation), hits, sort, groups);
    }

    /**
     * Rescores each sub-query's hits of each slice by every rescorer, as {@code
     * HybridSearchCollectorResultUtil} rescores them with Lucene's {@code QueryRescorer}: each list
     * apart, the window its first hits. Null when done, else the reason it cannot be.
     */
    static String rescore(Answer answer, long handle, List<NativeRescore> rescores) {
        for (int s = 0; s < answer.slices().size(); s++) {
            Collected collected = answer.slices().get(s);
            List<TopDocs> rescored = new ArrayList<>(collected.topDocs().size());
            for (TopDocs top : collected.topDocs()) {
                int n = top.scoreDocs.length;
                if (n == 0) {
                    rescored.add(top);
                    continue;
                }
                int[] docs = new int[n];
                float[] scores = new float[n];
                for (int i = 0; i < n; i++) {
                    docs[i] = top.scoreDocs[i].doc;
                    scores[i] = top.scoreDocs[i].score;
                }
                for (NativeRescore r : rescores) {
                    java.util.Set<Integer> window = new java.util.HashSet<>();
                    for (int i = 0; i < Math.min(n, r.window()); i++) {
                        window.add(docs[i]);
                    }
                    int rc = NativeBridge.rescore(handle, r.blob(), r.window(), r.queryWeight(), r.rescoreQueryWeight(), r.mode(), docs, scores);
                    if (rc != NativeBridge.OK) {
                        NativeMemory.check(rc);
                        return "native_error";
                    }
                    r.context().setRescoredDocs(java.util.Collections.unmodifiableSet(window));
                }
                ScoreDoc[] hits = new ScoreDoc[n];
                for (int i = 0; i < n; i++) {
                    hits[i] = new ScoreDoc(docs[i], scores[i]);
                }
                rescored.add(new TopDocs(top.totalHits, hits));
            }
            answer.slices().set(s, new Collected(rescored, collected.getTotalHits(), collected.getMaxScore()));
        }
        return null;
    }

    /**
     * Stores the native answer as {@code HybridCollectorManager.reduce} stores its collectors': each
     * slice's results formatted, then merged into the query result in slice order.
     */
    static void store(SearchContext ctx, Answer answer, boolean rescored) throws java.io.IOException {
        if (terminates(ctx)) {
            ctx.queryResult().terminatedEarly(answer.terminated());
        }
        HybridCollectorResultsUtilParams params = new HybridCollectorResultsUtilParams.Builder().searchContext(ctx).rescoredNatively(rescored).build();
        for (Collected collected : answer.slices()) {
            HybridSearchCollectorResultUtil util = new HybridSearchCollectorResultUtil(params, collected);
            util.reduceCollectorResults(ctx.queryResult(), util.getTopDocsAndMaxScore());
        }
    }

    /** One slice's collected results, standing in for its {@code HybridTopScoreDocCollector}. */
    record Collected(List<? extends TopDocs> topDocs, int getTotalHits, float getMaxScore) implements HybridSearchCollector {
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
