/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * A hybrid query's shard phase run natively ({@link NativeHybrid}): the sub-queries' top hits, the
 * aggregations over the documents the collection counted, the sort, collapse and rescore.
 */
final class HybridPhase {
    private static final Logger logger = LogManager.getLogger(HybridPhase.class);

    private final NativeReaders readers;
    private final SearchStats stats;

    HybridPhase(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
    }

    /**
     * A hybrid query's shard phase run natively ({@link NativeHybrid}), or why it cannot be: the
     * stock checks first (search type, sub-queries), which fail the request as they do in
     * Java.
     */
    String search(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        if (SearchFlags.of(ctx).enabled() == false) return FallbackReason.DISABLED.key();
        // The aggregations as the plain path plans them; they run over the documents the hybrid
        // collection counted.
        long searchStart = System.nanoTime();
        NativeAggregations.Plan aggs = ctx.queryCollectorManagers().isEmpty() ? null : NativeAggregations.plan(ctx);
        NativeAggregationTree.Tree tree = aggs != null || ctx.queryCollectorManagers().isEmpty()
            ? null
            : NativeAggregationTree.plan(ctx, field -> true);
        String reason = NativeHybrid.ineligible(ctx, query, collectors, hasFilterCollector, aggs != null || tree != null);
        if (reason != null) return reason;
        HybridQuery hybrid = (HybridQuery) query;
        org.lucenerust.opensearch.neural.util.HybridQueryUtil.validateHybridQuerySearchType(ctx.searchType());
        org.lucenerust.opensearch.neural.util.HybridQueryUtil.validateHybridQuery(hybrid);
        int[][] slices = NativeAggregations.slices(ctx);
        if (slices == null) return FallbackReason.INTRA_SEGMENT.key();
        if (LiveDirectoryReader.find(searcher.getIndexReader()) != null) return FallbackReason.REALTIME_VIEW.key();
        SimilarityEncoder similarities = new SimilarityEncoder(searcher.getSimilarity());
        // The post filter is rewritten as QueryPhase rewrites it; an approximate range in it matches
        // only the first documents it visits, which the native engine would not reproduce.
        Query postFilter = ctx.parsedPostFilter() == null ? null : searcher.rewrite(ctx.parsedPostFilter().query());
        if (postFilter != null && SearchSupport.approximated(postFilter) && SearchSupport.buffered(searcher) == false && SearchFlags.of(ctx).nativeApproximate() == false) return FallbackReason.APPROXIMATE.key();
        NativeHybrid.Encoded enc = NativeHybrid.encode(hybrid, similarities, searcher, SearchFlags.of(ctx).allShapes(), EncodeContext.of(ctx, searcher.getTopReaderContext()));
        if (enc.fallbackReason() != null) return enc.fallbackReason();
        byte[] filterBlob = null;
        if (postFilter != null) {
            Query filter = postFilter;
            QueryEncoder.Encoded encoded = QueryEncoder.encode(filter, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            if (encoded.fallbackReason() != null) return encoded.fallbackReason();
            filterBlob = similarities.prefix(encoded.blob());
        }
        // The rescore queries (HybridSearchCollectorResultUtil rescores each sub-query's hits), encoded
        // before anything runs.
        List<NativeRescore> rescores = null;
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) {
            if (NativeRescore.supported(ctx) == false) return FallbackReason.RESCORE.key();
            rescores = new ArrayList<>();
            reason = NativeRescore.encode(ctx, searcher, new SimilarityEncoder(searcher.getSimilarity()), rescores);
            if (reason != null) return reason;
        }
        NativeReaders.Acquired acquired = readers.acquire(searcher.getIndexReader());
        if (acquired.fallbackReason() != null) return acquired.fallbackReason();
        if (ctx.isCancelled()) return FallbackReason.CANCELLED.key();
        if (hasTimeout && SearchSupport.deadlinePassed(ctx)) return FallbackReason.TIMEOUT.key();
        // HybridCollectorManager's own checks and sizes (pagination_depth, from on one shard).
        int topN = NativeHybrid.numHits(ctx, hybrid);
        // The hits sorted: by the request's sort, after the search_after document when there is one.
        // A collapse (HybridCollectorFactory) is sorted by the request's sort or the score, ignores
        // search_after, and groups by a keyword or number field.
        SortField[] sortFields = null;
        byte[] sortBlob = null;
        byte[] collapseBlob = null;
        String collapseField = null;
        if (ctx.collapse() != null) {
            MappedFieldType type = ctx.collapse().getFieldType();
            byte kind;
            if (type instanceof KeywordFieldMapper.KeywordFieldType) {
                kind = NativeCollapse.KEYWORD;
            } else if (type instanceof NumberFieldMapper.NumberFieldType) {
                kind = NativeCollapse.NUMERIC;
            } else {
                throw new IllegalStateException(
                    "unknown type for collapse field " + ctx.collapse().getFieldName() + ", only keywords and numbers are accepted"
                );
            }
            collapseField = ctx.collapse().getFieldName();
            java.io.ByteArrayOutputStream blob = new java.io.ByteArrayOutputStream();
            byte[] name = type.name().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            NativeAggregations.writeInt(blob, name.length);
            blob.writeBytes(name);
            blob.write(kind);
            collapseBlob = blob.toByteArray();
        }
        if (ctx.sort() != null || collapseBlob != null) {
            Sort keys = ctx.sort() != null ? ctx.sort().sort : new Sort(new SortField(null, SortField.Type.SCORE));
            sortFields = keys.getSort();
            FieldDoc after = collapseBlob != null ? null : ctx.searchAfter();
            if (after != null) {
                if (after.fields == null) {
                    throw new IllegalArgumentException("after.fields wasn't set; you must pass fillFields=true for the previous search");
                }
                if (after.fields.length != sortFields.length) {
                    throw new IllegalArgumentException(
                        "after.fields has " + after.fields.length + " values but sort has " + sortFields.length
                    );
                }
            }
            byte[] indexSorted = SortEncoder.indexSorted(searcher.getIndexReader(), keys);
            SortEncoder.Encoded sorted = SortEncoder.encode(keys, after, false, new int[0][], 0, false, indexSorted, EncodeContext.of(ctx));
            if (sorted.fallbackReason() != null) return sorted.fallbackReason();
            sortBlob = sorted.blob();
        }
        long threshold = Math.max(topN, ctx.trackTotalHitsUpTo());
        boolean aggregating = aggs != null || tree != null;
        byte[][] out = new byte[1][];
        int rc;
        NativeMemory.Scope memory = NativeMemory.open(ctx, "accelerated_hybrid_search");
        try {
            rc = NativeSearch.searchHybrid(acquired.handle(), NativeHybrid.blob(enc.blobs(), topN, slices, aggregating, filterBlob, sortBlob, threshold, collapseBlob, NativeHybrid.terminates(ctx) ? ctx.terminateAfter() : -1), out);
            NativeMemory.check(rc);
            if (rc != NativeBridge.OK) {
                logger.debug("accelerated-search: native hybrid search failed ({}): {}", rc, NativeBridge.lastError());
                return FallbackReason.ACCELERATED_ERROR.key();
            }
            NativeHybrid.Answer answer = NativeHybrid.read(out[0], enc.blobs().size(), aggregating, sortFields, collapseField, NativeHybrid.terminates(ctx));
            InternalAggregations[] aggOut = new InternalAggregations[1];
            InternalAggregations aggResult = null;
            if (aggregating) {
                String failed = hybridAggregations(ctx, searcher, acquired.handle(), similarities, answer.counted(), aggs, tree, slices, aggOut);
                if (failed != null) return failed;
                aggResult = aggOut[0];
            }
            // A sorted result is never rescored (the result util's sorted branch).
            if (rescores != null && sortFields == null) {
                reason = NativeHybrid.rescore(answer, acquired.handle(), rescores);
                if (reason != null) return reason;
            }
            NativeHybrid.store(ctx, answer, rescores != null && sortFields == null);
            if (NativeProfile.requested(ctx)) {
                SearchProfiling.profileHybrid(ctx, searcher, hybrid, query, collectors, hasFilterCollector, hasTimeout, acquired.handle(), enc.blobs(), similarities, topN, threshold, System.nanoTime() - searchStart);
            }
            if (aggResult != null) {
                ctx.queryResult().aggregations(aggResult);
                if (tree != null && tree.global().isEmpty() == false) {
                    ctx.queryCollectorManagers().remove(GlobalAggCollectorManager.class);
                }
            }
        } finally {
            memory.close();
            stats.nativeMemory(memory.peak());
        }
        if (memory.timedOut() || hasTimeout && SearchSupport.deadlinePassed(ctx)) {
            SearchSupport.timedOut(ctx);
        }
        return null;
    }

    /**
     * The aggregations over the documents the hybrid collection counted (what the aggregators
     * see: the bulk scorer's windows, not the post filter's), each scored with its sub-queries'
     * scores summed. The result in {@code out[0]}, or the reason it cannot be had.
     */
    private String hybridAggregations(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        long handle,
        SimilarityEncoder similarities,
        ScoreDoc[] counted,
        NativeAggregations.Plan aggs,
        NativeAggregationTree.Tree tree,
        int[][] slices,
        InternalAggregations[] out
    ) throws IOException {
        QueryEncoder.Encoded docs = QueryEncoder.encode(KnnHitsQuery.of(searcher.getIndexReader(), counted), similarities, searcher.getTopReaderContext());
        if (docs.fallbackReason() != null) return docs.fallbackReason();
        byte[] blob = docs.blob();
        if (aggs != null) {
            int states = aggs.metrics().size() * Math.max(1, slices.length);
            long[] aggCounts = new long[states];
            double[] aggValues = new double[states * NativeAggregations.VALUES];
            byte[][] terms = new byte[1][];
            int rc = NativeSearch.aggregate(
                handle, blob, aggs.blob(slices), aggCounts, aggValues, terms, -1, new long[] { -1, 0, 0 },
                new long[searcher.getIndexReader().leaves().size()]
            );
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.ACCELERATED_ERROR.key();
            }
            out[0] = aggs.build(slices, aggCounts, aggValues, terms[0], ctx.partialOnShard());
            NativeAggregations.consumeBuckets(ctx, out[0]);
            return null;
        }
        byte[][] result = new byte[1][];
        int rc = NativeSearch.aggregateTree(handle, blob, tree.blob(slices, NativeAggregations.bucketLimit(ctx)), false, result);
        if (rc != NativeBridge.OK) {
            NativeAggregations.checkBuckets(ctx, rc);
            NativeMemory.check(rc);
            stats.nativeError();
            logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return FallbackReason.ACCELERATED_ERROR.key();
        }
        try {
            out[0] = tree.build(result[0], slices, ctx.partialOnShard());
        } catch (JavaRefuses e) {
            return FallbackReason.AGGREGATIONS.key();
        } catch (ReflectiveOperationException | RuntimeException e) {
            stats.nativeError();
            logger.warn("accelerated-search: building native aggregation results failed, re-running on Lucene", e);
            return FallbackReason.ACCELERATED_ERROR.key();
        }
        NativeAggregations.consumeBuckets(ctx, out[0]);
        return null;
    }
}
