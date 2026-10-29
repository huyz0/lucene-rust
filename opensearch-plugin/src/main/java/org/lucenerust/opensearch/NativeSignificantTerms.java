/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.opensearch.common.util.BigArrays;
import org.opensearch.common.util.LongArray;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.bucket.BucketsAggregator;
import org.opensearch.search.aggregations.bucket.terms.LongKeyedBucketOrds;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import static org.lucenerust.opensearch.NativeAggregationTree.get;
import static org.lucenerust.opensearch.NativeAggregationTree.writeString;

/**
 * Keyword {@code significant_terms} ({@code GlobalOrdinalsStringTermsAggregator} with its
 * {@code SignificantTermsResults}) as a native tree node: the native side does the collection --
 * each owning bucket's document count ({@code subsetSizes}) and each global ordinal's, in the order
 * Java would have collected them -- and that state is written into the aggregator, whose own
 * {@code buildAggregations} then scores the terms against their background frequencies and keeps the
 * best, so the heuristics, {@code background_filter}, {@code include}/{@code exclude} and thresholds
 * all behave as in the stock engine. Sub-aggregations, {@code missing}, the {@code map} execution and
 * numeric fields fall back.
 */
final class NativeSignificantTerms {
    private NativeSignificantTerms() {}

    static final byte SIGNIFICANT_TERMS = 13;

    private static final String TERMS = "org.opensearch.search.aggregations.bucket.terms.";
    private static final String AGG = TERMS + "GlobalOrdinalsStringTermsAggregator";

    /** The node for a built keyword significant-terms aggregator, or null. */
    static NativeAggregationTree.Node node(AggregatorBase agg, MappedFieldType type, String field, boolean missing, List<?> subs)
        throws ReflectiveOperationException {
        if (missing || subs.isEmpty() == false || agg.subAggregators().length != 0) {
            return null;
        }
        if (!(type instanceof KeywordFieldMapper.KeywordFieldType) || agg.getClass().getName().equals(AGG) == false) {
            return null;
        }
        if (!(agg instanceof BucketsAggregator buckets)) {
            return null;
        }
        Object results = get(agg, "resultStrategy");
        Object collection = get(agg, "collectionStrategy");
        Field subsetSizes = Reflect.inHierarchy(results.getClass(), "subsetSizes");
        if (results.getClass().getName().equals(AGG + "$SignificantTermsResults") == false || subsetSizes == null) {
            return null;
        }
        boolean dense = collection.getClass().getName().equals(AGG + "$DenseGlobalOrds");
        if (dense == false && collection.getClass().getName().equals(AGG + "$RemapGlobalOrds") == false) {
            return null;
        }
        LongKeyedBucketOrds bucketOrds = dense ? null : (LongKeyedBucketOrds) get(collection, "bucketOrds");
        return new Node(buckets, field, results, subsetSizes, bucketOrds);
    }

    static final class Node extends NativeAggregationTree.Node {
        final BucketsAggregator buckets;
        final String field;
        final Object results;
        final Field subsetSizes;
        /** {@code RemapGlobalOrds}' bucket ordinals; null for {@code DenseGlobalOrds}. */
        final LongKeyedBucketOrds bucketOrds;
        /**
         * The next owning bucket ordinal the remapping aggregator has not seen: each read (a slice, or
         * the owning buckets of a parent) gets fresh ones, so its buckets start empty.
         */
        long nextOwner;

        Node(BucketsAggregator buckets, String field, Object results, Field subsetSizes, LongKeyedBucketOrds bucketOrds) {
            super(buckets, List.of());
            this.buckets = buckets;
            this.field = field;
            this.results = results;
            this.subsetSizes = subsetSizes;
            this.bucketOrds = bucketOrds;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(SIGNIFICANT_TERMS);
            writeString(out, field);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            BigArrays bigArrays = buckets.context().bigArrays();
            long[] owners = new long[n];
            if (bucketOrds == null) {
                // DenseGlobalOrds: one owning bucket, its buckets the global ordinals themselves.
                if (n != 1) {
                    throw new IllegalStateException("lucene-rust: dense significant_terms for " + n + " owning buckets");
                }
                LongArray docCounts = (LongArray) get(buckets, "docCounts");
                docCounts.fill(0, docCounts.size(), 0);
            } else {
                for (int i = 0; i < n; i++) {
                    owners[i] = nextOwner++;
                }
            }
            LongArray sizes = (LongArray) subsetSizes.get(results);
            for (int i = 0; i < n; i++) {
                long subset = in.getLong();
                int count = in.getInt();
                if (count < 0) {
                    throw new IllegalStateException("lucene-rust: " + count + " significant_terms buckets");
                }
                sizes = bigArrays.grow(sizes, owners[i] + 1);
                sizes.set(owners[i], subset);
                for (int b = 0; b < count; b++) {
                    long ord = in.getLong();
                    long docs = in.getLong();
                    long bucket = bucketOrds == null ? ord : bucketOrds.add(owners[i], ord);
                    if (bucket < 0) {
                        throw new IllegalStateException("lucene-rust: significant_terms ordinal " + ord + " collected twice");
                    }
                    buckets.incrementBucketDocCount(bucket, docs);
                }
            }
            subsetSizes.set(results, sizes);
            try {
                return Arrays.asList(buckets.buildAggregations(owners));
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }
}
