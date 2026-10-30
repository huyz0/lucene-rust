/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.util.BytesRef;
import org.opensearch.common.util.LongArray;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.aggregations.Aggregator;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.AggregatorFactory;
import org.opensearch.search.aggregations.CardinalityUpperBound;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.InternalMultiBucketAggregation;
import org.opensearch.search.aggregations.bucket.BucketsAggregator;
import org.opensearch.search.aggregations.bucket.terms.BytesKeyedBucketOrds;
import org.opensearch.search.aggregations.bucket.terms.LongKeyedBucketOrds;
import org.opensearch.search.internal.SearchContext;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.lucenerust.opensearch.NativeAggregationTree.get;
import static org.lucenerust.opensearch.NativeAggregationTree.writeString;

/**
 * {@code significant_terms} as a native tree node: the native side does the collection -- each owning
 * bucket's document count ({@code subsetSizes}), each key's (a keyword's global ordinal or term, a
 * numeric field's long) in the order Java would have collected them, and each bucket's
 * sub-aggregations -- and that state is written into a fresh aggregator from the request's factory,
 * whose own {@code buildAggregations} then scores the terms against their background frequencies and
 * keeps the best. The heuristics, {@code background_filter}, {@code include}/{@code exclude} on a
 * keyword's ordinals and the thresholds are therefore Java's own; the kept buckets then take the
 * native sub-aggregations. The three collecting aggregators are covered:
 * {@code GlobalOrdinalsStringTermsAggregator} (dense or remapped ordinals),
 * {@code MapStringTermsAggregator} ({@code execution_hint: map}) and {@code NumericTermsAggregator}.
 * {@code include}/{@code exclude} under {@code map} or on a numeric field, floating-point and
 * unsigned fields, and {@code significant_text} fall back.
 */
final class NativeSignificantTerms {
    private NativeSignificantTerms() {}

    static final byte SIGNIFICANT_TERMS = 13;

    private static final String TERMS = "org.opensearch.search.aggregations.bucket.terms.";
    private static final String ORDS = TERMS + "GlobalOrdinalsStringTermsAggregator";
    private static final String MAP = TERMS + "MapStringTermsAggregator";
    private static final String NUMERIC = TERMS + "NumericTermsAggregator";
    private static final Field BUCKET_ORD = bucketField("bucketOrd");
    private static final Field BUCKET_AGGS = bucketField("aggregations");

    private static Field bucketField(String name) {
        try {
            Field f = Class.forName(TERMS + "InternalSignificantTerms$Bucket").getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** The node for a built significant-terms aggregator, or null. */
    static Node node(
        SearchContext ctx,
        AggregatorFactory f,
        AggregatorBase agg,
        MappedFieldType type,
        String field,
        byte valueKind,
        List<NativeAggregationTree.Node> subs
    ) throws ReflectiveOperationException {
        if (BUCKET_ORD == null || BUCKET_AGGS == null || !(agg instanceof BucketsAggregator)) {
            return null;
        }
        String cls = agg.getClass().getName();
        Object results = get(agg, "resultStrategy");
        boolean keyword = type instanceof KeywordFieldMapper.KeywordFieldType;
        boolean includeExclude = NativeAggregations.hasIncludeExclude(f);
        byte flags;
        if (cls.equals(ORDS) && keyword) {
            flags = 0;
        } else if (cls.equals(MAP) && keyword && includeExclude == false) {
            flags = 2;
        } else if (cls.equals(NUMERIC) && valueKind == NativeAggregations.LONG && includeExclude == false) {
            flags = 1;
        } else {
            return null;
        }
        if (results.getClass().getName().endsWith("$SignificantTermsResults") == false
            && results.getClass().getName().endsWith("$SignificantLongTermsResults") == false) {
            return null;
        }
        if (Reflect.inHierarchy(results.getClass(), "subsetSizes") == null) {
            return null;
        }
        return new Node(ctx, f, agg, subs, field, flags);
    }

    static final class Node extends NativeAggregationTree.Node {
        final SearchContext ctx;
        final AggregatorFactory factory;
        final String field;
        /** 1 numeric, 2 the keys as terms; 4 (set by {@link #missingTerm}) a keyword missing term. */
        final byte flags;
        /** A keyword's {@code missing} term, or null. */
        BytesRef missingTerm;

        Node(SearchContext ctx, AggregatorFactory factory, AggregatorBase agg, List<NativeAggregationTree.Node> subs, String field, byte flags) {
            super(agg, subs);
            this.ctx = ctx;
            this.factory = factory;
            this.field = field;
            this.flags = flags;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(SIGNIFICANT_TERMS);
            out.write(flags | (missingTerm != null ? 4 : 0));
            if (missingTerm != null) {
                NativeAggregations.writeInt(out, missingTerm.length);
                out.write(missingTerm.bytes, missingTerm.offset, missingTerm.length);
            }
            writeString(out, field);
            writeSubs(out);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            // A fresh aggregator for each read (a slice, or the owning buckets of a parent), so its
            // buckets start empty: one owning bucket at the top level, as Java's has.
            Aggregator parent = agg.parent();
            BucketsAggregator fresh;
            try {
                fresh = (BucketsAggregator) factory.create(
                    ctx,
                    parent,
                    parent == null && n == 1 ? CardinalityUpperBound.ONE : CardinalityUpperBound.MANY
                );
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
            Object results = get(fresh, "resultStrategy");
            Field subsetSizes = Reflect.inHierarchy(results.getClass(), "subsetSizes");
            LongKeyedBucketOrds longOrds = null;
            BytesKeyedBucketOrds bytesOrds = null;
            String cls = fresh.getClass().getName();
            if (cls.equals(ORDS)) {
                Object collection = get(fresh, "collectionStrategy");
                if (collection.getClass().getName().equals(ORDS + "$RemapGlobalOrds")) {
                    longOrds = (LongKeyedBucketOrds) get(collection, "bucketOrds");
                } else if (collection.getClass().getName().equals(ORDS + "$DenseGlobalOrds") == false || n != 1) {
                    throw new IllegalStateException("lucene-rust: significant_terms collected by " + collection.getClass().getName());
                }
            } else if (cls.equals(NUMERIC)) {
                longOrds = (LongKeyedBucketOrds) get(fresh, "bucketOrds");
            } else if (cls.equals(MAP)) {
                bytesOrds = (BytesKeyedBucketOrds) get(fresh, "bucketOrds");
            } else {
                throw new IllegalStateException("lucene-rust: significant_terms built as " + cls);
            }
            boolean terms = (flags & 2) != 0;
            if (terms != (bytesOrds != null)) {
                throw new IllegalStateException("lucene-rust: significant_terms keys as terms for " + cls);
            }
            long[] owners = new long[n];
            LongArray sizes = (LongArray) subsetSizes.get(results);
            // Each listed bucket's ordinal in the fresh aggregator, in the native order.
            List<Long> listed = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                owners[i] = i;
                long subset = in.getLong();
                int count = in.getInt();
                if (count < 0) {
                    throw new IllegalStateException("lucene-rust: " + count + " significant_terms buckets");
                }
                sizes = ctx.bigArrays().grow(sizes, i + 1);
                sizes.set(i, subset);
                for (int b = 0; b < count; b++) {
                    long bucket;
                    if (terms) {
                        int len = in.getInt();
                        if (len < 0 || len > in.remaining()) {
                            throw new IllegalStateException("lucene-rust: a significant_terms term of " + len + " bytes");
                        }
                        byte[] term = new byte[len];
                        in.get(term);
                        bucket = bytesOrds.add(i, new BytesRef(term));
                    } else {
                        long key = in.getLong();
                        bucket = longOrds == null ? key : longOrds.add(i, key);
                    }
                    if (bucket < 0) {
                        throw new IllegalStateException("lucene-rust: a significant_terms key collected twice");
                    }
                    fresh.incrementBucketDocCount(bucket, in.getLong());
                    listed.add(bucket);
                }
            }
            subsetSizes.set(results, sizes);
            List<InternalAggregations> subAggs = subs.isEmpty() ? List.of() : readSubs(in, listed.size());
            InternalAggregation[] built;
            try {
                built = fresh.buildAggregations(owners);
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
            if (subs.isEmpty() == false) {
                // The kept buckets take their native sub-aggregations, found by bucket ordinal (the
                // fresh aggregator never collected its own).
                java.util.Map<Long, Integer> at = new java.util.HashMap<>();
                for (int i = 0; i < listed.size(); i++) {
                    at.put(listed.get(i), i);
                }
                for (InternalAggregation r : built) {
                    for (Object b : ((InternalMultiBucketAggregation<?, ?>) r).getBuckets()) {
                        Integer i = at.get(BUCKET_ORD.getLong(b));
                        if (i == null) {
                            throw new IllegalStateException("lucene-rust: a significant_terms bucket the native side did not list");
                        }
                        BUCKET_AGGS.set(b, subAggs.get(i));
                    }
                }
            }
            return Arrays.asList(built);
        }
    }
}
