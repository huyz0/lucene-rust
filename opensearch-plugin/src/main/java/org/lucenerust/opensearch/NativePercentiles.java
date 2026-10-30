/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.HdrHistogram.DoubleHistogram;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.metrics.InternalHDRPercentileRanks;
import org.opensearch.search.aggregations.metrics.InternalHDRPercentiles;
import org.opensearch.search.aggregations.metrics.InternalTDigestPercentileRanks;
import org.opensearch.search.aggregations.metrics.InternalTDigestPercentiles;
import org.opensearch.search.aggregations.metrics.TDigestState;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lucenerust.opensearch.NativeAggregationTree.get;
import static org.lucenerust.opensearch.NativeAggregationTree.writeString;

/**
 * {@code percentiles} and {@code percentile_ranks} over a t-digest ({@code TDigestPercentilesAggregator},
 * {@code TDigestPercentileRanksAggregator}) as a native tree node: the native side feeds each bucket's
 * {@code MergingDigest} the values Java would, in the same order, and hands back the digest's whole
 * state, which is written into a {@code TDigestState} field by field -- the state the Java aggregator
 * would have built, so the reduce and the quantiles asked of it come out the same. HDR percentiles
 * ({@code HDRPercentilesAggregator}, {@code HDRPercentileRanksAggregator}) take each bucket's values
 * instead, in collection order, and record them into a {@code DoubleHistogram} as the Java aggregator
 * would. Scripts and the field types the metrics refuse fall back.
 */
final class NativePercentiles {
    private NativePercentiles() {}

    static final byte PERCENTILES = 11;
    static final byte HDR_PERCENTILES = 14;
    /** {@code MAX_COMPRESSION} in {@code jvm_aggs.rs}. */
    private static final double MAX_COMPRESSION = 10_000;

    private static final String METRICS = "org.opensearch.search.aggregations.metrics.";
    private static final Class<?> DIGEST = TDigestState.class.getSuperclass();
    private static final Class<?> TDIGEST = DIGEST == null ? null : DIGEST.getSuperclass().getSuperclass();
    private static final Field MERGE_COUNT = field(DIGEST, "mergeCount");
    private static final Field LAST_USED_CELL = field(DIGEST, "lastUsedCell");
    private static final Field TOTAL_WEIGHT = field(DIGEST, "totalWeight");
    private static final Field UNMERGED_WEIGHT = field(DIGEST, "unmergedWeight");
    private static final Field TEMP_USED = field(DIGEST, "tempUsed");
    private static final Field MEAN = field(DIGEST, "mean");
    private static final Field WEIGHT = field(DIGEST, "weight");
    private static final Field TEMP_MEAN = field(DIGEST, "tempMean");
    private static final Field TEMP_WEIGHT = field(DIGEST, "tempWeight");
    private static final Field MIN = field(TDIGEST, "min");
    private static final Field MAX = field(TDIGEST, "max");
    private static final boolean FIELDS_OK = MERGE_COUNT != null
        && LAST_USED_CELL != null
        && TOTAL_WEIGHT != null
        && UNMERGED_WEIGHT != null
        && TEMP_USED != null
        && MEAN != null
        && WEIGHT != null
        && TEMP_MEAN != null
        && TEMP_WEIGHT != null
        && MIN != null
        && MAX != null;

    private static Field field(Class<?> c, String name) {
        if (c == null) {
            return null;
        }
        try {
            Field f = c.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** The node for a built t-digest percentiles aggregator over {@code field}, or null. */
    static NativeAggregationTree.Node node(AggregatorBase agg, String field, byte valueKind) throws ReflectiveOperationException {
        String cls = agg.getClass().getName();
        boolean hdrRanks = cls.equals(METRICS + "HDRPercentileRanksAggregator");
        if (hdrRanks || cls.equals(METRICS + "HDRPercentilesAggregator")) {
            return valueKind < 0 ? null : new HdrNode(agg, field, valueKind, hdrRanks);
        }
        boolean ranks = cls.equals(METRICS + "TDigestPercentileRanksAggregator");
        if (FIELDS_OK == false || valueKind < 0 || (ranks == false && cls.equals(METRICS + "TDigestPercentilesAggregator") == false)) {
            return null;
        }
        double compression = (double) get(agg, "compression");
        if (!(compression > 0 && compression <= MAX_COMPRESSION)) {
            return null;
        }
        return new Node(agg, field, valueKind, compression, ranks);
    }

    /** An HDR percentiles node: each bucket's values back, recorded here. */
    static final class HdrNode extends NativeAggregationTree.Node {
        final String field;
        final byte valueKind;
        final boolean ranks;

        HdrNode(AggregatorBase agg, String field, byte valueKind, boolean ranks) {
            super(agg, List.of());
            this.field = field;
            this.valueKind = valueKind;
            this.ranks = ranks;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(HDR_PERCENTILES);
            out.write(valueKind);
            writeString(out, field);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            double[] keys = (double[]) get(agg, "keys");
            boolean keyed = (boolean) get(agg, "keyed");
            DocValueFormat format = (DocValueFormat) get(agg, "format");
            int digits = (int) get(agg, "numberOfSignificantValueDigits");
            List<InternalAggregation> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                if (in.get() == 0) {
                    // HDRPercentilesAggregator.buildAggregation: a bucket that collected nothing.
                    out.add(agg.buildEmptyAggregation());
                    continue;
                }
                int count = in.getInt();
                if (count < 0 || (long) count * Double.BYTES > in.remaining()) {
                    throw new IllegalStateException("lucene-rust: " + count + " HDR percentiles values");
                }
                // AbstractHDRPercentilesAggregator.getExistingOrNewHistogram, then collect's recordValue.
                DoubleHistogram state = new DoubleHistogram(digits);
                state.setAutoResize(true);
                try {
                    for (int v = 0; v < count; v++) {
                        state.recordValue(in.getDouble());
                    }
                } catch (ArrayIndexOutOfBoundsException e) {
                    // A negative value, which the Java aggregator's histogram refuses as well.
                    throw new JavaRefuses("lucene-rust: an HDR histogram refused a value", e);
                }
                out.add(
                    ranks
                        ? new InternalHDRPercentileRanks(agg.name(), keys, state, keyed, format, agg.metadata())
                        : new InternalHDRPercentiles(agg.name(), keys, state, keyed, format, agg.metadata())
                );
            }
            return out;
        }
    }

    static final class Node extends NativeAggregationTree.Node {
        final String field;
        final byte valueKind;
        final double compression;
        final boolean ranks;

        Node(AggregatorBase agg, String field, byte valueKind, double compression, boolean ranks) {
            super(agg, List.of());
            this.field = field;
            this.valueKind = valueKind;
            this.compression = compression;
            this.ranks = ranks;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(PERCENTILES);
            out.write(valueKind);
            NativeAggregationTree.writeLong(out, Double.doubleToRawLongBits(compression));
            writeString(out, field);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            if (in.get() != 0) {
                // MergingDigest.add on a NaN value, as the Java aggregator would have.
                throw new JavaRefuses("Cannot add NaN to t-digest", null);
            }
            double[] keys = (double[]) get(agg, "keys");
            boolean keyed = (boolean) get(agg, "keyed");
            DocValueFormat format = (DocValueFormat) get(agg, "formatter");
            List<InternalAggregation> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                if (in.get() == 0) {
                    // TDigestPercentilesAggregator.buildAggregation: a bucket that collected nothing.
                    out.add(agg.buildEmptyAggregation());
                    continue;
                }
                TDigestState state = state(in);
                out.add(
                    ranks
                        ? new InternalTDigestPercentileRanks(agg.name(), keys, state, keyed, format, agg.metadata())
                        : new InternalTDigestPercentiles(agg.name(), keys, state, keyed, format, agg.metadata())
                );
            }
            return out;
        }

        /** A {@code TDigestState(compression)} holding the native digest's every field. */
        private TDigestState state(ByteBuffer in) throws ReflectiveOperationException {
            TDigestState state = new TDigestState(compression);
            MERGE_COUNT.setInt(state, in.getInt());
            int last = in.getInt();
            TOTAL_WEIGHT.setDouble(state, in.getDouble());
            UNMERGED_WEIGHT.setDouble(state, in.getDouble());
            int temp = in.getInt();
            MIN.setDouble(state, in.getDouble());
            MAX.setDouble(state, in.getDouble());
            double[] mean = (double[]) MEAN.get(state);
            double[] weight = (double[]) WEIGHT.get(state);
            double[] tempMean = (double[]) TEMP_MEAN.get(state);
            double[] tempWeight = (double[]) TEMP_WEIGHT.get(state);
            if (last < 0 || last > mean.length || temp < 0 || temp > tempMean.length) {
                throw new IllegalStateException("lucene-rust: a t-digest of " + last + " and " + temp + " centroids");
            }
            for (int c = 0; c < last; c++) {
                mean[c] = in.getDouble();
                weight[c] = in.getDouble();
            }
            for (int c = 0; c < temp; c++) {
                tempMean[c] = in.getDouble();
                tempWeight[c] = in.getDouble();
            }
            LAST_USED_CELL.setInt(state, last);
            TEMP_USED.setInt(state, temp);
            return state;
        }
    }
}
