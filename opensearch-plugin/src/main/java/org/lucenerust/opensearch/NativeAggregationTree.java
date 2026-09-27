/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.Rounding;
import org.opensearch.common.hash.MurmurHash3;
import org.opensearch.common.util.BigArrays;
import org.opensearch.common.util.BitMixer;
import org.opensearch.index.mapper.DateFieldMapper;
import org.opensearch.index.mapper.DocCountFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.aggregations.Aggregator;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.AggregatorFactories;
import org.opensearch.search.aggregations.AggregatorFactory;
import org.opensearch.search.aggregations.BucketOrder;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.MultiBucketCollector;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.aggregations.bucket.filter.InternalFilters;
import org.opensearch.search.aggregations.bucket.histogram.DoubleBounds;
import org.opensearch.search.aggregations.bucket.histogram.InternalDateHistogram;
import org.opensearch.search.aggregations.bucket.histogram.InternalHistogram;
import org.opensearch.search.aggregations.bucket.histogram.LongBounds;
import org.opensearch.search.aggregations.bucket.range.InternalRange;
import org.opensearch.search.aggregations.bucket.range.RangeAggregator;
import org.opensearch.search.aggregations.bucket.terms.StringTerms;
import org.opensearch.search.aggregations.metrics.HyperLogLogPlusPlus;
import org.opensearch.search.aggregations.support.ValuesSourceConfig;
import org.opensearch.search.internal.SearchContext;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Aggregation trees that run natively (read path R5): {@code histogram}, {@code date_histogram},
 * {@code range}/{@code date_range}, {@code filter}, {@code filters}, {@code global}, keyword
 * {@code terms}, {@code cardinality} and the metrics of {@link NativeAggregations}, nested in any
 * combination -- what {@link NativeAggregations#plan} (flat metrics and {@code terms} only) does
 * not take.
 *
 * <p>{@link #plan} walks OpenSearch's own aggregator factories beside the aggregators its
 * collector manager built from them (so every parameter is the one OpenSearch resolved: the
 * sorted ranges, the rounding, the bounds, the {@code shard_size}), {@link Tree#blob} is the tree
 * blob {@code jvm_aggs.rs} decodes, and {@link Tree#build} makes the shard results those
 * aggregators would build -- through their own empty results and factories wherever OpenSearch
 * exposes them.
 */
@SuppressWarnings("deprecation") // Rounding.offset(), which DateHistogramAggregator itself reads
public final class NativeAggregationTree {
    static final byte METRIC = 0;
    static final byte CARDINALITY = 1;
    static final byte TERMS = 2;
    static final byte HISTOGRAM = 3;
    static final byte DATE_HISTOGRAM = 4;
    static final byte RANGE = 5;
    static final byte FILTERS = 6;
    static final byte GLOBAL = 7;
    /** {@code date_histogram} unit codes, in the order {@code jvm_aggs.rs} lists them. */
    private static final Rounding.DateTimeUnit[] UNITS = {
        Rounding.DateTimeUnit.WEEK_OF_WEEKYEAR,
        Rounding.DateTimeUnit.YEAR_OF_CENTURY,
        Rounding.DateTimeUnit.QUARTER_OF_YEAR,
        Rounding.DateTimeUnit.MONTH_OF_YEAR,
        Rounding.DateTimeUnit.DAY_OF_MONTH,
        Rounding.DateTimeUnit.HOUR_OF_DAY,
        Rounding.DateTimeUnit.MINUTES_OF_HOUR,
        Rounding.DateTimeUnit.SECOND_OF_MINUTE };
    private static final byte UNIT_INTERVAL = 8;

    private static final String AGGS = "org.opensearch.search.aggregations.";
    private static final Field METADATA = field(AggregatorFactory.class, "metadata");
    private static final Field SUB_FACTORIES = field(AggregatorFactory.class, "factories");
    private static final Field CONFIG = field(
        org.opensearch.search.aggregations.support.ValuesSourceAggregatorFactory.class,
        "config"
    );

    private NativeAggregationTree() {}

    /** One aggregation of the tree: its blob entry and how its result is built. */
    abstract static class Node {
        final AggregatorBase agg;
        final List<Node> subs;

        Node(AggregatorBase agg, List<Node> subs) {
            this.agg = agg;
            this.subs = subs;
        }

        abstract void write(ByteArrayOutputStream out);

        /** Its results for {@code n} owning buckets, read off the native result. */
        abstract List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException;

        void writeSubs(ByteArrayOutputStream out) {
            out.write(subs.size());
            for (Node s : subs) {
                s.write(out);
            }
        }

        /** Each sub-aggregation's results for {@code n} owning buckets, gathered per bucket. */
        List<InternalAggregations> readSubs(ByteBuffer in, int n) throws ReflectiveOperationException {
            List<List<InternalAggregation>> per = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                per.add(new ArrayList<>(subs.size()));
            }
            for (Node s : subs) {
                List<InternalAggregation> r = s.read(in, n);
                for (int i = 0; i < n; i++) {
                    per.get(i).add(r.get(i));
                }
            }
            List<InternalAggregations> out = new ArrayList<>(n);
            for (List<InternalAggregation> p : per) {
                out.add(InternalAggregations.from(p));
            }
            return out;
        }

        InternalAggregations emptySubs() {
            List<InternalAggregation> out = new ArrayList<>();
            for (Aggregator a : agg.subAggregators()) {
                out.add(a.buildEmptyAggregation());
            }
            return InternalAggregations.from(out);
        }

        static int count(ByteBuffer in, int n) {
            int got = in.getInt();
            if (got != n) {
                throw new IllegalStateException("native aggregation result for " + got + " buckets, expected " + n);
            }
            return got;
        }
    }

    static final class MetricNode extends Node {
        final NativeAggregations.Metric metric;

        MetricNode(AggregatorBase agg, NativeAggregations.Metric metric) {
            super(agg, List.of());
            this.metric = metric;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(METRIC);
            out.write(metric.valueKind());
            out.write(metric.source());
            out.write(metric.needs());
            writeString(out, metric.field());
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) {
            count(in, n);
            List<InternalAggregation> out = new ArrayList<>(n);
            long[] counts = new long[1];
            double[] values = new double[NativeAggregations.VALUES];
            for (int i = 0; i < n; i++) {
                counts[0] = in.getLong();
                for (int v = 0; v < values.length; v++) {
                    values[v] = in.getDouble();
                }
                out.add(NativeAggregations.Plan.metric(metric, counts, values, 0));
            }
            return out;
        }
    }

    static final class CardinalityNode extends Node {
        final String field;
        /** 0 keyword, else 1 + the value kind. */
        final byte kind;
        final int precision;

        CardinalityNode(AggregatorBase agg, String field, byte kind, int precision) {
            super(agg, List.of());
            this.field = field;
            this.kind = kind;
            this.precision = precision;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(CARDINALITY);
            out.write(kind);
            writeString(out, field);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            List<InternalAggregation> out = new ArrayList<>(n);
            MurmurHash3.Hash128 hash = new MurmurHash3.Hash128();
            for (int i = 0; i < n; i++) {
                int values = in.getInt();
                if (values == 0) {
                    // CardinalityAggregator.buildAggregation: a bucket that saw nothing.
                    out.add(agg.buildEmptyAggregation());
                    continue;
                }
                HyperLogLogPlusPlus counts = new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
                for (int v = 0; v < values; v++) {
                    if (kind == 0) {
                        byte[] term = new byte[in.getInt()];
                        in.get(term);
                        MurmurHash3.hash128(term, 0, term.length, 0, hash);
                        counts.collect(0, hash.h1);
                    } else {
                        // MurmurHash3Values: a long as is, a double as its doubleToLongBits.
                        counts.collect(0, BitMixer.mix64(in.getLong()));
                    }
                }
                out.add(CARDINALITY_CTOR.newInstance(agg.name(), counts, agg.metadata()));
            }
            return out;
        }
    }

    static final class TermsNode extends Node {
        final NativeAggregations.Terms terms;

        TermsNode(AggregatorBase agg, List<Node> subs, NativeAggregations.Terms terms) {
            super(agg, subs);
            this.terms = terms;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(TERMS);
            terms.write(out);
            writeSubs(out);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            long[] other = new long[n];
            List<List<Object[]>> per = new ArrayList<>(n);
            int total = 0;
            for (int i = 0; i < n; i++) {
                other[i] = in.getLong();
                int m = in.getInt();
                List<Object[]> list = new ArrayList<>(m);
                for (int b = 0; b < m; b++) {
                    long docs = in.getLong();
                    byte[] term = new byte[in.getInt()];
                    in.get(term);
                    list.add(new Object[] { docs, term });
                }
                per.add(list);
                total += m;
            }
            List<InternalAggregations> subAggs = readSubs(in, total);
            List<InternalAggregation> out = new ArrayList<>(n);
            int child = 0;
            for (int i = 0; i < n; i++) {
                List<StringTerms.Bucket> buckets = new ArrayList<>(per.get(i).size());
                for (Object[] b : per.get(i)) {
                    buckets.add(
                        new StringTerms.Bucket(
                            new BytesRef((byte[]) b[1]),
                            (long) b[0],
                            subAggs.get(child++),
                            terms.showTermDocCountError(),
                            0,
                            terms.format()
                        )
                    );
                }
                out.add(
                    new StringTerms(
                        terms.name(),
                        BucketOrder.key(true),
                        terms.order(),
                        terms.metadata(),
                        terms.format(),
                        terms.thresholds().getShardSize(),
                        terms.showTermDocCountError(),
                        other[i],
                        buckets,
                        0,
                        terms.thresholds()
                    )
                );
            }
            return out;
        }
    }

    static final class HistogramNode extends Node {
        final String field;
        final byte valueKind;
        final double interval;
        final double offset;
        final DoubleBounds hardBounds;

        HistogramNode(AggregatorBase agg, List<Node> subs, String field, byte valueKind, double interval, double offset, DoubleBounds hard) {
            super(agg, subs);
            this.field = field;
            this.valueKind = valueKind;
            this.interval = interval;
            this.offset = offset;
            this.hardBounds = hard;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(HISTOGRAM);
            writeString(out, field);
            out.write(valueKind);
            writeLong(out, Double.doubleToRawLongBits(interval));
            writeLong(out, Double.doubleToRawLongBits(offset));
            Double min = hardBounds == null ? null : hardBounds.getMin();
            Double max = hardBounds == null ? null : hardBounds.getMax();
            out.write((min != null ? 1 : 0) | (max != null ? 2 : 0));
            if (min != null) {
                writeLong(out, Double.doubleToRawLongBits(min));
            }
            if (max != null) {
                writeLong(out, Double.doubleToRawLongBits(max));
            }
            writeSubs(out);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            List<double[]> keys = new ArrayList<>(n);
            List<long[]> docs = new ArrayList<>(n);
            int total = 0;
            for (int i = 0; i < n; i++) {
                int m = in.getInt();
                double[] k = new double[m];
                long[] d = new long[m];
                for (int b = 0; b < m; b++) {
                    k[b] = in.getDouble();
                    d[b] = in.getLong();
                }
                keys.add(k);
                docs.add(d);
                total += m;
            }
            List<InternalAggregations> subAggs = readSubs(in, total);
            // The aggregator's own empty result carries its order, bounds, format and empty-bucket
            // info -- the same ones its buildAggregations would use.
            InternalHistogram empty = (InternalHistogram) agg.buildEmptyAggregation();
            DocValueFormat format = (DocValueFormat) get(agg, "formatter");
            boolean keyed = (boolean) get(agg, "keyed");
            List<InternalAggregation> out = new ArrayList<>(n);
            int child = 0;
            for (int i = 0; i < n; i++) {
                List<InternalHistogram.Bucket> buckets = new ArrayList<>(keys.get(i).length);
                for (int b = 0; b < keys.get(i).length; b++) {
                    double key = keys.get(i)[b] * interval + offset;
                    buckets.add(new InternalHistogram.Bucket(key, docs.get(i)[b], keyed, format, subAggs.get(child++)));
                }
                buckets.sort(BucketOrder.key(true).comparator());
                out.add(empty.create(buckets));
            }
            return out;
        }
    }

    static final class DateHistogramNode extends Node {
        final String field;
        final byte unit;
        final long interval;
        final long zoneMillis;
        final long offset;
        final LongBounds hardBounds;

        DateHistogramNode(
            AggregatorBase agg,
            List<Node> subs,
            String field,
            byte unit,
            long interval,
            long zoneMillis,
            long offset,
            LongBounds hard
        ) {
            super(agg, subs);
            this.field = field;
            this.unit = unit;
            this.interval = interval;
            this.zoneMillis = zoneMillis;
            this.offset = offset;
            this.hardBounds = hard;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(DATE_HISTOGRAM);
            writeString(out, field);
            out.write(unit);
            if (unit == UNIT_INTERVAL) {
                writeLong(out, interval);
            }
            writeLong(out, zoneMillis);
            writeLong(out, offset);
            Long min = hardBounds == null ? null : hardBounds.getMin();
            Long max = hardBounds == null ? null : hardBounds.getMax();
            out.write((min != null ? 1 : 0) | (max != null ? 2 : 0));
            if (min != null) {
                writeLong(out, min);
            }
            if (max != null) {
                writeLong(out, max);
            }
            writeSubs(out);
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            List<long[]> keys = new ArrayList<>(n);
            List<long[]> docs = new ArrayList<>(n);
            int total = 0;
            for (int i = 0; i < n; i++) {
                int m = in.getInt();
                long[] k = new long[m];
                long[] d = new long[m];
                for (int b = 0; b < m; b++) {
                    k[b] = in.getLong();
                    d[b] = in.getLong();
                }
                keys.add(k);
                docs.add(d);
                total += m;
            }
            List<InternalAggregations> subAggs = readSubs(in, total);
            // DateHistogramAggregator.buildAggregations: the empty-bucket info carries the
            // rounding without its offset (buildEmptyAggregation's keeps it), so it is built here.
            Rounding rounding = (Rounding) get(agg, "rounding");
            DocValueFormat format = (DocValueFormat) get(agg, "formatter");
            boolean keyed = (boolean) get(agg, "keyed");
            long minDocCount = (long) get(agg, "minDocCount");
            BucketOrder order = (BucketOrder) get(agg, "order");
            LongBounds extended = (LongBounds) get(agg, "extendedBounds");
            List<InternalAggregation> out = new ArrayList<>(n);
            int child = 0;
            for (int i = 0; i < n; i++) {
                List<InternalDateHistogram.Bucket> buckets = new ArrayList<>(keys.get(i).length);
                for (int b = 0; b < keys.get(i).length; b++) {
                    buckets.add(new InternalDateHistogram.Bucket(keys.get(i)[b], docs.get(i)[b], keyed, format, subAggs.get(child++)));
                }
                buckets.sort(BucketOrder.key(true).comparator());
                Object emptyInfo = minDocCount == 0 ? DATE_EMPTY_CTOR.newInstance(rounding.withoutOffset(), emptySubs(), extended) : null;
                out.add(
                    DATE_HISTOGRAM_CTOR.newInstance(
                        agg.name(),
                        buckets,
                        order,
                        minDocCount,
                        rounding.offset(),
                        emptyInfo,
                        format,
                        keyed,
                        agg.metadata()
                    )
                );
            }
            return out;
        }
    }

    /** {@code range}, {@code filters}, {@code filter} and {@code global}: fixed buckets per owner. */
    abstract static class FixedNode extends Node {
        final int width;

        FixedNode(AggregatorBase agg, List<Node> subs, int width) {
            super(agg, subs);
            this.width = width;
        }

        abstract InternalAggregation build(long[] docs, List<InternalAggregations> subs) throws ReflectiveOperationException;

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            count(in, n);
            int w = in.getInt();
            if (w != width) {
                throw new IllegalStateException("native aggregation result of width " + w + ", expected " + width);
            }
            long[] docs = new long[n * width];
            for (int i = 0; i < docs.length; i++) {
                docs[i] = in.getLong();
            }
            List<InternalAggregations> subAggs = readSubs(in, n * width);
            List<InternalAggregation> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(
                    build(Arrays.copyOfRange(docs, i * width, (i + 1) * width), subAggs.subList(i * width, (i + 1) * width))
                );
            }
            return out;
        }
    }

    static final class RangeNode extends FixedNode {
        final String field;
        final byte valueKind;
        final RangeAggregator.Range[] ranges;

        RangeNode(AggregatorBase agg, List<Node> subs, String field, byte valueKind, RangeAggregator.Range[] ranges) {
            super(agg, subs, ranges.length);
            this.field = field;
            this.valueKind = valueKind;
            this.ranges = ranges;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(RANGE);
            writeString(out, field);
            out.write(valueKind);
            NativeAggregations.writeInt(out, ranges.length);
            for (RangeAggregator.Range r : ranges) {
                writeLong(out, Double.doubleToRawLongBits(r.getFrom()));
                writeLong(out, Double.doubleToRawLongBits(r.getTo()));
            }
            writeSubs(out);
        }

        @Override
        @SuppressWarnings({ "unchecked", "rawtypes" })
        InternalAggregation build(long[] docs, List<InternalAggregations> subs) throws ReflectiveOperationException {
            // RangeAggregator.buildAggregations: its range factory, each range in its order.
            InternalRange.Factory factory = (InternalRange.Factory) get(agg, "rangeFactory");
            DocValueFormat format = (DocValueFormat) get(agg, "format");
            boolean keyed = (boolean) get(agg, "keyed");
            List buckets = new ArrayList<>(ranges.length);
            for (int i = 0; i < ranges.length; i++) {
                RangeAggregator.Range r = ranges[i];
                buckets.add(factory.createBucket(r.getKey(), r.getFrom(), r.getTo(), docs[i], subs.get(i), keyed, format));
            }
            return factory.create(agg.name(), buckets, format, keyed, agg.metadata());
        }
    }

    static final class FiltersNode extends FixedNode {
        final byte[][] filters;
        final boolean other;

        FiltersNode(AggregatorBase agg, List<Node> subs, byte[][] filters, boolean other) {
            super(agg, subs, filters.length + (other ? 1 : 0));
            this.filters = filters;
            this.other = other;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(FILTERS);
            NativeAggregations.writeInt(out, filters.length);
            for (byte[] f : filters) {
                NativeAggregations.writeInt(out, f.length);
                out.writeBytes(f);
            }
            out.write(other ? 1 : 0);
            writeSubs(out);
        }

        @Override
        InternalAggregation build(long[] docs, List<InternalAggregations> subs) throws ReflectiveOperationException {
            if (agg.getClass().getName().equals(AGGS + "bucket.filter.FilterAggregator")) {
                return FILTER_CTOR.newInstance(agg.name(), docs[0], subs.get(0), agg.metadata());
            }
            // FiltersAggregator.buildAggregations: a bucket per key, then the other bucket.
            String[] keys = (String[]) get(agg, "keys");
            boolean keyed = (boolean) get(agg, "keyed");
            List<InternalFilters.InternalBucket> buckets = new ArrayList<>(width);
            for (int i = 0; i < keys.length; i++) {
                buckets.add(new InternalFilters.InternalBucket(keys[i], docs[i], subs.get(i), keyed));
            }
            if (other) {
                String key = (String) get(agg, "otherBucketKey");
                buckets.add(new InternalFilters.InternalBucket(key, docs[keys.length], subs.get(keys.length), keyed));
            }
            return new InternalFilters(agg.name(), buckets, keyed, agg.metadata());
        }
    }

    static final class GlobalNode extends FixedNode {
        GlobalNode(AggregatorBase agg, List<Node> subs) {
            super(agg, subs, 1);
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(GLOBAL);
            writeSubs(out);
        }

        @Override
        InternalAggregation build(long[] docs, List<InternalAggregations> subs) throws ReflectiveOperationException {
            return GLOBAL_CTOR.newInstance(agg.name(), docs[0], subs.get(0), agg.metadata());
        }
    }

    /**
     * A request's aggregations as native trees: {@code main} over the query's matches (per slice),
     * {@code global} over every document (one pass), in the request's order within each.
     */
    public record Tree(List<Node> main, List<Node> global, byte[] globalQuery) {
        /** The tree blob ({@code decode_tree} in {@code jvm_aggs.rs}). */
        byte[] blob(int[][] slices) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(main.size());
            for (Node n : main) {
                n.write(out);
            }
            if (global.isEmpty()) {
                out.write(0);
            } else {
                out.write(1);
                NativeAggregations.writeInt(out, globalQuery.length);
                out.writeBytes(globalQuery);
                out.write(global.size());
                for (Node n : global) {
                    n.write(out);
                }
            }
            NativeAggregations.writeSlices(out, slices);
            return out.toByteArray();
        }

        /**
         * The shard results: each slice's main results reduced as {@code NonGlobalAggCollectorManager}
         * reduces its collectors' (not at all without slices), then the global ones after them, as
         * {@code DefaultAggregationProcessor.postProcess} appends them (each slice's reduced, as
         * {@code GlobalAggCollectorManager} reduces them, under concurrent search).
         */
        InternalAggregations build(byte[] result, int[][] slices, InternalAggregation.ReduceContext onShard)
            throws ReflectiveOperationException {
            ByteBuffer in = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
            List<InternalAggregation> all = new ArrayList<>();
            if (main.isEmpty() == false) {
                List<InternalAggregation> sliced = new ArrayList<>();
                for (int s = 0; s < Math.max(1, slices.length); s++) {
                    for (Node n : main) {
                        sliced.add(n.read(in, 1).get(0));
                    }
                }
                InternalAggregations mainAggs = InternalAggregations.from(sliced);
                if (slices.length > 0) {
                    mainAggs = InternalAggregations.reduce(List.of(mainAggs), onShard);
                }
                all.addAll(mainAggs.copyResults());
            }
            if (global.isEmpty() == false) {
                // The global search runs with the same slices (ConcurrentAggregationProcessor), its
                // results reduced as GlobalAggCollectorManager reduces them.
                List<InternalAggregation> sliced = new ArrayList<>();
                for (int s = 0; s < Math.max(1, slices.length); s++) {
                    for (Node n : global) {
                        sliced.add(n.read(in, 1).get(0));
                    }
                }
                InternalAggregations globalAggs = InternalAggregations.from(sliced);
                if (slices.length > 0) {
                    globalAggs = InternalAggregations.reduce(List.of(globalAggs), onShard);
                }
                all.addAll(globalAggs.copyResults());
            }
            if (in.hasRemaining()) {
                throw new IllegalStateException("native aggregation result: " + in.remaining() + " trailing bytes");
            }
            return InternalAggregations.from(all);
        }
    }

    /**
     * The native trees for {@code ctx}'s aggregations, or null when any of them is not supported.
     * {@code fieldOk} and the searcher are what {@link QueryEncoder} needs for the filters.
     */
    @SuppressWarnings("unchecked")
    public static Tree plan(SearchContext ctx, Predicate<String> fieldOk) {
        if (ctx.aggregations() == null || METADATA == null || SUB_FACTORIES == null || CONFIG == null || CTORS_OK == false) {
            return null;
        }
        if (ctx.getQueryShardContext() != null && ctx.getQueryShardContext().getStarTreeQueryContext() != null) {
            return null;
        }
        if (ctx.isStreamSearch()) {
            return null;
        }
        // Every bucket counts a document once: no `_doc_count` field in the index.
        for (LeafReaderContext leaf : ctx.searcher().getIndexReader().leaves()) {
            if (leaf.reader().getFieldInfos().fieldInfo(DocCountFieldMapper.NAME) != null) {
                return null;
            }
        }
        try {
            AggregatorFactories factories = ctx.aggregations().factories();
            List<AggregatorFactory> nonGlobal = new ArrayList<>();
            List<AggregatorFactory> globals = new ArrayList<>();
            for (AggregatorFactory f : factories.getFactories()) {
                (f.getClass().getName().equals(AGGS + "bucket.global.GlobalAggregatorFactory") ? globals : nonGlobal).add(f);
            }
            List<AggregatorBase> mainAggs = nonGlobal.isEmpty() ? List.of() : aggregators(ctx, NonGlobalAggCollectorManager.class);
            List<AggregatorBase> globalAggs = globals.isEmpty() ? List.of() : aggregators(ctx, GlobalAggCollectorManager.class);
            if (mainAggs == null || globalAggs == null || mainAggs.size() != nonGlobal.size() || globalAggs.size() != globals.size()) {
                return null;
            }
            List<Node> main = new ArrayList<>();
            for (int i = 0; i < nonGlobal.size(); i++) {
                Node n = node(ctx, nonGlobal.get(i), mainAggs.get(i), true, fieldOk);
                if (n == null) {
                    return null;
                }
                main.add(n);
            }
            List<Node> global = new ArrayList<>();
            for (int i = 0; i < globals.size(); i++) {
                Node n = node(ctx, globals.get(i), globalAggs.get(i), true, fieldOk);
                if (n == null) {
                    return null;
                }
                global.add(n);
            }
            byte[] globalQuery = null;
            if (global.isEmpty() == false) {
                // DefaultAggregationProcessor.postProcess searches this for the global aggregations.
                QueryEncoder.Encoded enc = QueryEncoder.encode(
                    ctx.searcher().rewrite(ctx.buildFilteredQuery(new MatchAllDocsQuery())),
                    fieldOk
                );
                if (enc.blob() == null) {
                    return null;
                }
                globalQuery = enc.blob();
            }
            return new Tree(List.copyOf(main), List.copyOf(global), globalQuery);
        } catch (Exception e) {
            return null;
        }
    }

    /** The top-level aggregators the collector manager registered under {@code key} built, in order. */
    private static List<AggregatorBase> aggregators(SearchContext ctx, Class<?> key) throws java.io.IOException {
        Object manager = ctx.queryCollectorManagers().get(key);
        if (!(manager instanceof org.apache.lucene.search.CollectorManager<?, ?> m)) {
            return null;
        }
        Object c = m.newCollector();
        List<AggregatorBase> out = new ArrayList<>();
        Collection<?> parts = c instanceof MultiBucketCollector mb ? Arrays.asList(mb.getCollectors()) : List.of(c);
        for (Object p : parts) {
            if (!(p instanceof AggregatorBase a)) {
                return null;
            }
            out.add(a);
        }
        return out;
    }

    private static final String TERMS_FACTORY = AGGS + "bucket.terms.TermsAggregatorFactory";

    /** One factory and the aggregator built from it, or null when either is not supported. */
    @SuppressWarnings("unchecked")
    private static Node node(SearchContext ctx, AggregatorFactory f, AggregatorBase agg, boolean top, Predicate<String> fieldOk)
        throws Exception {
        if (agg.name().equals(f.name()) == false) {
            return null;
        }
        String factory = f.getClass().getName();
        String aggClass = agg.getClass().getName();
        List<Node> subs = subs(ctx, f, agg, fieldOk);
        if (subs == null) {
            return null;
        }
        switch (factory) {
            case AGGS + "bucket.global.GlobalAggregatorFactory":
                return top ? new GlobalNode(agg, subs) : null;
            case AGGS + "bucket.filter.FilterAggregatorFactory": {
                byte[] q = filterBlob(ctx, (Query) declared(f, "filter"), fieldOk);
                return q == null ? null : new FiltersNode(agg, subs, new byte[][] { q }, false);
            }
            case AGGS + "bucket.filter.FiltersAggregatorFactory": {
                Query[] filters = (Query[]) declared(f, "filters");
                byte[][] blobs = new byte[filters.length][];
                for (int i = 0; i < filters.length; i++) {
                    blobs[i] = filterBlob(ctx, filters[i], fieldOk);
                    if (blobs[i] == null) {
                        return null;
                    }
                }
                if (filters.length == 0) {
                    return null;
                }
                return new FiltersNode(agg, subs, blobs, (boolean) declared(f, "otherBucket"));
            }
            default:
                break;
        }
        // The values-source aggregations: a mapped field, no script or missing value.
        ValuesSourceConfig config = (ValuesSourceConfig) CONFIG.get(f);
        if (config == null || config.script() != null || config.missing() != null || config.fieldContext() == null) {
            return null;
        }
        MappedFieldType type = config.fieldContext().fieldType();
        String field = config.fieldContext().field();
        byte valueKind = NativeAggregations.valueKind(type);
        switch (factory) {
            case TERMS_FACTORY: {
                NativeAggregations.Terms t = NativeAggregations.terms(ctx, f, config);
                return t == null ? null : new TermsNode(agg, subs, t);
            }
            case AGGS + "metrics.CardinalityAggregatorFactory": {
                if (aggClass.equals(AGGS + "metrics.CardinalityAggregator") == false) {
                    return null;
                }
                byte kind = type instanceof KeywordFieldMapper.KeywordFieldType ? 0 : valueKind >= 0 ? (byte) (1 + valueKind) : -1;
                if (kind < 0) {
                    return null;
                }
                return new CardinalityNode(agg, field, kind, (int) get(agg, "precision"));
            }
            case AGGS + "bucket.histogram.HistogramAggregatorFactory": {
                if (valueKind < 0 || aggClass.equals(AGGS + "bucket.histogram.NumericHistogramAggregator") == false) {
                    return null;
                }
                return new HistogramNode(
                    agg,
                    subs,
                    field,
                    valueKind,
                    (double) get(agg, "interval"),
                    (double) get(agg, "offset"),
                    (DoubleBounds) get(agg, "hardBounds")
                );
            }
            case AGGS + "bucket.histogram.DateHistogramAggregatorFactory": {
                if (!(type instanceof DateFieldMapper.DateFieldType d)
                    || d.resolution() != DateFieldMapper.Resolution.MILLISECONDS
                    || aggClass.equals(AGGS + "bucket.histogram.DateHistogramAggregator") == false) {
                    return null;
                }
                return dateHistogram(agg, subs, field, (Rounding) get(agg, "rounding"), (LongBounds) get(agg, "hardBounds"));
            }
            case AGGS + "bucket.range.RangeAggregatorFactory":
            case AGGS + "bucket.range.DateRangeAggregatorFactory": {
                if (valueKind < 0 || aggClass.equals(AGGS + "bucket.range.RangeAggregator") == false) {
                    return null;
                }
                return new RangeNode(agg, subs, field, valueKind, (RangeAggregator.Range[]) get(agg, "ranges"));
            }
            default:
                break;
        }
        // A metric: NativeAggregations' own, with the points shortcut only at the top level.
        NativeAggregations.Metric m = NativeAggregations.metric(ctx, f, config, top);
        return m == null || subs.isEmpty() == false ? null : new MetricNode(agg, m);
    }

    private static List<Node> subs(SearchContext ctx, AggregatorFactory f, AggregatorBase agg, Predicate<String> fieldOk) throws Exception {
        AggregatorFactory[] subFactories = ((AggregatorFactories) SUB_FACTORIES.get(f)).getFactories();
        Aggregator[] subAggs = agg.subAggregators();
        if (subFactories.length != subAggs.length) {
            return null;
        }
        List<Node> out = new ArrayList<>(subAggs.length);
        for (int i = 0; i < subAggs.length; i++) {
            AggregatorBase sub = unwrap(subAggs[i]);
            if (sub == null) {
                return null;
            }
            Node n = node(ctx, subFactories[i], sub, false, fieldOk);
            if (n == null) {
                return null;
            }
            out.add(n);
        }
        return out;
    }

    /**
     * The aggregator itself: a deferred sub-aggregation (a {@code terms} collected breadth first)
     * sits behind the deferring collector's {@code WrappedAggregator} once collection is set up,
     * which delegates everything this reads to the aggregator it holds.
     */
    private static AggregatorBase unwrap(Aggregator a) throws ReflectiveOperationException {
        for (int depth = 0; depth < 4 && a != null; depth++) {
            if (a instanceof AggregatorBase b) {
                return b;
            }
            Object in = get(a, "in");
            a = in instanceof Aggregator next ? next : null;
        }
        return null;
    }

    /** A {@code date_histogram} whose rounding is a unit or an interval in a fixed-offset zone. */
    private static Node dateHistogram(AggregatorBase agg, List<Node> subs, String field, Rounding rounding, LongBounds hard)
        throws ReflectiveOperationException {
        long offset = rounding.offset();
        Rounding inner = rounding.withoutOffset();
        ZoneId zone = (ZoneId) get(inner, "timeZone");
        if (zone == null || zone.getRules().isFixedOffset() == false) {
            return null;
        }
        long zoneMillis = zone.getRules().getOffset(Instant.EPOCH).getTotalSeconds() * 1000L;
        String kind = inner.getClass().getSimpleName();
        if (kind.equals("TimeUnitRounding")) {
            Rounding.DateTimeUnit unit = (Rounding.DateTimeUnit) get(inner, "unit");
            byte code = (byte) Arrays.asList(UNITS).indexOf(unit);
            return code < 0 ? null : new DateHistogramNode(agg, subs, field, code, 0, zoneMillis, offset, hard);
        }
        if (kind.equals("TimeIntervalRounding")) {
            return new DateHistogramNode(agg, subs, field, UNIT_INTERVAL, (long) get(inner, "interval"), zoneMillis, offset, hard);
        }
        return null;
    }

    /** A filter's query blob, rewritten as the filter aggregators rewrite it, or null. */
    private static byte[] filterBlob(SearchContext ctx, Query q, Predicate<String> fieldOk) throws java.io.IOException {
        if (q == null) {
            return null;
        }
        return QueryEncoder.encode(ctx.searcher().rewrite(q), fieldOk).blob();
    }

    // ---- reflection ----

    private static Field field(Class<?> c, String name) {
        try {
            Field f = c.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** A field of {@code o}'s class or a superclass. */
    static Object get(Object o, String name) throws ReflectiveOperationException {
        for (Class<?> c = o.getClass(); c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f.get(o);
            } catch (NoSuchFieldException e) {
                // next superclass
            }
        }
        throw new NoSuchFieldException(o.getClass().getName() + "." + name);
    }

    private static Object declared(AggregatorFactory f, String name) throws ReflectiveOperationException {
        return get(f, name);
    }

    private static <T> Constructor<T> ctor(Class<T> c, Class<?>... args) {
        try {
            Constructor<T> k = c.getDeclaredConstructor(args);
            k.setAccessible(true);
            return k;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private static Class<?> cls(String c) {
        try {
            return Class.forName(c);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return Object.class;
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> Constructor<T> ctor(String c, Class<?>... args) {
        try {
            return ctor((Class<T>) Class.forName(c), args);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private static final Constructor<InternalAggregation> FILTER_CTOR = ctor(
        AGGS + "bucket.filter.InternalFilter",
        String.class,
        long.class,
        InternalAggregations.class,
        Map.class
    );
    private static final Constructor<InternalAggregation> GLOBAL_CTOR = ctor(
        AGGS + "bucket.global.InternalGlobal",
        String.class,
        long.class,
        InternalAggregations.class,
        Map.class
    );
    private static final Constructor<InternalAggregation> CARDINALITY_CTOR = ctor(
        AGGS + "metrics.InternalCardinality",
        String.class,
        org.opensearch.search.aggregations.metrics.AbstractHyperLogLogPlusPlus.class,
        Map.class
    );
    private static final Constructor<InternalDateHistogram> DATE_HISTOGRAM_CTOR = ctor(
        InternalDateHistogram.class,
        String.class,
        List.class,
        BucketOrder.class,
        long.class,
        long.class,
        cls(AGGS + "bucket.histogram.InternalDateHistogram$EmptyBucketInfo"),
        DocValueFormat.class,
        boolean.class,
        Map.class
    );
    private static final Constructor<Object> DATE_EMPTY_CTOR = ctor(
        AGGS + "bucket.histogram.InternalDateHistogram$EmptyBucketInfo",
        Rounding.class,
        InternalAggregations.class,
        LongBounds.class
    );
    private static final boolean CTORS_OK = FILTER_CTOR != null
        && GLOBAL_CTOR != null
        && CARDINALITY_CTOR != null
        && DATE_HISTOGRAM_CTOR != null
        && DATE_EMPTY_CTOR != null;

    // ---- blob writing ----

    static void writeString(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        NativeAggregations.writeInt(out, b.length);
        out.writeBytes(b);
    }

    static void writeLong(ByteArrayOutputStream out, long v) {
        for (int i = 0; i < 8; i++) {
            out.write((int) (v >>> (8 * i)));
        }
    }
}
