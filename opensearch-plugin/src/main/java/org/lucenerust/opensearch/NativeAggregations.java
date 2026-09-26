/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.opensearch.index.mapper.DateFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.aggregations.AggregatorFactories;
import org.opensearch.search.aggregations.AggregatorFactory;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.metrics.InternalAvg;
import org.opensearch.search.aggregations.metrics.InternalMax;
import org.opensearch.search.aggregations.metrics.InternalMin;
import org.opensearch.search.aggregations.metrics.InternalStats;
import org.opensearch.search.aggregations.metrics.InternalSum;
import org.opensearch.search.aggregations.metrics.InternalValueCount;
import org.opensearch.search.aggregations.support.ValuesSourceAggregatorFactory;
import org.opensearch.search.aggregations.support.ValuesSourceConfig;
import org.opensearch.search.internal.SearchContext;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The aggregations of a search that run natively (read path R5): top-level {@code min}, {@code
 * max}, {@code sum}, {@code avg}, {@code value_count} and {@code stats} on a mapped numeric or date
 * field, without {@code missing}, scripts or sub-aggregations. {@link #plan} reads OpenSearch's own
 * aggregator factories -- so the name, metadata and resolved {@link DocValueFormat} are the ones its
 * aggregators would have used -- {@link Plan#blob} is the metrics blob {@code decode_metrics} in
 * {@code crates/lucene-ffi/src/jvm_reader.rs} reads, and {@link Plan#build} makes the shard results
 * those aggregators build ({@code InternalMin} and the rest) from the native state.
 */
public final class NativeAggregations {
    static final byte LONG = 0;
    static final byte DOUBLE = 1;
    static final byte FLOAT = 2;
    /** Doubles per field in the native output ({@code METRIC_VALUES}). */
    static final int VALUES = 6;

    enum Kind {
        MIN,
        MAX,
        SUM,
        AVG,
        VALUE_COUNT,
        STATS
    }

    /** Where a metric's minimum or maximum comes from ({@code aggs::Source} in Rust). */
    static final byte DOC_VALUES = 0;
    static final byte POINTS_MIN = 1;
    static final byte POINTS_MAX = 2;

    /** One aggregation; {@code source} is {@link #DOC_VALUES} or, see {@link #plan}, a points bound. */
    record Metric(
        String name,
        Kind kind,
        String field,
        byte valueKind,
        DocValueFormat format,
        Map<String, Object> metadata,
        byte source
    ) {}

    /** The native aggregations of a search, or null when some aggregation must run on Lucene. */
    public record Plan(List<Metric> metrics) {
        /** The metrics blob; {@code slices} as {@link NativeAggregations#slices} returns them. */
        byte[] blob(int[][] slices) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(metrics.size());
            for (Metric m : metrics) {
                out.write(m.valueKind());
                out.write(m.source());
                byte[] f = m.field().getBytes(StandardCharsets.UTF_8);
                for (int i = 0; i < 4; i++) {
                    out.write(f.length >>> (8 * i));
                }
                out.writeBytes(f);
            }
            writeSlices(out, slices);
            return out.toByteArray();
        }

        /**
         * The shard results from the native output: one set per slice (a single one when {@code
         * slices} is empty), and with slices reduced as {@code NonGlobalAggCollectorManager}
         * reduces its collectors' -- each slice's result carrying its sum without the delta.
         */
        InternalAggregations build(int[][] slices, long[] counts, double[] values, InternalAggregation.ReduceContext onShard) {
            if (slices.length == 0) {
                return InternalAggregations.from(build(counts, values, 0));
            }
            List<InternalAggregation> all = new ArrayList<>(metrics.size() * slices.length);
            for (int s = 0; s < slices.length; s++) {
                all.addAll(build(counts, values, s * metrics.size()));
            }
            return InternalAggregations.reduce(List.of(InternalAggregations.from(all)), onShard);
        }

        /** One slice's results, from the native counts and values starting at field {@code at}. */
        List<InternalAggregation> build(long[] counts, double[] values, int at) {
            List<InternalAggregation> aggs = new ArrayList<>(metrics.size());
            for (int k = 0; k < metrics.size(); k++) {
                Metric m = metrics.get(k);
                int i = at + k;
                long count = counts[i];
                double sum = values[i * VALUES];
                double min = values[i * VALUES + 2];
                double max = values[i * VALUES + 3];
                double minOfMins = values[i * VALUES + 4];
                double maxOfMaxes = values[i * VALUES + 5];
                aggs.add(switch (m.kind()) {
                    case MIN -> new InternalMin(m.name(), minOfMins, m.format(), m.metadata());
                    case MAX -> new InternalMax(m.name(), maxOfMaxes, m.format(), m.metadata());
                    case SUM -> new InternalSum(m.name(), sum, m.format(), m.metadata());
                    case AVG -> new InternalAvg(m.name(), sum, count, m.format(), m.metadata());
                    case VALUE_COUNT -> new InternalValueCount(m.name(), count, m.metadata());
                    case STATS -> new InternalStats(m.name(), count, sum, min, max, m.format(), m.metadata());
                });
            }
            return aggs;
        }
    }

    private static final String METRICS = "org.opensearch.search.aggregations.metrics.";
    private static final Field METADATA = field(AggregatorFactory.class, "metadata");
    private static final Field SUB_FACTORIES = field(AggregatorFactory.class, "factories");
    private static final Field CONFIG = field(ValuesSourceAggregatorFactory.class, "config");

    private static Field field(Class<?> c, String name) {
        try {
            Field f = c.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private NativeAggregations() {}

    /** The slices section ending a metrics or sort blob ({@code decode_slices} in Rust). */
    static void writeSlices(ByteArrayOutputStream out, int[][] slices) {
        writeInt(out, slices.length);
        for (int[] slice : slices) {
            writeInt(out, slice.length);
            for (int segment : slice) {
                writeInt(out, segment);
            }
        }
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        for (int i = 0; i < 4; i++) {
            out.write(v >>> (8 * i));
        }
    }

    /**
     * The segments (leaf ordinals) of each slice a concurrent segment search collects, in the order
     * its collector visits them; empty when the search is not concurrent; null when a slice holds
     * only part of a segment (intra-segment search), which the native pass cannot split.
     *
     * <p>A concurrent search gives every slice its own aggregators and reduces their results on the
     * shard, so a sum's rounding depends on the slices; the native pass reproduces them.
     */
    static int[][] slices(SearchContext ctx) {
        if (ctx.shouldUseConcurrentSearch() == false) {
            return new int[0][];
        }
        IndexSearcher.LeafSlice[] leafSlices = ctx.searcher().getSlices();
        int[][] out = new int[leafSlices.length][];
        for (int s = 0; s < leafSlices.length; s++) {
            IndexSearcher.LeafReaderContextPartition[] parts = leafSlices[s].partitions;
            out[s] = new int[parts.length];
            for (int p = 0; p < parts.length; p++) {
                if (parts[p].minDocId != 0 || parts[p].maxDocId < parts[p].ctx.reader().maxDoc()) {
                    return null;
                }
                out[s][p] = parts[p].ctx.ord;
            }
        }
        return out;
    }

    /** The native plan for {@code ctx}'s aggregations, or null when any of them is not supported. */
    @SuppressWarnings("unchecked")
    public static Plan plan(SearchContext ctx) {
        if (ctx.aggregations() == null || METADATA == null || SUB_FACTORIES == null || CONFIG == null) {
            return null;
        }
        // A star-tree index answers min/max/sum/avg from its pre-aggregated tree
        // (tryPrecomputeAggregationForLeaf), whose sums round differently.
        if (ctx.getQueryShardContext() != null && ctx.getQueryShardContext().getStarTreeQueryContext() != null) {
            return null;
        }
        AggregatorFactories factories = ctx.aggregations().factories();
        if (factories.hasGlobalAggregator()) {
            return null;
        }
        List<Metric> metrics = new ArrayList<>();
        try {
            for (AggregatorFactory f : factories.getFactories()) {
                Kind kind = kind(f);
                if (kind == null || ((AggregatorFactories) SUB_FACTORIES.get(f)).countAggregators() != 0) {
                    return null;
                }
                ValuesSourceConfig config = (ValuesSourceConfig) CONFIG.get(f);
                if (config == null || config.script() != null || config.missing() != null || config.fieldContext() == null) {
                    return null;
                }
                byte valueKind = valueKind(config.fieldContext().fieldType());
                if (valueKind < 0) {
                    return null;
                }
                // AggregatorBase.pointReaderIfAvailable: a top-level min or max whose query is a
                // bare MatchAllDocsQuery (a request without a query; an explicit match_all is an
                // ApproximateScoreQuery in 3.8), on a field with points, reads each segment's bound
                // off the points
                // (tryPrecomputeAggregationForLeaf) -- faster, and over a double field with a NaN
                // document a different answer (NaN sorts last among the points).
                byte source = DOC_VALUES;
                if ((kind == Kind.MIN || kind == Kind.MAX)
                    && (ctx.query() == null || ctx.query().getClass() == MatchAllDocsQuery.class)
                    && config.getPointReaderOrNull() != null) {
                    source = kind == Kind.MIN ? POINTS_MIN : POINTS_MAX;
                }
                metrics.add(
                    new Metric(
                        f.name(),
                        kind,
                        config.fieldContext().field(),
                        valueKind,
                        config.format(),
                        (Map<String, Object>) METADATA.get(f),
                        source
                    )
                );
            }
        } catch (IllegalAccessException | RuntimeException e) {
            return null;
        }
        return metrics.isEmpty() ? null : new Plan(List.copyOf(metrics));
    }

    /** The factories' classes are package-private, so they are told apart by name. */
    private static Kind kind(AggregatorFactory f) {
        return switch (f.getClass().getName()) {
            case METRICS + "MinAggregatorFactory" -> Kind.MIN;
            case METRICS + "MaxAggregatorFactory" -> Kind.MAX;
            case METRICS + "SumAggregatorFactory" -> Kind.SUM;
            case METRICS + "AvgAggregatorFactory" -> Kind.AVG;
            case METRICS + "ValueCountAggregatorFactory" -> Kind.VALUE_COUNT;
            case METRICS + "StatsAggregatorFactory" -> Kind.STATS;
            default -> null;
        };
    }

    /** How the field's stored longs become the doubles its aggregators read, or -1. */
    static byte valueKind(MappedFieldType type) {
        if (type instanceof NumberFieldMapper.NumberFieldType n) {
            return switch (n.numberType()) {
                case LONG, INTEGER, SHORT, BYTE -> LONG;
                case DOUBLE -> DOUBLE;
                case FLOAT -> FLOAT;
                default -> -1;
            };
        }
        if (type instanceof DateFieldMapper.DateFieldType d && d.resolution() == DateFieldMapper.Resolution.MILLISECONDS) {
            return LONG;
        }
        return -1;
    }
}
