/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

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
        byte[] blob() {
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
            return out.toByteArray();
        }

        /** The shard results, from the native per-field counts and values (see the class doc). */
        InternalAggregations build(long[] counts, double[] values) {
            List<InternalAggregation> aggs = new ArrayList<>(metrics.size());
            for (int i = 0; i < metrics.size(); i++) {
                Metric m = metrics.get(i);
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
            return InternalAggregations.from(aggs);
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

    /** The native plan for {@code ctx}'s aggregations, or null when any of them is not supported. */
    @SuppressWarnings("unchecked")
    public static Plan plan(SearchContext ctx) {
        if (ctx.aggregations() == null || METADATA == null || SUB_FACTORIES == null || CONFIG == null) {
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
                // AggregatorBase.pointReaderIfAvailable: a top-level min or max under a bare
                // match-all, on a field with points, reads each segment's bound off the points
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
