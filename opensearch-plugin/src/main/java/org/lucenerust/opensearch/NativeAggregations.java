/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
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
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

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

    /**
     * One aggregation. {@code minPoints} is set for a {@code min} that OpenSearch answers from the
     * field's points rather than its doc values (see {@link #plan}); null otherwise.
     */
    record Metric(
        String name,
        Kind kind,
        String field,
        byte valueKind,
        DocValueFormat format,
        Map<String, Object> metadata,
        Function<byte[], Number> minPoints
    ) {}

    /** The native aggregations of a search, or null when some aggregation must run on Lucene. */
    public record Plan(List<Metric> metrics) {
        byte[] blob() {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(metrics.size());
            for (Metric m : metrics) {
                out.write(m.valueKind());
                byte[] f = m.field().getBytes(StandardCharsets.UTF_8);
                for (int i = 0; i < 4; i++) {
                    out.write(f.length >>> (8 * i));
                }
                out.writeBytes(f);
            }
            return out.toByteArray();
        }

        /** The shard results, from the native per-field counts and values (see the class doc). */
        InternalAggregations build(IndexReader reader, long[] counts, double[] values) throws IOException {
            List<InternalAggregation> aggs = new ArrayList<>(metrics.size());
            for (int i = 0; i < metrics.size(); i++) {
                Metric m = metrics.get(i);
                long count = counts[i];
                double sum = values[i * VALUES];
                double min = values[i * VALUES + 2];
                double max = values[i * VALUES + 3];
                double minOfMins = values[i * VALUES + 4];
                if (m.minPoints() != null && Double.isNaN(minOfMins)) {
                    minOfMins = minFromPoints(reader, m);
                }
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

    /**
     * {@code MinAggregator}'s match-all answer: per segment the smallest live point ({@code
     * findLeafMinValue}), folded with {@code Math.min}. It differs from the doc-values answer only
     * over a {@code NaN} -- a document whose only value is {@code NaN} makes {@code Math.min} over
     * the documents {@code NaN}, while {@code NaN} sorts last among a double field's points -- so
     * {@link Plan#build} asks for it only when the native minimum is {@code NaN}.
     */
    static double minFromPoints(IndexReader reader, Metric m) throws IOException {
        double min = Double.POSITIVE_INFINITY;
        for (LeafReaderContext leaf : reader.leaves()) {
            Number segMin;
            try {
                segMin = (Number) FIND_LEAF_MIN.invoke(null, leaf.reader(), m.field(), m.minPoints());
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof IOException io) {
                    throw io;
                }
                throw new IllegalStateException(e.getCause());
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
            if (segMin != null) {
                min = Math.min(min, segMin.doubleValue());
            }
        }
        return min;
    }

    private static final String METRICS = "org.opensearch.search.aggregations.metrics.";
    private static final Method FIND_LEAF_MIN = findLeafMin();

    private static Method findLeafMin() {
        try {
            Method f = Class.forName(METRICS + "MinAggregator").getDeclaredMethod("findLeafMinValue", LeafReader.class, String.class, Function.class);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

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
                // AggregatorBase.pointReaderIfAvailable: a top-level min under a bare match-all
                // reads the points. Over longs and dates both answers agree, so only a double or
                // float field keeps the converter (for Plan.build's NaN case).
                Function<byte[], Number> minPoints = null;
                if (kind == Kind.MIN
                    && valueKind != LONG
                    && (ctx.query() == null || ctx.query().getClass() == MatchAllDocsQuery.class)) {
                    minPoints = config.getPointReaderOrNull();
                    if (minPoints != null && FIND_LEAF_MIN == null) {
                        return null;
                    }
                }
                metrics.add(
                    new Metric(
                        f.name(),
                        kind,
                        config.fieldContext().field(),
                        valueKind,
                        config.format(),
                        (Map<String, Object>) METADATA.get(f),
                        minPoints
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
