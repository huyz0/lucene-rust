/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.mapper.DateFieldMapper;
import org.opensearch.index.mapper.DocCountFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.search.aggregations.BucketOrder;
import org.opensearch.search.aggregations.bucket.BucketUtils;
import org.opensearch.search.aggregations.bucket.terms.StringTerms;
import org.opensearch.search.aggregations.bucket.terms.TermsAggregator;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    /**
     * A {@code terms} aggregation on a keyword field: OpenSearch's own effective thresholds (the
     * {@code shard_size} heuristic and {@code ensureValidity} applied, as {@code
     * TermsAggregatorFactory.doCreateInternal} applies them), order, format and metadata.
     */
    record Terms(
        String name,
        String field,
        DocValueFormat format,
        Map<String, Object> metadata,
        BucketOrder order,
        TermsAggregator.BucketCountThresholds thresholds,
        boolean showTermDocCountError
    ) {
        /** The terms blob ({@code decode_terms_spec} in Rust). */
        byte[] blob(int[][] slices) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] f = field.getBytes(StandardCharsets.UTF_8);
            writeInt(out, f.length);
            out.writeBytes(f);
            writeInt(out, thresholds.getShardSize());
            writeSlices(out, slices);
            return out.toByteArray();
        }

        /**
         * One slice's {@code StringTerms} from the native result at {@code in}: what {@code
         * StandardTermsResults.buildResult} builds -- reduce order {@code KEY_ASC}, the buckets by
         * term, no error.
         */
        StringTerms read(ByteBuffer in) {
            long other = in.getLong();
            int n = in.getInt();
            List<StringTerms.Bucket> buckets = new ArrayList<>(n);
            for (int b = 0; b < n; b++) {
                long docCount = in.getLong();
                byte[] term = new byte[in.getInt()];
                in.get(term);
                buckets.add(
                    new StringTerms.Bucket(new BytesRef(term), docCount, InternalAggregations.EMPTY, showTermDocCountError, 0, format)
                );
            }
            return new StringTerms(
                name,
                BucketOrder.key(true),
                order,
                metadata,
                format,
                thresholds.getShardSize(),
                showTermDocCountError,
                other,
                buckets,
                0,
                thresholds
            );
        }
    }

    /**
     * The native aggregations of a search, or null when some aggregation must run on Lucene:
     * {@code entries} in the request's order, each a {@link Metric} or a {@link Terms}.
     */
    public record Plan(List<Object> entries) {
        List<Metric> metrics() {
            return entries.stream().filter(e -> e instanceof Metric).map(e -> (Metric) e).toList();
        }

        List<Terms> terms() {
            return entries.stream().filter(e -> e instanceof Terms).map(e -> (Terms) e).toList();
        }

        /** The metrics blob; {@code slices} as {@link NativeAggregations#slices} returns them. */
        byte[] blob(int[][] slices) {
            List<Metric> metrics = metrics();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(metrics.size());
            for (Metric m : metrics) {
                out.write(m.valueKind());
                out.write(m.source());
                byte[] f = m.field().getBytes(StandardCharsets.UTF_8);
                writeInt(out, f.length);
                out.writeBytes(f);
            }
            writeSlices(out, slices);
            return out.toByteArray();
        }

        /**
         * The shard results from the native output: one set per slice (a single one when {@code
         * slices} is empty), and with slices reduced as {@code NonGlobalAggCollectorManager}
         * reduces its collectors' -- each slice's sum without its delta, each slice's terms
         * through {@code InternalTerms.reduce}. {@code terms} holds each terms entry's encoded
         * result, in entry order.
         */
        InternalAggregations build(
            int[][] slices,
            long[] counts,
            double[] values,
            List<byte[]> terms,
            InternalAggregation.ReduceContext onShard
        ) {
            int sliceCount = Math.max(1, slices.length);
            List<ByteBuffer> termsIn = terms.stream().map(b -> ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)).toList();
            int metricCount = metrics().size();
            List<InternalAggregation> all = new ArrayList<>(entries.size() * sliceCount);
            for (int s = 0; s < sliceCount; s++) {
                int metric = 0;
                int term = 0;
                for (Object e : entries) {
                    if (e instanceof Metric m) {
                        all.add(metric(m, counts, values, s * metricCount + metric++));
                    } else {
                        all.add(((Terms) e).read(termsIn.get(term++)));
                    }
                }
            }
            if (slices.length == 0) {
                return InternalAggregations.from(all);
            }
            return InternalAggregations.reduce(List.of(InternalAggregations.from(all)), onShard);
        }

        /** A metric's result from the native counts and values of state {@code i}. */
        static InternalAggregation metric(Metric m, long[] counts, double[] values, int i) {
            long count = counts[i];
            double sum = values[i * VALUES];
            double min = values[i * VALUES + 2];
            double max = values[i * VALUES + 3];
            double minOfMins = values[i * VALUES + 4];
            double maxOfMaxes = values[i * VALUES + 5];
            return switch (m.kind()) {
                case MIN -> new InternalMin(m.name(), minOfMins, m.format(), m.metadata());
                case MAX -> new InternalMax(m.name(), maxOfMaxes, m.format(), m.metadata());
                case SUM -> new InternalSum(m.name(), sum, m.format(), m.metadata());
                case AVG -> new InternalAvg(m.name(), sum, count, m.format(), m.metadata());
                case VALUE_COUNT -> new InternalValueCount(m.name(), count, m.metadata());
                case STATS -> new InternalStats(m.name(), count, sum, min, max, m.format(), m.metadata());
            };
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
        List<Object> entries = new ArrayList<>();
        try {
            for (AggregatorFactory f : factories.getFactories()) {
                if (((AggregatorFactories) SUB_FACTORIES.get(f)).countAggregators() != 0) {
                    return null;
                }
                ValuesSourceConfig config = (ValuesSourceConfig) CONFIG.get(f);
                if (config == null || config.script() != null || config.missing() != null || config.fieldContext() == null) {
                    return null;
                }
                if (f.getClass().getName().equals(TERMS_FACTORY)) {
                    Terms t = terms(ctx, f, config);
                    if (t == null) {
                        return null;
                    }
                    entries.add(t);
                    continue;
                }
                Kind kind = kind(f);
                if (kind == null) {
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
                entries.add(
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
        return entries.isEmpty() ? null : new Plan(List.copyOf(entries));
    }

    private static final String TERMS_FACTORY = "org.opensearch.search.aggregations.bucket.terms.TermsAggregatorFactory";
    private static final BucketOrder DEFAULT_TERMS_ORDER = BucketOrder.compound(BucketOrder.count(false), BucketOrder.key(true));

    /**
     * The native plan of a {@code terms} factory, or null: a keyword field, no include/exclude, the
     * default order, {@code min_doc_count} at least 1 and {@code shard_min_doc_count} 0 (so only
     * counted terms are candidates, and all of them), and no {@code _doc_count} field in the
     * index (whose documents count as many).
     */
    private static Terms terms(SearchContext ctx, AggregatorFactory f, ValuesSourceConfig config) throws IllegalAccessException {
        if (config.fieldContext().fieldType() instanceof KeywordFieldMapper.KeywordFieldType == false) {
            return null;
        }
        Object include = declared(f, "includeExclude");
        BucketOrder order = (BucketOrder) declared(f, "order");
        TermsAggregator.BucketCountThresholds declaredThresholds = (TermsAggregator.BucketCountThresholds) declared(
            f,
            "bucketCountThresholds"
        );
        Boolean showError = (Boolean) declared(f, "showTermDocCountError");
        if (include != null || DEFAULT_TERMS_ORDER.equals(order) == false || declaredThresholds == null || showError == null) {
            return null;
        }
        // TermsAggregatorFactory.doCreateInternal: the shard_size heuristic, then ensureValidity.
        TermsAggregator.BucketCountThresholds thresholds = new TermsAggregator.BucketCountThresholds(declaredThresholds);
        if (thresholds.getShardSize() == -1) {
            thresholds.setShardSize(BucketUtils.suggestShardSideQueueSize(thresholds.getRequiredSize()));
        }
        thresholds.ensureValidity();
        if (thresholds.getMinDocCount() < 1 || thresholds.getShardMinDocCount() != 0) {
            return null;
        }
        for (LeafReaderContext leaf : ctx.searcher().getIndexReader().leaves()) {
            if (leaf.reader().getFieldInfos().fieldInfo(DocCountFieldMapper.NAME) != null) {
                return null;
            }
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) METADATA.get(f);
        return new Terms(f.name(), config.fieldContext().field(), config.format(), metadata, order, thresholds, showError);
    }

    /** A private field of {@code f}'s own class, or null when it has none. */
    private static Object declared(AggregatorFactory f, String name) throws IllegalAccessException {
        try {
            Field field = f.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return field.get(f);
        } catch (NoSuchFieldException e) {
            return null;
        }
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
