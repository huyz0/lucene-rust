/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.BigArrays;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.DocCountFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.search.aggregations.AggregationBuilder;
import org.opensearch.search.aggregations.BucketOrder;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.bucket.BucketUtils;
import org.opensearch.search.aggregations.bucket.terms.TermsAggregationBuilder;
import org.opensearch.search.aggregations.bucket.terms.TermsAggregator;
import org.opensearch.search.aggregations.metrics.AvgAggregationBuilder;
import org.opensearch.search.aggregations.metrics.ExtendedStatsAggregationBuilder;
import org.opensearch.search.aggregations.metrics.MaxAggregationBuilder;
import org.opensearch.search.aggregations.metrics.MinAggregationBuilder;
import org.opensearch.search.aggregations.metrics.StatsAggregationBuilder;
import org.opensearch.search.aggregations.metrics.SumAggregationBuilder;
import org.opensearch.search.aggregations.metrics.ValueCountAggregationBuilder;
import org.opensearch.search.aggregations.pipeline.PipelineAggregator;
import org.opensearch.search.aggregations.support.ValuesSourceAggregationBuilder;
import org.opensearch.search.builder.SearchSourceBuilder;

/**
 * The aggregations a direct search answers: a {@code size: 0} request whose aggregations are top-level
 * numeric metrics ({@code min max sum avg value_count stats extended_stats}) and {@code terms} on keyword fields,
 * with no script, {@code missing}, format or sub-aggregation. They are planned from the request's builders with
 * the same parameters the aggregator factories carry, run in the one native aggregation pass, and the shard
 * result is reduced and written as the coordinator would.
 */
final class DirectAggs {
    private static final BucketOrder DEFAULT_TERMS_ORDER = BucketOrder.compound(BucketOrder.count(false), BucketOrder.key(true));
    private static final int MAX_BUCKETS = 65535;

    private DirectAggs() {}

    /** The aggregations' JSON member, the total {value, lower bound} the pass counted. */
    record Result(String json, long total, boolean lowerBound) {}

    /** Whether the request's aggregations have a shape this path can plan (decided without the mappings). */
    static boolean shape(SearchSourceBuilder source) {
        if (source.aggregations() == null) {
            return true;
        }
        if (source.size() != 0 || source.aggregations().getPipelineAggregatorFactories().isEmpty() == false) {
            return false;
        }
        for (AggregationBuilder b : source.aggregations().getAggregatorFactories()) {
            if (b.getSubAggregations().isEmpty() == false || kind(b) == null && b instanceof TermsAggregationBuilder == false) {
                return false;
            }
        }
        return source.aggregations().getAggregatorFactories().isEmpty() == false;
    }

    /**
     * Whether OpenSearch could slice the search's segments across threads: unless concurrent segment
     * search is switched off ({@code mode: none}, on the index, else in the cluster or node settings),
     * its default {@code auto} mode runs aggregations concurrently, and a concurrent search reports
     * {@code terminated_early} on a size-0 count and a terms error bound per slice, which this path does
     * not reproduce.
     */
    static boolean slicing(Settings node, Settings cluster, Settings index) {
        String mode = firstMode(index, cluster, node);
        if (mode != null) {
            return mode.equals("none") == false;
        }
        for (Settings s : new Settings[] { index, cluster, node }) {
            String enabled = s.get("search.concurrent_segment_search.enabled");
            if (enabled != null) {
                return Boolean.parseBoolean(enabled);
            }
        }
        return true;
    }

    private static String firstMode(Settings... levels) {
        for (Settings s : levels) {
            String mode = s.get("index.search.concurrent_segment_search.mode", s.get("search.concurrent_segment_search.mode"));
            if (mode != null) {
                return mode;
            }
        }
        return null;
    }

    /** The metadata the aggregator factory carries: none when the request gave none. */
    private static Map<String, Object> metadata(AggregationBuilder b) {
        Map<String, Object> m = b.getMetadata();
        return m == null || m.isEmpty() ? null : m;
    }

    private static NativeAggregations.Kind kind(AggregationBuilder b) {
        if (b instanceof MinAggregationBuilder) return NativeAggregations.Kind.MIN;
        if (b instanceof MaxAggregationBuilder) return NativeAggregations.Kind.MAX;
        if (b instanceof SumAggregationBuilder) return NativeAggregations.Kind.SUM;
        if (b instanceof AvgAggregationBuilder) return NativeAggregations.Kind.AVG;
        if (b instanceof ValueCountAggregationBuilder) return NativeAggregations.Kind.VALUE_COUNT;
        if (b instanceof StatsAggregationBuilder) return NativeAggregations.Kind.STATS;
        if (b instanceof ExtendedStatsAggregationBuilder) return NativeAggregations.Kind.EXTENDED;
        return null;
    }

    /** The native plan of the request's aggregations, or null when one of them is not one this path answers. */
    static NativeAggregations.Plan plan(SearchSourceBuilder source, MapperService mappers, IndexReader reader) {
        List<Object> entries = new ArrayList<>();
        for (AggregationBuilder b : source.aggregations().getAggregatorFactories()) {
            if (b instanceof ValuesSourceAggregationBuilder<?> == false) {
                return null;
            }
            ValuesSourceAggregationBuilder<?> vb = (ValuesSourceAggregationBuilder<?>) b;
            if (vb.field() == null || vb.script() != null || vb.missing() != null || vb.format() != null || vb.timeZone() != null
                || vb.userValueTypeHint() != null) {
                return null;
            }
            MappedFieldType type = mappers.fieldType(vb.field());
            if (type == null || type.hasDocValues() == false || NativeAggregations.hasOwnDocValues(type) == false) {
                return null;
            }
            Object entry = b instanceof TermsAggregationBuilder t ? terms(t, type, reader) : metric(b, vb, type, source.query() == null);
            if (entry == null) {
                return null;
            }
            entries.add(entry);
        }
        return entries.isEmpty() ? null : new NativeAggregations.Plan(List.copyOf(entries));
    }

    private static NativeAggregations.Metric metric(AggregationBuilder b, ValuesSourceAggregationBuilder<?> vb, MappedFieldType type, boolean matchAll) {
        NativeAggregations.Kind kind = kind(b);
        byte valueKind = NativeAggregations.valueKind(type);
        if (kind == null || valueKind < 0) {
            return null;
        }
        byte source = NativeAggregations.DOC_VALUES;
        if ((kind == NativeAggregations.Kind.MIN || kind == NativeAggregations.Kind.MAX)
            && matchAll
            && valueKind <= NativeAggregations.FLOAT
            && type.isSearchable()) {
            source = kind == NativeAggregations.Kind.MIN ? NativeAggregations.POINTS_MIN : NativeAggregations.POINTS_MAX;
        }
        double sigma = b instanceof ExtendedStatsAggregationBuilder e ? e.sigma() : 0;
        return new NativeAggregations.Metric(
            b.getName(),
            kind,
            type.name(),
            valueKind,
            type.docValueFormat(null, null),
            metadata(b),
            source,
            NativeAggregations.Plan.needs(kind),
            sigma,
            NativeAggregations.scale(type)
        );
    }

    private static NativeAggregations.Terms terms(TermsAggregationBuilder t, MappedFieldType type, IndexReader reader) {
        if (type instanceof KeywordFieldMapper.KeywordFieldType == false
            || t.includeExclude() != null
            || t.showTermDocCountError()
            || DEFAULT_TERMS_ORDER.equals(t.order()) == false) {
            return null;
        }
        TermsAggregator.BucketCountThresholds thresholds = new TermsAggregator.BucketCountThresholds(
            t.minDocCount(), t.shardMinDocCount(), t.size(), t.shardSize()
        );
        if (thresholds.getShardSize() == -1) {
            thresholds.setShardSize(BucketUtils.suggestShardSideQueueSize(thresholds.getRequiredSize()));
        }
        thresholds.ensureValidity();
        if (thresholds.getMinDocCount() <= 0 || thresholds.getShardMinDocCount() != 0) {
            return null;
        }
        for (LeafReaderContext leaf : reader.leaves()) {
            if (leaf.reader().getFieldInfos().fieldInfo(DocCountFieldMapper.NAME) != null) {
                return null;
            }
        }
        return new NativeAggregations.Terms(t.getName(), type.name(), type.docValueFormat(null, null), metadata(t), t.order(), thresholds, false);
    }

    /**
     * Runs the plan's aggregations over the query blob in one native pass and writes them as OpenSearch's
     * response does ({@code params} the request's: {@code typed_keys}); null when the native engine does not answer.
     * {@code countLimit} above zero also counts the total, as the size-0 search does.
     */
    static Result run(long handle, byte[] blob, NativeAggregations.Plan plan, IndexReader reader, long countLimit, ToXContent.Params params)
        throws IOException {
        int[][] slices = new int[0][];
        int states = plan.metrics().size();
        long[] counts = new long[states];
        double[] values = new double[states * NativeAggregations.VALUES];
        byte[][] terms = new byte[1][];
        long[] total = new long[] { -1, 0, 0 };
        long[] perSegment = new long[reader.leaves().size()];
        int rc = NativeSearch.aggregate(handle, blob, plan.blob(slices), counts, values, terms, countLimit > 0 ? countLimit : -1, total, perSegment);
        if (rc != NativeBridge.OK || countLimit > 0 && total[0] < 0) {
            return null;
        }
        InternalAggregations shard = plan.build(slices, counts, values, terms[0], null);
        int[] buckets = new int[1];
        InternalAggregation.ReduceContext reduce = InternalAggregation.ReduceContext.forFinalReduction(
            BigArrays.NON_RECYCLING_INSTANCE,
            null,
            n -> {
                buckets[0] += n;
                if (buckets[0] > MAX_BUCKETS) {
                    throw new IllegalStateException("too many buckets");
                }
            },
            PipelineAggregator.PipelineTree.EMPTY
        );
        InternalAggregations reduced = InternalAggregations.topLevelReduce(List.of(shard), reduce);
        XContentBuilder b = org.opensearch.common.xcontent.json.JsonXContent.contentBuilder();
        b.startObject();
        reduced.toXContent(b, params);
        b.endObject();
        String json = b.toString();
        return new Result(json.substring(1, json.length() - 1), total[0], total[1] != 0);
    }
}
