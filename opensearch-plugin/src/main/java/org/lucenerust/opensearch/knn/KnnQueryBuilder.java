/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.lucenerust.opensearch.knn.rest.KnnStats;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.common.lucene.search.Queries;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.Version;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilderVisitor;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * k-NN's {@code knn} query ({@code KNNQueryBuilder}): {@code vector} with one of {@code k}, {@code
 * max_distance} or {@code min_score}, and {@code filter}, {@code method_parameters.ef_search},
 * {@code rescore}, {@code expand_nested_docs}, {@code ignore_unmapped}, {@code boost} and {@code
 * _name}. A top-{@code k} search is the shard's top {@code k} ({@link KnnTopVectorQuery}); a radial
 * search runs segment by segment ({@link KnnLeafVectorQuery}). The vectors are searched at full precision,
 * so {@code rescore} has nothing to correct and is accepted as is.
 */
public final class KnnQueryBuilder extends AbstractQueryBuilder<KnnQueryBuilder> {
    public static final String NAME = "knn";
    public static final int MAX_K = 10_000;

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField K_FIELD = new ParseField("k");
    public static final ParseField FILTER_FIELD = new ParseField("filter");
    public static final ParseField MAX_DISTANCE_FIELD = new ParseField("max_distance");
    public static final ParseField MIN_SCORE_FIELD = new ParseField("min_score");
    public static final ParseField METHOD_PARAMETERS_FIELD = new ParseField("method_parameters");
    public static final ParseField METHOD_PARAMS_FIELD = METHOD_PARAMETERS_FIELD;
    public static final ParseField RESCORE_FIELD = new ParseField("rescore");
    public static final ParseField OVERSAMPLE_FIELD = new ParseField("oversample_factor");
    public static final ParseField EXPAND_NESTED_FIELD = new ParseField("expand_nested_docs");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");

    private String fieldName;
    private float[] vector;
    private Integer k;
    private Float maxDistance;
    private Float minScore;
    private Integer efSearch;
    private Integer nprobes;
    private QueryBuilder filter;
    /** {@code rescore}: null unset, false off, true on (with {@link #oversample} when given). */
    private Boolean rescore;
    private Float oversample;
    private Boolean expandNested;
    private boolean ignoreUnmapped;

    private KnnQueryBuilder() {}

    public KnnQueryBuilder(String fieldName, float[] vector, int k) {
        this.fieldName = Objects.requireNonNull(fieldName);
        this.vector = Objects.requireNonNull(vector);
        this.k = k;
    }

    public KnnQueryBuilder(StreamInput in) throws IOException {
        super(in);
        fieldName = in.readString();
        vector = in.readFloatArray();
        k = in.readOptionalInt();
        maxDistance = in.readOptionalFloat();
        minScore = in.readOptionalFloat();
        efSearch = in.readOptionalInt();
        nprobes = in.readOptionalInt();
        filter = in.readOptionalNamedWriteable(QueryBuilder.class);
        rescore = in.readOptionalBoolean();
        oversample = in.readOptionalFloat();
        expandNested = in.readOptionalBoolean();
        ignoreUnmapped = in.readBoolean();
    }

    /**
     * This query's own fields without the boost and name, as {@code KNNQueryBuilderParser.streamOutput}
     * writes them for a query that wraps a knn query; {@link #KnnQueryBuilder(StreamInput)} reads them
     * back after the wrapper's boost and name.
     */
    public void writeFieldsTo(StreamOutput out) throws IOException {
        doWriteTo(out);
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeFloatArray(vector);
        out.writeOptionalInt(k);
        out.writeOptionalFloat(maxDistance);
        out.writeOptionalFloat(minScore);
        out.writeOptionalInt(efSearch);
        out.writeOptionalInt(nprobes);
        out.writeOptionalNamedWriteable(filter);
        out.writeOptionalBoolean(rescore);
        out.writeOptionalFloat(oversample);
        out.writeOptionalBoolean(expandNested);
        out.writeBoolean(ignoreUnmapped);
    }

    /**
     * {@code KNNQueryBuilder.filter}: a new query with {@code more} added to its filter (as a bool
     * query's filter when it has one), so the knn search is filtered rather than its hits. Like
     * stock's, the new query has the default boost and no name.
     */
    @Override
    public KnnQueryBuilder filter(QueryBuilder more) {
        if (validateFilterParams(more) == false) {
            return this;
        }
        KnnQueryBuilder c = copy();
        c.filter = filter == null ? more : unshared(filter).filter(more);
        c.boost(DEFAULT_BOOST).queryName(null);
        return c;
    }

    /**
     * {@code filter}, or a copy of it when it is a bool query: a bool query's {@code filter} adds the
     * clause in place, and this query's own filter must not change.
     */
    private static QueryBuilder unshared(QueryBuilder filter) {
        if (filter instanceof org.opensearch.index.query.BoolQueryBuilder b) {
            org.opensearch.index.query.BoolQueryBuilder copy = new org.opensearch.index.query.BoolQueryBuilder();
            b.must().forEach(copy::must);
            b.should().forEach(copy::should);
            b.mustNot().forEach(copy::mustNot);
            b.filter().forEach(copy::filter);
            copy.minimumShouldMatch(b.minimumShouldMatch());
            copy.adjustPureNegative(b.adjustPureNegative());
            copy.boost(b.boost()).queryName(b.queryName());
            return copy;
        }
        return filter;
    }

    /** {@code KNNQueryBuilder.visit}: this query, then its filter as a filter clause. */
    @Override
    public void visit(QueryBuilderVisitor visitor) {
        visitor.accept(this);
        if (filter != null) {
            filter.visit(visitor.getChildVisitor(org.apache.lucene.search.BooleanClause.Occur.FILTER));
        }
    }

    public KnnQueryBuilder efSearch(int efSearch) {
        this.efSearch = efSearch;
        return this;
    }

    public String fieldName() {
        return fieldName;
    }

    public Float getMaxDistance() {
        return maxDistance;
    }

    public Float getMinScore() {
        return minScore;
    }

    /** k-NN's {@code setK}: MMR widens a query's k to its candidate count before the search runs. */
    public void setK(int k) {
        this.k = k;
    }

    public Integer getK() {
        return k;
    }

    /** This query's body, for a query that renders a knn query as its own ({@code NeuralKNNQueryBuilder}). */
    public void doXContentBody(XContentBuilder builder, Params params) throws IOException {
        doXContent(builder, params);
    }

    public static Builder builder() {
        return new Builder();
    }

    /** {@code KNNQueryBuilder.Builder}: a knn query from options given in code, checked as stock checks them. */
    public static final class Builder {
        private String fieldName;
        private float[] vector;
        private Integer k;
        private Map<String, ?> methodParameters;
        private Float maxDistance;
        private Float minScore;
        private QueryBuilder filter;
        private boolean ignoreUnmapped;
        private String queryName;
        private float boost = DEFAULT_BOOST;
        private RescoreContext rescoreContext;
        private Boolean expandNested;

        public Builder fieldName(String fieldName) {
            this.fieldName = fieldName;
            return this;
        }

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder k(Integer k) {
            this.k = k;
            return this;
        }

        public Builder methodParameters(Map<String, ?> methodParameters) {
            this.methodParameters = methodParameters;
            return this;
        }

        public Builder maxDistance(Float maxDistance) {
            this.maxDistance = maxDistance;
            return this;
        }

        public Builder minScore(Float minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder ignoreUnmapped(boolean ignoreUnmapped) {
            this.ignoreUnmapped = ignoreUnmapped;
            return this;
        }

        public Builder filter(QueryBuilder filter) {
            this.filter = filter;
            return this;
        }

        public Builder queryName(String queryName) {
            this.queryName = queryName;
            return this;
        }

        public Builder boost(float boost) {
            this.boost = boost;
            return this;
        }

        public Builder rescoreContext(RescoreContext rescoreContext) {
            this.rescoreContext = rescoreContext;
            return this;
        }

        public Builder expandNested(Boolean expandNested) {
            this.expandNested = expandNested;
            return this;
        }

        public KnnQueryBuilder build() {
            validate();
            KnnQueryBuilder b = new KnnQueryBuilder();
            b.fieldName = fieldName;
            b.vector = vector;
            b.k = k;
            b.maxDistance = maxDistance;
            b.minScore = minScore;
            b.filter = filter;
            b.ignoreUnmapped = ignoreUnmapped;
            b.expandNested = expandNested;
            if (methodParameters != null) {
                Object ef = methodParameters.get("ef_search");
                Object np = methodParameters.get("nprobes");
                b.efSearch = ef == null ? null : ((Number) ef).intValue();
                b.nprobes = np == null ? null : ((Number) np).intValue();
            }
            if (rescoreContext != null) {
                b.rescore = rescoreContext.isRescoreEnabled();
                b.oversample = rescoreContext.isRescoreEnabled() ? rescoreContext.getOversampleFactor() : null;
            }
            b.boost(boost).queryName(queryName);
            return b;
        }

        private void validate() {
            if (fieldName == null || fieldName.isEmpty()) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
            }
            if (vector == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
            } else if (vector.length == 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
            }
            int set = (k != null ? 1 : 0) + (maxDistance != null ? 1 : 0) + (minScore != null ? 1 : 0);
            if (set != 1) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME));
            }
            if (k != null && (k <= 0 || k > MAX_K)) {
                throw new IllegalArgumentException("[" + NAME + "] requires k to be in the range (0, " + MAX_K + "]");
            }
            if (minScore != null && minScore <= 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires minScore to be greater than 0", NAME));
            }
            if (methodParameters != null) {
                org.opensearch.common.ValidationException e = KnnQueryParsers.validateMethodParameters(methodParameters);
                if (e != null) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "[%s] errors in method parameter [%s]", NAME, e.getMessage())
                    );
                }
            }
            if (rescoreContext != null) {
                org.opensearch.common.ValidationException e = KnnQueryParsers.validateRescore(rescoreContext);
                if (e != null) {
                    throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] errors in rescore parameter [%s]", NAME, e.getMessage()));
                }
            }
        }
    }

    /** {@code KNNQueryBuilderParser}'s object parser: a field's parameters. */
    private static final ObjectParser<KnnQueryBuilder, Void> PARSER = new ObjectParser<>(NAME, KnnQueryBuilder::new);
    /** {@code RescoreParser}'s: {@code oversample_factor}, unset when absent. */
    private static final ObjectParser<Float[], Void> RESCORE_PARSER = new ObjectParser<>(RESCORE_FIELD.getPreferredName(), () -> new Float[1]);

    static {
        PARSER.declareFloat(KnnQueryBuilder::boost, BOOST_FIELD);
        PARSER.declareString(KnnQueryBuilder::queryName, NAME_FIELD);
        PARSER.declareFloatArray((b, v) -> b.vector = floats(v), VECTOR_FIELD);
        PARSER.declareInt((b, v) -> b.k = v, K_FIELD);
        PARSER.declareBoolean((b, v) -> b.ignoreUnmapped = v, IGNORE_UNMAPPED_FIELD);
        PARSER.declareFloat((b, v) -> b.maxDistance = v, MAX_DISTANCE_FIELD);
        PARSER.declareFloat((b, v) -> b.minScore = v, MIN_SCORE_FIELD);
        PARSER.declareObject((b, v) -> b.methodParameters = v, (p, c) -> KnnQueryParsers.methodParametersFromXContent(p), METHOD_PARAMETERS_FIELD);
        PARSER.declareObject((b, v) -> b.filter = v, (p, c) -> parseInnerQueryBuilder(p), FILTER_FIELD);
        PARSER.declareField((p, b, c) -> {
            if (p.currentToken() == XContentParser.Token.VALUE_BOOLEAN) {
                b.rescore = p.booleanValue();
                b.oversample = null;
            } else {
                b.rescore = true;
                b.oversample = RESCORE_PARSER.apply(p, null)[0];
            }
        }, RESCORE_FIELD, ObjectParser.ValueType.OBJECT_OR_BOOLEAN);
        PARSER.declareBoolean((b, v) -> b.expandNested = v, EXPAND_NESTED_FIELD);
        RESCORE_PARSER.declareFloat((f, v) -> f[0] = v, OVERSAMPLE_FIELD);
    }

    /** Method parameters parsed but not yet checked: stock checks them after k, distance and score. */
    private Map<String, ?> methodParameters;

    /** {@code KNNQueryBuilderParser.fromXContent}: one field, its parameters an object or the vector alone. */
    public static KnnQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String fieldName = null;
        String current = null;
        XContentParser.Token token;
        KnnQueryBuilder b = null;
        List<Object> vector = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                multipleFields(parser, fieldName, current);
                fieldName = current;
                b = PARSER.apply(parser, null);
            } else {
                multipleFields(parser, fieldName, parser.currentName());
                fieldName = parser.currentName();
                vector = parser.list();
            }
        }
        if (b == null) {
            b = new KnnQueryBuilder();
            b.vector = objects(vector);
        }
        b.fieldName = fieldName;
        b.validate();
        return b;
    }

    private static void multipleFields(XContentParser parser, String fieldName, String current) {
        if (fieldName != null) {
            throw new ParsingException(
                parser.getTokenLocation(),
                "[" + NAME + "] query doesn't support multiple fields, found [" + fieldName + "] and [" + current + "]"
            );
        }
    }

    private static float[] floats(List<Float> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] field 'vector' requires to be non-null and non-empty", NAME));
        }
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = values.get(i);
        }
        return v;
    }

    private static float[] objects(List<Object> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] field 'vector' requires to be non-null and non-empty", NAME));
        }
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            if (values.get(i) instanceof Number n) {
                v[i] = n.floatValue();
            } else {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] field 'vector' requires to be an array of numbers", NAME));
            }
        }
        return v;
    }

    /** {@code KNNQueryBuilder.Builder.validate}, in its order. */
    private void validate() {
        if (fieldName == null || fieldName.isEmpty()) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
        } else if (vector.length == 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
        }
        int set = (k != null ? 1 : 0) + (maxDistance != null ? 1 : 0) + (minScore != null ? 1 : 0);
        if (set != 1) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME));
        }
        if (k != null && (k <= 0 || k > MAX_K)) {
            throw new IllegalArgumentException("[" + NAME + "] requires k to be in the range (0, " + MAX_K + "]");
        }
        if (minScore != null && minScore <= 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires minScore to be greater than 0", NAME));
        }
        if (methodParameters != null) {
            org.opensearch.common.ValidationException e = KnnQueryParsers.validateMethodParameters(methodParameters);
            if (e != null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] errors in method parameter [%s]", NAME, e.getMessage()));
            }
            Object ef = methodParameters.get("ef_search");
            Object np = methodParameters.get("nprobes");
            efSearch = ef == null ? null : ((Number) ef).intValue();
            nprobes = np == null ? null : ((Number) np).intValue();
            methodParameters = null;
        }
        if (Boolean.TRUE.equals(rescore) && oversample != null) {
            org.opensearch.common.ValidationException e = KnnQueryParsers.validateRescore(new RescoreContext(oversample, true));
            if (e != null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] errors in rescore parameter [%s]", NAME, e.getMessage()));
            }
        }
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);
        builder.field(VECTOR_FIELD.getPreferredName(), vector);
        if (k != null) {
            builder.field(K_FIELD.getPreferredName(), k);
        }
        if (maxDistance != null) {
            builder.field(MAX_DISTANCE_FIELD.getPreferredName(), maxDistance);
        }
        if (minScore != null) {
            builder.field(MIN_SCORE_FIELD.getPreferredName(), minScore);
        }
        if (filter != null) {
            builder.field(FILTER_FIELD.getPreferredName(), filter);
        }
        if (efSearch != null || nprobes != null) {
            builder.startObject(METHOD_PARAMETERS_FIELD.getPreferredName());
            if (efSearch != null) {
                builder.field("ef_search", efSearch);
            }
            if (nprobes != null) {
                builder.field("nprobes", nprobes);
            }
            builder.endObject();
        }
        if (rescore != null) {
            if (rescore && oversample != null) {
                builder.startObject(RESCORE_FIELD.getPreferredName()).field(OVERSAMPLE_FIELD.getPreferredName(), oversample).endObject();
            } else {
                builder.field(RESCORE_FIELD.getPreferredName(), rescore);
            }
        }
        if (expandNested != null) {
            builder.field(EXPAND_NESTED_FIELD.getPreferredName(), expandNested);
        }
        if (ignoreUnmapped) {
            builder.field(IGNORE_UNMAPPED_FIELD.getPreferredName(), true);
        }
        printBoostAndQueryName(builder);
        builder.endObject();
        builder.endObject();
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext context) throws IOException {
        if (filter == null) {
            return this;
        }
        QueryBuilder rewritten = filter.rewrite(context);
        if (rewritten == filter) {
            return this;
        }
        KnnQueryBuilder copy = copy();
        copy.filter = rewritten;
        return copy;
    }

    private KnnQueryBuilder copy() {
        KnnQueryBuilder c = new KnnQueryBuilder();
        c.fieldName = fieldName;
        c.vector = vector;
        c.k = k;
        c.maxDistance = maxDistance;
        c.minScore = minScore;
        c.efSearch = efSearch;
        c.nprobes = nprobes;
        c.filter = filter;
        c.rescore = rescore;
        c.oversample = oversample;
        c.expandNested = expandNested;
        c.ignoreUnmapped = ignoreUnmapped;
        c.boost(boost).queryName(queryName);
        return c;
    }

    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {
        MappedFieldType ft = context.fieldMapper(fieldName);
        if (ft == null && ignoreUnmapped) {
            return MatchNoDocsQuery.INSTANCE;
        }
        if (ft instanceof KnnVectorFieldMapper.KnnVectorFieldType == false) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not knn_vector type.", fieldName));
        }
        KnnVectorFieldMapper.KnnVectorFieldType knn = (KnnVectorFieldMapper.KnnVectorFieldType) ft;
        if (knn.isIndexedForAnn() == false) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not built for ANN search.", fieldName));
        }
        // k-NN's getVectorQueryType: k 0 (as MMR's candidates 0 leaves it) counts as unset.
        if (maxDistance == null && minScore == null && (k == null || k == 0)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME));
        }
        // Counted where the stock plugin counts: once the field is a knn one, before the vector is checked.
        KnnStats.query(
            k != null ? KnnStats.QueryKind.K
                : maxDistance != null ? KnnStats.QueryKind.MAX_DISTANCE
                : KnnStats.QueryKind.MIN_SCORE,
            filter != null
        );
        BitSetProducer parents = context.getParentFilter();
        boolean expand = Boolean.TRUE.equals(expandNested);
        if (k != null && expand && parents == null) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for the [%s] field. [%s] is only supported with a nested field.",
                    EXPAND_NESTED_FIELD.getPreferredName(),
                    EXPAND_NESTED_FIELD.getPreferredName()
                )
            );
        }
        KnnFieldConfig config = knn.config();
        if (config.binary() && k == null) {
            throw new UnsupportedOperationException(
                String.format(
                    Locale.ROOT,
                    "Radial search is not supported for this configuration: engine=LUCENE, data_type=BINARY, method=hnsw, compression=%s",
                    config.compression() == null ? "1x" : config.compression()
                )
            );
        }
        int queryDimension = config.binary() ? vector.length * Byte.SIZE : vector.length;
        if (queryDimension != config.dimension()) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Query vector has invalid dimension: %d. Dimension should be: %d", queryDimension, config.dimension())
            );
        }
        SpaceType space = config.spaceType();
        KnnTarget target;
        if (config.dataType() == VectorEncoding.BYTE) {
            byte[] bytes = new byte[vector.length];
            for (int i = 0; i < bytes.length; i++) {
                bytes[i] = KnnVectors.toByte(vector[i]);
            }
            target = KnnTarget.of(bytes);
        } else {
            KnnVectors.checkFloats(vector, space);
            float[] query = vector.clone();
            // KNNVectorFieldType.transformQueryVector: half-float cosine queries are normalized, as the vectors are.
            if (config.halfFloat() && space == SpaceType.COSINESIMIL) {
                VectorUtil.l2normalize(query);
            }
            target = KnnTarget.of(query);
        }
        Query filterQuery = filterQuery(context);
        if (k != null) {
            KnnTopVectorQuery.RecallTarget recall = recallTarget(context);
            if (config.ivf()) {
                // IVF sizes its own rerank from k; ef_search widens it only when asked for.
                int candidates = efSearch == null ? k : Math.max(k, efSearch);
                return new KnnTopVectorQuery(fieldName, target, k, candidates, filterQuery, parents, expand).withNprobes(
                    nprobes == null ? 0 : nprobes
                ).withRecallTarget(recall);
            }
            int candidates = Math.max(k, efSearch == null ? defaultEfSearch(context) : efSearch);
            return new KnnTopVectorQuery(fieldName, target, k, candidates, filterQuery, parents, expand).withRecallTarget(recall);
        }
        float threshold;
        if (maxDistance != null) {
            if (maxDistance < 0 && space != SpaceType.INNER_PRODUCT) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires distance to be non-negative for space type: %s", NAME, space.name())
                );
            }
            threshold = space.scoreForDistance(maxDistance);
        } else {
            if (minScore > 1 && space != SpaceType.INNER_PRODUCT) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires score to be in the range [0, 1] for space type: %s", NAME, space.name())
                );
            }
            threshold = minScore;
        }
        return KnnLeafVectorQuery.radial(fieldName, target, threshold, filterQuery);
    }

    /**
     * {@code BaseQueryFactory.getFilterQuery}: the filter, evaluated at the root level. Under a
     * {@code nested} query it matches parents, so it is joined down to their children (a to-parent
     * join over the root is unwrapped instead).
     */
    private Query filterQuery(QueryShardContext context) throws IOException {
        if (filter == null) {
            return null;
        }
        Deque<ObjectMapper> levels = new ArrayDeque<>();
        if (context.nestedScope() != null) {
            ObjectMapper level;
            while ((level = context.nestedScope().getObjectMapper()) != null) {
                levels.push(level);
                context.nestedScope().previousLevel();
            }
        }
        Query q;
        try {
            q = filter.toQuery(context);
        } finally {
            while (levels.isEmpty() == false) {
                context.nestedScope().nextLevel(levels.pop());
            }
        }
        if (q == null || context.getParentFilter() == null) {
            return q;
        }
        if (q instanceof OpenSearchToParentBlockJoinQuery toParent && toParent.getPath() == null) {
            return toParent.getChildQuery();
        }
        Query parentQuery = Queries.newNonNestedFilter();
        BitSetProducer parents = context.bitsetFilter(parentQuery);
        Query roots = new NestedHelper(context.getMapperService()).mightMatchNestedDocs(q) ? Queries.filtered(q, parentQuery) : q;
        return new ToChildBlockJoinQuery(roots, parents);
    }

    /**
     * The index's {@code index.knn.target_recall} for this shard's field, or null when it
     * is off or the query asks for its own depth ({@code ef_search} or {@code nprobes}), which is
     * then searched as asked.
     */
    private KnnTopVectorQuery.RecallTarget recallTarget(QueryShardContext context) {
        if (efSearch != null || nprobes != null) {
            return null;
        }
        Settings settings = context.getIndexSettings().getSettings();
        float recall = RustEngineSupport.KNN_TARGET_RECALL.get(settings);
        if (recall <= 0f) {
            return null;
        }
        String key = context.index().getUUID() + "/" + context.getShardId() + "/" + fieldName;
        return new KnnTopVectorQuery.RecallTarget(
            recall,
            RustEngineSupport.KNN_RECALL_SAMPLE_RATE.get(settings),
            RustEngineSupport.KNN_RECALL_CHECK_BUDGET.get(settings),
            key
        );
    }

    /**
     * {@code IndexHyperParametersUtil.getHNSWEFSearchValue}: the index's {@code
     * knn.algo_param.ef_search}, whose default is 512 for indexes created before 2.12.
     */
    private static int defaultEfSearch(QueryShardContext context) {
        Settings settings = context.getIndexSettings().getSettings();
        if (KnnSettings.EF_SEARCH.exists(settings) == false && context.indexVersionCreated().before(Version.V_2_12_0)) {
            return 512;
        }
        return KnnSettings.EF_SEARCH.get(settings);
    }

    @Override
    protected boolean doEquals(KnnQueryBuilder o) {
        return fieldName.equals(o.fieldName)
            && Arrays.equals(vector, o.vector)
            && Objects.equals(k, o.k)
            && Objects.equals(maxDistance, o.maxDistance)
            && Objects.equals(minScore, o.minScore)
            && Objects.equals(efSearch, o.efSearch)
            && Objects.equals(nprobes, o.nprobes)
            && Objects.equals(filter, o.filter)
            && Objects.equals(rescore, o.rescore)
            && Objects.equals(oversample, o.oversample)
            && Objects.equals(expandNested, o.expandNested)
            && ignoreUnmapped == o.ignoreUnmapped;
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.hashCode(vector), k, maxDistance, minScore, efSearch, nprobes, filter, rescore, oversample, expandNested, ignoreUnmapped);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
