/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.lucenerust.opensearch.knn.rest.KnnStats;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.opensearch.Version;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
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

    static final ParseField VECTOR_FIELD = new ParseField("vector");
    static final ParseField K_FIELD = new ParseField("k");
    static final ParseField FILTER_FIELD = new ParseField("filter");
    static final ParseField MAX_DISTANCE_FIELD = new ParseField("max_distance");
    static final ParseField MIN_SCORE_FIELD = new ParseField("min_score");
    static final ParseField METHOD_PARAMETERS_FIELD = new ParseField("method_parameters");
    static final ParseField RESCORE_FIELD = new ParseField("rescore");
    static final ParseField OVERSAMPLE_FIELD = new ParseField("oversample_factor");
    static final ParseField EXPAND_NESTED_FIELD = new ParseField("expand_nested_docs");
    static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");

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

    public KnnQueryBuilder filter(QueryBuilder filter) {
        this.filter = filter;
        return this;
    }

    public KnnQueryBuilder efSearch(int efSearch) {
        this.efSearch = efSearch;
        return this;
    }

    /** {@code KNNQueryBuilder.fromXContent}: one field, its parameters an object. */
    public static KnnQueryBuilder fromXContent(XContentParser parser) throws IOException {
        KnnQueryBuilder b = new KnnQueryBuilder();
        String current = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                if (b.fieldName != null) {
                    throw new ParsingException(
                        parser.getTokenLocation(),
                        "[" + NAME + "] query doesn't support multiple fields, found [" + b.fieldName + "] and [" + current + "]"
                    );
                }
                b.fieldName = current;
                parseField(parser, b);
            } else {
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] unknown token [" + token + "] after [" + current + "]");
            }
        }
        if (b.fieldName == null) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] requires a field");
        }
        b.validate(parser);
        return b;
    }

    private static void parseField(XContentParser parser, KnnQueryBuilder b) throws IOException {
        String current = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (VECTOR_FIELD.match(current, parser.getDeprecationHandler())) {
                b.vector = floats(parser);
            } else if (FILTER_FIELD.match(current, parser.getDeprecationHandler())) {
                b.filter = parseInnerQueryBuilder(parser);
            } else if (METHOD_PARAMETERS_FIELD.match(current, parser.getDeprecationHandler())) {
                methodParameters(parser, b);
            } else if (RESCORE_FIELD.match(current, parser.getDeprecationHandler())) {
                if (token == XContentParser.Token.START_OBJECT) {
                    b.rescore = true;
                    String inner = null;
                    while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
                        if (token == XContentParser.Token.FIELD_NAME) {
                            inner = parser.currentName();
                        } else if (OVERSAMPLE_FIELD.match(inner, parser.getDeprecationHandler())) {
                            b.oversample = parser.floatValue();
                            if (b.oversample < 1 || b.oversample > 100) {
                                throw new ParsingException(parser.getTokenLocation(), "Oversample factor must be between 1.0 and 100.0");
                            }
                        } else {
                            throw new ParsingException(parser.getTokenLocation(), "Unknown field in rescore: " + inner);
                        }
                    }
                } else {
                    b.rescore = parser.booleanValue();
                }
            } else if (token.isValue()) {
                if (K_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.k = parser.intValue(false);
                } else if (MAX_DISTANCE_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.maxDistance = parser.floatValue();
                } else if (MIN_SCORE_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.minScore = parser.floatValue();
                } else if (EXPAND_NESTED_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.expandNested = parser.booleanValue();
                } else if (IGNORE_UNMAPPED_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.ignoreUnmapped = parser.booleanValue();
                } else if (BOOST_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.boost(parser.floatValue());
                } else if (NAME_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.queryName(parser.text());
                } else {
                    throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] query does not support [" + current + "]");
                }
            } else {
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] unknown token [" + token + "] after [" + current + "]");
            }
        }
    }

    private static float[] floats(XContentParser parser) throws IOException {
        if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [vector] must be an array of numbers");
        }
        List<Float> values = new ArrayList<>();
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            values.add(parser.floatValue());
        }
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = values.get(i);
        }
        return v;
    }

    private static void methodParameters(XContentParser parser, KnnQueryBuilder b) throws IOException {
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [method_parameters] must be an object");
        }
        String current = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if ("ef_search".equals(current)) {
                b.efSearch = parser.intValue(false);
                if (b.efSearch <= 0) {
                    throw new ParsingException(parser.getTokenLocation(), "Validation Failed: 1: ef_search should be greater than 0;");
                }
            } else if ("nprobes".equals(current)) {
                b.nprobes = parser.intValue(false);
                if (b.nprobes <= 0) {
                    throw new ParsingException(parser.getTokenLocation(), "Validation Failed: 1: nprobes should be greater than 0;");
                }
            } else {
                throw new ParsingException(parser.getTokenLocation(), "Validation Failed: 1: unknown method parameter found [" + current + "];");
            }
        }
    }

    /** {@code KNNQueryBuilder}'s checks that need no mapping. */
    private void validate(XContentParser parser) {
        if (vector == null || vector.length == 0) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] query vector is empty");
        }
        int set = (k != null ? 1 : 0) + (maxDistance != null ? 1 : 0) + (minScore != null ? 1 : 0);
        if (set != 1) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] requires exactly one of k, distance or score to be set");
        }
        if (k != null && (k <= 0 || k > MAX_K)) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] requires k to be in the range (0, " + MAX_K + "]");
        }
        if (minScore != null && minScore <= 0) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] requires minScore to be greater than 0");
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
        // Counted where the stock plugin counts: once the field is a knn one, before the vector is checked.
        KnnStats.query(
            k != null ? KnnStats.QueryKind.K
                : maxDistance != null ? KnnStats.QueryKind.MAX_DISTANCE
                : KnnStats.QueryKind.MIN_SCORE,
            filter != null
        );
        if (Boolean.TRUE.equals(expandNested)) {
            throw new IllegalArgumentException("[" + NAME + "] [expand_nested_docs] is not supported by this node yet");
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
            target = KnnTarget.of(vector.clone());
        }
        Query filterQuery = filter == null ? null : filter.toQuery(context);
        if (k != null) {
            return new KnnTopVectorQuery(fieldName, target, k, Math.max(k, efSearch == null ? defaultEfSearch(context) : efSearch), filterQuery);
        }
        float threshold;
        if (maxDistance != null) {
            if (maxDistance < 0 && space != SpaceType.INNER_PRODUCT) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires distance to be non-negative for space type: %s", NAME, space)
                );
            }
            threshold = space.scoreForDistance(maxDistance);
        } else {
            if (minScore > 1 && space != SpaceType.INNER_PRODUCT) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires score to be in the range [0, 1] for space type: %s", NAME, space)
                );
            }
            threshold = minScore;
        }
        return KnnLeafVectorQuery.radial(fieldName, target, threshold, filterQuery);
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
