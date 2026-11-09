/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.lucenerust.opensearch.neural.stats.events.EventStatName;
import org.lucenerust.opensearch.neural.stats.events.EventStatsManager;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
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
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * neural-search's {@code neural_sparse} query with {@code query_tokens}: on a {@code rank_features}
 * field the sum of linear {@code FeatureField} queries, on a {@code sparse_vector} field its
 * approximate search ({@link SparseVectorQuery}, with {@code method_parameters} {@code top_n},
 * {@code k}, {@code heap_factor} and {@code filter}). {@code query_text} needs a model or a sparse
 * analyzer, which this plugin does not run, and is refused.
 */
public final class NeuralSparseQueryBuilder extends AbstractQueryBuilder<NeuralSparseQueryBuilder> {
    public static final String NAME = "neural_sparse";

    static final ParseField QUERY_TOKENS_FIELD = new ParseField("query_tokens");
    static final ParseField METHOD_PARAMETERS_FIELD = new ParseField("method_parameters");
    static final ParseField TOP_N_FIELD = new ParseField("top_n");
    static final ParseField K_FIELD = new ParseField("k");
    static final ParseField HEAP_FACTOR_FIELD = new ParseField("heap_factor");
    static final ParseField FILTER_FIELD = new ParseField("filter");
    private static final String[] INFERENCE_FIELDS = { "query_text", "model_id", "analyzer", "max_token_score" };

    private static final int DEFAULT_TOP_N = 10;
    private static final int DEFAULT_K = 10;
    private static final float DEFAULT_HEAP_FACTOR = 1.0f;

    private String fieldName;
    private Map<String, Float> queryTokens;
    private Integer topN;
    private Integer k;
    private Float heapFactor;
    private QueryBuilder filter;

    public NeuralSparseQueryBuilder(String fieldName, Map<String, Float> queryTokens) {
        this.fieldName = fieldName;
        this.queryTokens = queryTokens;
    }

    private NeuralSparseQueryBuilder() {}

    public NeuralSparseQueryBuilder(StreamInput in) throws IOException {
        super(in);
        fieldName = in.readString();
        queryTokens = in.readMap(StreamInput::readString, StreamInput::readFloat);
        topN = in.readOptionalInt();
        k = in.readOptionalInt();
        heapFactor = in.readOptionalFloat();
        filter = in.readOptionalNamedWriteable(QueryBuilder.class);
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeMap(queryTokens, StreamOutput::writeString, StreamOutput::writeFloat);
        out.writeOptionalInt(topN);
        out.writeOptionalInt(k);
        out.writeOptionalFloat(heapFactor);
        out.writeOptionalNamedWriteable(filter);
    }

    public NeuralSparseQueryBuilder topN(int topN) {
        this.topN = topN;
        return this;
    }

    public NeuralSparseQueryBuilder k(int k) {
        this.k = k;
        return this;
    }

    public NeuralSparseQueryBuilder heapFactor(float heapFactor) {
        this.heapFactor = heapFactor;
        return this;
    }

    public NeuralSparseQueryBuilder filter(QueryBuilder filter) {
        this.filter = filter;
        return this;
    }

    /** {@code NeuralSparseQueryBuilder.fromXContent}: one field, its parameters an object. */
    public static NeuralSparseQueryBuilder fromXContent(XContentParser parser) throws IOException {
        EventStatsManager.increment(EventStatName.NEURAL_SPARSE_QUERY_REQUESTS);
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), "First token of " + NAME + "query must be START_OBJECT");
        }
        NeuralSparseQueryBuilder b = new NeuralSparseQueryBuilder();
        parser.nextToken();
        b.fieldName = parser.currentName();
        parser.nextToken();
        parseParams(parser, b);
        if (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                String.format(
                    Locale.ROOT,
                    "[%s] query doesn't support multiple fields, found [%s] and [%s]",
                    NAME,
                    b.fieldName,
                    parser.currentName()
                )
            );
        }
        if (b.fieldName == null) {
            throw new IllegalArgumentException("Field name must be provided for " + NAME + " query");
        }
        if (b.queryTokens == null) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "%s field must be provided for [%s] query", QUERY_TOKENS_FIELD.getPreferredName(), NAME)
            );
        }
        return b;
    }

    private static void parseParams(XContentParser parser, NeuralSparseQueryBuilder b) throws IOException {
        String current = "";
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (token.isValue()) {
                if (NAME_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.queryName(parser.text());
                } else if (BOOST_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.boost(parser.floatValue());
                } else if (isInferenceField(current)) {
                    throw new ParsingException(
                        parser.getTokenLocation(),
                        String.format(
                            Locale.ROOT,
                            "[%s] [%s] needs neural-search's model inference, which this node does not run; give [%s]",
                            NAME,
                            current,
                            QUERY_TOKENS_FIELD.getPreferredName()
                        )
                    );
                } else {
                    throw new ParsingException(
                        parser.getTokenLocation(),
                        String.format(Locale.ROOT, "[%s] query does not support [%s] field", NAME, current)
                    );
                }
            } else if (QUERY_TOKENS_FIELD.match(current, parser.getDeprecationHandler())) {
                b.queryTokens = parser.map(HashMap::new, XContentParser::floatValue);
            } else if (METHOD_PARAMETERS_FIELD.match(current, parser.getDeprecationHandler())) {
                parseMethodParameters(parser, b);
            } else {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    String.format(Locale.ROOT, "[%s] unknown token [%s] after [%s]", NAME, token, current)
                );
            }
        }
    }

    private static boolean isInferenceField(String name) {
        for (String f : INFERENCE_FIELDS) {
            if (f.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /** {@code SparseAnnQueryBuilder.fromXContent}. */
    private static void parseMethodParameters(XContentParser parser, NeuralSparseQueryBuilder b) throws IOException {
        EventStatsManager.increment(EventStatName.SEISMIC_QUERY_REQUESTS);
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                String.format(Locale.ROOT, "[%s] %s must be an object", NAME, METHOD_PARAMETERS_FIELD.getPreferredName())
            );
        }
        String current = "";
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (token.isValue()) {
                if (TOP_N_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.topN = positive(parser, parser.intValue(), TOP_N_FIELD);
                } else if (K_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.k = positive(parser, parser.intValue(), K_FIELD);
                } else if (HEAP_FACTOR_FIELD.match(current, parser.getDeprecationHandler())) {
                    b.heapFactor = parser.floatValue();
                    if (b.heapFactor <= 0) {
                        throw new ParsingException(
                            parser.getTokenLocation(),
                            String.format(Locale.ROOT, "[%s] %s must be a positive float", NAME, HEAP_FACTOR_FIELD.getPreferredName())
                        );
                    }
                } else {
                    throw new ParsingException(parser.getTokenLocation(), String.format(Locale.ROOT, "[%s] unknown field [%s]", NAME, current));
                }
            } else if (FILTER_FIELD.match(current, parser.getDeprecationHandler())) {
                b.filter = parseInnerQueryBuilder(parser);
            } else {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    String.format(Locale.ROOT, "[%s] unknown token [%s] after [%s]", NAME, token, current)
                );
            }
        }
    }

    private static int positive(XContentParser parser, int v, ParseField field) {
        if (v <= 0) {
            throw new ParsingException(
                parser.getTokenLocation(),
                String.format(Locale.ROOT, "[%s] %s must be a positive integer", NAME, field.getPreferredName())
            );
        }
        return v;
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);
        builder.field(QUERY_TOKENS_FIELD.getPreferredName(), queryTokens);
        if (topN != null || k != null || heapFactor != null || filter != null) {
            builder.startObject(METHOD_PARAMETERS_FIELD.getPreferredName());
            if (topN != null) {
                builder.field(TOP_N_FIELD.getPreferredName(), topN);
            }
            if (k != null) {
                builder.field(K_FIELD.getPreferredName(), k);
            }
            if (heapFactor != null) {
                builder.field(HEAP_FACTOR_FIELD.getPreferredName(), heapFactor);
            }
            if (filter != null) {
                builder.field(FILTER_FIELD.getPreferredName(), filter);
            }
            builder.endObject();
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
        NeuralSparseQueryBuilder copy = new NeuralSparseQueryBuilder(fieldName, queryTokens);
        copy.topN = topN;
        copy.k = k;
        copy.heapFactor = heapFactor;
        copy.filter = rewritten;
        copy.boost(boost).queryName(queryName);
        return copy;
    }

    /** {@code NeuralSparseQueryBuilder.doToQuery} and {@code SparseAnnQueryBuilder.doToQuery}. */
    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {
        MappedFieldType ft = context.fieldMapper(fieldName);
        boolean seismic = ft instanceof SparseVectorFieldMapper.SparseVectorFieldType;
        if (seismic == false) {
            if (ft == null || ft.typeName().equals("rank_features") == false) {
                throw new IllegalArgumentException("[" + NAME + "] query only works on [rank_features] fields");
            }
            if (filter != null) {
                throw new IllegalArgumentException("[" + NAME + "] [method_parameters.filter] only works on [sparse_vector] fields");
            }
        }
        BooleanQuery.Builder fallback = new BooleanQuery.Builder();
        for (Map.Entry<String, Float> e : queryTokens.entrySet()) {
            fallback.add(FeatureField.newLinearQuery(fieldName, e.getKey(), e.getValue()), BooleanClause.Occur.SHOULD);
        }
        if (seismic == false) {
            return fallback.build();
        }
        Query filterQuery = filter == null ? null : filter.toQuery(context);
        if (filterQuery != null) {
            fallback.add(filterQuery, BooleanClause.Occur.FILTER);
        }
        return new SparseVectorQuery(
            fieldName,
            integerTokens(queryTokens),
            topN == null ? DEFAULT_TOP_N : topN,
            k == null || k == 0 ? DEFAULT_K : k,
            heapFactor == null ? DEFAULT_HEAP_FACTOR : heapFactor,
            filterQuery,
            fallback.build()
        );
    }

    /** {@code SparseAnnQueryBuilder.preprocessQueryTokens}: non-negative integers, duplicates at their highest weight. */
    private static Map<Integer, Float> integerTokens(Map<String, Float> tokens) {
        Map<Integer, Float> out = new HashMap<>();
        try {
            for (Map.Entry<String, Float> e : tokens.entrySet()) {
                int token = Integer.parseInt(e.getKey());
                if (token < 0) {
                    throw new IllegalArgumentException("Query tokens should be non-negative integer!");
                }
                out.merge(token, e.getValue(), Math::max);
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Query tokens should be valid integer");
        }
        return out;
    }

    @Override
    protected boolean doEquals(NeuralSparseQueryBuilder o) {
        return fieldName.equals(o.fieldName)
            && Objects.equals(queryTokens, o.queryTokens)
            && Objects.equals(topN, o.topN)
            && Objects.equals(k, o.k)
            && Objects.equals(heapFactor, o.heapFactor)
            && Objects.equals(filter, o.filter);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, queryTokens, topN, k, heapFactor, filter);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
