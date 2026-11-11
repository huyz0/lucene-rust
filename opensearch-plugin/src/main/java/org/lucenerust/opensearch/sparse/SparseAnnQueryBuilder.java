/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.search.Query;
import org.lucenerust.opensearch.neural.stats.events.EventStatName;
import org.lucenerust.opensearch.neural.stats.events.EventStatsManager;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.XContentParser.Token;
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
 * neural-search's {@code SparseAnnQueryBuilder}: the {@code method_parameters} of a {@code
 * neural_sparse} query on a {@code sparse_vector} field ({@code top_n}, {@code k}, {@code
 * heap_factor}, {@code filter}), which becomes a {@link SparseVectorQuery} searched in Rust.
 */
public class SparseAnnQueryBuilder extends AbstractQueryBuilder<SparseAnnQueryBuilder> {
    public static final String NAME = "neural_sparse";
    public static final ParseField TOP_N_FIELD = new ParseField("top_n");
    public static final ParseField TOP_K_FIELD = new ParseField("k");
    public static final ParseField HEAP_FACTOR_FIELD = new ParseField("heap_factor");
    public static final ParseField METHOD_PARAMETERS_FIELD = new ParseField("method_parameters");
    public static final ParseField FILTER_FIELD = new ParseField("filter");

    private static final int DEFAULT_TOP_K = 10;
    private static final int DEFAULT_QUERY_CUT = 10;
    private static final float DEFAULT_HEAP_FACTOR = 1.0F;

    private String fieldName;
    private Integer queryCut;
    private Integer k;
    private Float heapFactor;
    private QueryBuilder filter;
    private Query fallbackQuery;
    private Map<String, Float> queryTokens;

    public SparseAnnQueryBuilder() {}

    public SparseAnnQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.queryCut = in.readOptionalInt();
        this.k = in.readOptionalInt();
        this.heapFactor = in.readOptionalFloat();
        this.filter = in.readOptionalNamedWriteable(QueryBuilder.class);
    }

    /** {@code SparseAnnQueryBuilder.fromXContent}: the {@code method_parameters} object. */
    public static SparseAnnQueryBuilder fromXContent(XContentParser parser) throws IOException {
        EventStatsManager.increment(EventStatName.SEISMIC_QUERY_REQUESTS);
        String methodFieldName = "";
        Token token = parser.currentToken();
        if (token != Token.START_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                String.format(Locale.ROOT, "[%s] %s must be an object", NAME, METHOD_PARAMETERS_FIELD.getPreferredName())
            );
        }
        SparseAnnQueryBuilder builder = new SparseAnnQueryBuilder();
        while ((token = parser.nextToken()) != Token.END_OBJECT) {
            if (token == Token.FIELD_NAME) {
                methodFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (TOP_N_FIELD.match(methodFieldName, parser.getDeprecationHandler())) {
                    builder.queryCut = positive(parser, parser.intValue(), TOP_N_FIELD);
                } else if (TOP_K_FIELD.match(methodFieldName, parser.getDeprecationHandler())) {
                    builder.k = positive(parser, parser.intValue(), TOP_K_FIELD);
                } else if (HEAP_FACTOR_FIELD.match(methodFieldName, parser.getDeprecationHandler())) {
                    builder.heapFactor = parser.floatValue();
                    if (builder.heapFactor <= 0) {
                        throw new ParsingException(
                            parser.getTokenLocation(),
                            String.format(Locale.ROOT, "[%s] %s must be a positive float", NAME, HEAP_FACTOR_FIELD.getPreferredName())
                        );
                    }
                } else {
                    throw new ParsingException(
                        parser.getTokenLocation(),
                        String.format(Locale.ROOT, "[%s] unknown field [%s]", NAME, methodFieldName)
                    );
                }
            } else if (FILTER_FIELD.match(methodFieldName, parser.getDeprecationHandler())) {
                builder.filter = parseInnerQueryBuilder(parser);
            } else {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    String.format(Locale.ROOT, "[%s] unknown token [%s] after [%s]", NAME, token, methodFieldName)
                );
            }
        }
        return builder;
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
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeOptionalInt(this.queryCut);
        out.writeOptionalInt(this.k);
        out.writeOptionalFloat(this.heapFactor);
        out.writeOptionalNamedWriteable(this.filter);
    }

    @Override
    protected void doXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        if (Objects.nonNull(queryCut)) {
            xContentBuilder.field(TOP_N_FIELD.getPreferredName(), queryCut);
        }
        if (Objects.nonNull(k)) {
            xContentBuilder.field(TOP_K_FIELD.getPreferredName(), k);
        }
        if (Objects.nonNull(heapFactor)) {
            xContentBuilder.field(HEAP_FACTOR_FIELD.getPreferredName(), heapFactor);
        }
        if (Objects.nonNull(filter)) {
            xContentBuilder.field(FILTER_FIELD.getPreferredName(), filter);
        }
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryRewriteContext) {
        return new SparseAnnQueryBuilder().fieldName(fieldName)
            .queryCut(queryCut)
            .k(k)
            .filter(filter)
            .fallbackQuery(fallbackQuery)
            .heapFactor(heapFactor);
    }

    /** {@code SparseAnnQueryBuilder.doToQuery}: the approximate search over the field's clustered postings. */
    @Override
    public Query doToQuery(QueryShardContext context) throws IOException {
        final MappedFieldType fieldType = context.fieldMapper(fieldName);
        validateFieldType(fieldType);
        Query filterQuery = filter == null ? null : filter.toQuery(context);
        return new SparseVectorQuery(
            fieldName,
            integerTokens(queryTokens),
            queryCut == null ? DEFAULT_QUERY_CUT : queryCut,
            k == null || k == 0 ? DEFAULT_TOP_K : k,
            heapFactor == null ? DEFAULT_HEAP_FACTOR : heapFactor,
            filterQuery,
            fallbackQuery
        );
    }

    public static void validateFieldType(MappedFieldType fieldType) {
        if (Objects.isNull(fieldType) || SparseFieldUtils.isSparseVectorType(fieldType.typeName()) == false) {
            throw new IllegalArgumentException(
                "["
                    + NAME
                    + "] query with ["
                    + METHOD_PARAMETERS_FIELD.getPreferredName()
                    + "] only works on ["
                    + SparseVectorFieldMapper.CONTENT_TYPE
                    + "] fields"
            );
        }
    }

    /** {@code SparseAnnQueryBuilder.preprocessQueryTokens}: non-negative integers, duplicates at their highest weight. */
    private static Map<Integer, Float> integerTokens(Map<String, Float> tokens) {
        Map<Integer, Float> out = new HashMap<>();
        if (tokens == null) {
            return out;
        }
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
    protected boolean doEquals(SparseAnnQueryBuilder obj) {
        return Objects.equals(queryCut, obj.queryCut)
            && Objects.equals(heapFactor, obj.heapFactor)
            && Objects.equals(k, obj.k)
            && Objects.equals(filter, obj.filter);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(queryCut, heapFactor, k, filter);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    public SparseAnnQueryBuilder queryTokens(Map<String, Float> queryTokens) {
        integerTokens(queryTokens);
        this.queryTokens = queryTokens;
        return this;
    }

    public String fieldName() {
        return fieldName;
    }

    public Integer queryCut() {
        return queryCut;
    }

    public Integer k() {
        return k;
    }

    public Float heapFactor() {
        return heapFactor;
    }

    public QueryBuilder filter() {
        return filter;
    }

    public Query fallbackQuery() {
        return fallbackQuery;
    }

    public Map<String, Float> queryTokens() {
        return queryTokens;
    }

    public SparseAnnQueryBuilder fieldName(final String fieldName) {
        this.fieldName = fieldName;
        return this;
    }

    public SparseAnnQueryBuilder queryCut(final Integer queryCut) {
        this.queryCut = queryCut;
        return this;
    }

    public SparseAnnQueryBuilder k(final Integer k) {
        this.k = k;
        return this;
    }

    public SparseAnnQueryBuilder heapFactor(final Float heapFactor) {
        this.heapFactor = heapFactor;
        return this;
    }

    public SparseAnnQueryBuilder filter(final QueryBuilder filter) {
        this.filter = filter;
        return this;
    }

    public SparseAnnQueryBuilder fallbackQuery(final Query fallbackQuery) {
        this.fallbackQuery = fallbackQuery;
        return this;
    }
}
