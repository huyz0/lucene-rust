/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.tensor;

import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.lucenerust.opensearch.knn.KnnTarget;
import org.lucenerust.opensearch.knn.KnnTopVectorQuery;
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
 * The {@code max_sim} query: the documents with a value in a {@code multi_vector} field (and
 * matching the optional {@code filter}), each scored by the {@code MaxSim} of {@code
 * query_vectors} against its tokens ({@link MaxSimQuery}).
 *
 * <pre>{@code
 * "max_sim": { "<field>": { "query_vectors": [[...], [...]], "candidates": 100, "filter": { ... }, "boost": 1.0 } }
 * }</pre>
 *
 * With {@code candidates} (a {@code pooled} field) it matches only the {@code candidates} documents
 * whose pooled mean is nearest the query's (an HNSW search of the field's hidden vector field, the
 * filter applied during it), each still scored by {@code MaxSim}. In a {@code bool} {@code should}
 * beside a lexical query it adds those documents to the first pass, for the rescore to rerank.
 *
 * As a {@code rescore_query} with {@code query_weight: 0} it reranks a first-pass window, ColBERT's
 * late interaction.
 */
public final class MaxSimQueryBuilder extends AbstractQueryBuilder<MaxSimQueryBuilder> {
    public static final String NAME = "max_sim";

    static final ParseField QUERY_VECTORS_FIELD = new ParseField("query_vectors");
    static final ParseField FILTER_FIELD = new ParseField("filter");
    static final ParseField CANDIDATES_FIELD = new ParseField("candidates");
    /** k-NN's limit on {@code k}. */
    static final int MAX_CANDIDATES = 10000;
    /** The HNSW search's width at least, k-NN's default {@code ef_search}. */
    static final int MIN_SEARCH_WIDTH = 100;

    private String fieldName;
    private float[][] queryVectors;
    private QueryBuilder filter;
    private int candidates;

    public MaxSimQueryBuilder(String fieldName, float[][] queryVectors) {
        this.fieldName = Objects.requireNonNull(fieldName, "[" + NAME + "] requires a field name");
        this.queryVectors = checked(queryVectors);
    }

    /** A copy of {@code v}: at least one token, every one the same non-zero length, finite. */
    private static float[][] checked(float[][] v) {
        if (v == null || v.length == 0) {
            throw new IllegalArgumentException("[" + NAME + "] requires [query_vectors], an array of token vectors");
        }
        float[][] out = new float[v.length][];
        for (int i = 0; i < v.length; i++) {
            if (v[i] == null || v[i].length == 0 || v[i].length != v[0].length) {
                throw new IllegalArgumentException("[" + NAME + "] every vector of [query_vectors] must have the same, non-zero length");
            }
            for (float f : v[i]) {
                if (Float.isFinite(f) == false) {
                    throw new IllegalArgumentException("[" + NAME + "] [query_vectors] must hold finite numbers");
                }
            }
            out[i] = v[i].clone();
        }
        return out;
    }

    private MaxSimQueryBuilder() {}

    public MaxSimQueryBuilder(StreamInput in) throws IOException {
        super(in);
        fieldName = in.readString();
        int n = in.readVInt();
        float[][] v = new float[n][];
        for (int i = 0; i < n; i++) {
            v[i] = in.readFloatArray();
        }
        queryVectors = checked(v);
        filter = in.readOptionalNamedWriteable(QueryBuilder.class);
        candidates(in.readVInt());
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeVInt(queryVectors.length);
        for (float[] v : queryVectors) {
            out.writeFloatArray(v);
        }
        out.writeOptionalNamedWriteable(filter);
        out.writeVInt(candidates);
    }

    public MaxSimQueryBuilder filter(QueryBuilder filter) {
        this.filter = filter;
        return this;
    }

    /** Match only the {@code n} documents nearest by pooled mean; 0 matches every document with a value. */
    public MaxSimQueryBuilder candidates(int n) {
        if (n < 0 || n > MAX_CANDIDATES) {
            throw new IllegalArgumentException("[" + NAME + "] [candidates] must be in [0, " + MAX_CANDIDATES + "], got " + n);
        }
        this.candidates = n;
        return this;
    }

    public static MaxSimQueryBuilder fromXContent(XContentParser parser) throws IOException {
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), "First token of " + NAME + " query must be START_OBJECT");
        }
        MaxSimQueryBuilder b = new MaxSimQueryBuilder();
        if (parser.nextToken() != XContentParser.Token.FIELD_NAME) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] query must name a field");
        }
        b.fieldName = parser.currentName();
        if (parser.nextToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] field [" + b.fieldName + "] must hold an object");
        }
        String current = "";
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                current = parser.currentName();
            } else if (QUERY_VECTORS_FIELD.match(current, parser.getDeprecationHandler())) {
                if (token != XContentParser.Token.START_ARRAY) {
                    throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [query_vectors] must be an array of arrays of numbers");
                }
                b.queryVectors = vectors(parser);
            } else if (FILTER_FIELD.match(current, parser.getDeprecationHandler()) && token == XContentParser.Token.START_OBJECT) {
                b.filter = parseInnerQueryBuilder(parser);
            } else if (CANDIDATES_FIELD.match(current, parser.getDeprecationHandler())) {
                b.candidates = candidates(parser, token);
            } else if (token.isValue() && NAME_FIELD.match(current, parser.getDeprecationHandler())) {
                b.queryName(parser.text());
            } else if (token.isValue() && BOOST_FIELD.match(current, parser.getDeprecationHandler())) {
                b.boost(parser.floatValue());
            } else {
                throw new ParsingException(parser.getTokenLocation(), String.format(Locale.ROOT, "[%s] unknown field [%s]", NAME, current));
            }
        }
        if (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                String.format(Locale.ROOT, "[%s] query doesn't support multiple fields, found [%s] and [%s]", NAME, b.fieldName, parser.currentName())
            );
        }
        if (b.queryVectors == null || b.queryVectors.length == 0) {
            throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] requires [query_vectors], an array of token vectors");
        }
        return b;
    }

    /** {@code candidates}: a whole number in {@code [1, MAX_CANDIDATES]}. */
    private static int candidates(XContentParser parser, XContentParser.Token token) throws IOException {
        String bad = "[" + NAME + "] [candidates] must be a whole number in [1, " + MAX_CANDIDATES + "]";
        if (token != XContentParser.Token.VALUE_NUMBER) {
            throw new ParsingException(parser.getTokenLocation(), bad + ", got " + token);
        }
        double d = parser.doubleValue();
        if (d != Math.rint(d) || d < 1 || d > MAX_CANDIDATES) {
            throw new ParsingException(parser.getTokenLocation(), bad + ", got " + parser.text());
        }
        return (int) d;
    }

    /** {@code [[f, ...], ...]}: at least one token, every one the same non-zero length. */
    private static float[][] vectors(XContentParser parser) throws IOException {
        List<float[]> tokens = new ArrayList<>();
        for (XContentParser.Token t = parser.nextToken(); t != XContentParser.Token.END_ARRAY; t = parser.nextToken()) {
            if (t != XContentParser.Token.START_ARRAY) {
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [query_vectors] must be an array of arrays of numbers");
            }
            List<Float> v = new ArrayList<>();
            for (XContentParser.Token c = parser.nextToken(); c != XContentParser.Token.END_ARRAY; c = parser.nextToken()) {
                if (c != XContentParser.Token.VALUE_NUMBER) {
                    throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [query_vectors] must be an array of arrays of numbers");
                }
                float f = parser.floatValue();
                if (Float.isFinite(f) == false) {
                    throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] [query_vectors] must hold finite numbers");
                }
                v.add(f);
            }
            float[] a = new float[v.size()];
            for (int i = 0; i < a.length; i++) {
                a[i] = v.get(i);
            }
            if (a.length == 0 || (tokens.isEmpty() == false && a.length != tokens.get(0).length)) {
                throw new ParsingException(parser.getTokenLocation(), "[" + NAME + "] every vector of [query_vectors] must have the same, non-zero length");
            }
            tokens.add(a);
        }
        return tokens.toArray(new float[0][]);
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);
        builder.startArray(QUERY_VECTORS_FIELD.getPreferredName());
        for (float[] v : queryVectors) {
            builder.value(v);
        }
        builder.endArray();
        if (candidates > 0) {
            builder.field(CANDIDATES_FIELD.getPreferredName(), candidates);
        }
        if (filter != null) {
            builder.field(FILTER_FIELD.getPreferredName(), filter);
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
        MaxSimQueryBuilder copy = new MaxSimQueryBuilder(fieldName, queryVectors).filter(rewritten).candidates(candidates);
        copy.boost(boost).queryName(queryName);
        return copy;
    }

    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {
        MappedFieldType ft = context.fieldMapper(fieldName);
        if (ft == null) {
            // As match and term: an index without the field matches nothing.
            return new MatchNoDocsQuery("[" + NAME + "] field [" + fieldName + "] is not mapped");
        }
        if ((ft instanceof MultiVectorFieldMapper.MultiVectorFieldType) == false) {
            throw new IllegalArgumentException("[" + NAME + "] query only works on [" + MultiVectorFieldMapper.CONTENT_TYPE + "] fields, not on [" + fieldName + "]");
        }
        MultiVectorFieldMapper.MultiVectorFieldType mv = (MultiVectorFieldMapper.MultiVectorFieldType) ft;
        if (queryVectors[0].length != mv.dimension()) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] query vectors have dimension %d, but field [%s] has dimension %d",
                    NAME,
                    queryVectors[0].length,
                    fieldName,
                    mv.dimension()
                )
            );
        }
        if (candidates > 0) {
            if (mv.pooled() == false) {
                throw new IllegalArgumentException(
                    "[" + NAME + "] [candidates] needs field [" + fieldName + "] mapped with [" + MultiVectorFieldMapper.POOLED + ": true]"
                );
            }
            float[] mean = MultiVectorEncoding.pooledMean(queryVectors);
            if (mean == null) {
                return new MatchNoDocsQuery("[" + NAME + "] query vectors with a zero mean have no nearest documents");
            }
            Query f = filter == null ? null : filter.toQuery(context);
            return new MaxSimQuery(new KnnTopVectorQuery(mv.pooledField(), KnnTarget.of(mean), candidates, Math.max(candidates, MIN_SEARCH_WIDTH), f), fieldName, queryVectors);
        }
        Query sub = new FieldExistsQuery(fieldName);
        if (filter != null) {
            sub = new BooleanQuery.Builder().add(sub, BooleanClause.Occur.FILTER)
                .add(filter.toQuery(context), BooleanClause.Occur.FILTER)
                .build();
        }
        return new MaxSimQuery(sub, fieldName, queryVectors);
    }

    @Override
    protected boolean doEquals(MaxSimQueryBuilder o) {
        return fieldName.equals(o.fieldName) && Arrays.deepEquals(queryVectors, o.queryVectors)
            && Objects.equals(filter, o.filter)
            && candidates == o.candidates;
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, Arrays.deepHashCode(queryVectors), filter, candidates);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
