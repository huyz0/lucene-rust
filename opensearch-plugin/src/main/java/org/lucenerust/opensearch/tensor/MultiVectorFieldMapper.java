/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.tensor;

import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceValueFetcher;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The {@code multi_vector} field: a document's token vectors (ColBERT's late-interaction
 * embeddings), an array of arrays of {@code dimension} numbers, kept as one binary doc value in the
 * field's {@code encoding} ({@link MultiVectorEncoding}): {@code float16} by default, {@code float}
 * for Lucene's own {@code LateInteractionField} bytes, or {@code int8}. Searched by {@code max_sim}
 * ({@link MaxSimQueryBuilder}); {@code _source} keeps the values as sent. With {@code pooled}, each
 * document's mean token (unit length, {@link MultiVectorEncoding#pooledMean}) is also indexed in a
 * hidden HNSW vector field ({@code <field>._pooled}, dot product), where {@code max_sim}'s {@code
 * candidates} finds the nearest documents (a document whose tokens sum to zero has no mean, so
 * no candidate search finds it).
 *
 * <pre>{@code
 * "colbert": { "type": "multi_vector", "dimension": 96, "encoding": "float16", "pooled": true }
 * }</pre>
 */
public final class MultiVectorFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "multi_vector";
    static final String DIMENSION = "dimension";
    static final String ENCODING = "encoding";
    static final String POOLED = "pooled";
    /** The suffix of the hidden vector field a {@code pooled} field keeps its documents' means in. */
    public static final String POOLED_SUFFIX = "._pooled";
    /** The k-NN plugin's limit on a vector's dimension. */
    static final int MAX_DIMENSION = 16000;
    /** Lucene's {@code KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS}, the hidden pooled field's limit. */
    static final int MAX_POOLED_DIMENSION = 1024;

    private static MultiVectorFieldType ft(Mapper m) {
        return ((MultiVectorFieldMapper) m).fieldType();
    }

    public static final class Builder extends ParametrizedFieldMapper.Builder {
        final Parameter<Integer> dimension = new Parameter<>(
            DIMENSION,
            false,
            () -> -1,
            (n, c, o) -> XContentMapValues.nodeIntegerValue(o),
            m -> ft(m).dimension
        );
        final Parameter<String> encoding = new Parameter<>(
            ENCODING,
            false,
            () -> MultiVectorEncoding.FLOAT16.getName(),
            (n, c, o) -> MultiVectorEncoding.of(XContentMapValues.nodeStringValue(o, null)).getName(),
            m -> ft(m).encoding.getName()
        ).alwaysSerialize();
        final Parameter<Boolean> pooled = Parameter.boolParam(POOLED, false, m -> ft(m).pooled, false).alwaysSerialize();
        final Parameter<Map<String, String>> meta = Parameter.metaParam();

        Builder(String name) {
            super(name);
            builder = this;
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(dimension, encoding, pooled, meta);
        }

        @Override
        public ParametrizedFieldMapper build(BuilderContext context) {
            return new MultiVectorFieldMapper(
                name,
                new MultiVectorFieldType(
                    buildFullName(context),
                    dimension.getValue(),
                    MultiVectorEncoding.of(encoding.getValue()),
                    pooled.getValue(),
                    meta.getValue()
                ),
                multiFieldsBuilder.build(this, context),
                copyTo.build()
            );
        }
    }

    public static final class TypeParser implements Mapper.TypeParser {
        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) {
            if (node.containsKey("fields")) {
                throw new MapperParsingException("[" + CONTENT_TYPE + "] field [" + name + "] takes no multi-fields");
            }
            Builder builder = new Builder(name);
            builder.parse(name, parserContext, node);
            int dim = builder.dimension.getValue();
            if (dim == -1) {
                throw new MapperParsingException("[" + CONTENT_TYPE + "] field [" + name + "] requires [" + DIMENSION + "]");
            }
            if (dim < 1 || dim > MAX_DIMENSION) {
                throw new MapperParsingException(
                    String.format(Locale.ROOT, "[%s] field [%s] [%s] must be in [1, %d], got %d", CONTENT_TYPE, name, DIMENSION, MAX_DIMENSION, dim)
                );
            }
            if (builder.pooled.getValue() && dim > MAX_POOLED_DIMENSION) {
                throw new MapperParsingException(
                    String.format(
                        Locale.ROOT,
                        "[%s] field [%s] with [%s] takes a [%s] up to %d, got %d",
                        CONTENT_TYPE,
                        name,
                        POOLED,
                        DIMENSION,
                        MAX_POOLED_DIMENSION,
                        dim
                    )
                );
            }
            return builder;
        }
    }

    private MultiVectorFieldMapper(String simpleName, MappedFieldType type, MultiFields multiFields, CopyTo copyTo) {
        super(simpleName, type, multiFields, copyTo);
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName()).init(this);
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    protected MultiVectorFieldMapper clone() {
        return (MultiVectorFieldMapper) super.clone();
    }

    @Override
    public MultiVectorFieldType fieldType() {
        return (MultiVectorFieldType) super.fieldType();
    }

    /** The whole array of token vectors is one value. */
    @Override
    public boolean parsesArrayValue() {
        return true;
    }

    /** An array of {@code dimension}-long arrays of numbers, at least one; null indexes nothing. */
    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        if (context.externalValueSet()) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields can't be used in multi-fields");
        }
        XContentParser parser = context.parser();
        if (parser.currentToken() == XContentParser.Token.VALUE_NULL) {
            return;
        }
        if (context.doc().getField(name()) != null) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes one value per document");
        }
        int dim = fieldType().dimension;
        if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes an array of token vectors, got " + parser.currentToken());
        }
        List<float[]> tokens = new ArrayList<>();
        for (XContentParser.Token t = parser.nextToken(); t != XContentParser.Token.END_ARRAY; t = parser.nextToken()) {
            if (t != XContentParser.Token.START_ARRAY) {
                throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes an array of token vectors, got " + t);
            }
            float[] v = new float[dim];
            int k = 0;
            for (XContentParser.Token c = parser.nextToken(); c != XContentParser.Token.END_ARRAY; c = parser.nextToken()) {
                if (c != XContentParser.Token.VALUE_NUMBER && c != XContentParser.Token.VALUE_STRING) {
                    throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes arrays of numbers, got " + c);
                }
                float f = parser.floatValue();
                if (Float.isFinite(f) == false) {
                    throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes finite numbers, got " + f);
                }
                if (k < dim) {
                    v[k] = f;
                }
                k++;
            }
            if (k != dim) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Vector dimension mismatch. Expected: %d, Given: %d", dim, k)
                );
            }
            tokens.add(v);
        }
        if (tokens.isEmpty()) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] needs at least one token vector");
        }
        float[][] value = tokens.toArray(new float[0][]);
        byte[] bytes = fieldType().encoding.encode(value, dim);
        context.doc().add(new BinaryDocValuesField(name(), new BytesRef(bytes)));
        if (fieldType().pooled) {
            float[] mean = MultiVectorEncoding.pooledMean(value);
            if (mean != null) {
                context.doc().add(new KnnFloatVectorField(fieldType().pooledField(), mean, VectorSimilarityFunction.DOT_PRODUCT));
            }
        }
    }

    /** Not searchable but through {@code max_sim}. */
    public static final class MultiVectorFieldType extends MappedFieldType {
        final int dimension;
        final MultiVectorEncoding encoding;
        final boolean pooled;

        MultiVectorFieldType(String name, int dimension, MultiVectorEncoding encoding, boolean pooled, Map<String, String> meta) {
            super(name, false, false, true, TextSearchInfo.NONE, meta);
            this.dimension = dimension;
            this.encoding = encoding;
            this.pooled = pooled;
        }

        /** Whether the documents' pooled means are kept, for {@code max_sim}'s {@code candidates}. */
        public boolean pooled() {
            return pooled;
        }

        /** The hidden vector field holding the documents' pooled means. */
        public String pooledField() {
            return name() + POOLED_SUFFIX;
        }

        public int dimension() {
            return dimension;
        }

        public MultiVectorEncoding encoding() {
            return encoding;
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        @Override
        public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
            return SourceValueFetcher.identity(name(), context, format);
        }

        @Override
        public Query termQuery(Object value, QueryShardContext context) {
            throw new IllegalArgumentException("Queries on [" + CONTENT_TYPE + "] fields are not supported but through [max_sim]");
        }

        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields do not support sorting, scripting or aggregating");
        }
    }
}
