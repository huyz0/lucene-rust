/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.document.FeatureField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.search.Query;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.FieldMapper;
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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * neural-search's {@code sparse_vector} field: each token a {@code FeatureField} term, and the whole
 * vector as binary doc values (big-endian {@code int} token, {@code float} weight pairs), the field
 * carrying its {@code method}'s attributes. The Rust writer clusters the field's postings from
 * those doc values in every segment of {@code index.sparse} index at or over
 * {@code approximate_threshold} documents, in neural-search's file format.
 */
public final class SparseVectorFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "sparse_vector";
    static final String METHOD = "method";
    /** {@code SparseVectorField.SPARSE_FIELD}. */
    static final String SPARSE_FIELD = "sparse_vector_field";

    private static final FieldType DEFAULT_FIELD_TYPE = new FieldType();

    static {
        DEFAULT_FIELD_TYPE.setTokenized(false);
        DEFAULT_FIELD_TYPE.setIndexOptions(IndexOptions.NONE);
        DEFAULT_FIELD_TYPE.putAttribute(SPARSE_FIELD, "true");
        DEFAULT_FIELD_TYPE.freeze();
    }

    private final SparseMethod method;
    private final FieldType vectorType;

    private SparseVectorFieldMapper(String simpleName, MappedFieldType type, MultiFields multiFields, CopyTo copyTo, SparseMethod method) {
        super(simpleName, type, multiFields, copyTo);
        this.method = method;
        FieldType ft = new FieldType(DEFAULT_FIELD_TYPE);
        ft.setDocValuesType(DocValuesType.BINARY);
        for (Map.Entry<String, String> a : method.attributes().entrySet()) {
            ft.putAttribute(a.getKey(), a.getValue());
        }
        ft.freeze();
        this.vectorType = ft;
    }

    private static SparseVectorFieldType ft(FieldMapper in) {
        return ((SparseVectorFieldMapper) in).fieldType();
    }

    /** {@code SparseVectorFieldMapper.Builder}. */
    public static final class Builder extends ParametrizedFieldMapper.Builder {
        final Parameter<SparseMethod> method = new Parameter<>(
            METHOD,
            false,
            () -> null,
            (n, c, o) -> SparseMethod.parse(o),
            m -> ft(m).method
        ).setSerializer((b, n, v) -> {
            b.startObject(n);
            v.toXContent(b, ToXContent.EMPTY_PARAMS);
            b.endObject();
        }, SparseMethod::toString);

        Builder(String name) {
            super(name);
            builder = this;
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(method);
        }

        @Override
        public ParametrizedFieldMapper build(BuilderContext context) {
            return new SparseVectorFieldMapper(
                name,
                new SparseVectorFieldType(buildFullName(context), method.getValue()),
                multiFieldsBuilder.build(this, context),
                copyTo.build(),
                method.getValue()
            );
        }
    }

    /** {@code SparseVectorFieldMapper.SparseTypeParser}. */
    public static final class TypeParser implements Mapper.TypeParser {
        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) {
            Builder builder = new Builder(name);
            builder.parse(name, parserContext, node);
            SparseMethod method = builder.method.getValue();
            if (method == null) {
                throw new MapperParsingException("[" + CONTENT_TYPE + "] requires [method] parameter");
            }
            method.validate();
            return builder;
        }
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
    protected SparseVectorFieldMapper clone() {
        return (SparseVectorFieldMapper) super.clone();
    }

    @Override
    public SparseVectorFieldType fieldType() {
        return (SparseVectorFieldType) super.fieldType();
    }

    /** {@code SparseVectorFieldMapper.parseCreateField}. */
    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        if (context.externalValueSet()) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields can't be used in multi-fields");
        }
        XContentParser parser = context.parser();
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new IllegalArgumentException(
                "[" + CONTENT_TYPE + "] fields must be json objects, expected a START_OBJECT but got: " + parser.currentToken()
            );
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bytes);
        String feature = "";
        for (XContentParser.Token token = parser.nextToken(); token != XContentParser.Token.END_OBJECT; token = parser.nextToken()) {
            if (token == XContentParser.Token.FIELD_NAME) {
                feature = parser.currentName();
            } else if (token == XContentParser.Token.VALUE_NULL) {
                // ignored, as numeric fields ignore a null
            } else if (token == XContentParser.Token.VALUE_NUMBER || token == XContentParser.Token.VALUE_STRING) {
                String key = name() + "." + feature;
                float value = parser.floatValue(true);
                if (context.doc().getByKey(key) != null) {
                    throw new IllegalArgumentException(
                        "["
                            + CONTENT_TYPE
                            + "] fields do not support indexing multiple values for the same key ["
                            + key
                            + "] in the same document"
                    );
                }
                context.doc().addWithKey(key, new FeatureField(name(), feature, value));
                try {
                    int tokenIndex = Integer.parseInt(feature);
                    if (tokenIndex < 0) {
                        throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields should be text of non-negative integer");
                    }
                    out.writeInt(tokenIndex);
                    out.writeFloat(value);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields should be valid integer");
                }
            } else {
                throw new IllegalArgumentException(
                    "["
                        + CONTENT_TYPE
                        + "] fields take hashes that map a feature to a strictly positive float, but got unexpected token "
                        + token
                );
            }
        }
        out.flush();
        context.doc().add(new Field(name(), bytes.toByteArray(), vectorType));
    }

    /** {@code SparseVectorFieldType}: not searchable but through {@code neural_sparse}. */
    public static final class SparseVectorFieldType extends MappedFieldType {
        final SparseMethod method;

        SparseVectorFieldType(String name, SparseMethod method) {
            super(name, false, false, false, TextSearchInfo.NONE, Map.of());
            this.method = method;
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
            throw new IllegalArgumentException("Queries on [" + CONTENT_TYPE + "] fields are not supported");
        }

        @Override
        public Query existsQuery(QueryShardContext context) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields do not support [exists] queries");
        }

        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields do not support sorting, scripting or aggregating");
        }
    }
}
