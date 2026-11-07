/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.VectorUtil;
import org.lucenerust.opensearch.engine.VectorsFormatSpec;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceValueFetcher;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.time.ZoneId;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * k-NN's {@code knn_vector} field. In an {@code index.knn} index each value is a Lucene vector field
 * ({@code KnnFloatVectorField} or {@code KnnByteVectorField}) whose similarity is the space type's,
 * carrying the stock plugin's field attributes and {@link VectorsFormatSpec#ATTRIBUTE}, from which the
 * Rust writer builds its HNSW graph; elsewhere it is doc values only, as the stock plugin's flat
 * mapper writes it. {@code doc_values} and {@code stored} keep the vector as the stock plugin
 * serializes it (big-endian floats, or the bytes).
 */
public final class KnnVectorFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "knn_vector";
    /** The stock plugin's field attributes. */
    static final String KNN_FIELD = "knn_field";
    static final String DIMENSION = "dimension";
    static final String SPACE_TYPE = "space_type";
    static final String KNN_ENGINE = "knn_engine";
    static final String DATA_TYPE = "data_type";

    private static KnnVectorFieldMapper toType(FieldMapper in) {
        return (KnnVectorFieldMapper) in;
    }

    private static String engineOrDefault(String engine) {
        return engine.equals("undefined") ? KnnFieldConfig.LUCENE : engine;
    }

    /**
     * A method as persisted (engine, space type and parameters filled in) and as a request repeats
     * it are the same method: keys only one names are the resolved defaults.
     */
    static boolean sameMethod(Map<String, Object> a, Map<String, Object> b) {
        if (a == null || b == null) {
            return a == b;
        }
        for (String k : List.of("engine", SPACE_TYPE, "name")) {
            Object x = a.get(k), y = b.get(k);
            if (x != null && y != null && String.valueOf(x).equalsIgnoreCase(String.valueOf(y)) == false) {
                return false;
            }
        }
        Object pa = a.getOrDefault("parameters", Map.of()), pb = b.getOrDefault("parameters", Map.of());
        return Objects.equals(pa, pb);
    }

    /** {@code KNNVectorFieldMapper.Builder}. */
    public static final class Builder extends ParametrizedFieldMapper.Builder {
        final Parameter<Integer> dimension = new Parameter<>(
            DIMENSION,
            false,
            () -> -1,
            (n, c, o) -> XContentMapValues.nodeIntegerValue(o),
            m -> toType(m).dimension
        );
        // Parsed in any case and persisted upper case, as stock's VectorDataType.
        final Parameter<String> dataType = new Parameter<>(
            DATA_TYPE,
            false,
            () -> "float",
            (n, c, o) -> XContentMapValues.nodeStringValue(o, null).toLowerCase(Locale.ROOT),
            m -> toType(m).dataType
        ).setSerializer((b, n, v) -> b.field(n, v.toUpperCase(Locale.ROOT)), v -> v);
        final Parameter<String> spaceType = Parameter.stringParam(SPACE_TYPE, false, m -> toType(m).spaceType, SpaceType.UNDEFINED);
        final Parameter<String> engine = Parameter.stringParam("engine", false, m -> toType(m).engine, "undefined")
            .setMergeValidator((a, b) -> engineOrDefault(a).equalsIgnoreCase(engineOrDefault(b)));
        final Parameter<Map<String, Object>> method = new Parameter<Map<String, Object>>(
            "method",
            false,
            () -> null,
            (n, c, o) -> XContentMapValues.nodeMapValue(o, n),
            m -> toType(m).method
        ).setSerializer((b, n, v) -> {
            if (v != null) {
                b.field(n, v);
            }
        }, v -> String.valueOf(v)).setMergeValidator(KnnVectorFieldMapper::sameMethod);
        final Parameter<String> mode = Parameter.stringParam("mode", false, m -> toType(m).mode, null).acceptsNull();
        final Parameter<String> compression = Parameter.stringParam("compression_level", false, m -> toType(m).compression, null)
            .acceptsNull();
        final Parameter<String> modelId = Parameter.stringParam("model_id", false, m -> null, null).acceptsNull();
        final Parameter<Boolean> docValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, false);
        final Parameter<Boolean> stored = Parameter.storeParam(m -> toType(m).stored, false);
        final Parameter<Map<String, String>> meta = Parameter.metaParam();

        private final Settings indexSettings;

        Builder(String name, Settings indexSettings) {
            super(name);
            this.indexSettings = indexSettings;
            builder = this;
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(dimension, dataType, spaceType, engine, method, mode, compression, modelId, docValues, stored, meta);
        }

        KnnFieldConfig config() {
            if (modelId.getValue() != null) {
                throw new IllegalArgumentException("[" + CONTENT_TYPE + "] [model_id] needs the k-NN models API, which this node does not serve");
            }
            return KnnFieldConfig.resolve(
                name,
                dimension.getValue(),
                dataType.getValue(),
                spaceType.getValue(),
                engine.getValue(),
                method.getValue(),
                mode.getValue(),
                compression.getValue(),
                indexSettings.get(KnnSettings.SPACE_TYPE.getKey())
            );
        }

        @Override
        public KnnVectorFieldMapper build(BuilderContext context) {
            boolean knnIndex = KnnSettings.KNN_INDEX.get(indexSettings);
            KnnFieldConfig config = config();
            if (method.getValue() != null) {
                // Persisted resolved, as stock does; the engine so a stock node, whose default is faiss, reads it as lucene.
                Map<String, Object> resolved = new LinkedHashMap<>();
                resolved.put("engine", config.engine());
                resolved.put(SPACE_TYPE, config.spaceType().value());
                resolved.put("name", method.getValue().get("name"));
                resolved.put("parameters", method.getValue().getOrDefault("parameters", Map.of()));
                method.setValue(resolved);
            } else if (engine.isConfigured() == false) {
                engine.setValue(config.engine());
            }
            // The stock plugin's flat mapper keeps doc values by default.
            boolean dv = docValues.isConfigured() ? docValues.getValue() : knnIndex == false;
            KnnVectorFieldType ft = new KnnVectorFieldType(buildFullName(context), meta.getValue(), config, knnIndex, dv);
            return new KnnVectorFieldMapper(name, ft, multiFieldsBuilder.build(this, context), copyTo.build(), this, config, knnIndex, dv);
        }
    }

    /** {@code KNNVectorFieldMapper.TypeParser}: the mapping is resolved here, so a bad one fails the request. */
    public static final class TypeParser implements Mapper.TypeParser {
        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) {
            Builder builder = new Builder(name, parserContext.getSettings());
            builder.parse(name, parserContext, node);
            builder.config();
            return builder;
        }
    }

    final int dimension;
    final String dataType;
    final String spaceType;
    final String engine;
    final Map<String, Object> method;
    final String mode;
    final String compression;
    final boolean hasDocValues;
    final boolean stored;
    /** {@code FlatVectorFieldMapper}'s half-float doc values: binary, the field marked {@code data_type: half_float}. */
    private static final FieldType HALF_FLOAT_DOC_VALUES = halfFloatDocValues();

    private static FieldType halfFloatDocValues() {
        FieldType ft = new FieldType();
        ft.setDocValuesType(DocValuesType.BINARY);
        ft.putAttribute("data_type", "half_float");
        ft.freeze();
        return ft;
    }

    private final Settings indexSettings;
    private final KnnFieldConfig config;
    private final FieldType vectorType;
    /** {@link #vectorType} marked for derived source, in an index that masks it. */
    private final FieldType derivedType;
    /** {@link #derivedType} for the field's values in nested documents. */
    private final FieldType nestedDerivedType;
    private volatile Boolean derivedField;

    private KnnVectorFieldMapper(
        String simpleName,
        KnnVectorFieldType type,
        MultiFields multiFields,
        CopyTo copyTo,
        Builder b,
        KnnFieldConfig config,
        boolean knnIndex,
        boolean docValues
    ) {
        super(simpleName, type, multiFields, copyTo);
        this.dimension = b.dimension.getValue();
        this.dataType = b.dataType.getValue();
        this.spaceType = b.spaceType.getValue();
        this.engine = b.engine.getValue();
        this.method = b.method.getValue();
        this.mode = b.mode.getValue();
        this.compression = b.compression.getValue();
        this.hasDocValues = docValues;
        this.stored = b.stored.getValue();
        this.indexSettings = b.indexSettings;
        this.config = config;
        if (knnIndex) {
            FieldType ft = new FieldType();
            ft.setVectorAttributes(config.vectorDimension(), config.dataType(), config.spaceType().similarity());
            ft.putAttribute(KNN_FIELD, "true");
            ft.putAttribute(DIMENSION, Integer.toString(config.dimension()));
            ft.putAttribute(SPACE_TYPE, config.spaceType().value());
            ft.putAttribute(KNN_ENGINE, config.engine());
            ft.putAttribute(
                DATA_TYPE,
                config.binary() ? "binary" : config.halfFloat() ? "half_float" : config.dataType() == VectorEncoding.BYTE ? "byte" : "float"
            );
            ft.putAttribute(VectorsFormatSpec.ATTRIBUTE, config.format().attribute());
            ft.freeze();
            this.vectorType = ft;
            boolean derived = DerivedSource.forIndex(indexSettings);
            this.derivedType = derived ? DerivedSource.marked(ft, false) : null;
            this.nestedDerivedType = derived ? DerivedSource.marked(ft, true) : null;
        } else {
            this.vectorType = null;
            this.derivedType = null;
            this.nestedDerivedType = null;
        }
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName(), indexSettings).init(this);
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    protected KnnVectorFieldMapper clone() {
        return (KnnVectorFieldMapper) super.clone();
    }

    @Override
    public KnnVectorFieldType fieldType() {
        return (KnnVectorFieldType) super.fieldType();
    }

    /** The whole array is one vector. */
    @Override
    public boolean parsesArrayValue() {
        return true;
    }

    /** {@code KNNVectorFieldMapper.parseCreateField}: one array of numbers, checked as the stock plugin checks it. */
    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        if (context.externalValueSet()) {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] fields can't be used in multi-fields");
        }
        XContentParser parser = context.parser();
        if (parser.currentToken() == XContentParser.Token.VALUE_NULL) {
            return;
        }
        List<Float> values = new ArrayList<>(config.vectorDimension());
        if (parser.currentToken() == XContentParser.Token.START_ARRAY) {
            for (XContentParser.Token t = parser.nextToken(); t != XContentParser.Token.END_ARRAY; t = parser.nextToken()) {
                if (t != XContentParser.Token.VALUE_NUMBER && t != XContentParser.Token.VALUE_STRING) {
                    throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes an array of numbers, got " + t);
                }
                values.add(parser.floatValue());
            }
        } else if (parser.currentToken() == XContentParser.Token.VALUE_NUMBER) {
            values.add(parser.floatValue());
        } else {
            throw new IllegalArgumentException("[" + CONTENT_TYPE + "] field [" + name() + "] takes an array of numbers, got " + parser.currentToken());
        }
        if (values.size() != config.vectorDimension()) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Vector dimension mismatch. Expected: %d, Given: %d", config.vectorDimension(), values.size())
            );
        }
        if (config.dataType() == VectorEncoding.BYTE) {
            byte[] v = KnnVectors.toBytes(values);
            if (vectorType != null) {
                FieldType type = vectorType(context);
                context.doc().add(masked(context) ? new DerivedSource.MaskedByte(name(), v, type) : new KnnByteVectorField(name(), v, type));
            }
            keep(context, v);
        } else {
            float[] v = KnnVectors.toFloats(values, config.spaceType());
            if (config.halfFloat()) {
                KnnVectors.checkHalfFloats(v);
                // VectorTransformerFactory: k-NN normalizes half-float cosine vectors before storing them.
                if (config.spaceType() == SpaceType.COSINESIMIL) {
                    VectorUtil.l2normalize(v);
                }
            }
            if (vectorType != null) {
                FieldType type = vectorType(context);
                context.doc().add(masked(context) ? new DerivedSource.MaskedFloat(name(), v, type) : new KnnFloatVectorField(name(), v, type));
            }
            if (config.halfFloat()) {
                // LuceneFieldStrategy: no doc values on a knn index; stored as little-endian half
                // floats. FlatVectorFieldMapper (not a knn index): half-float doc values.
                if (vectorType == null && hasDocValues) {
                    context.doc().add(new Field(name(), new org.apache.lucene.util.BytesRef(KnnVectors.halfFloatBytes(v)), HALF_FLOAT_DOC_VALUES));
                }
                if (stored) {
                    context.doc().add(new StoredField(name(), KnnVectors.halfFloatBytes(v)));
                }
            } else if (hasDocValues || stored) {
                ByteBuffer bytes = ByteBuffer.allocate(v.length * Float.BYTES);
                for (float f : v) {
                    bytes.putFloat(f);
                }
                keep(context, bytes.array());
            }
        }
    }

    /** The marked type for every field of a masking index: the nested one in a nested document. */
    private FieldType vectorType(ParseContext context) {
        if (derivedType == null) {
            return vectorType;
        }
        return context.doc() == context.rootDoc() ? derivedType : nestedDerivedType;
    }

    /** Whether the field's value is masked in {@code _source} ({@code isDerivedEnabled}). */
    private boolean masked(ParseContext context) {
        if (derivedType == null) {
            return false;
        }
        if (derivedField == null) {
            derivedField = DerivedSource.forField(name(), context.docMapper());
        }
        return derivedField;
    }

    private void keep(ParseContext context, byte[] serialized) {
        if (hasDocValues) {
            context.doc().add(new BinaryDocValuesField(name(), new org.apache.lucene.util.BytesRef(serialized)));
        }
        if (stored) {
            context.doc().add(new StoredField(name(), serialized));
        }
    }

    /** {@code KNNVectorFieldType}: searched by the {@code knn} query only. */
    public static final class KnnVectorFieldType extends MappedFieldType {
        private final KnnFieldConfig config;
        private final boolean knnIndex;

        KnnVectorFieldType(String name, Map<String, String> meta, KnnFieldConfig config, boolean knnIndex, boolean docValues) {
            super(name, false, false, docValues, TextSearchInfo.NONE, meta);
            this.config = config;
            this.knnIndex = knnIndex;
        }

        public KnnFieldConfig config() {
            return config;
        }

        /** Whether the field has a vector index ({@code index.knn}). */
        public boolean isIndexedForAnn() {
            return knnIndex;
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
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "KNN vector do not support exact searching, use KNN queries instead: [%s]", name())
            );
        }

        @Override
        public Query existsQuery(QueryShardContext context) {
            return new FieldExistsQuery(name());
        }

        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            return new KnnVectorFieldData.Builder(name(), config);
        }

        @Override
        public DocValueFormat docValueFormat(String format, ZoneId timeZone) {
            if (timeZone != null) {
                throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] does not support custom time zones");
            }
            return KnnVectorFieldData.Format.of(format);
        }

        @Override
        public Object valueForDisplay(Object value) {
            return value instanceof BytesRef b ? KnnVectorFieldData.stored(b, config) : value;
        }
    }
}
