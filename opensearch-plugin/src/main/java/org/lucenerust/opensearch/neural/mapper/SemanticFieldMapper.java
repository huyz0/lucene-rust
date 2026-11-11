/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.mapper;

import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.FilterFieldType;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.MatchOnlyTextFieldMapper;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.TextFieldMapper;
import org.lucenerust.opensearch.neural.constants.MappingConstants;
import org.opensearch.index.mapper.WildcardFieldMapper;
import org.lucenerust.opensearch.neural.mapper.dto.ChunkingConfig;
import org.lucenerust.opensearch.neural.mapper.dto.ModelSelection;
import org.lucenerust.opensearch.neural.mapper.dto.SemanticParameters;
import org.lucenerust.opensearch.neural.processor.chunker.ChunkerValidatorFactory;
import org.lucenerust.opensearch.neural.mapper.dto.SparseEncodingConfig;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import static org.lucenerust.opensearch.neural.constants.MappingConstants.PATH_SEPARATOR;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.CHUNKING;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.DENSE_EMBEDDING_CONFIG;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.MODEL_ID;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.MODEL_SELECTION;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.RAW_FIELD_TYPE;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SKIP_EXISTING_EMBEDDING;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SEARCH_MODEL_ID;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SEMANTIC_FIELD_SEARCH_ANALYZER;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SPARSE_ENCODING_CONFIG;

/**
 * FieldMapper for the semantic field. It will hold a delegate field mapper to delegate the data parsing and query work
 * based on the raw_field_type.
 */
public class SemanticFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "semantic";
    private final SemanticParameters semanticParameters;
    private ParametrizedFieldMapper delegateFieldMapper;

    protected SemanticFieldMapper(String simpleName, MappedFieldType mappedFieldType, MultiFields multiFields, CopyTo copyTo, ParametrizedFieldMapper delegateFieldMapper, SemanticParameters semanticParameters) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.delegateFieldMapper = delegateFieldMapper;
        this.semanticParameters = semanticParameters;
    }

    @Override
    public Builder getMergeBuilder() {
        Builder semanticFieldMapperBuilder = (Builder) new Builder(simpleName()).init(this);
        ParametrizedFieldMapper.Builder delegateBuilder = delegateFieldMapper.getMergeBuilder();
        semanticFieldMapperBuilder.setDelegateBuilder(delegateBuilder);
        return semanticFieldMapperBuilder;
    }

    @Override
    public Iterator<Mapper> iterator() {
        return delegateFieldMapper.iterator();
    }

    @Override
    public final ParametrizedFieldMapper merge(Mapper mergeWith) {
        if (mergeWith instanceof SemanticFieldMapper) {
            try {
                delegateFieldMapper = delegateFieldMapper.merge(((SemanticFieldMapper) mergeWith).delegateFieldMapper);
            } catch (IllegalArgumentException e) {
                final String err = String.format(Locale.ROOT, "Failed to update the mapper %s because failed to update the delegate mapper for the raw_field_type %s due to %s", this.name(), this.semanticParameters.getRawFieldType(), e.getMessage());
                throw new IllegalArgumentException(err, e);
            }
        }
        return super.merge(mergeWith);
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        delegateFieldMapper.parse(context);
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }


    public static class Builder extends ParametrizedFieldMapper.Builder {
        protected final Parameter<String> modelId = Parameter.stringParam(MODEL_ID, true, m -> ((SemanticFieldMapper) m).semanticParameters.getModelId(), null);
        protected final Parameter<String> searchModelId = Parameter.stringParam(SEARCH_MODEL_ID, true, m -> ((SemanticFieldMapper) m).semanticParameters.getSearchModelId(), null);
        protected final Parameter<String> rawFieldType = Parameter.stringParam(RAW_FIELD_TYPE, false, m -> ((SemanticFieldMapper) m).semanticParameters.getRawFieldType(), TextFieldMapper.CONTENT_TYPE);
        protected final Parameter<String> semanticInfoFieldName = Parameter.stringParam(SEMANTIC_INFO_FIELD_NAME, false, m -> ((SemanticFieldMapper) m).semanticParameters.getSemanticInfoFieldName(), null);
        protected final Parameter<ChunkingConfig> chunkingConfig = new Parameter<>(CHUNKING, false, () -> null, (name, ctx, value) -> new ChunkingConfig(name, value, new ChunkerValidatorFactory()), m -> ((SemanticFieldMapper) m).semanticParameters.getChunkingConfig()).setSerializer((builder, name, value) -> {
            if (value == null) {
                builder.nullField(name);
            } else {
                value.toXContent(builder, name);
            }
        }, value -> value == null ? null : value.toString());
        protected final Parameter<String> semanticFieldSearchAnalyzer = Parameter.stringParam(SEMANTIC_FIELD_SEARCH_ANALYZER, true, m -> ((SemanticFieldMapper) m).semanticParameters.getSemanticFieldSearchAnalyzer(), null);
        protected final Parameter<Map<String, Object>> denseEmbeddingConfig = new Parameter<>(DENSE_EMBEDDING_CONFIG, false, () -> null, (name, ctx, o) -> {
            if (o == null) {
                return null;
            } else if (o instanceof Map<?, ?>) {
                return asStringObjectMap(o);
            } else {
                throw new MapperParsingException("[" + DENSE_EMBEDDING_CONFIG + "] must be an object");
            }
        }, m -> ((SemanticFieldMapper) m).semanticParameters.getDenseEmbeddingConfig()).setSerializer((builder, name, value) -> {
            if (value == null) {
                builder.nullField(name);
            } else {
                builder.startObject(name);
                builder.mapContents(value);
                builder.endObject();
            }
        }, v -> v == null ? null : v.toString());
        protected final Parameter<SparseEncodingConfig> sparseEncodingConfig = new Parameter<>(SPARSE_ENCODING_CONFIG, false, () -> null, (name, ctx, value) -> new SparseEncodingConfig(name, value), m -> ((SemanticFieldMapper) m).semanticParameters.getSparseEncodingConfig()).setSerializer((builder, name, value) -> {
            if (value == null) {
                builder.nullField(name);
            } else {
                value.toXContent(builder, name);
            }
        }, value -> value == null ? null : value.toString());
        protected final Parameter<Boolean> skipExistingEmbedding = Parameter.boolParam(SKIP_EXISTING_EMBEDDING, true, m -> ((SemanticFieldMapper) m).semanticParameters.getSkipExistingEmbedding(), false);
        protected final Parameter<ModelSelection> modelSelection = new Parameter<>(MODEL_SELECTION, true, () -> null, (name, ctx, value) -> value == null ? null : new ModelSelection(name, value), m -> ((SemanticFieldMapper) m).semanticParameters.getModelSelection()).setSerializer((builder, name, value) -> {
            if (value == null) {
                builder.nullField(name);
            } else {
                value.toXContent(builder, name);
            }
        }, value -> value == null ? null : value.toString());
        protected ParametrizedFieldMapper.Builder delegateBuilder;

        protected Builder(String name) {
            super(name);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(modelId, searchModelId, rawFieldType, semanticInfoFieldName, chunkingConfig, semanticFieldSearchAnalyzer, denseEmbeddingConfig, sparseEncodingConfig, skipExistingEmbedding, modelSelection);
        }

        @Override
        public SemanticFieldMapper build(BuilderContext context) {
            final ParametrizedFieldMapper delegateMapper = delegateBuilder.build(context);
            final SemanticParameters semanticParameters = this.getSemanticParameters();
            final MappedFieldType semanticFieldType = new SemanticFieldType(delegateMapper.fieldType(), semanticParameters);
            return new SemanticFieldMapper(name, semanticFieldType, multiFieldsBuilder.build(this, context), copyTo.build(), delegateMapper, semanticParameters);
        }

        public SemanticParameters getSemanticParameters() {
            return SemanticParameters.builder().modelId(modelId.getValue()).searchModelId(searchModelId.getValue()).rawFieldType(rawFieldType.getValue()).semanticInfoFieldName(semanticInfoFieldName.getValue()).chunkingConfig(chunkingConfig.getValue()).semanticFieldSearchAnalyzer(semanticFieldSearchAnalyzer.getValue()).denseEmbeddingConfig(denseEmbeddingConfig.getValue()).sparseEncodingConfig(sparseEncodingConfig.getValue()).skipExistingEmbedding(skipExistingEmbedding.getValue()).modelSelection(modelSelection.getValue()).build();
        }

        public Parameter<String> getModelId() {
            return this.modelId;
        }

        public Parameter<String> getSearchModelId() {
            return this.searchModelId;
        }

        public Parameter<String> getRawFieldType() {
            return this.rawFieldType;
        }

        public Parameter<String> getSemanticInfoFieldName() {
            return this.semanticInfoFieldName;
        }

        public Parameter<ChunkingConfig> getChunkingConfig() {
            return this.chunkingConfig;
        }

        public Parameter<Boolean> getSkipExistingEmbedding() {
            return this.skipExistingEmbedding;
        }

        public Parameter<ModelSelection> getModelSelection() {
            return this.modelSelection;
        }

        public void setDelegateBuilder(final ParametrizedFieldMapper.Builder delegateBuilder) {
            this.delegateBuilder = delegateBuilder;
        }
    }


    public static class TypeParser implements Mapper.TypeParser {
        private static final Set<String> SUPPORTED_RAW_FIELD_TYPE = Set.of(TextFieldMapper.CONTENT_TYPE, KeywordFieldMapper.CONTENT_TYPE, MatchOnlyTextFieldMapper.CONTENT_TYPE, WildcardFieldMapper.CONTENT_TYPE, MappingConstants.TOKEN_COUNT, BinaryFieldMapper.CONTENT_TYPE);

        @Override
        public Builder parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            final String rawFieldType = (String) node.getOrDefault(RAW_FIELD_TYPE, TextFieldMapper.CONTENT_TYPE);
            final String searchModelId = (String) node.getOrDefault(SEARCH_MODEL_ID, null);
            final String semanticFieldSearchAnalyzer = (String) node.getOrDefault(SEMANTIC_FIELD_SEARCH_ANALYZER, null);
            validateRawFieldType(rawFieldType);
            validateSearchModelIdAndSemanticFieldSearchAnalyzer(searchModelId, semanticFieldSearchAnalyzer, name);
            final ParametrizedFieldMapper.TypeParser typeParser = (ParametrizedFieldMapper.TypeParser) parserContext.typeParser(rawFieldType);
            final Builder semanticFieldMapperBuilder = new Builder(name);
            // semantic field mapper builder parse semantic fields
            Map<String, Object> semanticConfig = extractSemanticConfig(node, semanticFieldMapperBuilder.getParameters(), rawFieldType);
            semanticFieldMapperBuilder.parse(name, parserContext, semanticConfig);
            // delegate field mapper builder parse remaining fields
            ParametrizedFieldMapper.Builder delegateBuilder = typeParser.parse(name, node, parserContext);
            semanticFieldMapperBuilder.setDelegateBuilder(delegateBuilder);
            return semanticFieldMapperBuilder;
        }

        private void validateRawFieldType(final String rawFieldType) {
            if (rawFieldType == null || !SUPPORTED_RAW_FIELD_TYPE.contains(rawFieldType)) {
                final String err = String.format(Locale.ROOT, "raw_field_type %s is not supported. It should be one of [%s]", rawFieldType, String.join(", ", SUPPORTED_RAW_FIELD_TYPE));
                throw new IllegalArgumentException(err);
            }
        }

        private void validateSearchModelIdAndSemanticFieldSearchAnalyzer(final String searchModelId, final String semanticFieldSearchAnalyzer, final String name) {
            if (searchModelId != null && semanticFieldSearchAnalyzer != null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "%s can not coexist with %s in semantic field %s", SEARCH_MODEL_ID, SEMANTIC_FIELD_SEARCH_ANALYZER, name));
            }
            if (searchModelId != null && searchModelId.isEmpty()) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "%s can not be empty string in semantic field %s", SEARCH_MODEL_ID, name));
            }
            if (semanticFieldSearchAnalyzer != null && semanticFieldSearchAnalyzer.isEmpty()) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "%s can not be empty string in semantic field %s", SEMANTIC_FIELD_SEARCH_ANALYZER, name));
            }
        }

        /**
         * In this function we will extract all the parameters defined in the semantic field mapper builder and parse it
         * later. The remaining parameters will be processed by the type parser of the raw field type. Here we cannot
         * pass the parameters defined by semantic field to the delegate type parser of the raw field type because it
         * cannot recognize them.
         * @param node field config
         * @param parameters parameters for semantic field
         * @param rawFieldType field type of the raw data
         * @return semantic field config
         */
        private Map<String, Object> extractSemanticConfig(Map<String, Object> node, List<Parameter<?>> parameters, String rawFieldType) {
            final Map<String, Object> semanticConfig = new HashMap<>();
            for (Parameter<?> parameter : parameters) {
                Object config = node.get(parameter.name);
                if (config != null) {
                    semanticConfig.put(parameter.name, config);
                    node.remove(parameter.name);
                }
            }
            semanticConfig.put(MappingConstants.TYPE, SemanticFieldMapper.CONTENT_TYPE);
            node.put(MappingConstants.TYPE, rawFieldType);
            return semanticConfig;
        }
    }


    public static class SemanticFieldType extends FilterFieldType {
        private SemanticParameters semanticParameters;

        public SemanticFieldType(final MappedFieldType delegate, final SemanticParameters semanticParameters) {
            super(delegate);
            java.util.Objects.requireNonNull(delegate, "delegate is marked non-null but is null");
            java.util.Objects.requireNonNull(semanticParameters, "semanticParameters is marked non-null but is null");
            this.semanticParameters = semanticParameters;
        }

        @Override
        public String typeName() {
            return SemanticFieldMapper.CONTENT_TYPE;
        }

        public String getSemanticInfoFieldPath() {
            final String[] paths = name().split("\\.");
            final String semanticInfoFieldName = semanticParameters.getSemanticInfoFieldName();
            paths[paths.length - 1] = semanticInfoFieldName == null ? paths[paths.length - 1] + DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX : semanticInfoFieldName;
            return String.join(PATH_SEPARATOR, paths);
        }

        public SemanticParameters getSemanticParameters() {
            return this.semanticParameters;
        }
    }

    @Override
    protected void doXContentBody(XContentBuilder builder, boolean includeDefaults, Params params) throws IOException {
        builder.field(MappingConstants.TYPE, contentType());
        // semantic parameters
        final List<Parameter<?>> parameters = getMergeBuilder().getParameters();
        for (Parameter<?> parameter : parameters) {
            // By default, we will not return the default value. But raw_field_type is useful info to let users know how
            // we will handle the raw data. So we explicitly return it even it is using the default value.
            if (RAW_FIELD_TYPE.equals(parameter.name)) {
                parameter.toXContent(builder, true);
            } else {
                parameter.toXContent(builder, includeDefaults);
            }
        }
        // non-semantic parameters
        // semantic field mapper itself does not handle multi fields or copy to. The delegate field mapper will handle it.
        delegateFieldMapper.multiFields().toXContent(builder, params);
        delegateFieldMapper.copyTo().toXContent(builder, params);
        delegateFieldMapper.getMergeBuilder().toXContent(builder, includeDefaults);
    }

    public void setDelegateFieldMapper(final ParametrizedFieldMapper delegateFieldMapper) {
        this.delegateFieldMapper = delegateFieldMapper;
    }

    public ParametrizedFieldMapper getDelegateFieldMapper() {
        return this.delegateFieldMapper;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asStringObjectMap(Object o) {
        return (Map<String, Object>) o;
    }
}
