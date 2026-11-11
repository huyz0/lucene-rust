/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.mappingtransformer;

import org.lucenerust.opensearch.neural.ml.FunctionName;
import org.lucenerust.opensearch.neural.ml.MLModel;

import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.mapper.TextFieldMapper;
import reactor.util.annotation.NonNull;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import static org.lucenerust.opensearch.neural.constants.MappingConstants.PROPERTIES;
import static org.lucenerust.opensearch.neural.constants.MappingConstants.TYPE;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.DENSE_EMBEDDING_CONFIG;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SEMANTIC_FIELD_SEARCH_ANALYZER;
import static org.lucenerust.opensearch.neural.constants.SemanticFieldConstants.SPARSE_ENCODING_CONFIG;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.EMBEDDING_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.CHUNKS_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.CHUNKS_TEXT_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.DEFAULT_MODEL_CONFIG;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_DATA_TYPE_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_DIMENSION_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_DEFAULT_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_NAME_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.MODEL_FIELD_NAME;
import static org.lucenerust.opensearch.neural.mappingtransformer.SemanticMappingTransformer.SUPPORTED_MODEL_ALGORITHMS;
import static org.lucenerust.opensearch.neural.mappingtransformer.SemanticMappingTransformer.SUPPORTED_REMOTE_MODEL_TYPES;

/**
 * SemanticInfoConfigBuilder helps build the semantic info fields config based on the ML model config
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class SemanticInfoConfigBuilder {
    private final NamedXContentRegistry xContentRegistry;
    private String embeddingFieldType;
    private String spaceType;
    private String knnMethodName = KNN_VECTOR_METHOD_DEFAULT_NAME;
    private Integer embeddingDimension;
    private Boolean chunkingEnabled;
    private String semanticFieldSearchAnalyzer;
    private Map<String, Object> denseEmbeddingConfig;
    private boolean sparseEncodingConfigDefined;
    private static final List<String> UNSUPPORTED_DENSE_EMBEDDING_CONFIG = List.of(KNN_VECTOR_DIMENSION_FIELD_NAME, KNN_VECTOR_DATA_TYPE_FIELD_NAME, KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME);

    public SemanticInfoConfigBuilder(final NamedXContentRegistry xContentRegistry) {
        this.xContentRegistry = xContentRegistry;
    }

    /**
     * It will construct the index mapping config for the semantic info fields.
     * {
     *     "chunks":{
     *         "type": "nested",
     *         "properties":{
     *             "text":{
     *                 "type": "text"
     *             },
     *             "embedding":{
     *                 ... // embedding config
     *             }
     *         }
     *     },
     *     "model":{
     *         ... // model config
     *     }
     * }
     *
     * @return Config of the semantic info fields.
     */
    public Map<String, Object> build() {
        validate();
        final Map<String, Object> embeddingFieldConfig = switch (embeddingFieldType) {
            case "knn_vector" -> buildKnnFieldConfig();
            case "rank_features" -> buildRankFeaturesFieldConfig();
            default -> throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot build the semantic info config because the embedding field type %s is not supported.", embeddingFieldType));
        };
        if (Boolean.TRUE.equals(chunkingEnabled)) {
            final Map<String, Object> chunksConfig = Map.of(TYPE, ObjectMapper.NESTED_CONTENT_TYPE, PROPERTIES, Map.of(CHUNKS_TEXT_FIELD_NAME, Map.of(TYPE, TextFieldMapper.CONTENT_TYPE), EMBEDDING_FIELD_NAME, embeddingFieldConfig));
            return Map.of(PROPERTIES, Map.of(CHUNKS_FIELD_NAME, chunksConfig, MODEL_FIELD_NAME, DEFAULT_MODEL_CONFIG));
        } else {
            return Map.of(PROPERTIES, Map.of(EMBEDDING_FIELD_NAME, embeddingFieldConfig, MODEL_FIELD_NAME, DEFAULT_MODEL_CONFIG));
        }
    }

    private void validate() {
        if ("knn_vector".equals(embeddingFieldType)) {
            validateSearchAnalyzerNotDefined();
            validateSparseEncodingConfigNotDefined();
        }
        if ("rank_features".equals(embeddingFieldType)) {
            validateDenseEmbeddingConfigNotDefined();
        }
    }

    private void validateSparseEncodingConfigNotDefined() {
        if (sparseEncodingConfigDefined) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot build the semantic info config because the dense(text embedding) model cannot support %s.", SPARSE_ENCODING_CONFIG));
        }
    }

    private void validateDenseEmbeddingConfigNotDefined() {
        if (denseEmbeddingConfig != null && "rank_features".equals(embeddingFieldType)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot build the semantic info config because %s is not supported by the sparse model.", DENSE_EMBEDDING_CONFIG));
        }
    }

    private void validateSearchAnalyzerNotDefined() {
        if (semanticFieldSearchAnalyzer != null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot build the semantic info config because the dense(text embedding) model cannot support %s", SEMANTIC_FIELD_SEARCH_ANALYZER));
        }
    }

    private Map<String, Object> buildKnnFieldConfig() {
        final Map<String, Object> config = new HashMap<>();
        config.put(TYPE, "knn_vector");
        config.put(KNN_VECTOR_DIMENSION_FIELD_NAME, this.embeddingDimension);
        final Map<String, Object> methodConfig = new HashMap<>();
        if (denseEmbeddingConfig != null) {
            validateDenseEmbeddingConfig(denseEmbeddingConfig);
            config.putAll(denseEmbeddingConfig);
            final Object methodConfigObj = denseEmbeddingConfig.get(KNN_VECTOR_METHOD_FIELD_NAME);
            if (methodConfigObj != null) {
                if (methodConfigObj instanceof Map<?, ?> methodConfigMap) {
                    // Create a shallow copy so we don't modify the original
                    methodConfig.putAll((Map<String, Object>) methodConfigMap);
                } else {
                    throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot build the semantic info config because %s must be a Map when provided.", KNN_VECTOR_METHOD_FIELD_NAME));
                }
            }
        }
        // Safe to override or add new method values
        methodConfig.putIfAbsent(KNN_VECTOR_METHOD_NAME_FIELD_NAME, this.knnMethodName);
        // Always use the one we found from the model config
        methodConfig.put(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME, this.spaceType);
        config.put(KNN_VECTOR_METHOD_FIELD_NAME, methodConfig);
        return config;
    }

    private void validateDenseEmbeddingConfig(final Map<String, Object> denseEmbeddingConfig) {
        UNSUPPORTED_DENSE_EMBEDDING_CONFIG.forEach(name -> {
            if (denseEmbeddingConfig.get(name) != null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot configure %s in %s in the semantic field.", name, DENSE_EMBEDDING_CONFIG));
            }
        });
        if (denseEmbeddingConfig.get(KNN_VECTOR_METHOD_FIELD_NAME) != null && denseEmbeddingConfig.get(KNN_VECTOR_METHOD_FIELD_NAME) instanceof Map methodConfigMap) {
            if (methodConfigMap.get(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME) != null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Cannot configure %s in %s in the semantic field.", KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME, DENSE_EMBEDDING_CONFIG));
            }
        }
    }

    private Map<String, Object> buildRankFeaturesFieldConfig() {
        final Map<String, Object> config = new HashMap<>();
        config.put(TYPE, "rank_features");
        return config;
    }

    // Here we also require the model id because sometimes the MLModel does not have that info.
    /**
     * Extract the info from the ML model and set it in the SemanticInfoConfigBuilder for build later.
     * @param mlModel ML model info
     * @param modelId ID of the ML model
     * @return this
     */
    public SemanticInfoConfigBuilder mlModel(final MLModel mlModel, final String modelId) {
        switch (mlModel.getFunctionName()) {
            case FunctionName.TEXT_EMBEDDING -> extractInfoForTextEmbeddingModel(mlModel, modelId, false);
            case FunctionName.SPARSE_ENCODING, FunctionName.SPARSE_TOKENIZE -> extractInfoForSparseModel();
            case FunctionName.REMOTE -> extractInfoForRemoteModel(mlModel, modelId);
            default -> throw new IllegalArgumentException(String.format(Locale.ROOT, "The algorithm %s of the model %s is not supported in the semantic field. Supported algorithms: [%s].", mlModel.getAlgorithm(), modelId, String.join(",", SUPPORTED_MODEL_ALGORITHMS)));
        }
        return this;
    }

    private void extractInfoForTextEmbeddingModel(final MLModel mlModel, final String modelId, final boolean isRemote) {
        this.embeddingFieldType = "knn_vector";
        // ml-commons parses a text embedding model's model_config as a TextEmbeddingModelConfig and a
        // remote model's as a RemoteModelConfig; both carry embedding_dimension and additional_config.
        TextEmbeddingInfo textEmbeddingInfo = new TextEmbeddingInfo(mlModel.getEmbeddingDimension(), mlModel.getAdditionalConfig());
        validateAndSetTextEmbeddingInfo(textEmbeddingInfo, modelId);
    }

    private void validateAndSetTextEmbeddingInfo(final TextEmbeddingInfo textEmbeddingInfo, final String modelId) {
        if (textEmbeddingInfo.getEmbeddingDimension() == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Model %s is a text embedding model, but the embedding dimension is not defined in the model config.", modelId));
        }
        this.embeddingDimension = textEmbeddingInfo.getEmbeddingDimension();
        if (textEmbeddingInfo.getAdditionalConfig() == null || textEmbeddingInfo.getAdditionalConfig().get(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME) instanceof String == false) {
            throw createInvalidSpaceTypeException(modelId);
        }
        this.spaceType = (String) textEmbeddingInfo.getAdditionalConfig().get(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME);
    }

    private IllegalArgumentException createInvalidSpaceTypeException(final String modelId) {
        return new IllegalArgumentException(String.format(Locale.ROOT, "space_type is not defined or not a string in the additional_config of the model %s.", modelId));
    }

    private void extractInfoForSparseModel() {
        this.embeddingFieldType = "rank_features";
    }

    private void extractInfoForRemoteModel(final MLModel mlModel, final String modelId) {
        if (mlModel.getModelConfig() == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Model config is null for the remote model %s.", modelId));
        }
        final String modelType = mlModel.getModelType();
        final FunctionName modelTypeFunctionName;
        try {
            modelTypeFunctionName = FunctionName.from(modelType);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(getUnsupportedRemoteModelError(modelType, modelId));
        }
        switch (modelTypeFunctionName) {
            case FunctionName.TEXT_EMBEDDING -> extractInfoForTextEmbeddingModel(mlModel, modelId, true);
            case FunctionName.SPARSE_ENCODING, FunctionName.SPARSE_TOKENIZE -> extractInfoForSparseModel();
            default -> throw new IllegalArgumentException(getUnsupportedRemoteModelError(modelType, modelId));
        }
    }

    private String getUnsupportedRemoteModelError(final String modelType, final String modelId) {
        java.util.Objects.requireNonNull(modelId, "modelId is marked non-null but is null");
        return String.format(Locale.ROOT, "The model type %s of the model %s is not supported in the semantic field. Supported remote model types: [%s]", modelType, modelId, String.join(",", SUPPORTED_REMOTE_MODEL_TYPES));
    }

    public SemanticInfoConfigBuilder chunkingEnabled(final Boolean chunkingEnabled) {
        this.chunkingEnabled = chunkingEnabled;
        return this;
    }

    public SemanticInfoConfigBuilder semanticFieldSearchAnalyzer(final String semanticFieldSearchAnalyzer) {
        this.semanticFieldSearchAnalyzer = semanticFieldSearchAnalyzer;
        return this;
    }

    public SemanticInfoConfigBuilder denseEmbeddingConfig(final Map<String, Object> denseEmbeddingConfig) {
        this.denseEmbeddingConfig = denseEmbeddingConfig;
        return this;
    }

    public SemanticInfoConfigBuilder sparseEncodingConfigDefined(final boolean sparseEncodingConfigDefined) {
        this.sparseEncodingConfigDefined = sparseEncodingConfigDefined;
        return this;
    }
}
