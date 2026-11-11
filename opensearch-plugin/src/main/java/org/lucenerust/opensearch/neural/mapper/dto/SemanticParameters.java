/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.mapper.dto;

import java.util.Map;

/**
 * A DTO to hold all the semantic parameters.
 */
public class SemanticParameters {
    private final String modelId;
    private final String searchModelId;
    private final String rawFieldType;
    private final String semanticInfoFieldName;
    private final ChunkingConfig chunkingConfig;
    private final String semanticFieldSearchAnalyzer;
    private final Map<String, Object> denseEmbeddingConfig;
    private final SparseEncodingConfig sparseEncodingConfig;
    private final ModelSelection modelSelection;

    public boolean isChunkingEnabled() {
        if (chunkingConfig == null) {
            return false;
        }
        return chunkingConfig.isEnabled();
    }

    private final Boolean skipExistingEmbedding;

    SemanticParameters(final String modelId, final String searchModelId, final String rawFieldType, final String semanticInfoFieldName, final ChunkingConfig chunkingConfig, final String semanticFieldSearchAnalyzer, final Map<String, Object> denseEmbeddingConfig, final SparseEncodingConfig sparseEncodingConfig, final ModelSelection modelSelection, final Boolean skipExistingEmbedding) {
        this.modelId = modelId;
        this.searchModelId = searchModelId;
        this.rawFieldType = rawFieldType;
        this.semanticInfoFieldName = semanticInfoFieldName;
        this.chunkingConfig = chunkingConfig;
        this.semanticFieldSearchAnalyzer = semanticFieldSearchAnalyzer;
        this.denseEmbeddingConfig = denseEmbeddingConfig;
        this.sparseEncodingConfig = sparseEncodingConfig;
        this.modelSelection = modelSelection;
        this.skipExistingEmbedding = skipExistingEmbedding;
    }


    public static class SemanticParametersBuilder {
        private String modelId;
        private String searchModelId;
        private String rawFieldType;
        private String semanticInfoFieldName;
        private ChunkingConfig chunkingConfig;
        private String semanticFieldSearchAnalyzer;
        private Map<String, Object> denseEmbeddingConfig;
        private SparseEncodingConfig sparseEncodingConfig;
        private ModelSelection modelSelection;
        private Boolean skipExistingEmbedding;

        SemanticParametersBuilder() {
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder modelId(final String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder searchModelId(final String searchModelId) {
            this.searchModelId = searchModelId;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder rawFieldType(final String rawFieldType) {
            this.rawFieldType = rawFieldType;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder semanticInfoFieldName(final String semanticInfoFieldName) {
            this.semanticInfoFieldName = semanticInfoFieldName;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder chunkingConfig(final ChunkingConfig chunkingConfig) {
            this.chunkingConfig = chunkingConfig;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder semanticFieldSearchAnalyzer(final String semanticFieldSearchAnalyzer) {
            this.semanticFieldSearchAnalyzer = semanticFieldSearchAnalyzer;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder denseEmbeddingConfig(final Map<String, Object> denseEmbeddingConfig) {
            this.denseEmbeddingConfig = denseEmbeddingConfig;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder sparseEncodingConfig(final SparseEncodingConfig sparseEncodingConfig) {
            this.sparseEncodingConfig = sparseEncodingConfig;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder modelSelection(final ModelSelection modelSelection) {
            this.modelSelection = modelSelection;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public SemanticParameters.SemanticParametersBuilder skipExistingEmbedding(final Boolean skipExistingEmbedding) {
            this.skipExistingEmbedding = skipExistingEmbedding;
            return this;
        }

        public SemanticParameters build() {
            return new SemanticParameters(this.modelId, this.searchModelId, this.rawFieldType, this.semanticInfoFieldName, this.chunkingConfig, this.semanticFieldSearchAnalyzer, this.denseEmbeddingConfig, this.sparseEncodingConfig, this.modelSelection, this.skipExistingEmbedding);
        }

        @java.lang.Override
        public java.lang.String toString() {
            return "SemanticParameters.SemanticParametersBuilder(modelId=" + this.modelId + ", searchModelId=" + this.searchModelId + ", rawFieldType=" + this.rawFieldType + ", semanticInfoFieldName=" + this.semanticInfoFieldName + ", chunkingConfig=" + this.chunkingConfig + ", semanticFieldSearchAnalyzer=" + this.semanticFieldSearchAnalyzer + ", denseEmbeddingConfig=" + this.denseEmbeddingConfig + ", sparseEncodingConfig=" + this.sparseEncodingConfig + ", modelSelection=" + this.modelSelection + ", skipExistingEmbedding=" + this.skipExistingEmbedding + ")";
        }
    }

    public static SemanticParameters.SemanticParametersBuilder builder() {
        return new SemanticParameters.SemanticParametersBuilder();
    }

    public String getModelId() {
        return this.modelId;
    }

    public String getSearchModelId() {
        return this.searchModelId;
    }

    public String getRawFieldType() {
        return this.rawFieldType;
    }

    public String getSemanticInfoFieldName() {
        return this.semanticInfoFieldName;
    }

    public ChunkingConfig getChunkingConfig() {
        return this.chunkingConfig;
    }

    public String getSemanticFieldSearchAnalyzer() {
        return this.semanticFieldSearchAnalyzer;
    }

    public Map<String, Object> getDenseEmbeddingConfig() {
        return this.denseEmbeddingConfig;
    }

    public SparseEncodingConfig getSparseEncodingConfig() {
        return this.sparseEncodingConfig;
    }

    public ModelSelection getModelSelection() {
        return this.modelSelection;
    }

    public Boolean getSkipExistingEmbedding() {
        return this.skipExistingEmbedding;
    }
}
