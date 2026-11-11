/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.query.dto;

public class NeuralQueryTargetFieldConfig {
    private final Boolean isSemanticField;
    private final Boolean isUnmappedField;
    private final String searchModelId;
    private final String embeddingFieldType;
    private final String embeddingFieldPath;
    private final String chunksPath;
    private final Boolean chunkingEnabled;
    private final String semanticFieldSearchAnalyzer;

    NeuralQueryTargetFieldConfig(final Boolean isSemanticField, final Boolean isUnmappedField, final String searchModelId, final String embeddingFieldType, final String embeddingFieldPath, final String chunksPath, final Boolean chunkingEnabled, final String semanticFieldSearchAnalyzer) {
        this.isSemanticField = isSemanticField;
        this.isUnmappedField = isUnmappedField;
        this.searchModelId = searchModelId;
        this.embeddingFieldType = embeddingFieldType;
        this.embeddingFieldPath = embeddingFieldPath;
        this.chunksPath = chunksPath;
        this.chunkingEnabled = chunkingEnabled;
        this.semanticFieldSearchAnalyzer = semanticFieldSearchAnalyzer;
    }


    public static class NeuralQueryTargetFieldConfigBuilder {
        private Boolean isSemanticField;
        private Boolean isUnmappedField;
        private String searchModelId;
        private String embeddingFieldType;
        private String embeddingFieldPath;
        private String chunksPath;
        private Boolean chunkingEnabled;
        private String semanticFieldSearchAnalyzer;

        NeuralQueryTargetFieldConfigBuilder() {
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder isSemanticField(final Boolean isSemanticField) {
            this.isSemanticField = isSemanticField;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder isUnmappedField(final Boolean isUnmappedField) {
            this.isUnmappedField = isUnmappedField;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder searchModelId(final String searchModelId) {
            this.searchModelId = searchModelId;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder embeddingFieldType(final String embeddingFieldType) {
            this.embeddingFieldType = embeddingFieldType;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder embeddingFieldPath(final String embeddingFieldPath) {
            this.embeddingFieldPath = embeddingFieldPath;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder chunksPath(final String chunksPath) {
            this.chunksPath = chunksPath;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder chunkingEnabled(final Boolean chunkingEnabled) {
            this.chunkingEnabled = chunkingEnabled;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder semanticFieldSearchAnalyzer(final String semanticFieldSearchAnalyzer) {
            this.semanticFieldSearchAnalyzer = semanticFieldSearchAnalyzer;
            return this;
        }

        public NeuralQueryTargetFieldConfig build() {
            return new NeuralQueryTargetFieldConfig(this.isSemanticField, this.isUnmappedField, this.searchModelId, this.embeddingFieldType, this.embeddingFieldPath, this.chunksPath, this.chunkingEnabled, this.semanticFieldSearchAnalyzer);
        }

        @java.lang.Override
        public java.lang.String toString() {
            return "NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder(isSemanticField=" + this.isSemanticField + ", isUnmappedField=" + this.isUnmappedField + ", searchModelId=" + this.searchModelId + ", embeddingFieldType=" + this.embeddingFieldType + ", embeddingFieldPath=" + this.embeddingFieldPath + ", chunksPath=" + this.chunksPath + ", chunkingEnabled=" + this.chunkingEnabled + ", semanticFieldSearchAnalyzer=" + this.semanticFieldSearchAnalyzer + ")";
        }
    }

    public static NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder builder() {
        return new NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder();
    }

    public Boolean getIsSemanticField() {
        return this.isSemanticField;
    }

    public Boolean getIsUnmappedField() {
        return this.isUnmappedField;
    }

    public String getSearchModelId() {
        return this.searchModelId;
    }

    public String getEmbeddingFieldType() {
        return this.embeddingFieldType;
    }

    public String getEmbeddingFieldPath() {
        return this.embeddingFieldPath;
    }

    public String getChunksPath() {
        return this.chunksPath;
    }

    public Boolean getChunkingEnabled() {
        return this.chunkingEnabled;
    }

    public String getSemanticFieldSearchAnalyzer() {
        return this.semanticFieldSearchAnalyzer;
    }

    @java.lang.Override
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof NeuralQueryTargetFieldConfig)) return false;
        final NeuralQueryTargetFieldConfig other = (NeuralQueryTargetFieldConfig) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        final java.lang.Object this$isSemanticField = this.getIsSemanticField();
        final java.lang.Object other$isSemanticField = other.getIsSemanticField();
        if (this$isSemanticField == null ? other$isSemanticField != null : !this$isSemanticField.equals(other$isSemanticField)) return false;
        final java.lang.Object this$isUnmappedField = this.getIsUnmappedField();
        final java.lang.Object other$isUnmappedField = other.getIsUnmappedField();
        if (this$isUnmappedField == null ? other$isUnmappedField != null : !this$isUnmappedField.equals(other$isUnmappedField)) return false;
        final java.lang.Object this$chunkingEnabled = this.getChunkingEnabled();
        final java.lang.Object other$chunkingEnabled = other.getChunkingEnabled();
        if (this$chunkingEnabled == null ? other$chunkingEnabled != null : !this$chunkingEnabled.equals(other$chunkingEnabled)) return false;
        final java.lang.Object this$searchModelId = this.getSearchModelId();
        final java.lang.Object other$searchModelId = other.getSearchModelId();
        if (this$searchModelId == null ? other$searchModelId != null : !this$searchModelId.equals(other$searchModelId)) return false;
        final java.lang.Object this$embeddingFieldType = this.getEmbeddingFieldType();
        final java.lang.Object other$embeddingFieldType = other.getEmbeddingFieldType();
        if (this$embeddingFieldType == null ? other$embeddingFieldType != null : !this$embeddingFieldType.equals(other$embeddingFieldType)) return false;
        final java.lang.Object this$embeddingFieldPath = this.getEmbeddingFieldPath();
        final java.lang.Object other$embeddingFieldPath = other.getEmbeddingFieldPath();
        if (this$embeddingFieldPath == null ? other$embeddingFieldPath != null : !this$embeddingFieldPath.equals(other$embeddingFieldPath)) return false;
        final java.lang.Object this$chunksPath = this.getChunksPath();
        final java.lang.Object other$chunksPath = other.getChunksPath();
        if (this$chunksPath == null ? other$chunksPath != null : !this$chunksPath.equals(other$chunksPath)) return false;
        final java.lang.Object this$semanticFieldSearchAnalyzer = this.getSemanticFieldSearchAnalyzer();
        final java.lang.Object other$semanticFieldSearchAnalyzer = other.getSemanticFieldSearchAnalyzer();
        if (this$semanticFieldSearchAnalyzer == null ? other$semanticFieldSearchAnalyzer != null : !this$semanticFieldSearchAnalyzer.equals(other$semanticFieldSearchAnalyzer)) return false;
        return true;
    }

    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof NeuralQueryTargetFieldConfig;
    }

    @java.lang.Override
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final java.lang.Object $isSemanticField = this.getIsSemanticField();
        result = result * PRIME + ($isSemanticField == null ? 43 : $isSemanticField.hashCode());
        final java.lang.Object $isUnmappedField = this.getIsUnmappedField();
        result = result * PRIME + ($isUnmappedField == null ? 43 : $isUnmappedField.hashCode());
        final java.lang.Object $chunkingEnabled = this.getChunkingEnabled();
        result = result * PRIME + ($chunkingEnabled == null ? 43 : $chunkingEnabled.hashCode());
        final java.lang.Object $searchModelId = this.getSearchModelId();
        result = result * PRIME + ($searchModelId == null ? 43 : $searchModelId.hashCode());
        final java.lang.Object $embeddingFieldType = this.getEmbeddingFieldType();
        result = result * PRIME + ($embeddingFieldType == null ? 43 : $embeddingFieldType.hashCode());
        final java.lang.Object $embeddingFieldPath = this.getEmbeddingFieldPath();
        result = result * PRIME + ($embeddingFieldPath == null ? 43 : $embeddingFieldPath.hashCode());
        final java.lang.Object $chunksPath = this.getChunksPath();
        result = result * PRIME + ($chunksPath == null ? 43 : $chunksPath.hashCode());
        final java.lang.Object $semanticFieldSearchAnalyzer = this.getSemanticFieldSearchAnalyzer();
        result = result * PRIME + ($semanticFieldSearchAnalyzer == null ? 43 : $semanticFieldSearchAnalyzer.hashCode());
        return result;
    }

    @java.lang.Override
    public java.lang.String toString() {
        return "NeuralQueryTargetFieldConfig(isSemanticField=" + this.getIsSemanticField() + ", isUnmappedField=" + this.getIsUnmappedField() + ", searchModelId=" + this.getSearchModelId() + ", embeddingFieldType=" + this.getEmbeddingFieldType() + ", embeddingFieldPath=" + this.getEmbeddingFieldPath() + ", chunksPath=" + this.getChunksPath() + ", chunkingEnabled=" + this.getChunkingEnabled() + ", semanticFieldSearchAnalyzer=" + this.getSemanticFieldSearchAnalyzer() + ")";
    }
}
