/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.dto;

import org.opensearch.index.analysis.AnalysisRegistry;
import org.lucenerust.opensearch.neural.mapper.dto.ChunkingConfig;
import org.lucenerust.opensearch.neural.mapper.dto.SparseEncodingConfig;
import org.lucenerust.opensearch.neural.processor.chunker.Chunker;
import org.lucenerust.opensearch.neural.processor.chunker.ChunkerFactory;
import org.lucenerust.opensearch.neural.processor.chunker.FixedTokenLengthChunker;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import static org.lucenerust.opensearch.neural.constants.MappingConstants.PATH_SEPARATOR;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.EMBEDDING_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.CHUNKS_FIELD_NAME;
import static org.lucenerust.opensearch.neural.constants.SemanticInfoFieldConstants.MODEL_FIELD_NAME;
import static org.lucenerust.opensearch.neural.mapper.dto.ChunkingConfig.ALGORITHM_FIELD;
import static org.lucenerust.opensearch.neural.mapper.dto.ChunkingConfig.PARAMETERS_FIELD;

/**
 * SemanticFieldInfo is a data transfer object to help hold semantic field info
 */
@SuppressWarnings("unchecked")
public class SemanticFieldInfo {
    /**
     * The raw string value of the semantic field
     */
    private String value;
    /**
     * The model id of the semantic field which will be used to generate the embedding
     */
    private String modelId;
    /**
     * The full path to the semantic field in the index mapping
     */
    private String semanticFieldFullPathInMapping;
    /**
     * The full path to the semantic field in the doc
     */
    private String semanticFieldFullPathInDoc;
    /**
     * The full path to the semantic info fields in the doc. The path in the doc will contain the index of the inter
     * nested object.
     */
    private String semanticInfoFullPathInDoc;
    /**
     * If the chunking is enabled for the field
     */
    private Boolean chunkingEnabled;
    /**
     * A list of chunkers that will be used to chunk the semantic field. e.g. If we have chunker1 and chunker2 then
     * we will use chunker1 to chunk the original text "test text" as ["test", "text"]. Then we will use the chunker2
     * to further chunk the chunked text as ["te", "st", "te", "xt"].
     */
    private List<Chunker> chunkers;
    /**
     * The chunked strings of the original string value of the semantic field
     */
    private List<String> chunks;
    private SparseEncodingConfig sparseEncodingConfig;
    /**
     * If we should skip the embedding generation for the semantic field with exist reusable embedding. We think the
     * embedding is reusable if it exists and semantic field value and model are not changed.
     */
    private Boolean skipExistingEmbedding;
    /**
     * The id of the doc of the semantic field
     */
    private String docId;

    /**
     * @return full path to the chunks field of the semantic field in a doc
     */
    public String getFullPathForChunksInDoc() {
        if (Boolean.TRUE.equals(chunkingEnabled)) {
            return new StringBuilder().append(semanticInfoFullPathInDoc).append(PATH_SEPARATOR).append(CHUNKS_FIELD_NAME).toString();
        }
        throw new IllegalStateException(String.format(Locale.ROOT, "Should not try to get full path to chunks for the semantic field at %s when the chunking is not enabled.", semanticFieldFullPathInMapping));
    }

    /**
     * Return the full path to the embedding field in the doc.
     * @param index index of the chunk the embedding is in. If the chunking is not enabled then the index must be 0.
     * @return full path to the embedding field of the semantic field
     */
    public String getFullPathForEmbeddingInDoc(int index) {
        if (Boolean.TRUE.equals(chunkingEnabled)) {
            return new StringBuilder().append(semanticInfoFullPathInDoc).append(PATH_SEPARATOR).append(CHUNKS_FIELD_NAME).append(PATH_SEPARATOR).append(index).append(PATH_SEPARATOR).append(EMBEDDING_FIELD_NAME).toString();
        } else {
            if (index != 0) {
                throw new IllegalStateException(String.format(Locale.ROOT, "Should not try to get the full path for the embedding with index %d when the chunking is not enabled for the semantic field at %s.", index, semanticFieldFullPathInMapping));
            }
            return new StringBuilder().append(semanticInfoFullPathInDoc).append(PATH_SEPARATOR).append(EMBEDDING_FIELD_NAME).toString();
        }
    }

    /**
     * @return full path to the model info fields in the doc
     */
    public String getFullPathForModelInfoInDoc() {
        return new StringBuilder().append(semanticInfoFullPathInDoc).append(PATH_SEPARATOR).append(MODEL_FIELD_NAME).toString();
    }

    public void setChunkingConfig(final ChunkingConfig chunkingConfig, final AnalysisRegistry analysisRegistry) {
        java.util.Objects.requireNonNull(analysisRegistry, "analysisRegistry is marked non-null but is null");
        if (chunkingConfig == null) {
            chunkingEnabled = false;
            return;
        }
        chunkingEnabled = chunkingConfig.isEnabled();
        if (chunkingConfig.getConfigs() != null && chunkingConfig.getConfigs().isEmpty() == false) {
            chunkers = new ArrayList<>();
            for (Map<String, Object> config : chunkingConfig.getConfigs()) {
                final String algorithm = (String) config.get(ALGORITHM_FIELD);
                final Object paramObject = config.get(PARAMETERS_FIELD);
                final Map<String, Object> parameters = paramObject == null ? new HashMap<>() : new HashMap<>((Map<String, Object>) paramObject);
                if (FixedTokenLengthChunker.ALGORITHM_NAME.equals(algorithm)) {
                    parameters.put(FixedTokenLengthChunker.ANALYSIS_REGISTRY_FIELD, analysisRegistry);
                }
                chunkers.add(ChunkerFactory.create(algorithm, parameters));
            }
        }
    }

    /**
     * Creates a new {@code SemanticFieldInfo} instance.
     *
     * @param value The raw string value of the semantic field
     * @param modelId The model id of the semantic field which will be used to generate the embedding
     * @param semanticFieldFullPathInMapping The full path to the semantic field in the index mapping
     * @param semanticFieldFullPathInDoc The full path to the semantic field in the doc
     * @param semanticInfoFullPathInDoc The full path to the semantic info fields in the doc. The path in the doc will contain the index of the inter
     * nested object.
     * @param chunkingEnabled If the chunking is enabled for the field
     * @param chunkers A list of chunkers that will be used to chunk the semantic field. e.g. If we have chunker1 and chunker2 then
     * we will use chunker1 to chunk the original text "test text" as ["test", "text"]. Then we will use the chunker2
     * to further chunk the chunked text as ["te", "st", "te", "xt"].
     * @param chunks The chunked strings of the original string value of the semantic field
     * @param sparseEncodingConfig
     * @param skipExistingEmbedding If we should skip the embedding generation for the semantic field with exist reusable embedding. We think the
     * embedding is reusable if it exists and semantic field value and model are not changed.
     * @param docId The id of the doc of the semantic field
     */
    
    
    SemanticFieldInfo(final String value, final String modelId, final String semanticFieldFullPathInMapping, final String semanticFieldFullPathInDoc, final String semanticInfoFullPathInDoc, final Boolean chunkingEnabled, final List<Chunker> chunkers, final List<String> chunks, final SparseEncodingConfig sparseEncodingConfig, final Boolean skipExistingEmbedding, final String docId) {
        this.value = value;
        this.modelId = modelId;
        this.semanticFieldFullPathInMapping = semanticFieldFullPathInMapping;
        this.semanticFieldFullPathInDoc = semanticFieldFullPathInDoc;
        this.semanticInfoFullPathInDoc = semanticInfoFullPathInDoc;
        this.chunkingEnabled = chunkingEnabled;
        this.chunkers = chunkers;
        this.chunks = chunks;
        this.sparseEncodingConfig = sparseEncodingConfig;
        this.skipExistingEmbedding = skipExistingEmbedding;
        this.docId = docId;
    }


    
    
    public static class SemanticFieldInfoBuilder {
        
        
        private String value;
        
        
        private String modelId;
        
        
        private String semanticFieldFullPathInMapping;
        
        
        private String semanticFieldFullPathInDoc;
        
        
        private String semanticInfoFullPathInDoc;
        
        
        private Boolean chunkingEnabled;
        
        
        private List<Chunker> chunkers;
        
        
        private List<String> chunks;
        
        
        private SparseEncodingConfig sparseEncodingConfig;
        
        
        private Boolean skipExistingEmbedding;
        
        
        private String docId;

        
        
        SemanticFieldInfoBuilder() {
        }

        /**
         * The raw string value of the semantic field
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder value(final String value) {
            this.value = value;
            return this;
        }

        /**
         * The model id of the semantic field which will be used to generate the embedding
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder modelId(final String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * The full path to the semantic field in the index mapping
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder semanticFieldFullPathInMapping(final String semanticFieldFullPathInMapping) {
            this.semanticFieldFullPathInMapping = semanticFieldFullPathInMapping;
            return this;
        }

        /**
         * The full path to the semantic field in the doc
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder semanticFieldFullPathInDoc(final String semanticFieldFullPathInDoc) {
            this.semanticFieldFullPathInDoc = semanticFieldFullPathInDoc;
            return this;
        }

        /**
         * The full path to the semantic info fields in the doc. The path in the doc will contain the index of the inter
         * nested object.
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder semanticInfoFullPathInDoc(final String semanticInfoFullPathInDoc) {
            this.semanticInfoFullPathInDoc = semanticInfoFullPathInDoc;
            return this;
        }

        /**
         * If the chunking is enabled for the field
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder chunkingEnabled(final Boolean chunkingEnabled) {
            this.chunkingEnabled = chunkingEnabled;
            return this;
        }

        /**
         * A list of chunkers that will be used to chunk the semantic field. e.g. If we have chunker1 and chunker2 then
         * we will use chunker1 to chunk the original text "test text" as ["test", "text"]. Then we will use the chunker2
         * to further chunk the chunked text as ["te", "st", "te", "xt"].
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder chunkers(final List<Chunker> chunkers) {
            this.chunkers = chunkers;
            return this;
        }

        /**
         * The chunked strings of the original string value of the semantic field
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder chunks(final List<String> chunks) {
            this.chunks = chunks;
            return this;
        }

        /**
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder sparseEncodingConfig(final SparseEncodingConfig sparseEncodingConfig) {
            this.sparseEncodingConfig = sparseEncodingConfig;
            return this;
        }

        /**
         * If we should skip the embedding generation for the semantic field with exist reusable embedding. We think the
         * embedding is reusable if it exists and semantic field value and model are not changed.
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder skipExistingEmbedding(final Boolean skipExistingEmbedding) {
            this.skipExistingEmbedding = skipExistingEmbedding;
            return this;
        }

        /**
         * The id of the doc of the semantic field
         * @return {@code this}.
         */
        
        
        public SemanticFieldInfo.SemanticFieldInfoBuilder docId(final String docId) {
            this.docId = docId;
            return this;
        }

        
        
        public SemanticFieldInfo build() {
            return new SemanticFieldInfo(this.value, this.modelId, this.semanticFieldFullPathInMapping, this.semanticFieldFullPathInDoc, this.semanticInfoFullPathInDoc, this.chunkingEnabled, this.chunkers, this.chunks, this.sparseEncodingConfig, this.skipExistingEmbedding, this.docId);
        }

        @java.lang.Override
        
        
        public java.lang.String toString() {
            return "SemanticFieldInfo.SemanticFieldInfoBuilder(value=" + this.value + ", modelId=" + this.modelId + ", semanticFieldFullPathInMapping=" + this.semanticFieldFullPathInMapping + ", semanticFieldFullPathInDoc=" + this.semanticFieldFullPathInDoc + ", semanticInfoFullPathInDoc=" + this.semanticInfoFullPathInDoc + ", chunkingEnabled=" + this.chunkingEnabled + ", chunkers=" + this.chunkers + ", chunks=" + this.chunks + ", sparseEncodingConfig=" + this.sparseEncodingConfig + ", skipExistingEmbedding=" + this.skipExistingEmbedding + ", docId=" + this.docId + ")";
        }
    }

    
    
    public static SemanticFieldInfo.SemanticFieldInfoBuilder builder() {
        return new SemanticFieldInfo.SemanticFieldInfoBuilder();
    }

    /**
     * The raw string value of the semantic field
     */
    
    
    public String getValue() {
        return this.value;
    }

    /**
     * The model id of the semantic field which will be used to generate the embedding
     */
    
    
    public String getModelId() {
        return this.modelId;
    }

    /**
     * The full path to the semantic field in the index mapping
     */
    
    
    public String getSemanticFieldFullPathInMapping() {
        return this.semanticFieldFullPathInMapping;
    }

    /**
     * The full path to the semantic field in the doc
     */
    
    
    public String getSemanticFieldFullPathInDoc() {
        return this.semanticFieldFullPathInDoc;
    }

    /**
     * The full path to the semantic info fields in the doc. The path in the doc will contain the index of the inter
     * nested object.
     */
    
    
    public String getSemanticInfoFullPathInDoc() {
        return this.semanticInfoFullPathInDoc;
    }

    /**
     * If the chunking is enabled for the field
     */
    
    
    public Boolean getChunkingEnabled() {
        return this.chunkingEnabled;
    }

    /**
     * A list of chunkers that will be used to chunk the semantic field. e.g. If we have chunker1 and chunker2 then
     * we will use chunker1 to chunk the original text "test text" as ["test", "text"]. Then we will use the chunker2
     * to further chunk the chunked text as ["te", "st", "te", "xt"].
     */
    
    
    public List<Chunker> getChunkers() {
        return this.chunkers;
    }

    /**
     * The chunked strings of the original string value of the semantic field
     */
    
    
    public List<String> getChunks() {
        return this.chunks;
    }

    
    
    public SparseEncodingConfig getSparseEncodingConfig() {
        return this.sparseEncodingConfig;
    }

    /**
     * If we should skip the embedding generation for the semantic field with exist reusable embedding. We think the
     * embedding is reusable if it exists and semantic field value and model are not changed.
     */
    
    
    public Boolean getSkipExistingEmbedding() {
        return this.skipExistingEmbedding;
    }

    /**
     * The id of the doc of the semantic field
     */
    
    
    public String getDocId() {
        return this.docId;
    }

    /**
     * The raw string value of the semantic field
     */
    
    
    public void setValue(final String value) {
        this.value = value;
    }

    /**
     * The model id of the semantic field which will be used to generate the embedding
     */
    
    
    public void setModelId(final String modelId) {
        this.modelId = modelId;
    }

    /**
     * The full path to the semantic field in the index mapping
     */
    
    
    public void setSemanticFieldFullPathInMapping(final String semanticFieldFullPathInMapping) {
        this.semanticFieldFullPathInMapping = semanticFieldFullPathInMapping;
    }

    /**
     * The full path to the semantic field in the doc
     */
    
    
    public void setSemanticFieldFullPathInDoc(final String semanticFieldFullPathInDoc) {
        this.semanticFieldFullPathInDoc = semanticFieldFullPathInDoc;
    }

    /**
     * The full path to the semantic info fields in the doc. The path in the doc will contain the index of the inter
     * nested object.
     */
    
    
    public void setSemanticInfoFullPathInDoc(final String semanticInfoFullPathInDoc) {
        this.semanticInfoFullPathInDoc = semanticInfoFullPathInDoc;
    }

    /**
     * If the chunking is enabled for the field
     */
    
    
    public void setChunkingEnabled(final Boolean chunkingEnabled) {
        this.chunkingEnabled = chunkingEnabled;
    }

    /**
     * A list of chunkers that will be used to chunk the semantic field. e.g. If we have chunker1 and chunker2 then
     * we will use chunker1 to chunk the original text "test text" as ["test", "text"]. Then we will use the chunker2
     * to further chunk the chunked text as ["te", "st", "te", "xt"].
     */
    
    
    public void setChunkers(final List<Chunker> chunkers) {
        this.chunkers = chunkers;
    }

    /**
     * The chunked strings of the original string value of the semantic field
     */
    
    
    public void setChunks(final List<String> chunks) {
        this.chunks = chunks;
    }

    
    
    public void setSparseEncodingConfig(final SparseEncodingConfig sparseEncodingConfig) {
        this.sparseEncodingConfig = sparseEncodingConfig;
    }

    /**
     * If we should skip the embedding generation for the semantic field with exist reusable embedding. We think the
     * embedding is reusable if it exists and semantic field value and model are not changed.
     */
    
    
    public void setSkipExistingEmbedding(final Boolean skipExistingEmbedding) {
        this.skipExistingEmbedding = skipExistingEmbedding;
    }

    /**
     * The id of the doc of the semantic field
     */
    
    
    public void setDocId(final String docId) {
        this.docId = docId;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof SemanticFieldInfo)) return false;
        final SemanticFieldInfo other = (SemanticFieldInfo) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        final java.lang.Object this$chunkingEnabled = this.getChunkingEnabled();
        final java.lang.Object other$chunkingEnabled = other.getChunkingEnabled();
        if (this$chunkingEnabled == null ? other$chunkingEnabled != null : !this$chunkingEnabled.equals(other$chunkingEnabled)) return false;
        final java.lang.Object this$skipExistingEmbedding = this.getSkipExistingEmbedding();
        final java.lang.Object other$skipExistingEmbedding = other.getSkipExistingEmbedding();
        if (this$skipExistingEmbedding == null ? other$skipExistingEmbedding != null : !this$skipExistingEmbedding.equals(other$skipExistingEmbedding)) return false;
        final java.lang.Object this$value = this.getValue();
        final java.lang.Object other$value = other.getValue();
        if (this$value == null ? other$value != null : !this$value.equals(other$value)) return false;
        final java.lang.Object this$modelId = this.getModelId();
        final java.lang.Object other$modelId = other.getModelId();
        if (this$modelId == null ? other$modelId != null : !this$modelId.equals(other$modelId)) return false;
        final java.lang.Object this$semanticFieldFullPathInMapping = this.getSemanticFieldFullPathInMapping();
        final java.lang.Object other$semanticFieldFullPathInMapping = other.getSemanticFieldFullPathInMapping();
        if (this$semanticFieldFullPathInMapping == null ? other$semanticFieldFullPathInMapping != null : !this$semanticFieldFullPathInMapping.equals(other$semanticFieldFullPathInMapping)) return false;
        final java.lang.Object this$semanticFieldFullPathInDoc = this.getSemanticFieldFullPathInDoc();
        final java.lang.Object other$semanticFieldFullPathInDoc = other.getSemanticFieldFullPathInDoc();
        if (this$semanticFieldFullPathInDoc == null ? other$semanticFieldFullPathInDoc != null : !this$semanticFieldFullPathInDoc.equals(other$semanticFieldFullPathInDoc)) return false;
        final java.lang.Object this$semanticInfoFullPathInDoc = this.getSemanticInfoFullPathInDoc();
        final java.lang.Object other$semanticInfoFullPathInDoc = other.getSemanticInfoFullPathInDoc();
        if (this$semanticInfoFullPathInDoc == null ? other$semanticInfoFullPathInDoc != null : !this$semanticInfoFullPathInDoc.equals(other$semanticInfoFullPathInDoc)) return false;
        final java.lang.Object this$chunkers = this.getChunkers();
        final java.lang.Object other$chunkers = other.getChunkers();
        if (this$chunkers == null ? other$chunkers != null : !this$chunkers.equals(other$chunkers)) return false;
        final java.lang.Object this$chunks = this.getChunks();
        final java.lang.Object other$chunks = other.getChunks();
        if (this$chunks == null ? other$chunks != null : !this$chunks.equals(other$chunks)) return false;
        final java.lang.Object this$sparseEncodingConfig = this.getSparseEncodingConfig();
        final java.lang.Object other$sparseEncodingConfig = other.getSparseEncodingConfig();
        if (this$sparseEncodingConfig == null ? other$sparseEncodingConfig != null : !this$sparseEncodingConfig.equals(other$sparseEncodingConfig)) return false;
        final java.lang.Object this$docId = this.getDocId();
        final java.lang.Object other$docId = other.getDocId();
        if (this$docId == null ? other$docId != null : !this$docId.equals(other$docId)) return false;
        return true;
    }

    
    
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof SemanticFieldInfo;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final java.lang.Object $chunkingEnabled = this.getChunkingEnabled();
        result = result * PRIME + ($chunkingEnabled == null ? 43 : $chunkingEnabled.hashCode());
        final java.lang.Object $skipExistingEmbedding = this.getSkipExistingEmbedding();
        result = result * PRIME + ($skipExistingEmbedding == null ? 43 : $skipExistingEmbedding.hashCode());
        final java.lang.Object $value = this.getValue();
        result = result * PRIME + ($value == null ? 43 : $value.hashCode());
        final java.lang.Object $modelId = this.getModelId();
        result = result * PRIME + ($modelId == null ? 43 : $modelId.hashCode());
        final java.lang.Object $semanticFieldFullPathInMapping = this.getSemanticFieldFullPathInMapping();
        result = result * PRIME + ($semanticFieldFullPathInMapping == null ? 43 : $semanticFieldFullPathInMapping.hashCode());
        final java.lang.Object $semanticFieldFullPathInDoc = this.getSemanticFieldFullPathInDoc();
        result = result * PRIME + ($semanticFieldFullPathInDoc == null ? 43 : $semanticFieldFullPathInDoc.hashCode());
        final java.lang.Object $semanticInfoFullPathInDoc = this.getSemanticInfoFullPathInDoc();
        result = result * PRIME + ($semanticInfoFullPathInDoc == null ? 43 : $semanticInfoFullPathInDoc.hashCode());
        final java.lang.Object $chunkers = this.getChunkers();
        result = result * PRIME + ($chunkers == null ? 43 : $chunkers.hashCode());
        final java.lang.Object $chunks = this.getChunks();
        result = result * PRIME + ($chunks == null ? 43 : $chunks.hashCode());
        final java.lang.Object $sparseEncodingConfig = this.getSparseEncodingConfig();
        result = result * PRIME + ($sparseEncodingConfig == null ? 43 : $sparseEncodingConfig.hashCode());
        final java.lang.Object $docId = this.getDocId();
        result = result * PRIME + ($docId == null ? 43 : $docId.hashCode());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "SemanticFieldInfo(value=" + this.getValue() + ", modelId=" + this.getModelId() + ", semanticFieldFullPathInMapping=" + this.getSemanticFieldFullPathInMapping() + ", semanticFieldFullPathInDoc=" + this.getSemanticFieldFullPathInDoc() + ", semanticInfoFullPathInDoc=" + this.getSemanticInfoFullPathInDoc() + ", chunkingEnabled=" + this.getChunkingEnabled() + ", chunkers=" + this.getChunkers() + ", chunks=" + this.getChunks() + ", sparseEncodingConfig=" + this.getSparseEncodingConfig() + ", skipExistingEmbedding=" + this.getSkipExistingEmbedding() + ", docId=" + this.getDocId() + ")";
    }
}
