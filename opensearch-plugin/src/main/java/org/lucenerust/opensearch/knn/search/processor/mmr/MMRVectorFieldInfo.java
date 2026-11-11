/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.search.processor.mmr;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.lucenerust.opensearch.knn.index.SpaceType;
import org.lucenerust.opensearch.knn.index.VectorDataType;
import org.lucenerust.opensearch.knn.KnnVectorFieldMapper;
import java.util.Map;
import static org.lucenerust.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.lucenerust.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.lucenerust.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.lucenerust.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.lucenerust.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.lucenerust.opensearch.knn.index.engine.SpaceTypeResolver.getDefaultSpaceType;

/**
 * A DTO to hold the info of the vector field used for MMR
 */
public class MMRVectorFieldInfo {
    private String indexName;
    private String fieldPath;
    private VectorDataType vectorDataType;
    private String modelId;
    private SpaceType spaceType;
    private boolean unmapped;
    private String fieldType;

    public MMRVectorFieldInfo(SpaceType spaceType, VectorDataType vectorDataType) {
        this.spaceType = spaceType;
        this.vectorDataType = vectorDataType;
    }

    public boolean isKNNVectorField() {
        return KnnVectorFieldMapper.CONTENT_TYPE.equals(fieldType);
    }

    public void setKnnConfig(final Map<String, Object> knnConfig) {
        if (knnConfig == null) {
            throw new java.lang.NullPointerException("knnConfig is marked non-null but is null");
        }
        setVectorDataTypeByConfig(knnConfig);
        if (setModelIdIfPresent(knnConfig)) {
            return;
        }
        if (setSpaceTypeIfPresent(knnConfig)) {
            return;
        }
        this.spaceType = getDefaultSpaceType(vectorDataType);
    }

    public void setIndexNameByIndexMetadata(final IndexMetadata indexMetadata) {
        if (indexMetadata == null) {
            throw new java.lang.NullPointerException("indexMetadata is marked non-null but is null");
        }
        this.indexName = indexMetadata.getIndex().getName();
    }

    private void setVectorDataTypeByConfig(Map<String, Object> knnConfig) {
        String dataType = (String) knnConfig.get(VECTOR_DATA_TYPE_FIELD);
        this.vectorDataType = (dataType == null) ? VectorDataType.DEFAULT : VectorDataType.get(dataType);
    }

    private boolean setModelIdIfPresent(Map<String, Object> knnConfig) {
        String modelId = (String) knnConfig.get(MODEL_ID);
        if (modelId != null) {
            this.modelId = modelId;
            return true;
        }
        return false;
    }

    private boolean setSpaceTypeIfPresent(Map<String, Object> knnConfig) {
        String topLevelSpaceType = (String) knnConfig.get(TOP_LEVEL_PARAMETER_SPACE_TYPE);
        if (topLevelSpaceType != null) {
            this.spaceType = SpaceType.getSpace(topLevelSpaceType);
            return true;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> knnMethod = (Map<String, Object>) knnConfig.get(KNN_METHOD);
        if (knnMethod != null) {
            String spaceType = (String) knnMethod.get(METHOD_PARAMETER_SPACE_TYPE);
            if (spaceType != null) {
                this.spaceType = SpaceType.getSpace(spaceType);
                return true;
            }
        }
        return false;
    }

    
    
    public String getIndexName() {
        return this.indexName;
    }

    
    
    public String getFieldPath() {
        return this.fieldPath;
    }

    
    
    public VectorDataType getVectorDataType() {
        return this.vectorDataType;
    }

    
    
    public String getModelId() {
        return this.modelId;
    }

    
    
    public SpaceType getSpaceType() {
        return this.spaceType;
    }

    
    
    public boolean isUnmapped() {
        return this.unmapped;
    }

    
    
    public String getFieldType() {
        return this.fieldType;
    }

    
    
    public void setIndexName(final String indexName) {
        this.indexName = indexName;
    }

    
    
    public void setFieldPath(final String fieldPath) {
        this.fieldPath = fieldPath;
    }

    
    
    public void setVectorDataType(final VectorDataType vectorDataType) {
        this.vectorDataType = vectorDataType;
    }

    
    
    public void setModelId(final String modelId) {
        this.modelId = modelId;
    }

    
    
    public void setSpaceType(final SpaceType spaceType) {
        this.spaceType = spaceType;
    }

    
    
    public void setUnmapped(final boolean unmapped) {
        this.unmapped = unmapped;
    }

    
    
    public void setFieldType(final String fieldType) {
        this.fieldType = fieldType;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof MMRVectorFieldInfo)) return false;
        final MMRVectorFieldInfo other = (MMRVectorFieldInfo) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        if (this.isUnmapped() != other.isUnmapped()) return false;
        final java.lang.Object this$indexName = this.getIndexName();
        final java.lang.Object other$indexName = other.getIndexName();
        if (this$indexName == null ? other$indexName != null : !this$indexName.equals(other$indexName)) return false;
        final java.lang.Object this$fieldPath = this.getFieldPath();
        final java.lang.Object other$fieldPath = other.getFieldPath();
        if (this$fieldPath == null ? other$fieldPath != null : !this$fieldPath.equals(other$fieldPath)) return false;
        final java.lang.Object this$vectorDataType = this.getVectorDataType();
        final java.lang.Object other$vectorDataType = other.getVectorDataType();
        if (this$vectorDataType == null ? other$vectorDataType != null : !this$vectorDataType.equals(other$vectorDataType)) return false;
        final java.lang.Object this$modelId = this.getModelId();
        final java.lang.Object other$modelId = other.getModelId();
        if (this$modelId == null ? other$modelId != null : !this$modelId.equals(other$modelId)) return false;
        final java.lang.Object this$spaceType = this.getSpaceType();
        final java.lang.Object other$spaceType = other.getSpaceType();
        if (this$spaceType == null ? other$spaceType != null : !this$spaceType.equals(other$spaceType)) return false;
        final java.lang.Object this$fieldType = this.getFieldType();
        final java.lang.Object other$fieldType = other.getFieldType();
        if (this$fieldType == null ? other$fieldType != null : !this$fieldType.equals(other$fieldType)) return false;
        return true;
    }

    
    
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof MMRVectorFieldInfo;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + (this.isUnmapped() ? 79 : 97);
        final java.lang.Object $indexName = this.getIndexName();
        result = result * PRIME + ($indexName == null ? 43 : $indexName.hashCode());
        final java.lang.Object $fieldPath = this.getFieldPath();
        result = result * PRIME + ($fieldPath == null ? 43 : $fieldPath.hashCode());
        final java.lang.Object $vectorDataType = this.getVectorDataType();
        result = result * PRIME + ($vectorDataType == null ? 43 : $vectorDataType.hashCode());
        final java.lang.Object $modelId = this.getModelId();
        result = result * PRIME + ($modelId == null ? 43 : $modelId.hashCode());
        final java.lang.Object $spaceType = this.getSpaceType();
        result = result * PRIME + ($spaceType == null ? 43 : $spaceType.hashCode());
        final java.lang.Object $fieldType = this.getFieldType();
        result = result * PRIME + ($fieldType == null ? 43 : $fieldType.hashCode());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "MMRVectorFieldInfo(indexName=" + this.getIndexName() + ", fieldPath=" + this.getFieldPath() + ", vectorDataType=" + this.getVectorDataType() + ", modelId=" + this.getModelId() + ", spaceType=" + this.getSpaceType() + ", unmapped=" + this.isUnmapped() + ", fieldType=" + this.getFieldType() + ")";
    }

    
    
    public MMRVectorFieldInfo() {
    }
}
