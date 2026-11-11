/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.search.processor.mmr;

import org.lucenerust.opensearch.knn.index.SpaceType;
import org.lucenerust.opensearch.knn.index.VectorDataType;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import java.util.Map;

/**
 * A DTO to hold the context for MMR rerank
 */
public class MMRRerankContext {
    private Integer originalQuerySize;
    private Float diversity;
    private FetchSourceContext originalFetchSourceContext;
    private SpaceType spaceType;
    // The default path if we cannot find the path based on the index
    private String vectorFieldPath;
    private VectorDataType vectorDataType;
    // To support the case that we have different vector field paths in different indices
    private Map<String, String> indexToVectorFieldPathMap;
    // Flag to enable MMR explain info in the response
    private Boolean explain;

    
    
    public Integer getOriginalQuerySize() {
        return this.originalQuerySize;
    }

    
    
    public Float getDiversity() {
        return this.diversity;
    }

    
    
    public FetchSourceContext getOriginalFetchSourceContext() {
        return this.originalFetchSourceContext;
    }

    
    
    public SpaceType getSpaceType() {
        return this.spaceType;
    }

    
    
    public String getVectorFieldPath() {
        return this.vectorFieldPath;
    }

    
    
    public VectorDataType getVectorDataType() {
        return this.vectorDataType;
    }

    
    
    public Map<String, String> getIndexToVectorFieldPathMap() {
        return this.indexToVectorFieldPathMap;
    }

    
    
    public Boolean getExplain() {
        return this.explain;
    }

    
    
    public void setOriginalQuerySize(final Integer originalQuerySize) {
        this.originalQuerySize = originalQuerySize;
    }

    
    
    public void setDiversity(final Float diversity) {
        this.diversity = diversity;
    }

    
    
    public void setOriginalFetchSourceContext(final FetchSourceContext originalFetchSourceContext) {
        this.originalFetchSourceContext = originalFetchSourceContext;
    }

    
    
    public void setSpaceType(final SpaceType spaceType) {
        this.spaceType = spaceType;
    }

    
    
    public void setVectorFieldPath(final String vectorFieldPath) {
        this.vectorFieldPath = vectorFieldPath;
    }

    
    
    public void setVectorDataType(final VectorDataType vectorDataType) {
        this.vectorDataType = vectorDataType;
    }

    
    
    public void setIndexToVectorFieldPathMap(final Map<String, String> indexToVectorFieldPathMap) {
        this.indexToVectorFieldPathMap = indexToVectorFieldPathMap;
    }

    
    
    public void setExplain(final Boolean explain) {
        this.explain = explain;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof MMRRerankContext)) return false;
        final MMRRerankContext other = (MMRRerankContext) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        final java.lang.Object this$originalQuerySize = this.getOriginalQuerySize();
        final java.lang.Object other$originalQuerySize = other.getOriginalQuerySize();
        if (this$originalQuerySize == null ? other$originalQuerySize != null : !this$originalQuerySize.equals(other$originalQuerySize)) return false;
        final java.lang.Object this$diversity = this.getDiversity();
        final java.lang.Object other$diversity = other.getDiversity();
        if (this$diversity == null ? other$diversity != null : !this$diversity.equals(other$diversity)) return false;
        final java.lang.Object this$explain = this.getExplain();
        final java.lang.Object other$explain = other.getExplain();
        if (this$explain == null ? other$explain != null : !this$explain.equals(other$explain)) return false;
        final java.lang.Object this$originalFetchSourceContext = this.getOriginalFetchSourceContext();
        final java.lang.Object other$originalFetchSourceContext = other.getOriginalFetchSourceContext();
        if (this$originalFetchSourceContext == null ? other$originalFetchSourceContext != null : !this$originalFetchSourceContext.equals(other$originalFetchSourceContext)) return false;
        final java.lang.Object this$spaceType = this.getSpaceType();
        final java.lang.Object other$spaceType = other.getSpaceType();
        if (this$spaceType == null ? other$spaceType != null : !this$spaceType.equals(other$spaceType)) return false;
        final java.lang.Object this$vectorFieldPath = this.getVectorFieldPath();
        final java.lang.Object other$vectorFieldPath = other.getVectorFieldPath();
        if (this$vectorFieldPath == null ? other$vectorFieldPath != null : !this$vectorFieldPath.equals(other$vectorFieldPath)) return false;
        final java.lang.Object this$vectorDataType = this.getVectorDataType();
        final java.lang.Object other$vectorDataType = other.getVectorDataType();
        if (this$vectorDataType == null ? other$vectorDataType != null : !this$vectorDataType.equals(other$vectorDataType)) return false;
        final java.lang.Object this$indexToVectorFieldPathMap = this.getIndexToVectorFieldPathMap();
        final java.lang.Object other$indexToVectorFieldPathMap = other.getIndexToVectorFieldPathMap();
        if (this$indexToVectorFieldPathMap == null ? other$indexToVectorFieldPathMap != null : !this$indexToVectorFieldPathMap.equals(other$indexToVectorFieldPathMap)) return false;
        return true;
    }

    
    
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof MMRRerankContext;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final java.lang.Object $originalQuerySize = this.getOriginalQuerySize();
        result = result * PRIME + ($originalQuerySize == null ? 43 : $originalQuerySize.hashCode());
        final java.lang.Object $diversity = this.getDiversity();
        result = result * PRIME + ($diversity == null ? 43 : $diversity.hashCode());
        final java.lang.Object $explain = this.getExplain();
        result = result * PRIME + ($explain == null ? 43 : $explain.hashCode());
        final java.lang.Object $originalFetchSourceContext = this.getOriginalFetchSourceContext();
        result = result * PRIME + ($originalFetchSourceContext == null ? 43 : $originalFetchSourceContext.hashCode());
        final java.lang.Object $spaceType = this.getSpaceType();
        result = result * PRIME + ($spaceType == null ? 43 : $spaceType.hashCode());
        final java.lang.Object $vectorFieldPath = this.getVectorFieldPath();
        result = result * PRIME + ($vectorFieldPath == null ? 43 : $vectorFieldPath.hashCode());
        final java.lang.Object $vectorDataType = this.getVectorDataType();
        result = result * PRIME + ($vectorDataType == null ? 43 : $vectorDataType.hashCode());
        final java.lang.Object $indexToVectorFieldPathMap = this.getIndexToVectorFieldPathMap();
        result = result * PRIME + ($indexToVectorFieldPathMap == null ? 43 : $indexToVectorFieldPathMap.hashCode());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "MMRRerankContext(originalQuerySize=" + this.getOriginalQuerySize() + ", diversity=" + this.getDiversity() + ", originalFetchSourceContext=" + this.getOriginalFetchSourceContext() + ", spaceType=" + this.getSpaceType() + ", vectorFieldPath=" + this.getVectorFieldPath() + ", vectorDataType=" + this.getVectorDataType() + ", indexToVectorFieldPathMap=" + this.getIndexToVectorFieldPathMap() + ", explain=" + this.getExplain() + ")";
    }

    
    
    public MMRRerankContext() {
    }
}
