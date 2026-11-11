/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.mappingtransformer;

import java.util.Map;

public class TextEmbeddingInfo {
    private Integer embeddingDimension;
    private Map<String, Object> additionalConfig;

    
    
    public Integer getEmbeddingDimension() {
        return this.embeddingDimension;
    }

    
    
    public Map<String, Object> getAdditionalConfig() {
        return this.additionalConfig;
    }

    
    
    public void setEmbeddingDimension(final Integer embeddingDimension) {
        this.embeddingDimension = embeddingDimension;
    }

    
    
    public void setAdditionalConfig(final Map<String, Object> additionalConfig) {
        this.additionalConfig = additionalConfig;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof TextEmbeddingInfo)) return false;
        final TextEmbeddingInfo other = (TextEmbeddingInfo) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        final java.lang.Object this$embeddingDimension = this.getEmbeddingDimension();
        final java.lang.Object other$embeddingDimension = other.getEmbeddingDimension();
        if (this$embeddingDimension == null ? other$embeddingDimension != null : !this$embeddingDimension.equals(other$embeddingDimension)) return false;
        final java.lang.Object this$additionalConfig = this.getAdditionalConfig();
        final java.lang.Object other$additionalConfig = other.getAdditionalConfig();
        if (this$additionalConfig == null ? other$additionalConfig != null : !this$additionalConfig.equals(other$additionalConfig)) return false;
        return true;
    }

    
    
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof TextEmbeddingInfo;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final java.lang.Object $embeddingDimension = this.getEmbeddingDimension();
        result = result * PRIME + ($embeddingDimension == null ? 43 : $embeddingDimension.hashCode());
        final java.lang.Object $additionalConfig = this.getAdditionalConfig();
        result = result * PRIME + ($additionalConfig == null ? 43 : $additionalConfig.hashCode());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "TextEmbeddingInfo(embeddingDimension=" + this.getEmbeddingDimension() + ", additionalConfig=" + this.getAdditionalConfig() + ")";
    }

    
    
    public TextEmbeddingInfo(final Integer embeddingDimension, final Map<String, Object> additionalConfig) {
        this.embeddingDimension = embeddingDimension;
        this.additionalConfig = additionalConfig;
    }
}
