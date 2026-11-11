/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * ml-commons' {@code AsymmetricTextEmbeddingParameters}: whether a text is embedded as a query or
 * a passage, and the key format of sparse vectors.
 */
public final class AsymmetricTextEmbeddingParameters implements MLAlgoParams {
    public enum EmbeddingContentType {
        QUERY,
        PASSAGE
    }

    private final EmbeddingContentType embeddingContentType;
    private final SparseEmbeddingFormat sparseEmbeddingFormat;

    private AsymmetricTextEmbeddingParameters(EmbeddingContentType embeddingContentType, SparseEmbeddingFormat sparseEmbeddingFormat) {
        this.embeddingContentType = embeddingContentType;
        this.sparseEmbeddingFormat = sparseEmbeddingFormat;
    }

    public EmbeddingContentType getEmbeddingContentType() {
        return embeddingContentType;
    }

    public SparseEmbeddingFormat getSparseEmbeddingFormat() {
        return sparseEmbeddingFormat;
    }

    @Override
    public Map<String, Object> toParameters() {
        Map<String, Object> p = new LinkedHashMap<>();
        if (embeddingContentType != null) {
            p.put("content_type", embeddingContentType.name());
        }
        if (sparseEmbeddingFormat != null) {
            p.put("sparse_embedding_format", sparseEmbeddingFormat.name());
        }
        return p;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder().embeddingContentType(embeddingContentType).sparseEmbeddingFormat(sparseEmbeddingFormat);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof AsymmetricTextEmbeddingParameters p
            && embeddingContentType == p.embeddingContentType
            && sparseEmbeddingFormat == p.sparseEmbeddingFormat;
    }

    @Override
    public int hashCode() {
        return Objects.hash(embeddingContentType, sparseEmbeddingFormat);
    }

    public static final class Builder {
        private EmbeddingContentType embeddingContentType;
        private SparseEmbeddingFormat sparseEmbeddingFormat;

        public Builder embeddingContentType(EmbeddingContentType embeddingContentType) {
            this.embeddingContentType = embeddingContentType;
            return this;
        }

        public Builder sparseEmbeddingFormat(SparseEmbeddingFormat sparseEmbeddingFormat) {
            this.sparseEmbeddingFormat = sparseEmbeddingFormat;
            return this;
        }

        public AsymmetricTextEmbeddingParameters build() {
            return new AsymmetricTextEmbeddingParameters(embeddingContentType, sparseEmbeddingFormat);
        }
    }
}
