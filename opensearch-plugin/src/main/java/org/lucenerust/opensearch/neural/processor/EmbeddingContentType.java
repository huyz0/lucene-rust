/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import org.lucenerust.opensearch.neural.ml.AsymmetricTextEmbeddingParameters;

/**
 * Enum to represent the content type for text embedding.
 * Used to distinguish between query and passage/document embeddings in asymmetric models.
 */
public enum EmbeddingContentType {
    QUERY,
    PASSAGE;

    /**
     * Converts to the corresponding AsymmetricTextEmbeddingParameters.EmbeddingContentType
     * @return the ML commons equivalent content type
     */
    public AsymmetricTextEmbeddingParameters.EmbeddingContentType toMLContentType() {
        return this == QUERY
            ? AsymmetricTextEmbeddingParameters.EmbeddingContentType.QUERY
            : AsymmetricTextEmbeddingParameters.EmbeddingContentType.PASSAGE;
    }
}
