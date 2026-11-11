/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import java.util.List;
import org.lucenerust.opensearch.neural.ml.MLAlgoParams;

/**
 * Base abstract class for inference requests.
 *  This class contains common fields and behaviors shared across different types of inference requests.
 */
public abstract class InferenceRequest {
    /**
     * Unique identifier for the model to be used for inference.
     * This field is required and cannot be null.
     */
    
    private String modelId;
    /**
     * List of targetResponseFilters to be applied.
     * Defaults value if not specified.
     */
    private List<String> targetResponseFilters;
    /**
     * ML algorithm parameters for models.
     * For asymmetric models, use AsymmetricTextEmbeddingParameters with embeddingContentType set.
     */
    private MLAlgoParams mlAlgoParams;
    /**
     * Content type for embedding (QUERY or PASSAGE).
     * Used as indicator for asymmetric models to determine which prefix to apply.
     */
    private EmbeddingContentType embeddingContentType;

    private static List<String> $default$targetResponseFilters() {
        return List.of("sentence_embedding");
    }


    public static abstract class InferenceRequestBuilder<C extends InferenceRequest, B extends InferenceRequest.InferenceRequestBuilder<C, B>> {
        private String modelId;
        private boolean targetResponseFilters$set;
        private List<String> targetResponseFilters$value;
        private MLAlgoParams mlAlgoParams;
        private EmbeddingContentType embeddingContentType;

        /**
         * Unique identifier for the model to be used for inference.
         * This field is required and cannot be null.
         * @return {@code this}.
         */
        public B modelId(final String modelId) {
            java.util.Objects.requireNonNull(modelId, "modelId is marked non-null but is null");
            this.modelId = modelId;
            return self();
        }

        /**
         * List of targetResponseFilters to be applied.
         * Defaults value if not specified.
         * @return {@code this}.
         */
        public B targetResponseFilters(final List<String> targetResponseFilters) {
            this.targetResponseFilters$value = targetResponseFilters;
            targetResponseFilters$set = true;
            return self();
        }

        /**
         * ML algorithm parameters for models.
         * For asymmetric models, use AsymmetricTextEmbeddingParameters with embeddingContentType set.
         * @return {@code this}.
         */
        public B mlAlgoParams(final MLAlgoParams mlAlgoParams) {
            this.mlAlgoParams = mlAlgoParams;
            return self();
        }

        /**
         * Content type for embedding (QUERY or PASSAGE).
         * Used as indicator for asymmetric models to determine which prefix to apply.
         * @return {@code this}.
         */
        public B embeddingContentType(final EmbeddingContentType embeddingContentType) {
            this.embeddingContentType = embeddingContentType;
            return self();
        }

        protected abstract B self();

        public abstract C build();

        @java.lang.Override
        public java.lang.String toString() {
            return "InferenceRequest.InferenceRequestBuilder(modelId=" + this.modelId + ", targetResponseFilters$value=" + this.targetResponseFilters$value + ", mlAlgoParams=" + this.mlAlgoParams + ", embeddingContentType=" + this.embeddingContentType + ")";
        }
    }

    protected InferenceRequest(final InferenceRequest.InferenceRequestBuilder<?, ?> b) {
        this.modelId = b.modelId;
        java.util.Objects.requireNonNull(modelId, "modelId is marked non-null but is null");
        if (b.targetResponseFilters$set) this.targetResponseFilters = b.targetResponseFilters$value;
         else this.targetResponseFilters = InferenceRequest.$default$targetResponseFilters();
        this.mlAlgoParams = b.mlAlgoParams;
        this.embeddingContentType = b.embeddingContentType;
    }

    public InferenceRequest() {
        this.targetResponseFilters = InferenceRequest.$default$targetResponseFilters();
    }

    /**
     * Unique identifier for the model to be used for inference.
     * This field is required and cannot be null.
     */
    
    public String getModelId() {
        return this.modelId;
    }

    /**
     * List of targetResponseFilters to be applied.
     * Defaults value if not specified.
     */
    public List<String> getTargetResponseFilters() {
        return this.targetResponseFilters;
    }

    /**
     * ML algorithm parameters for models.
     * For asymmetric models, use AsymmetricTextEmbeddingParameters with embeddingContentType set.
     */
    public MLAlgoParams getMlAlgoParams() {
        return this.mlAlgoParams;
    }

    /**
     * Content type for embedding (QUERY or PASSAGE).
     * Used as indicator for asymmetric models to determine which prefix to apply.
     */
    public EmbeddingContentType getEmbeddingContentType() {
        return this.embeddingContentType;
    }

    /**
     * Unique identifier for the model to be used for inference.
     * This field is required and cannot be null.
     */
    public void setModelId(final String modelId) {
        java.util.Objects.requireNonNull(modelId, "modelId is marked non-null but is null");
        this.modelId = modelId;
    }

    /**
     * List of targetResponseFilters to be applied.
     * Defaults value if not specified.
     */
    public void setTargetResponseFilters(final List<String> targetResponseFilters) {
        this.targetResponseFilters = targetResponseFilters;
    }

    /**
     * ML algorithm parameters for models.
     * For asymmetric models, use AsymmetricTextEmbeddingParameters with embeddingContentType set.
     */
    public void setMlAlgoParams(final MLAlgoParams mlAlgoParams) {
        this.mlAlgoParams = mlAlgoParams;
    }

    /**
     * Content type for embedding (QUERY or PASSAGE).
     * Used as indicator for asymmetric models to determine which prefix to apply.
     */
    public void setEmbeddingContentType(final EmbeddingContentType embeddingContentType) {
        this.embeddingContentType = embeddingContentType;
    }
}
