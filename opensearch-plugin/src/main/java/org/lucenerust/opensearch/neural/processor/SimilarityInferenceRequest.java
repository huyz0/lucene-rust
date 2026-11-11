/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

/**
 * Implementation of InferenceRequest for similarity based text inference requests.
 *
 * @see TextInferenceRequest
 */
public class SimilarityInferenceRequest extends TextInferenceRequest {
    private String queryText;


    public static abstract class SimilarityInferenceRequestBuilder<C extends SimilarityInferenceRequest, B extends SimilarityInferenceRequest.SimilarityInferenceRequestBuilder<C, B>> extends TextInferenceRequest.TextInferenceRequestBuilder<C, B> {
        private String queryText;

        /**
         * @return {@code this}.
         */
        public B queryText(final String queryText) {
            this.queryText = queryText;
            return self();
        }

        @java.lang.Override
        protected abstract B self();

        @java.lang.Override
        public abstract C build();

        @java.lang.Override
        public java.lang.String toString() {
            return "SimilarityInferenceRequest.SimilarityInferenceRequestBuilder(super=" + super.toString() + ", queryText=" + this.queryText + ")";
        }
    }


    private static final class SimilarityInferenceRequestBuilderImpl extends SimilarityInferenceRequest.SimilarityInferenceRequestBuilder<SimilarityInferenceRequest, SimilarityInferenceRequest.SimilarityInferenceRequestBuilderImpl> {
        private SimilarityInferenceRequestBuilderImpl() {
        }

        @java.lang.Override
        protected SimilarityInferenceRequest.SimilarityInferenceRequestBuilderImpl self() {
            return this;
        }

        @java.lang.Override
        public SimilarityInferenceRequest build() {
            return new SimilarityInferenceRequest(this);
        }
    }

    protected SimilarityInferenceRequest(final SimilarityInferenceRequest.SimilarityInferenceRequestBuilder<?, ?> b) {
        super(b);
        this.queryText = b.queryText;
    }

    public static SimilarityInferenceRequest.SimilarityInferenceRequestBuilder<?, ?> builder() {
        return new SimilarityInferenceRequest.SimilarityInferenceRequestBuilderImpl();
    }

    public SimilarityInferenceRequest() {
    }

    public String getQueryText() {
        return this.queryText;
    }

    public void setQueryText(final String queryText) {
        this.queryText = queryText;
    }
}
