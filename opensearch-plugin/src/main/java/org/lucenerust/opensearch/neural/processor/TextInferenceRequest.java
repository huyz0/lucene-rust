/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import java.util.List;

/**
 * Implementation of InferenceRequest for inputTexts based inference requests.
 * Use this class when the input data consists of list of strings.
 *
 * @see InferenceRequest
 */
public class TextInferenceRequest extends InferenceRequest {
    private List<String> inputTexts; // on which inference needs to happen


    public static abstract class TextInferenceRequestBuilder<C extends TextInferenceRequest, B extends TextInferenceRequest.TextInferenceRequestBuilder<C, B>> extends InferenceRequest.InferenceRequestBuilder<C, B> {
        private List<String> inputTexts;

        /**
         * @return {@code this}.
         */
        public B inputTexts(final List<String> inputTexts) {
            this.inputTexts = inputTexts;
            return self();
        }

        @java.lang.Override
        protected abstract B self();

        @java.lang.Override
        public abstract C build();

        @java.lang.Override
        public java.lang.String toString() {
            return "TextInferenceRequest.TextInferenceRequestBuilder(super=" + super.toString() + ", inputTexts=" + this.inputTexts + ")";
        }
    }


    private static final class TextInferenceRequestBuilderImpl extends TextInferenceRequest.TextInferenceRequestBuilder<TextInferenceRequest, TextInferenceRequest.TextInferenceRequestBuilderImpl> {
        private TextInferenceRequestBuilderImpl() {
        }

        @java.lang.Override
        protected TextInferenceRequest.TextInferenceRequestBuilderImpl self() {
            return this;
        }

        @java.lang.Override
        public TextInferenceRequest build() {
            return new TextInferenceRequest(this);
        }
    }

    protected TextInferenceRequest(final TextInferenceRequest.TextInferenceRequestBuilder<?, ?> b) {
        super(b);
        this.inputTexts = b.inputTexts;
    }

    public static TextInferenceRequest.TextInferenceRequestBuilder<?, ?> builder() {
        return new TextInferenceRequest.TextInferenceRequestBuilderImpl();
    }

    public TextInferenceRequest() {
    }

    public List<String> getInputTexts() {
        return this.inputTexts;
    }

    public void setInputTexts(final List<String> inputTexts) {
        this.inputTexts = inputTexts;
    }
}
