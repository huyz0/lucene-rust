/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.explain;

import java.util.Map;

/**
 * DTO class to hold explain details for normalization and combination
 */
public class ExplanationPayload {
    private final Map<PayloadType, Object> explainPayload;


    public enum PayloadType {
        NORMALIZATION_PROCESSOR;
    }


    @java.lang.SuppressWarnings("all")
    public static class ExplanationPayloadBuilder {
        @java.lang.SuppressWarnings("all")
        private Map<PayloadType, Object> explainPayload;

        @java.lang.SuppressWarnings("all")
        ExplanationPayloadBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public ExplanationPayload.ExplanationPayloadBuilder explainPayload(final Map<PayloadType, Object> explainPayload) {
            this.explainPayload = explainPayload;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public ExplanationPayload build() {
            return new ExplanationPayload(this.explainPayload);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "ExplanationPayload.ExplanationPayloadBuilder(explainPayload=" + this.explainPayload + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static ExplanationPayload.ExplanationPayloadBuilder builder() {
        return new ExplanationPayload.ExplanationPayloadBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public ExplanationPayload(final Map<PayloadType, Object> explainPayload) {
        this.explainPayload = explainPayload;
    }

    @java.lang.SuppressWarnings("all")
    public Map<PayloadType, Object> getExplainPayload() {
        return this.explainPayload;
    }
}
