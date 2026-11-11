/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import java.util.Map;

/**
 * Implementation of InferenceRequest for inputObjects based inference requests.
 * Use this class when the input data consists of key-value pairs.
 *
 * @see InferenceRequest
 */
public class MapInferenceRequest extends InferenceRequest {
    private Map<String, String> inputObjects;


    public static abstract class MapInferenceRequestBuilder<C extends MapInferenceRequest, B extends MapInferenceRequest.MapInferenceRequestBuilder<C, B>> extends InferenceRequest.InferenceRequestBuilder<C, B> {
        private Map<String, String> inputObjects;

        /**
         * @return {@code this}.
         */
        public B inputObjects(final Map<String, String> inputObjects) {
            this.inputObjects = inputObjects;
            return self();
        }

        @java.lang.Override
        protected abstract B self();

        @java.lang.Override
        public abstract C build();

        @java.lang.Override
        public java.lang.String toString() {
            return "MapInferenceRequest.MapInferenceRequestBuilder(super=" + super.toString() + ", inputObjects=" + this.inputObjects + ")";
        }
    }


    private static final class MapInferenceRequestBuilderImpl extends MapInferenceRequest.MapInferenceRequestBuilder<MapInferenceRequest, MapInferenceRequest.MapInferenceRequestBuilderImpl> {
        private MapInferenceRequestBuilderImpl() {
        }

        @java.lang.Override
        protected MapInferenceRequest.MapInferenceRequestBuilderImpl self() {
            return this;
        }

        @java.lang.Override
        public MapInferenceRequest build() {
            return new MapInferenceRequest(this);
        }
    }

    protected MapInferenceRequest(final MapInferenceRequest.MapInferenceRequestBuilder<?, ?> b) {
        super(b);
        this.inputObjects = b.inputObjects;
    }

    public static MapInferenceRequest.MapInferenceRequestBuilder<?, ?> builder() {
        return new MapInferenceRequest.MapInferenceRequestBuilderImpl();
    }

    public MapInferenceRequest() {
    }

    public Map<String, String> getInputObjects() {
        return this.inputObjects;
    }

    public void setInputObjects(final Map<String, String> inputObjects) {
        this.inputObjects = inputObjects;
    }
}
