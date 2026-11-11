/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.Map;

/** A model document of ml-commons' model index, as neural search reads it. */
public final class MLModel {
    private final String modelId;
    private final Map<String, Object> source;

    public MLModel(Map<String, Object> source) {
        this(null, source);
    }

    public MLModel(String modelId, Map<String, Object> source) {
        this.modelId = modelId;
        this.source = source;
    }

    /** {@code name}, or null. */
    public String getName() {
        Object n = source.get("name");
        return n == null ? null : n.toString();
    }

    public String getModelId() {
        return modelId;
    }

    /** {@code MLModel.getAlgorithm()} as ml-commons' enum. */
    public FunctionName getFunctionName() {
        return FunctionName.from(getAlgorithm());
    }

    /** {@code model_config.model_type}, or null. */
    public String getModelType() {
        Map<String, Object> config = getModelConfig();
        return config == null || config.get("model_type") == null ? null : config.get("model_type").toString();
    }

    /** {@code model_config.embedding_dimension}, or null. */
    public Integer getEmbeddingDimension() {
        Map<String, Object> config = getModelConfig();
        Object d = config == null ? null : config.get("embedding_dimension");
        // ml-commons parses it with XContentParser.intValue(), which takes a number or a numeric string.
        if (d instanceof Number n) {
            return n.intValue();
        }
        return d instanceof String str ? (int) Double.parseDouble(str) : null;
    }

    /** {@code MLModel.getAlgorithm().name()}: TEXT_EMBEDDING, SPARSE_ENCODING, REMOTE, ... */
    public String getAlgorithm() {
        Object a = source.get("algorithm");
        return a == null ? null : a.toString();
    }

    public boolean isRemote() {
        return "REMOTE".equals(getAlgorithm());
    }

    /** {@code model_config}, or null. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getModelConfig() {
        return source.get("model_config") instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
    }

    /** {@code model_config.additional_config}, or null. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getAdditionalConfig() {
        Map<String, Object> config = getModelConfig();
        return config != null && config.get("additional_config") instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
    }

    public Map<String, Object> source() {
        return source;
    }
}
