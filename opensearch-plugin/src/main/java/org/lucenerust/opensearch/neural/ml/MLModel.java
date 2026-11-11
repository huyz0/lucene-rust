/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.Map;

/** A model document of ml-commons' model index, as neural search reads it. */
public final class MLModel {
    private final Map<String, Object> source;

    public MLModel(Map<String, Object> source) {
        this.source = source;
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
