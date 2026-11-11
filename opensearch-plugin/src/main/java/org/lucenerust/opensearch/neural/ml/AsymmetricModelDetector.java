/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.Map;

/** neural-search's {@code AsymmetricModelDetector}: a model with query or passage prefixes. */
public final class AsymmetricModelDetector {
    private AsymmetricModelDetector() {}

    /** A local text embedding model whose config has a query or passage prefix, or a remote one flagged asymmetric. */
    public static boolean isAsymmetricModel(MLModel model) {
        if (model == null || model.getModelConfig() == null) {
            return false;
        }
        Map<String, Object> config = model.getModelConfig();
        if (model.isRemote() == false) {
            // ml-commons reads a prefix as asymmetric only from a text embedding model's config.
            return "TEXT_EMBEDDING".equals(model.getAlgorithm())
                && (config.get("passage_prefix") != null || config.get("query_prefix") != null);
        }
        Map<String, Object> additional = model.getAdditionalConfig();
        if (additional != null) {
            if (Boolean.TRUE.equals(additional.get("is_asymmetric"))) {
                return true;
            }
            return additional.containsKey(AsymmetricTextEmbeddingConstants.QUERY_PREFIX_CONFIG_KEY)
                || additional.containsKey(AsymmetricTextEmbeddingConstants.PASSAGE_PREFIX_CONFIG_KEY);
        }
        return false;
    }
}
