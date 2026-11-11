/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.settings;

import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import static org.lucenerust.opensearch.neural.NeuralSearchSettings.SEMANTIC_MODEL_SELECTION_MODEL_ID;
import static org.lucenerust.opensearch.neural.NeuralSearchSettings.SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX;

/**
 * Resolves the {@link NeuralSearchSettings#SEMANTIC_MODEL_SELECTION_MODEL_ID} affix setting for a given profile. It
 * reads node settings (opensearch.yml) merged with the current persistent and transient cluster settings, so it honors
 * both static configuration and the latest dynamically updated value.
 */
public class SemanticModelSelectionSettingsAccessor {
    private final ClusterService clusterService;

    public SemanticModelSelectionSettingsAccessor(final ClusterService clusterService) {
        java.util.Objects.requireNonNull(clusterService, "clusterService is marked non-null but is null");
        this.clusterService = clusterService;
    }

    /**
     * Return the configured model id for the given profile key, or null if none is configured.
     *
     * @param profileKey profile key of the form {@code <model_type>.<language_option>} (lower cased), e.g. {@code sparse.english}
     * @return the configured model id, or null when the operator has not configured this profile
     */
    public String getModelId(final String profileKey) {
        java.util.Objects.requireNonNull(profileKey, "profileKey is marked non-null but is null");
        final String fullKey = SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX + profileKey;
        // Merge node settings (opensearch.yml) with the merged persistent+transient cluster settings, so both static
        // (opensearch.yml) and dynamically updated values are honored. Metadata#settings() already merges the
        // persistent and transient cluster settings.
        final Settings settings = Settings.builder().put(clusterService.getSettings()).put(clusterService.state().getMetadata().settings()).build();
        final Setting<String> concreteSetting = SEMANTIC_MODEL_SELECTION_MODEL_ID.getConcreteSetting(fullKey);
        final String modelId = concreteSetting.get(settings);
        if (modelId == null || modelId.isEmpty()) {
            return null;
        }
        return modelId;
    }
}
