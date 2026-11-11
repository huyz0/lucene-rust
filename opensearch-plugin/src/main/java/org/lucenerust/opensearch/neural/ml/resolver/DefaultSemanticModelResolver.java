/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml.resolver;

import org.lucenerust.opensearch.neural.mapper.dto.ModelSelection;
import org.lucenerust.opensearch.neural.settings.SemanticModelSelectionSettingsAccessor;
import java.util.Locale;
import static org.lucenerust.opensearch.neural.NeuralSearchSettings.SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX;

/**
 * Default {@link SemanticModelResolver} implementation. It resolves a {@link ModelSelection} to a model_id by reading
 * the operator configured {@code plugins.neural_search.model_selection.model_id.<model_type>.<language_option>} cluster
 * setting. It does not register, deploy, or fetch any model; the operator is expected to deploy the model and configure
 * the setting. Verification that the resolved model exists and its type matches the requested model type happens later,
 * where the model is loaded to build the semantic info field.
 */
public class DefaultSemanticModelResolver implements SemanticModelResolver {
    
    
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(DefaultSemanticModelResolver.class);
    private final SemanticModelSelectionSettingsAccessor settingsAccessor;

    public DefaultSemanticModelResolver(final SemanticModelSelectionSettingsAccessor settingsAccessor) {
        java.util.Objects.requireNonNull(settingsAccessor, "settingsAccessor is marked non-null but is null");
        this.settingsAccessor = settingsAccessor;
    }

    @Override
    public String resolve(final ModelSelection modelSelection) {
        java.util.Objects.requireNonNull(modelSelection, "modelSelection is marked non-null but is null");
        final String profileKey = modelSelection.getProfileKey();
        final String modelId = settingsAccessor.getModelId(profileKey);
        if (modelId == null) {
            final String settingKey = SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX + profileKey;
            throw new IllegalArgumentException(String.format(Locale.ROOT, "No model_id is configured for the model_selection [language_option=%s, model_type=%s]. " + "Deploy a %s model and set the cluster setting [%s] to its model_id, then retry.", modelSelection.getLanguageOption(), modelSelection.getModelType(), modelSelection.getModelType(), settingKey));
        }
        return modelId;
    }
}
