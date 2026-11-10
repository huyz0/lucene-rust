/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural;

import org.opensearch.common.settings.Setting;

import java.util.List;

/** The neural-search settings and names the hybrid query and the stats API use, under their stock keys. */
public final class NeuralSearchSettings {
    private NeuralSearchSettings() {}

    /** The pipeline context attribute the normalization workflow leaves its explanations under. */
    public static final String EXPLANATION_RESPONSE_KEY = "explanation_response";

    public static final Setting<Integer> HYBRID_COLLAPSE_DOCS_PER_GROUP_PER_SUBQUERY = Setting.intSetting(
        "index.neural_search.hybrid_collapse_docs_per_group_per_subquery",
        0,
        0,
        1000,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic,
        Setting.Property.Deprecated
    );

    /** {@code NeuralSearch.NEURAL_BASE_URI}. */
    public static final String NEURAL_BASE_URI = "/_plugins/_neural";

    /** The stats API and its event counters: off, the API answers 403 and counting is skipped. */
    public static final Setting<Boolean> NEURAL_STATS_ENABLED = Setting.boolSetting(
        "plugins.neural_search.stats_enabled",
        false,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static List<Setting<?>> all() {
        return List.of(HYBRID_COLLAPSE_DOCS_PER_GROUP_PER_SUBQUERY, NEURAL_STATS_ENABLED);
    }
}
