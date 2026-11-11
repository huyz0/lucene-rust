/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural;

import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.unit.ByteSizeValue;

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

    /** The most fields a rerank processor's {@code document_fields} may list. */
    public static final Setting<Integer> RERANKER_MAX_DOC_FIELDS = Setting.intSetting(
        "plugins.neural_search.reranker_max_document_fields",
        50,
        Setting.Property.NodeScope
    );

    /** The semantic field's ingest batch size. */
    public static final Setting<Integer> SEMANTIC_INGEST_BATCH_SIZE = Setting.intSetting(
        "index.neural_search.semantic_ingest_batch_size",
        10,
        1,
        100,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * neural-search's breaker settings. The plugin keeps no JVM caches of forward indexes or
     * clustered postings, which is all that breaker accounts, so they are accepted and read by
     * nothing.
     */
    public static final Setting<Double> NEURAL_CIRCUIT_BREAKER_OVERHEAD = Setting.doubleSetting(
        "plugins.neural_search.circuit_breaker.overhead",
        1.0,
        0.0,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<ByteSizeValue> NEURAL_CIRCUIT_BREAKER_LIMIT = Setting.memorySizeSetting(
        "plugins.neural_search.circuit_breaker.limit",
        "10%",
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final String SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX = "plugins.neural_search.model_selection.model_id.";

    /** The semantic field's model choice per name. */
    public static final Setting.AffixSetting<String> SEMANTIC_MODEL_SELECTION_MODEL_ID = Setting.prefixKeySetting(
        SEMANTIC_MODEL_SELECTION_MODEL_ID_PREFIX,
        key -> Setting.simpleString(key, Setting.Property.NodeScope, Setting.Property.Dynamic)
    );

    /** neural-search's sparse settings for its {@code native} engine, which the plugin refuses; accepted for configuration compatibility. */
    public static final Setting<Integer> SPARSE_ALGO_PARAM_INDEX_THREAD_QTY = Setting.intSetting(
        "plugins.neural_search.sparse.algo_param.index_thread_qty",
        1,
        1,
        1024,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<Boolean> SPARSE_NATIVE_ENGINE_FEATURE_ENABLED = Setting.boolSetting(
        "plugins.neural_search.sparse.native_engine_feature_enabled",
        true,
        Setting.Property.NodeScope
    );

    public static final Setting<Boolean> SPARSE_NATIVE_ENGINE_ENABLED = Setting.boolSetting(
        "plugins.neural_search.sparse.native_engine_enabled",
        false,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static List<Setting<?>> all() {
        return List.of(
            HYBRID_COLLAPSE_DOCS_PER_GROUP_PER_SUBQUERY,
            NEURAL_STATS_ENABLED,
            RERANKER_MAX_DOC_FIELDS,
            SEMANTIC_INGEST_BATCH_SIZE,
            NEURAL_CIRCUIT_BREAKER_OVERHEAD,
            NEURAL_CIRCUIT_BREAKER_LIMIT,
            SEMANTIC_MODEL_SELECTION_MODEL_ID,
            SPARSE_ALGO_PARAM_INDEX_THREAD_QTY,
            SPARSE_NATIVE_ENGINE_FEATURE_ENABLED,
            SPARSE_NATIVE_ENGINE_ENABLED
        );
    }
}
