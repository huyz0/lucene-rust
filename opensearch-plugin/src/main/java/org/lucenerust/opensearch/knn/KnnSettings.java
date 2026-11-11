/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Setting.Property;

import java.util.List;

/**
 * k-NN's {@code KNNSettings} under their own names, so indexes and clusters configured for the stock
 * plugin are accepted. {@link #KNN_INDEX} turns the vector index of {@code knn_vector} fields on, and
 * {@link #EF_SEARCH} is the default search beam; the others tune the stock plugin's native-library
 * caches, circuit breakers and remote graph builds, which this node does not have, and are accepted
 * without effect.
 */
public final class KnnSettings {
    /** {@code index.knn}: whether {@code knn_vector} fields are indexed for approximate search. */
    public static final Setting<Boolean> KNN_INDEX = Setting.boolSetting("index.knn", false, Property.IndexScope, Property.Final);

    /** {@code index.knn.algo_param.ef_search}: the HNSW search beam when neither mapping nor query sets one. */
    public static final Setting<Integer> EF_SEARCH = Setting.intSetting(
        "index.knn.algo_param.ef_search",
        100,
        2,
        Property.IndexScope,
        Property.Dynamic
    );

    public static final Setting<Integer> APPROXIMATE_THRESHOLD = Setting.intSetting(
        "index.knn.advanced.approximate_threshold",
        0,
        -1,
        Integer.MAX_VALUE - 2,
        Property.IndexScope,
        Property.Dynamic
    );

    public static final Setting<Integer> FILTERED_EXACT_SEARCH_THRESHOLD = Setting.intSetting(
        "index.knn.advanced.filtered_exact_search_threshold",
        -1,
        Property.IndexScope,
        Property.Dynamic
    );

    /**
     * {@code index.knn.derived_source.enabled}: vectors are masked in the stored {@code _source} and
     * derived back from the vector files on read. On by default for a new knn index, as stock sets
     * it ({@code RustSearchPlugin.getAdditionalIndexSettingProviders}). Segments stock wrote that way
     * are read here ({@code DerivedSourceStoredFieldsFormat}); this node masks its own writes where
     * {@link DerivedSource#forIndex} allows, and keeps {@code _source} whole elsewhere.
     */
    public static final Setting<Boolean> DERIVED_SOURCE = Setting.boolSetting(
        "index.knn.derived_source.enabled",
        false,
        Property.IndexScope,
        Property.Final
    );

    public static final Setting<Boolean> MEMORY_OPTIMIZED_SEARCH = Setting.boolSetting(
        "index.knn.memory_optimized_search",
        false,
        Property.IndexScope
    );

    public static final Setting<Boolean> SHARD_LEVEL_RESCORING_DISABLED = Setting.boolSetting(
        "index.knn.disk.vector.shard_level_rescoring_disabled",
        false,
        Property.IndexScope,
        Property.Dynamic
    );

    public static final Setting<Boolean> FAISS_FILTER_DISABLE_EXACT_SEARCH = Setting.boolSetting(
        "index.knn.faiss.efficient_filter.disable_exact_search",
        false,
        Property.IndexScope,
        Property.Dynamic
    );

    public static final Setting<Boolean> INDEX_REMOTE_BUILD = Setting.boolSetting(
        "index.knn.remote_index_build.enabled",
        true,
        Property.IndexScope,
        Property.Dynamic
    );

    private static Setting<Integer> nodeInt(String key, int def) {
        return Setting.intSetting(key, def, 0, Property.NodeScope, Property.Dynamic);
    }

    /**
     * {@code knn.quantization.cache.size.limit}: a size or heap ratio, at most 10% of the heap. The
     * remote index build and quantization cache settings configure features of the faiss engine,
     * which the plugin does not have; they are accepted and validated as stock does, and read by
     * nothing.
     */
    static final Setting<org.opensearch.core.common.unit.ByteSizeValue> QUANTIZATION_STATE_CACHE_SIZE_LIMIT = new Setting<>(
        "knn.quantization.cache.size.limit",
        "5%",
        s -> {
            org.opensearch.core.common.unit.ByteSizeValue limit = org.opensearch.common.unit.MemorySizeValue.parseBytesSizeValueOrHeapRatio(
                s,
                "knn.quantization.cache.size.limit"
            );
            org.opensearch.core.common.unit.ByteSizeValue heap = org.opensearch.monitor.jvm.JvmInfo.jvmInfo().getMem().getHeapMax();
            if (limit.getKbFrac() / heap.getKbFrac() > 0.10) {
                throw new org.opensearch.OpenSearchParseException(
                    "{} ({} KB) cannot exceed {}% of the heap ({} KB).",
                    "knn.quantization.cache.size.limit",
                    limit.getKb(),
                    10,
                    heap.getKb()
                );
            }
            return limit;
        },
        Property.NodeScope,
        Property.Dynamic
    );

    private static Setting<Boolean> nodeBool(String key, boolean def, boolean dynamic) {
        return dynamic
            ? Setting.boolSetting(key, def, Property.NodeScope, Property.Dynamic)
            : Setting.boolSetting(key, def, Property.NodeScope);
    }

    /** Every setting, for {@code Plugin.getSettings}. */
    public static List<Setting<?>> settings() {
        return List.of(
            KNN_INDEX,
            EF_SEARCH,
            APPROXIMATE_THRESHOLD,
            FILTERED_EXACT_SEARCH_THRESHOLD,
            DERIVED_SOURCE,
            MEMORY_OPTIMIZED_SEARCH,
            SHARD_LEVEL_RESCORING_DISABLED,
            FAISS_FILTER_DISABLE_EXACT_SEARCH,
            INDEX_REMOTE_BUILD,
            nodeInt("knn.algo_param.index_thread_qty", 1),
            nodeBool("knn.memory.circuit_breaker.enabled", true, true),
            Setting.simpleString("knn.memory.circuit_breaker.limit", "50%", Property.NodeScope, Property.Dynamic),
            nodeBool("knn.circuit_breaker.triggered", false, true),
            Setting.doubleSetting("knn.circuit_breaker.unset.percentage", 75, 0, 100, Property.NodeScope, Property.Dynamic),
            nodeBool("knn.feature.cache.force_evict.enabled", false, true),
            nodeBool("knn.feature.prefetch.enabled", true, true),
            nodeBool("knn.cache.item.expiry.enabled", false, true),
            Setting.positiveTimeSetting(
                "knn.cache.item.expiry.minutes",
                org.opensearch.common.unit.TimeValue.timeValueHours(3),
                Property.NodeScope,
                Property.Dynamic
            ),
            nodeInt("knn.model.index.number_of_shards", 1),
            nodeInt("knn.model.index.number_of_replicas", 1),
            Setting.simpleString("knn.model.cache.size.limit", "10%", Property.NodeScope, Property.Dynamic),
            nodeBool("knn.faiss.avx2.disabled", false, false),
            nodeBool("knn.faiss.avx512.disabled", false, false),
            nodeBool("knn.faiss.avx512_spr.disabled", false, false),
            nodeBool("knn.remote_index_build.enabled", false, true),
            nodeBool("knn.dynamic_mapping.enabled", false, true),
            Setting.memorySizeSetting("knn.vector_streaming_memory.limit", "1%", Property.Dynamic, Property.NodeScope),
            QUANTIZATION_STATE_CACHE_SIZE_LIMIT,
            Setting.positiveTimeSetting(
                "knn.quantization.cache.expiry.minutes",
                org.opensearch.common.unit.TimeValue.timeValueMinutes(60),
                Property.NodeScope,
                Property.Dynamic
            ),
            Setting.byteSizeSetting(
                "index.knn.remote_index_build.size.min",
                new org.opensearch.core.common.unit.ByteSizeValue(50, org.opensearch.core.common.unit.ByteSizeUnit.MB),
                Property.Dynamic,
                Property.IndexScope
            ),
            Setting.simpleString("knn.remote_index_build.repository", Property.Dynamic, Property.NodeScope),
            Setting.byteSizeSetting(
                "knn.remote_index_build.size.max",
                new org.opensearch.core.common.unit.ByteSizeValue(0, org.opensearch.core.common.unit.ByteSizeUnit.MB),
                Property.Dynamic,
                Property.NodeScope
            ),
            Setting.simpleString("knn.remote_index_build.service.endpoint", Property.NodeScope, Property.Dynamic),
            Setting.timeSetting(
                "knn.remote_index_build.client.timeout",
                org.opensearch.common.unit.TimeValue.timeValueMinutes(60),
                Property.NodeScope,
                Property.Dynamic
            ),
            Setting.timeSetting(
                "knn.remote_index_build.poll.interval",
                org.opensearch.common.unit.TimeValue.timeValueSeconds(5),
                Property.NodeScope,
                Property.Dynamic
            ),
            org.opensearch.common.settings.SecureSetting.secureString("knn.remote_index_build.service.username", null),
            org.opensearch.common.settings.SecureSetting.secureString("knn.remote_index_build.service.password", null)
        );
    }

    private KnnSettings() {}
}
