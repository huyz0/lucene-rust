/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.settings.SettingsFilter;
import org.lucenerust.opensearch.engine.RustEngineFactory;
import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.indices.breaker.BreakerSettings;
import org.opensearch.monitor.jvm.JvmInfo;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.CircuitBreakerPlugin;
import org.opensearch.plugins.EnginePlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.SearchPlugin;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.search.query.QueryPhaseSearcher;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * lucene-rust for OpenSearch: serves the query phase of supported searches from the Rust engine
 * ({@link RustQueryPhaseSearcher}) and everything else from Lucene, and -- for an index created with
 * {@code index.lucene_rust.engine: true} -- indexes through the Rust writer ({@link
 * RustEngineFactory}).
 *
 * <p>The native library is loaded, and its ABI checked, in the constructor: a plugin whose library
 * is missing or stale stops the node at startup with one clear message rather than failing
 * searches later.
 */
public class RustSearchPlugin extends Plugin implements SearchPlugin, ActionPlugin, EnginePlugin, CircuitBreakerPlugin {
    /**
     * Whether this node serves Rust-engine indices with the Rust writer; {@code false} serves them
     * with OpenSearch's own engine -- a mixed cluster, one index.
     */
    public static final Setting<Boolean> NODE_ENGINE_ENABLED = Setting.boolSetting(
        "lucene_rust.engine.node_enabled",
        true,
        Setting.Property.NodeScope
    );

    private final NativeReaders readers = new NativeReaders();
    private final SearchStats stats = new SearchStats();
    private final String libraryPath;
    private final boolean nodeEngineEnabled;

    public RustSearchPlugin(Settings settings) {
        this.libraryPath = NativeLibrary.load(pluginDir());
        this.nodeEngineEnabled = NODE_ENGINE_ENABLED.get(settings);
    }

    /**
     * The breaker the Rust writers' buffers are accounted to -- native memory the JVM's own breakers
     * cannot see. {@code breaker.lucene_rust_writer.limit} sets it (default: 20% of the heap, the
     * same order as OpenSearch's own indexing buffer). It trips on its own limit. The parent
     * breaker adds it in only with {@code indices.breaker.total.use_real_memory: false}; with the
     * default real-memory parent, native bytes are outside the heap it measures.
     */
    @Override
    public BreakerSettings getCircuitBreaker(Settings settings) {
        long defaultLimit = JvmInfo.jvmInfo().getMem().getHeapMax().getBytes() / 5;
        return BreakerSettings.updateFromSettings(
            new BreakerSettings(
                RustEngineSupport.BREAKER_NAME,
                defaultLimit,
                1.0,
                CircuitBreaker.Type.MEMORY,
                CircuitBreaker.Durability.TRANSIENT
            ),
            settings
        );
    }

    @Override
    public void setCircuitBreaker(CircuitBreaker circuitBreaker) {
        RustEngineSupport.setBreaker(circuitBreaker);
    }

    @Override
    public Optional<EngineFactory> getEngineFactory(IndexSettings indexSettings) {
        if (indexSettings.getValue(RustEngineSupport.ENGINE_ENABLED) == false) {
            return Optional.empty();
        }
        // An index that asks for the Rust engine gets it, or a creation error saying why not; one
        // that only inherits the node default falls back to OpenSearch's engine where the Rust
        // writer cannot serve it.
        boolean asked = RustEngineSupport.ENGINE_ENABLED.exists(indexSettings.getSettings());
        boolean sorted = indexSettings.getIndexSortConfig().hasIndexSort();
        if (asked == false) {
            if (RustEngineSupport.unsupported(indexSettings, sorted) != null) {
                return Optional.empty();
            }
            MappingMetadata mapping = indexSettings.getIndexMetadata().mapping();
            if (mapping != null && RustEngineSupport.unsupportedField(mapping.sourceAsMap()) != null) {
                return Optional.empty();
            }
        }
        return Optional.of(new RustEngineFactory(nodeEngineEnabled));
    }

    private static Path pluginDir() {
        try {
            return Path.of(RustSearchPlugin.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("lucene-rust: cannot locate the plugin directory", e);
        }
    }

    @Override
    public Optional<QueryPhaseSearcher> getQueryPhaseSearcher() {
        return Optional.of(new RustQueryPhaseSearcher(readers, stats));
    }

    @Override
    public List<Setting<?>> getSettings() {
        return List.of(
            RustQueryPhaseSearcher.ENABLED,
            RustQueryPhaseSearcher.NATIVE_SHAPES,
            RustEngineSupport.ENGINE_ENABLED,
            RustEngineSupport.ENGINE_DEFAULT,
            RustEngineSupport.FAULT_INJECTION,
            NODE_ENGINE_ENABLED
        );
    }

    @Override
    public List<RestHandler> getRestHandlers(
        Settings settings,
        RestController restController,
        ClusterSettings clusterSettings,
        IndexScopedSettings indexScopedSettings,
        SettingsFilter settingsFilter,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<DiscoveryNodes> nodesInCluster
    ) {
        return List.of(new RestStatsAction(stats, readers, libraryPath));
    }

    @Override
    public void close() {
        readers.closeAll();
    }
}
