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
import org.lucenerust.opensearch.engine.RustIndexerFactory;
import org.opensearch.index.IndexModule;
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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.SetOnce;

import java.net.URISyntaxException;
import java.nio.file.Files;
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

    /**
     * Whether this node installs the stored-fields reader wrapper (read path R6). An index takes one
     * reader wrapper ({@code IndexModule.setReaderWrapper} is set-once), which the security plugin
     * also claims for field- and document-level security; the default is on unless the security
     * plugin is installed beside this one.
     */
    public static final Setting<Boolean> NODE_FETCH_WRAPPER = Setting.boolSetting(
        "lucene_rust.fetch.reader_wrapper",
        true,
        Setting.Property.NodeScope
    );

    private static final Logger logger = LogManager.getLogger(RustSearchPlugin.class);

    private final NativeReaders readers = new NativeReaders();
    private final SearchStats stats = new SearchStats();
    private final String libraryPath;
    private final boolean nodeEngineEnabled;
    private final boolean fetchWrapper;

    public RustSearchPlugin(Settings settings) {
        this.libraryPath = NativeLibrary.load(pluginDir());
        this.nodeEngineEnabled = NODE_ENGINE_ENABLED.get(settings);
        boolean security = Files.isDirectory(pluginDir().resolveSibling("opensearch-security"));
        this.fetchWrapper = NODE_FETCH_WRAPPER.exists(settings) ? NODE_FETCH_WRAPPER.get(settings) : security == false;
        if (security && fetchWrapper == false) {
            logger.info("lucene-rust: the security plugin is installed; stored fields are read by Lucene");
        }
        RustIndexerFactory.verify();
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

    /** Every shard of a Rust-engine index gets the indexer that lets it be a segment-replication primary. */
    @Override
    public void onIndexModule(IndexModule indexModule) {
        RustIndexerFactory.install(indexModule);
        // Read path R6: every searcher OpenSearch hands out for the index -- the fetch phase's, the
        // get API's -- reads stored fields natively while index.lucene_rust.fetch.enabled is on, and
        // with Lucene otherwise (either way timed, SearchStats.fetchTime).
        if (fetchWrapper) {
            FetchFlag flag = new FetchFlag(FETCH_ENABLED.get(indexModule.getSettings()));
            indexModule.addSettingsUpdateConsumer(FETCH_ENABLED, flag::set);
            try {
                indexModule.setReaderWrapper(indexService -> reader -> NativeStoredFieldsReader.wrap(reader, readers, stats, flag::enabled));
            } catch (SetOnce.AlreadySetException e) {
                // Another plugin wrapped the index's readers first: its wrapper stays, Lucene reads.
                logger.warn("lucene-rust: [{}] already has a reader wrapper; stored fields are read by Lucene", indexModule.getIndex());
            }
        }
    }

    /** {@link #FETCH_ENABLED} for one index, updated by the index's settings consumer. */
    private static final class FetchFlag {
        private volatile boolean enabled;

        FetchFlag(boolean enabled) {
            this.enabled = enabled;
        }

        void set(boolean enabled) {
            this.enabled = enabled;
        }

        boolean enabled() {
            return enabled;
        }
    }

    /** Whether the index's stored fields are read natively (read path R6). */
    public static final Setting<Boolean> FETCH_ENABLED = Setting.boolSetting(
        "index.lucene_rust.fetch.enabled",
        true,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

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
            FETCH_ENABLED,
            NODE_FETCH_WRAPPER,
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
