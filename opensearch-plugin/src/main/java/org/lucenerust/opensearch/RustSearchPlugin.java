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
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.lucenerust.opensearch.engine.RustEngineFactory;
import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.lucenerust.opensearch.engine.RustIndexerFactory;
import org.lucenerust.opensearch.store.LocalKeySource;
import org.lucenerust.opensearch.store.RustEncryptedDirectoryFactory;
import org.lucenerust.opensearch.store.StoreCrypto;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.indices.IndexingMemoryController;
import org.opensearch.indices.breaker.BreakerSettings;
import org.opensearch.index.analysis.AnalyzerProvider;
import org.opensearch.index.analysis.TokenizerFactory;
import org.opensearch.indices.analysis.AnalysisModule;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.AnalysisPlugin;
import org.opensearch.plugins.CircuitBreakerPlugin;
import org.opensearch.plugins.EnginePlugin;
import org.opensearch.plugins.IndexStorePlugin;
import org.opensearch.plugins.MapperPlugin;
import org.opensearch.index.mapper.Mapper;
import org.lucenerust.opensearch.sparse.NeuralSparseQueryBuilder;
import org.lucenerust.opensearch.sparse.SparseSettings;
import org.lucenerust.opensearch.sparse.SparseVectorFieldMapper;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
public class RustSearchPlugin extends Plugin implements
    SearchPlugin,
    ActionPlugin,
    EnginePlugin,
    CircuitBreakerPlugin,
    AnalysisPlugin,
    IndexStorePlugin,
    MapperPlugin {
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
     * Whether this node installs the stored-fields reader wrapper. An index takes one
     * reader wrapper ({@code IndexModule.setReaderWrapper} is set-once), which the security plugin
     * also claims for field- and document-level security; the default is on unless the security
     * plugin is installed beside this one.
     */
    public static final Setting<Boolean> NODE_FETCH_WRAPPER = Setting.boolSetting(
        "lucene_rust.fetch.reader_wrapper",
        true,
        Setting.Property.NodeScope
    );

    /**
     * Whether the Rust engine decrypts {@code rust_encrypted} shard files page by page on first touch,
     * keeping at most {@link #NODE_DECRYPT_CACHE} of them resident, instead of decrypting each file
     * whole into memory when it opens. Needs Linux with {@code userfaultfd} permitted (container
     * runtimes' default seccomp profiles block it); elsewhere files are decrypted whole.
     */
    public static final Setting<Boolean> NODE_LAZY_DECRYPT = Setting.boolSetting(
        "lucene_rust.store.encrypted.lazy_decrypt",
        false,
        Setting.Property.NodeScope
    );

    /** Decrypted bytes kept resident across all demand-decrypted shard files on this node. */
    public static final Setting<ByteSizeValue> NODE_DECRYPT_CACHE = Setting.byteSizeSetting(
        "lucene_rust.store.encrypted.decrypt_cache_size",
        new ByteSizeValue(512, ByteSizeUnit.MB),
        new ByteSizeValue(1, ByteSizeUnit.MB),
        new ByteSizeValue(Long.MAX_VALUE, ByteSizeUnit.BYTES),
        Setting.Property.NodeScope
    );

    private static final Logger logger = LogManager.getLogger(RustSearchPlugin.class);

    private final SetOnce<org.opensearch.cluster.service.ClusterService> clusterService = new SetOnce<>();
    private final SetOnce<IndexNameExpressionResolver> resolver = new SetOnce<>();

    private final NativeReaders readers = new NativeReaders();
    private final SearchStats stats = new SearchStats();
    private final String libraryPath;
    private final boolean nodeEngineEnabled;
    private final boolean fetchWrapper;
    private final LocalKeySource.Config localKey;

    public RustSearchPlugin(Settings settings, Path configPath) {
        this.libraryPath = NativeLibrary.load(pluginDir());
        addDataPackDirs(configPath);
        this.nodeEngineEnabled = NODE_ENGINE_ENABLED.get(settings);
        boolean security = Files.isDirectory(pluginDir().resolveSibling("opensearch-security"));
        this.fetchWrapper = NODE_FETCH_WRAPPER.exists(settings) ? NODE_FETCH_WRAPPER.get(settings) : security == false;
        if (security && fetchWrapper == false) {
            logger.info("lucene-rust: the security plugin is installed; stored fields are read by Lucene");
        }
        RustIndexerFactory.verify();
        // The keystore is only open during startup.
        this.localKey = LocalKeySource.Config.fromNode(settings);
        // The FIPS build checks its crypto now: a node that cannot encrypt with validated modules
        // does not start.
        if (StoreCrypto.buildIsFips()) {
            logger.info("lucene-rust: encrypted store crypto is {}", StoreCrypto.get().description());
        }
        configureLazyDecrypt(settings);
    }

    private static void configureLazyDecrypt(Settings settings) {
        boolean wanted = NODE_LAZY_DECRYPT.get(settings);
        int rc = NativeBridge.storeSetLazyDecrypt(wanted, NODE_DECRYPT_CACHE.get(settings).getBytes());
        if (rc != 0) {
            throw new IllegalStateException("lucene-rust: cannot configure demand decryption (status " + rc + ")");
        }
        if (wanted) {
            if (NativeBridge.storeLazyDecryptActive()) {
                logger.info(
                    "lucene-rust: encrypted shard files are demand-decrypted, {} resident at most",
                    NODE_DECRYPT_CACHE.get(settings)
                );
            } else {
                logger.warn(
                    "lucene-rust: [{}] is set but userfaultfd is unavailable here; encrypted shard files are decrypted whole",
                    NODE_LAZY_DECRYPT.getKey()
                );
            }
        }
    }

    /**
     * The {@code rust_encrypted} store type: encryption at rest with keys from a KMS, which Lucene
     * and the Rust engine both read and write ({@link RustEncryptedDirectoryFactory}).
     */
    @Override
    public Map<String, DirectoryFactory> getDirectoryFactories() {
        return Map.of(RustEncryptedDirectoryFactory.TYPE, new RustEncryptedDirectoryFactory(localKey));
    }

    /**
     * The breaker the Rust writers' buffers are accounted to -- native memory the JVM's own breakers
     * cannot see. {@code breaker.lucene_rust_writer.limit} sets it. The default is three times
     * {@code indices.memory.index_buffer_size} (30% of the heap by default): a writer stalls
     * indexing once its buffered and flushing bytes pass twice its RAM buffer, as Lucene's does, so
     * under steady load the buffer sits near that bound, and the breaker must sit above it. It trips
     * on its own limit. The parent breaker adds it in only with
     * {@code indices.breaker.total.use_real_memory: false}; with the default real-memory parent,
     * native bytes are outside the heap it measures.
     */
    @Override
    public BreakerSettings getCircuitBreaker(Settings settings) {
        long defaultLimit = 3 * indexBufferBytes(settings);
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

    /** {@code indices.memory.index_buffer_size} in bytes, clamped as IndexingMemoryController clamps a percentage. */
    static long indexBufferBytes(Settings settings) {
        long bytes = IndexingMemoryController.INDEX_BUFFER_SIZE_SETTING.get(settings).getBytes();
        String raw = settings.get(IndexingMemoryController.INDEX_BUFFER_SIZE_SETTING.getKey());
        if (raw == null || raw.endsWith("%")) {
            bytes = Math.max(bytes, IndexingMemoryController.MIN_INDEX_BUFFER_SIZE_SETTING.get(settings).getBytes());
            long max = IndexingMemoryController.MAX_INDEX_BUFFER_SIZE_SETTING.get(settings).getBytes();
            if (max >= 0) {
                bytes = Math.min(bytes, max);
            }
        }
        return bytes;
    }

    @Override
    public void setCircuitBreaker(CircuitBreaker circuitBreaker) {
        RustEngineSupport.setBreaker(circuitBreaker);
    }

    /** Every shard of a Rust-engine index gets the indexer that lets it be a segment-replication primary. */
    @Override
    public void onIndexModule(IndexModule indexModule) {
        RustIndexerFactory.install(indexModule);
        // The fuzzy-set _id postings format is refused at creation (RustEngineSupport.unsupported); the
        // setting is dynamic, so turning it on later is refused too, rather than ignored.
        if (RustEngineSupport.servedByRust(indexModule.getSettings())) {
            indexModule.addSettingsUpdateConsumer(IndexSettings.INDEX_DOC_ID_FUZZY_SET_ENABLED_SETTING, enabled -> {}, enabled -> {
                if (enabled) {
                    throw new IllegalArgumentException(RustEngineSupport.FUZZY_SET_REFUSED);
                }
            });
        }
        // Every searcher OpenSearch hands out for the index -- the fetch
        // phase's, the get API's -- reads stored fields natively while
        // index.lucene_rust.fetch.enabled is on, and with Lucene otherwise
        // (either way timed, SearchStats.fetchTime).
        if (fetchWrapper) {
            FetchFlag flag = new FetchFlag(FETCH_ENABLED.get(indexModule.getSettings()));
            indexModule.addSettingsUpdateConsumer(FETCH_ENABLED, flag::set);
            try {
                indexModule.setReaderWrapper(indexService -> reader -> NativeStoredFieldsReader.wrap(reader, readers, stats, flag::enabled));
            } catch (SetOnce.AlreadySetException e) {
                // Another plugin wrapped the index's readers first: its wrapper
                // stays, Lucene reads.
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

    /** Whether the index's stored fields are read natively. */
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

    /**
     * Where Rust looks for the analysis modules' data packs (their dictionaries, see
     * {@link RustAnalysis}): {@code <config>/lucene-rust/packs}, then the plugin's own {@code packs}
     * directory (a zip built with {@code -PdataPacks}).
     */
    static void addDataPackDirs(Path configPath) {
        List<Path> dirs = new ArrayList<>();
        if (configPath != null) {
            dirs.add(configPath.resolve("lucene-rust").resolve("packs"));
        }
        dirs.add(pluginDir().resolve("packs"));
        for (Path dir : dirs) {
            if (NativeBridge.addDataPackDir(dir.toAbsolutePath().toString()) != 0) {
                logger.warn("lucene-rust: cannot add data pack directory [{}]: {}", dir, NativeBridge.lastError());
            }
        }
    }

    private static Path pluginDir() {
        try {
            return Path.of(RustSearchPlugin.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("lucene-rust: cannot locate the plugin directory", e);
        }
    }

    /** The {@code rust_*} tokenizers ({@link RustAnalysis#TOKENIZERS}): Rust on the index and query side alike. */
    @Override
    public Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> getTokenizers() {
        return RustAnalysis.tokenizers();
    }

    @Override
    public Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends org.apache.lucene.analysis.Analyzer>>> getAnalyzers() {
        return RustAnalysis.analyzers();
    }

    @Override
    public Optional<QueryPhaseSearcher> getQueryPhaseSearcher() {
        return Optional.of(new RustQueryPhaseSearcher(readers, stats));
    }

    @Override
    public List<Setting<?>> getSettings() {
        List<Setting<?>> settings = new ArrayList<>(RustEncryptedDirectoryFactory.settings());
        settings.addAll(List.of(
            RustQueryPhaseSearcher.ENABLED,
            FETCH_ENABLED,
            NODE_FETCH_WRAPPER,
            RustQueryPhaseSearcher.NATIVE_SHAPES,
            RustEngineSupport.ENGINE_ENABLED,
            RustEngineSupport.ENGINE_DEFAULT,
            RustEngineSupport.FAULT_INJECTION,
            RustEngineSupport.REALTIME_SEARCH,
            RustEngineSupport.ADAPTIVE_DOC_VALUES,
            RustEngineSupport.PARTITION_FIELD,
            NODE_ENGINE_ENABLED,
            LocalKeySource.KEY,
            LocalKeySource.KEY_ID,
            NODE_LAZY_DECRYPT,
            NODE_DECRYPT_CACHE,
            SparseSettings.SPARSE_INDEX
        ));
        settings.addAll(org.lucenerust.opensearch.engine.BpReorderSettings.settings());
        return settings;
    }

    /** neural-search's {@code sparse_vector} field, which the Rust engine clusters and searches. */
    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return Map.of(SparseVectorFieldMapper.CONTENT_TYPE, new SparseVectorFieldMapper.TypeParser());
    }

    /** neural-search's {@code neural_sparse} query, by {@code query_tokens}. */
    @Override
    public List<QuerySpec<?>> getQueries() {
        return List.of(new QuerySpec<>(NeuralSparseQueryBuilder.NAME, NeuralSparseQueryBuilder::new, NeuralSparseQueryBuilder::fromXContent));
    }

    @Override
    public java.util.Collection<Object> createComponents(
        org.opensearch.transport.client.Client client,
        org.opensearch.cluster.service.ClusterService clusterService,
        org.opensearch.threadpool.ThreadPool threadPool,
        org.opensearch.watcher.ResourceWatcherService resourceWatcherService,
        org.opensearch.script.ScriptService scriptService,
        org.opensearch.core.xcontent.NamedXContentRegistry xContentRegistry,
        org.opensearch.env.Environment environment,
        org.opensearch.env.NodeEnvironment nodeEnvironment,
        org.opensearch.core.common.io.stream.NamedWriteableRegistry namedWriteableRegistry,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<org.opensearch.repositories.RepositoriesService> repositoriesServiceSupplier
    ) {
        this.clusterService.set(clusterService);
        this.resolver.set(indexNameExpressionResolver);
        return List.of();
    }

    @Override
    public List<org.opensearch.action.support.ActionFilter> getActionFilters() {
        return List.of(new RealtimePrimaryFilter(clusterService::get, resolver::get));
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
