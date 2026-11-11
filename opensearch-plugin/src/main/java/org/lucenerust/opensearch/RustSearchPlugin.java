/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.lucenerust.opensearch.hexid.HexIdDocValueFormat;
import org.lucenerust.opensearch.hexid.HexIdFieldMapper;
import org.lucenerust.opensearch.tensor.MaxSimQueryBuilder;
import org.lucenerust.opensearch.tensor.MultiVectorFieldMapper;
import org.lucenerust.opensearch.ml.MlActions;
import org.lucenerust.opensearch.ml.MlCommons;
import org.lucenerust.opensearch.ml.MlIndices;
import org.lucenerust.opensearch.ml.MlRestHandlers;
import org.lucenerust.opensearch.ml.MlSettings;
import org.opensearch.cluster.node.DiscoveryNodeRole;
import org.opensearch.indices.SystemIndexDescriptor;
import org.opensearch.plugins.SystemIndexPlugin;
import org.lucenerust.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.lucenerust.opensearch.knn.search.processor.KNNSourceExcludesProcessor;
import org.lucenerust.opensearch.knn.search.processor.mmr.MMRKnnQueryTransformer;
import org.lucenerust.opensearch.knn.search.processor.mmr.MMROverSampleProcessor;
import org.lucenerust.opensearch.knn.search.processor.mmr.MMRQueryTransformer;
import org.lucenerust.opensearch.knn.search.processor.mmr.MMRRerankProcessor;
import org.lucenerust.opensearch.neural.processor.mmr.MMRNeuralQueryTransformer;
import org.lucenerust.opensearch.neural.mapper.SemanticFieldMapper;
import org.lucenerust.opensearch.neural.mappingtransformer.SemanticMappingTransformer;
import org.lucenerust.opensearch.neural.ml.resolver.DefaultSemanticModelResolver;
import org.lucenerust.opensearch.neural.ml.resolver.SemanticModelResolver;
import org.lucenerust.opensearch.neural.processor.factory.SemanticFieldProcessorFactory;
import org.lucenerust.opensearch.neural.settings.SemanticModelSelectionSettingsAccessor;
import org.lucenerust.opensearch.neural.HybridSearch;
import org.lucenerust.opensearch.neural.NeuralSearchSettings;
import org.opensearch.plugins.SearchPipelinePlugin;
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
import org.opensearch.index.shard.SearchOperationListener;
import org.opensearch.search.internal.SearchContext;
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
import org.opensearch.index.codec.CodecServiceFactory;
import org.lucenerust.opensearch.knn.codec.KnnCodecService;
import org.opensearch.plugins.IndexStorePlugin;
import org.opensearch.plugins.MapperPlugin;
import org.opensearch.index.mapper.Mapper;
import org.lucenerust.opensearch.knn.KnnQueryBuilder;
import org.lucenerust.opensearch.knn.KnnSettings;
import org.lucenerust.opensearch.knn.KnnTopVectorQuery;
import org.lucenerust.opensearch.knn.KnnVectorFieldMapper;
import org.lucenerust.opensearch.knn.DerivedSource;
import org.opensearch.index.TieredMergePolicyProvider;
import org.opensearch.index.shard.IndexSettingProvider;
import org.lucenerust.opensearch.knn.rest.KnnModelHandlers;
import org.lucenerust.opensearch.knn.rest.KnnRestHandlers;
import org.lucenerust.opensearch.knn.rest.KnnShardsTransport;
import org.lucenerust.opensearch.knn.rest.KnnStatsTransport;
import org.opensearch.action.ActionRequest;
import org.opensearch.core.action.ActionResponse;
import org.lucenerust.opensearch.neural.ml.MLCommonsClientAccessor;
import org.lucenerust.opensearch.neural.processor.NeuralQueryEnricherProcessor;
import org.lucenerust.opensearch.neural.processor.NeuralSparseTwoPhaseProcessor;
import org.lucenerust.opensearch.neural.processor.factory.RerankProcessorFactory;
import org.lucenerust.opensearch.neural.processor.rerank.RerankProcessor;
import org.lucenerust.opensearch.neural.query.ext.RerankSearchExtBuilder;
import org.lucenerust.opensearch.neural.processor.SparseEncodingProcessor;
import org.lucenerust.opensearch.neural.processor.TextChunkingProcessor;
import org.lucenerust.opensearch.neural.processor.TextEmbeddingProcessor;
import org.lucenerust.opensearch.neural.processor.TextImageEmbeddingProcessor;
import org.lucenerust.opensearch.neural.processor.factory.SparseEncodingProcessorFactory;
import org.lucenerust.opensearch.neural.processor.factory.TextChunkingProcessorFactory;
import org.lucenerust.opensearch.neural.processor.factory.TextEmbeddingProcessorFactory;
import org.lucenerust.opensearch.neural.processor.factory.TextImageEmbeddingProcessorFactory;
import org.lucenerust.opensearch.neural.query.NeuralKNNQueryBuilder;
import org.lucenerust.opensearch.neural.query.NeuralQueryBuilder;
import org.lucenerust.opensearch.neural.query.NeuralSparseQueryBuilder;
import org.lucenerust.opensearch.ml.HfAnalysis;
import org.opensearch.index.analysis.PreBuiltAnalyzerProviderFactory;
import org.opensearch.index.analysis.PreConfiguredTokenizer;
import org.opensearch.plugins.IngestPlugin;
import org.lucenerust.opensearch.sparse.SparseSettings;
import org.lucenerust.opensearch.sparse.SparseVectorFieldMapper;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.ScriptPlugin;
import org.opensearch.plugins.SearchPlugin;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.lucenerust.opensearch.knn.script.KnnScoringScriptEngine;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.search.query.QueryPhaseSearcher;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.SetOnce;

import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
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
    MapperPlugin,
    IngestPlugin,
    ScriptPlugin,
    SearchPipelinePlugin,
    SystemIndexPlugin {
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

    /**
     * Bytes the page cache keeps resident across every shard whose {@code index.lucene_rust.store_mode}
     * is {@code page_cache}; pages past it are released and read again from disk on their next touch.
     */
    public static final Setting<ByteSizeValue> NODE_PAGE_CACHE_SIZE = Setting.byteSizeSetting(
        "lucene_rust.store.page_cache.size",
        new ByteSizeValue(1, ByteSizeUnit.GB),
        new ByteSizeValue(1, ByteSizeUnit.MB),
        new ByteSizeValue(Long.MAX_VALUE, ByteSizeUnit.BYTES),
        Setting.Property.NodeScope
    );

    /**
     * Bytes the aggregation segment cache keeps collected parts in, across every index on this node
     * that opts in with {@code index.lucene_rust.search.aggregation_segment_cache}; least recently
     * used parts past it are dropped. 0 turns the cache off.
     */
    public static final Setting<ByteSizeValue> NODE_AGG_SEGMENT_CACHE_SIZE = Setting.byteSizeSetting(
        "lucene_rust.search.aggregation_segment_cache.size",
        new ByteSizeValue(256, ByteSizeUnit.MB),
        new ByteSizeValue(0, ByteSizeUnit.BYTES),
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
    /**
     * Whether the stock k-NN plugin is installed beside this one: it then owns {@code knn_vector},
     * {@code knn} and the {@code knn} settings, and the Rust writer leaves its fields alone.
     */
    private final boolean stockKnn;
    /** Whether the stock ml-commons plugin is installed beside this one: it then serves {@code _plugins/_ml}. */
    private final boolean stockMl;

    private MLCommonsClientAccessor mlClient;
    private org.opensearch.core.xcontent.NamedXContentRegistry xContentRegistry;
    private SemanticMappingTransformer semanticMappingTransformer;
    private volatile SemanticModelResolver modelResolver;

    public RustSearchPlugin(Settings settings, Path configPath) {
        this.libraryPath = NativeLibrary.load(pluginDir());
        this.stockKnn = stockKnnInstalled();
        this.stockMl = MlCommons.stockInstalled(pluginDir());
        if (stockKnn) {
            logger.info("lucene-rust: the k-NN plugin is installed; it serves knn_vector and knn");
            org.lucenerust.opensearch.engine.RustEngineSupport.refuseKnnVectorFields();
        } else {
            KnnTopVectorQuery.setNative(new KnnTopVectorQuery.NativeSearch() {
                @Override
                public org.lucenerust.opensearch.knn.KnnHitsQuery search(org.apache.lucene.search.IndexSearcher searcher, KnnTopVectorQuery q) {
                    return KnnEncoder.topHits(readers, searcher, q);
                }

                @Override
                public void readBuffersByLucene(org.apache.lucene.search.IndexSearcher searcher) throws java.io.IOException {
                    KnnEncoder.readBuffersByLucene(readers, searcher);
                }
            });
        }
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
        configurePageCache(settings);
        configureAggSegmentCache(settings);
    }

    private static void configureAggSegmentCache(Settings settings) {
        int rc = NativeBridge.aggCacheSetBudget(NODE_AGG_SEGMENT_CACHE_SIZE.get(settings).getBytes());
        if (rc != 0) {
            throw new IllegalStateException("lucene-rust: cannot configure the aggregation segment cache (status " + rc + ")");
        }
    }

    private static void configurePageCache(Settings settings) {
        int rc = NativeBridge.storeSetPageCacheBudget(NODE_PAGE_CACHE_SIZE.get(settings).getBytes());
        if (rc != 0) {
            throw new IllegalStateException("lucene-rust: cannot configure the page cache (status " + rc + ")");
        }
        if (NativeBridge.storePageCacheAvailable() == false) {
            logger.info("lucene-rust: userfaultfd is unavailable here; indices with store_mode [page_cache] stay mapped");
        }
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

    /**
     * k-NN's settings for a new knn index ({@code KNNPlugin.getAdditionalIndexSettingProviders}):
     * derived source on, and gentler merges (vector merges are CPU heavy). The request's own
     * settings still win.
     */
    @Override
    public Collection<IndexSettingProvider> getAdditionalIndexSettingProviders() {
        if (stockKnn) {
            return List.of();
        }
        return List.of(new IndexSettingProvider() {
            @Override
            public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
                if (templateAndRequestSettings.getAsBoolean(KnnSettings.KNN_INDEX.getKey(), false) == false) {
                    return Settings.EMPTY;
                }
                return Settings.builder()
                    .put(KnnSettings.DERIVED_SOURCE.getKey(), true)
                    .put(TieredMergePolicyProvider.INDEX_MERGE_POLICY_MAX_MERGE_AT_ONCE_SETTING.getKey(), 10)
                    .put(TieredMergePolicyProvider.INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING.getKey(), new ByteSizeValue(2, ByteSizeUnit.MB))
                    .build();
            }
        });
    }

    /** Every shard of a Rust-engine index gets the indexer that lets it be a segment-replication primary. */
    @Override
    public void onIndexModule(IndexModule indexModule) {
        RustIndexerFactory.install(indexModule);
        if (stockKnn == false && DerivedSource.forIndex(indexModule.getSettings())) {
            indexModule.addIndexOperationListener(new DerivedSource.Listener(RustEngineSupport.REALTIME_SEARCH.get(indexModule.getSettings())));
        }
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
                indexModule.setReaderWrapper(indexService -> {
                    boolean derivedSource = indexService.getIndexSettings().isDerivedSourceEnabled();
                    return reader -> NativeStoredFieldsReader.wrap(reader, readers, stats, flag::enabled, derivedSource);
                });
                // The fetch phase reads its hits one at a time; announcing them first lets a
                // segment read by range fetch their stored fields together.
                indexModule.addSearchOperationListener(new SearchOperationListener() {
                    @Override
                    public void onPreFetchPhase(SearchContext context) {
                        NativeStoredFieldsReader.fetchStarted(stockKnn == false && DerivedSource.vectorsDropped(context));
                        if (flag.enabled() == false) {
                            return;
                        }
                        // `stored_fields: _none_`: the fetch phase reads no stored fields.
                        if (context.storedFieldsContext() != null && context.storedFieldsContext().fetchFields() == false) {
                            return;
                        }
                        try {
                            NativeStoredFieldsReader.prefetch(
                                context.searcher().getIndexReader(),
                                context.docIdsToLoad(),
                                context.docIdsToLoadFrom(),
                                context.docIdsToLoadSize()
                            );
                        } catch (RuntimeException e) {
                            // Only a hint: the fetch phase reads every document itself.
                            logger.debug("lucene-rust: stored-fields prefetch failed", e);
                        }
                    }

                    @Override
                    public void onFetchPhase(SearchContext context, long tookInNanos) {
                        NativeStoredFieldsReader.fetchEnded();
                    }

                    @Override
                    public void onFailedFetchPhase(SearchContext context) {
                        NativeStoredFieldsReader.fetchEnded();
                    }
                });
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

    /**
     * {@code KNNPlugin.getCustomCodecServiceFactory}: OpenSearch's engine writes a k-NN index's
     * vector fields in the formats their mappings resolve to, as the Rust writer does.
     */
    @Override
    public Optional<CodecServiceFactory> getCustomCodecServiceFactory(IndexSettings indexSettings) {
        if (stockKnn || KnnSettings.KNN_INDEX.get(indexSettings.getSettings()) == false) {
            return Optional.empty();
        }
        return Optional.of(KnnCodecService::new);
    }

    @Override
    public Optional<EngineFactory> getEngineFactory(IndexSettings indexSettings) {
        if (indexSettings.getValue(RustEngineSupport.ENGINE_ENABLED) == false) {
            return Optional.empty();
        }
        // An index that asks for the Rust engine gets it, or a creation error saying why not; one
        // that only inherits the node default falls back to OpenSearch's engine where the Rust
        // writer cannot serve it. Realtime search is the Rust engine's alone, so asking for it
        // asks for the engine: it is never dropped silently.
        boolean asked = RustEngineSupport.ENGINE_ENABLED.exists(indexSettings.getSettings())
            || indexSettings.getValue(RustEngineSupport.REALTIME_SEARCH);
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

    /** Whether the stock k-NN plugin is installed beside this one. */
    public static boolean stockKnnInstalled() {
        return Files.isDirectory(pluginDir().resolveSibling("opensearch-knn"));
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
        if (stockMl) {
            return RustAnalysis.tokenizers();
        }
        Map<String, AnalysisModule.AnalysisProvider<TokenizerFactory>> tokenizers = new java.util.HashMap<>(RustAnalysis.tokenizers());
        tokenizers.putAll(HfAnalysis.tokenizers());
        return tokenizers;
    }

    /** ml-commons' pre-built {@code bert-uncased} and {@code mbert-uncased} tokenizers. */
    @Override
    public List<PreConfiguredTokenizer> getPreConfiguredTokenizers() {
        return stockMl ? List.of() : HfAnalysis.preConfiguredTokenizers();
    }

    /** ml-commons' pre-built {@code bert-uncased} and {@code mbert-uncased} analyzers. */
    @Override
    public List<PreBuiltAnalyzerProviderFactory> getPreBuiltAnalyzerProviderFactories() {
        return stockMl ? List.of() : HfAnalysis.preBuiltAnalyzers();
    }

    @Override
    public Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends org.apache.lucene.analysis.Analyzer>>> getAnalyzers() {
        if (stockMl) {
            return RustAnalysis.analyzers();
        }
        Map<String, AnalysisModule.AnalysisProvider<AnalyzerProvider<? extends org.apache.lucene.analysis.Analyzer>>> analyzers = new java.util.HashMap<>(RustAnalysis.analyzers());
        analyzers.putAll(HfAnalysis.analyzers());
        return analyzers;
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
            RustQueryPhaseSearcher.AGG_SEGMENT_CACHE,
            NODE_AGG_SEGMENT_CACHE_SIZE,
            RustEngineSupport.ENGINE_ENABLED,
            RustEngineSupport.ENGINE_DEFAULT,
            RustEngineSupport.FAULT_INJECTION,
            RustEngineSupport.REALTIME_SEARCH,
            RustEngineSupport.REALTIME_MAX_STALENESS,
            RustEngineSupport.ADAPTIVE_DOC_VALUES,
            RustEngineSupport.ZSTD_TERMS,
            RustEngineSupport.MERGE_LZ4_HIGH_COMPRESSION,
            RustEngineSupport.PAGED_ZSTD_STORED_FIELDS,
            RustEngineSupport.STORE_MODE,
            RustEngineSupport.PARTITION_FIELD,
            RustEngineSupport.IVF_PARTITION_CLUSTERS,
            NODE_ENGINE_ENABLED,
            LocalKeySource.KEY,
            LocalKeySource.KEY_ID,
            NODE_LAZY_DECRYPT,
            NODE_DECRYPT_CACHE,
            NODE_PAGE_CACHE_SIZE,
            org.lucenerust.opensearch.logs.LogsPreset.ENABLED,
            SparseSettings.SPARSE_INDEX
        ));
        settings.addAll(NeuralSearchSettings.all());
        if (stockKnn == false) {
            settings.addAll(KnnSettings.settings());
        }
        settings.addAll(org.lucenerust.opensearch.engine.BpReorderSettings.settings());
        if (stockMl == false) {
            settings.addAll(MlSettings.all());
        }
        return settings;
    }

    /**
     * neural-search's {@code sparse_vector} field, which the Rust engine clusters and searches, and
     * k-NN's {@code knn_vector}, whose HNSW graphs it builds and searches.
     */
    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        Map<String, Mapper.TypeParser> mappers = new HashMap<>();
        mappers.put(SparseVectorFieldMapper.CONTENT_TYPE, new SparseVectorFieldMapper.TypeParser());
        mappers.put(MultiVectorFieldMapper.CONTENT_TYPE, new MultiVectorFieldMapper.TypeParser());
        mappers.put(org.lucenerust.opensearch.patterntext.PatternTextFieldMapper.CONTENT_TYPE, org.lucenerust.opensearch.patterntext.PatternTextFieldMapper.PARSER);
        mappers.put(SemanticFieldMapper.CONTENT_TYPE, new SemanticFieldMapper.TypeParser());
        mappers.put(HexIdFieldMapper.CONTENT_TYPE, HexIdFieldMapper.PARSER);
        if (stockKnn == false) {
            mappers.put(KnnVectorFieldMapper.CONTENT_TYPE, new KnnVectorFieldMapper.TypeParser());
        }
        return mappers;
    }

    /** neural-search's semantic field: adds the {@code semantic_info} fields its model needs to the mapping. */
    @Override
    public List<org.opensearch.index.mapper.MappingTransformer> getMappingTransformers() {
        semanticMappingTransformer = new SemanticMappingTransformer(mlClient(), xContentRegistry);
        if (modelResolver != null) {
            semanticMappingTransformer.setModelResolver(modelResolver);
        }
        return List.of(semanticMappingTransformer);
    }

    /** neural-search's system ingest processor, which writes a semantic field's embeddings. */
    @Override
    public Map<String, org.opensearch.ingest.Processor.Factory> getSystemIngestProcessors(org.opensearch.ingest.Processor.Parameters parameters) {
        return Map.of(
            SemanticFieldProcessorFactory.PROCESSOR_FACTORY_TYPE,
            new SemanticFieldProcessorFactory(
                mlClient(),
                parameters.env,
                parameters.ingestService.getClusterService(),
                parameters.analysisRegistry,
                parameters.client
            )
        );
    }

    /** k-NN's {@code knn_vector} doc-value format, which a shard's {@code docvalue_fields} results carry. */
    @Override
    public List<org.opensearch.core.common.io.stream.NamedWriteableRegistry.Entry> getNamedWriteables() {
        var hexId = new org.opensearch.core.common.io.stream.NamedWriteableRegistry.Entry(
            org.opensearch.search.DocValueFormat.class,
            HexIdDocValueFormat.NAME,
            HexIdDocValueFormat::fromStream
        );
        if (stockKnn) {
            return List.of(hexId);
        }
        return List.of(
            hexId,
            new org.opensearch.core.common.io.stream.NamedWriteableRegistry.Entry(
                org.opensearch.search.DocValueFormat.class,
                org.lucenerust.opensearch.knn.KnnVectorFieldData.Format.NAME,
                org.lucenerust.opensearch.knn.KnnVectorFieldData.Format::fromStream
            )
        );
    }

    /**
     * neural-search's {@code neural_sparse}, {@code hybrid}, and, over the plugin's k-NN,
     * {@code neural} and {@code neural_knn_query}; k-NN's {@code knn}.
     */
    @Override
    public List<QuerySpec<?>> getQueries() {
        List<QuerySpec<?>> queries = new ArrayList<>();
        queries.add(new QuerySpec<>(NeuralSparseQueryBuilder.NAME, NeuralSparseQueryBuilder::new, NeuralSparseQueryBuilder::fromXContent));
        queries.add(new QuerySpec<>(MaxSimQueryBuilder.NAME, MaxSimQueryBuilder::new, MaxSimQueryBuilder::fromXContent));
        queries.add(
            new QuerySpec<>(
                org.lucenerust.opensearch.join.NativeJoinQueryBuilder.NAME,
                org.lucenerust.opensearch.join.NativeJoinQueryBuilder::new,
                org.lucenerust.opensearch.join.NativeJoinQueryBuilder::fromXContent
            )
        );
        queries.add(HybridSearch.query());
        if (stockKnn == false) {
            queries.add(new QuerySpec<>(KnnQueryBuilder.NAME, KnnQueryBuilder::new, KnnQueryBuilder::fromXContent));
            queries.add(new QuerySpec<>(NeuralQueryBuilder.NAME, NeuralQueryBuilder::new, NeuralQueryBuilder::fromXContent));
            queries.add(new QuerySpec<>(NeuralKNNQueryBuilder.NAME, NeuralKNNQueryBuilder::new, NeuralKNNQueryBuilder::fromXContent));
        }
        return queries;
    }

    /** neural-search's model-backed ingest processors and {@code text_chunking}. */
    @Override
    public Map<String, org.opensearch.ingest.Processor.Factory> getProcessors(org.opensearch.ingest.Processor.Parameters parameters) {
        MLCommonsClientAccessor accessor = mlClient();
        org.opensearch.cluster.service.ClusterService cs = parameters.ingestService.getClusterService();
        return Map.of(
            TextEmbeddingProcessor.TYPE,
            new TextEmbeddingProcessorFactory(parameters.client, accessor, parameters.env, cs),
            SparseEncodingProcessor.TYPE,
            new SparseEncodingProcessorFactory(parameters.client, accessor, parameters.env, cs),
            TextImageEmbeddingProcessor.TYPE,
            new TextImageEmbeddingProcessorFactory(parameters.client, accessor, parameters.env, cs),
            TextChunkingProcessor.TYPE,
            new TextChunkingProcessorFactory(parameters.env, cs, parameters.analysisRegistry)
        );
    }

    /** neural-search's {@code ext.rerank}, the query context a rerank processor reads. */
    @Override
    public List<SearchExtSpec<?>> getSearchExts() {
        List<SearchExtSpec<?>> exts = new ArrayList<>();
        exts.add(new SearchExtSpec<>(RerankSearchExtBuilder.PARAM_FIELD_NAME, RerankSearchExtBuilder::new, RerankSearchExtBuilder::parse));
        if (stockKnn == false) {
            exts.add(new SearchExtSpec<>(MMRSearchExtBuilder.NAME, MMRSearchExtBuilder::new, MMRSearchExtBuilder::parse));
        }
        return exts;
    }

    /**
     * k-NN's system search processors: {@code knn_default_excludes}, which leaves knn_vector fields out
     * of a search's {@code _source} unless asked for, and MMR's oversampling request processor.
     */
    @Override
    public Map<String, org.opensearch.search.pipeline.SystemGeneratedProcessor.SystemGeneratedFactory<org.opensearch.search.pipeline.SearchRequestProcessor>> getSystemGeneratedRequestProcessors(
        Parameters parameters
    ) {
        if (stockKnn) {
            return Map.of();
        }
        Map<String, MMRQueryTransformer<? extends org.opensearch.index.query.QueryBuilder>> transformers = Map.of(
            KnnQueryBuilder.NAME,
            new MMRKnnQueryTransformer(),
            NeuralQueryBuilder.NAME,
            new MMRNeuralQueryTransformer()
        );
        return Map.of(
            KNNSourceExcludesProcessor.Factory.TYPE,
            new KNNSourceExcludesProcessor.Factory(clusterService.get(), resolver.get()),
            MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE,
            new MMROverSampleProcessor.MMROverSampleProcessorFactory(parameters.client, transformers)
        );
    }

    /** k-NN's MMR rerank response processor. */
    @Override
    public Map<String, org.opensearch.search.pipeline.SystemGeneratedProcessor.SystemGeneratedFactory<org.opensearch.search.pipeline.SearchResponseProcessor>> getSystemGeneratedResponseProcessors(
        Parameters parameters
    ) {
        return stockKnn ? Map.of() : Map.of(MMRRerankProcessor.MMRRerankProcessorFactory.TYPE, new MMRRerankProcessor.MMRRerankProcessorFactory());
    }

    /** neural-search's {@code neural_query_enricher} and {@code neural_sparse_two_phase_processor}. */
    @Override
    public Map<String, org.opensearch.search.pipeline.Processor.Factory<org.opensearch.search.pipeline.SearchRequestProcessor>> getRequestProcessors(
        Parameters parameters
    ) {
        org.lucenerust.opensearch.neural.util.NeuralSearchClusterUtil.instance().setSearchPipelineService(parameters.searchPipelineService);
        return Map.of(
            NeuralQueryEnricherProcessor.TYPE,
            new NeuralQueryEnricherProcessor.Factory(),
            NeuralSparseTwoPhaseProcessor.TYPE,
            new NeuralSparseTwoPhaseProcessor.Factory()
        );
    }

    /** The node's neural-search view of ml-commons; one per node, made on first use. */
    private synchronized MLCommonsClientAccessor mlClient() {
        if (mlClient == null) {
            mlClient = new MLCommonsClientAccessor(() -> {
                if (stockMl) {
                    throw new IllegalStateException(
                        "neural search runs models through the lucene-rust plugin's ml-commons, which stands down when opensearch-ml is installed; remove opensearch-ml to use model inference"
                    );
                }
                return MlCommons.get();
            });
        }
        return mlClient;
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
        HybridSearch.initialize(threadPool);
        NeuralQueryBuilder.initialize(mlClient());
        NeuralSparseQueryBuilder.initialize(mlClient());
        this.xContentRegistry = xContentRegistry;
        modelResolver = new DefaultSemanticModelResolver(new SemanticModelSelectionSettingsAccessor(clusterService));
        if (semanticMappingTransformer != null) {
            semanticMappingTransformer.setModelResolver(modelResolver);
        }
        org.lucenerust.opensearch.painless.PainlessScore.initialize(scriptService);
        if (org.lucenerust.opensearch.logs.LogsPreset.ENABLED.get(environment.settings())) {
            clusterService.addListener(new org.lucenerust.opensearch.logs.LogsPreset(client));
        }
        if (stockMl == false) {
            Path runtime = Path.of(libraryPath).resolveSibling(System.mapLibraryName("onnxruntime"));
            MlCommons.initialize(client, clusterService, threadPool, environment.dataFiles()[0], runtime);
        }
        return org.lucenerust.opensearch.neural.NeuralStats.initialize(clusterService, indexNameExpressionResolver, environment.settings());
    }

    @Override
    public List<org.opensearch.action.support.ActionFilter> getActionFilters() {
        return List.of(
            new RealtimePrimaryFilter(clusterService::get, resolver::get),
            HybridSearch.requestFilter(),
            new org.lucenerust.opensearch.join.JoinRequestFilter()
        );
    }

    @Override
    public List<org.opensearch.threadpool.ExecutorBuilder<?>> getExecutorBuilders(Settings settings) {
        List<org.opensearch.threadpool.ExecutorBuilder<?>> builders = new ArrayList<>();
        builders.add(HybridSearch.executorBuilder(settings));
        if (stockKnn == false) {
            builders.add(new org.opensearch.threadpool.FixedExecutorBuilder(settings, "training", 1, 1, "knn", false));
        }
        builders.add(
            new org.opensearch.threadpool.FixedExecutorBuilder(
                settings,
                org.lucenerust.opensearch.sparse.SparseShardsTransport.THREAD_POOL,
                org.lucenerust.opensearch.neural.NeuralSearchSettings.SPARSE_ALGO_PARAM_INDEX_THREAD_QTY.getDefault(Settings.EMPTY),
                -1,
                org.lucenerust.opensearch.sparse.SparseShardsTransport.THREAD_POOL,
                false
            )
        );
        if (stockMl == false) {
            builders.addAll(MlCommons.executorBuilders(settings));
        }
        return builders;
    }

    /** k-NN's concurrent search decider: {@code auto} mode searches a knn query's segments concurrently. */
    @Override
    public Optional<org.opensearch.search.deciders.ConcurrentSearchRequestDecider.Factory> getConcurrentSearchRequestDeciderFactory() {
        return stockKnn ? Optional.empty() : Optional.of(new org.lucenerust.opensearch.knn.KnnConcurrentSearchDecider.Factory());
    }

    @Override
    public List<org.opensearch.search.query.QueryCollectorContextSpecFactory> getCollectorContextSpecFactories() {
        return List.of(HybridSearch.collectorContextSpecFactory());
    }

    @Override
    public Map<String, org.opensearch.search.pipeline.Processor.Factory<org.opensearch.search.pipeline.SearchPhaseResultsProcessor>> getSearchPhaseResultsProcessors(
        Parameters parameters
    ) {
        return HybridSearch.phaseResultsProcessors();
    }

    @Override
    public Map<String, org.opensearch.search.pipeline.Processor.Factory<org.opensearch.search.pipeline.SearchResponseProcessor>> getResponseProcessors(
        Parameters parameters
    ) {
        Map<String, org.opensearch.search.pipeline.Processor.Factory<org.opensearch.search.pipeline.SearchResponseProcessor>> processors =
            new java.util.HashMap<>(HybridSearch.responseProcessors());
        processors.put(RerankProcessor.TYPE, new RerankProcessorFactory(mlClient(), parameters.searchPipelineService.getClusterService()));
        return processors;
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
        List<RestHandler> handlers = new ArrayList<>();
        handlers.add(new RestStatsAction(stats, readers, libraryPath));
        handlers.add(org.lucenerust.opensearch.neural.NeuralStats.restHandler());
        handlers.add(new org.lucenerust.opensearch.sparse.SparseShardsTransport.Rest(true));
        handlers.add(new org.lucenerust.opensearch.sparse.SparseShardsTransport.Rest(false));
        if (stockKnn == false) {
            handlers.addAll(KnnRestHandlers.all(clusterService::get, indexNameExpressionResolver));
            handlers.addAll(KnnModelHandlers.all(clusterService::get));
        }
        if (stockMl == false) {
            handlers.addAll(MlRestHandlers.all());
        }
        return handlers;
    }

    @Override
    public List<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
        List<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> actions = new ArrayList<>();
        actions.add(org.lucenerust.opensearch.neural.NeuralStats.actionHandler());
        actions.add(
            new ActionHandler<>(
                org.lucenerust.opensearch.sparse.SparseShardsTransport.WarmupAction.INSTANCE,
                org.lucenerust.opensearch.sparse.SparseShardsTransport.WarmupTransportAction.class
            )
        );
        actions.add(
            new ActionHandler<>(
                org.lucenerust.opensearch.sparse.SparseShardsTransport.ClearCacheAction.INSTANCE,
                org.lucenerust.opensearch.sparse.SparseShardsTransport.ClearCacheTransportAction.class
            )
        );
        if (stockKnn == false) {
            actions.add(new ActionHandler<>(KnnStatsTransport.Action.INSTANCE, KnnStatsTransport.TransportAction.class));
            actions.add(new ActionHandler<>(KnnShardsTransport.WarmupAction.INSTANCE, KnnShardsTransport.WarmupTransportAction.class));
            actions.add(new ActionHandler<>(KnnShardsTransport.ClearCacheAction.INSTANCE, KnnShardsTransport.ClearCacheTransportAction.class));
        }
        if (stockMl == false) {
            actions.addAll(MlActions.handlers());
        }
        return actions;
    }

    /** ml-commons' system indices, unless the stock ml-commons plugin registers them. */
    @Override
    public Collection<SystemIndexDescriptor> getSystemIndexDescriptors(Settings settings) {
        return stockMl ? List.of() : MlIndices.descriptors();
    }

    /** ml-commons' {@code ml} node role. */
    @Override
    public Set<DiscoveryNodeRole> getRoles() {
        return stockMl ? Set.of() : Set.of(MlCommons.ML_ROLE);
    }

    @Override
    public void close() {
        readers.closeAll();
        if (stockMl == false) {
            MlCommons.closeIfInitialized();
        }
    }

    /** k-NN's {@code knn} script language ({@code knn_score}), unless the stock k-NN plugin serves it. */
    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return stockKnn ? null : new KnnScoringScriptEngine();
    }
}
