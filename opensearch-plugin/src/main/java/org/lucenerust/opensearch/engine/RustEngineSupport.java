/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldDocValuesFormat;
import org.apache.lucene.codecs.perfield.PerFieldPostingsFormat;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.IndexDeletionPolicy;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.TieredMergePolicy;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.MergeSchedulerConfig;
import org.opensearch.index.engine.DefaultPrimaryOperationPolicy;
import org.opensearch.index.engine.DocumentIndexWriter;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.store.Store;
import org.lucenerust.opensearch.NativeBridge;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;

/**
 * What {@link RustEngine} adds to the {@code InternalEngine} it is derived from: opening the Rust
 * writer for a shard, and refusing, at engine creation, the index configurations the Rust writer
 * cannot produce -- so an unsupported index fails to start rather than indexing something else.
 */
public final class RustEngineSupport {
    /**
     * What an index that does not set {@code index.lucene_rust.engine} gets, on this node -- so a
     * node can serve every index, system indices included, with the Rust engine, as the REST
     * suites' run of {@code scripts/verify-opensearch.sh --engine} does.
     */
    public static final Setting<Boolean> ENGINE_DEFAULT = Setting.boolSetting(
        "lucene_rust.engine.default",
        false,
        Setting.Property.NodeScope
    );

    /** Serves the index's primaries (and document-replication replicas) with {@link RustEngine}. */
    public static final Setting<Boolean> ENGINE_ENABLED = Setting.boolSetting(
        "index.lucene_rust.engine",
        ENGINE_DEFAULT,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * Whether searches read the documents still in the writer's buffers, without a refresh: a
     * plain scored query sees every document indexed before it started.
     */
    public static final Setting<Boolean> REALTIME_SEARCH = Setting.boolSetting(
        "index.lucene_rust.realtime_search",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * With {@link #REALTIME_SEARCH}, how old a view of the buffers a search may read: the writer
     * opens one every few milliseconds with the buffers written ahead, and with what the recent
     * requests read of them made ahead, and searches share it instead of each opening its own. A
     * search then misses at most the writes of the last this long, and {@code refresh=wait_for}
     * makes the searches after it read a fresh view. The default, 20 ms, favours search latency;
     * zero gives every search a view of every write acknowledged before it (read after write), at
     * a higher search latency while indexing.
     */
    public static final Setting<org.opensearch.common.unit.TimeValue> REALTIME_MAX_STALENESS = Setting.timeSetting(
        "index.lucene_rust.realtime_max_staleness",
        org.opensearch.common.unit.TimeValue.timeValueMillis(20),
        org.opensearch.common.unit.TimeValue.ZERO,
        org.opensearch.common.unit.TimeValue.timeValueSeconds(1),
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * Whether the Rust engine writes doc values in the {@code RustAdaptive} format: numeric and
     * date columns as adaptive blocks (delta, GCD, decimal and bit-packing per 128 values), several
     * times smaller for timestamps and decimal doubles. Off by default, which writes stock
     * Lucene90 doc values. Only nodes with this plugin read {@code RustAdaptive} segments; to go
     * back, close the index, set this to false, reopen it and force-merge, which rewrites every
     * segment stock-readable.
     */
    public static final Setting<Boolean> ADAPTIVE_DOC_VALUES = Setting.boolSetting(
        "index.lucene_rust.adaptive_doc_values",
        false,
        Setting.Property.IndexScope
    );

    /**
     * Whether the Rust engine writes terms dictionaries in the {@code RustZstd104} postings format:
     * Lucene104's postings under a block-tree dictionary whose term blocks are zstd-compressed where
     * that saves an eighth or more, a large field's against a dictionary trained on its own blocks
     * (about a third smaller for text, more for keywords, ids and log fields). Off by default, which
     * writes stock Lucene104 postings. Only nodes with this plugin read {@code RustZstd104}
     * segments; to go back, close the index, set this to false, reopen it and force-merge, which
     * rewrites every segment stock-readable.
     */
    public static final Setting<Boolean> ZSTD_TERMS = Setting.boolSetting(
        "index.lucene_rust.zstd_terms",
        false,
        Setting.Property.IndexScope
    );

    /**
     * Whether merges write {@code index.codec: default} stored fields with a high-compression LZ4
     * match finder: the same LZ4 format, which OpenSearch's own engine and stock Lucene read, about
     * 8-33% smaller, and several times slower to compress. Flushes keep the fast compressor, and a
     * segment already written this way is copied by later merges rather than compressed again. Off
     * by default, like stock.
     */
    public static final Setting<Boolean> MERGE_LZ4_HIGH_COMPRESSION = Setting.boolSetting(
        "index.lucene_rust.stored_fields.merge_lz4_high_compression",
        false,
        Setting.Property.IndexScope
    );

    /**
     * Whether the Rust engine writes its paged zstd stored fields (codec {@code
     * LuceneRustZstdPaged104}) instead of {@code index.codec}'s: 16 KiB chunks of one zstd frame each,
     * packed so no chunk crosses a 16 KiB page, which a page cache then decodes page by page. A merge
     * of more than 16 MiB of stored fields trains a dictionary for its segment. At {@code
     * index.codec.compression_level} when {@code index.codec} is a zstd codec, level 3 otherwise.
     * About as small as {@code zstd} and 3x faster to fetch a document from; only this plugin reads
     * these segments.
     */
    public static final Setting<Boolean> PAGED_ZSTD_STORED_FIELDS = Setting.boolSetting(
        "index.lucene_rust.stored_fields.paged_zstd",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * How the Rust engine reads this index's files from local disk. {@code mmap} (the default) maps
     * them, as stock OpenSearch's {@code mmapfs} and {@code hybridfs} stores do. {@code page_cache}
     * reads them through the plugin's own page cache instead: pages read on first touch with direct
     * I/O, small hot files (terms index, norms, metadata) held whole, and every shard's pages within
     * one node budget ({@code lucene_rust.store.page_cache.size}) rather than the kernel's page cache.
     * {@code page_cache_vectors} reads only the vector formats' files (HNSW and IVF) that way and maps
     * the rest. Needs Linux with {@code userfaultfd} permitted; elsewhere the index stays mapped. The
     * files on disk are the same either way.
     */
    public static final Setting<String> STORE_MODE = new Setting<>(
        "index.lucene_rust.store_mode",
        "mmap",
        RustEngineSupport::parseStoreMode,
        Setting.Property.IndexScope
    );

    static String parseStoreMode(String value) {
        if (value.equals("mmap") || value.equals("page_cache") || value.equals("page_cache_vectors")) {
            return value;
        }
        throw new IllegalArgumentException(
            "unknown value ["
                + value
                + "] for [index.lucene_rust.store_mode]: expected [mmap], [page_cache] or [page_cache_vectors]"
        );
    }

    /**
     * Binds or unbinds the shard's path to the native page cache by {@link #STORE_MODE}, before the
     * engine's writer and readers open their directories there.
     */
    static void bindStoreMode(EngineConfig config, Store store) {
        // The key the writer and the native readers look the binding up by.
        String path = org.lucenerust.opensearch.store.RustEncryptedDirectory.nativePath(indexPath(store));
        String mode = STORE_MODE.get(config.getIndexSettings().getSettings());
        int rc = mode.equals("mmap")
            ? NativeBridge.storeUnbindPageCache(path)
            : NativeBridge.storeBindPageCache(path, 0, mode.equals("page_cache_vectors"));
        if (rc != 0) {
            throw new IllegalStateException("lucene-rust: cannot set the store mode of [" + path + "] (status " + rc + ")");
        }
    }

    /**
     * The mean recall at {@code k} a {@code knn} query on this index's {@code lucene_rust} or
     * {@code lucene} engine fields should reach. A value in {@code (0, 1]} turns on recall-targeted
     * search: each shard tunes each field's search depth (an HNSW search's candidates, an IVF
     * search's probes and rerank depth, from a quarter to 64 times the asked one) to the
     * shallowest one that reaches the target, measured by checking sampled queries against an exact
     * scan of the field on a thread of their own. {@code 0} (the default) searches at the depth asked,
     * as stock does. A query that sets {@code method_parameters.ef_search} or {@code nprobes} is
     * never tuned. On GloVe-100 (390k vectors, 4 segments) a 0.95 target ran IVF 1.9x faster than its
     * calibrated default (0.968 recall where the default over-reached to 0.988) and picked an HNSW
     * candidate count of 200; every target was met on held-out queries. See {@code
     * GET /_plugins/lucene_rust/stats}, {@code knn_recall}, for what each tuner measured.
     */
    public static final Setting<Float> KNN_TARGET_RECALL = new Setting<>(
        "index.lucene_rust.knn.target_recall",
        "0",
        v -> unitFloat("index.lucene_rust.knn.target_recall", v, true),
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * The most queries a recall-targeted search ({@link #KNN_TARGET_RECALL}) checks, as a share of
     * them, in {@code (0, 1]}.
     */
    public static final Setting<Float> KNN_RECALL_SAMPLE_RATE = new Setting<>(
        "index.lucene_rust.knn.recall_sample_rate",
        "0.01",
        v -> unitFloat("index.lucene_rust.knn.recall_sample_rate", v, false),
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * The most CPU time a recall-targeted search's checks ({@link #KNN_TARGET_RECALL}) take, as a
     * share of the time its searches took, once the first 64 checks are done (those take up to as
     * long as the searches): an exact scan reads every vector of the field, so on a large shard this
     * budget, not the sample rate, sets how often one runs.
     */
    public static final Setting<Float> KNN_RECALL_CHECK_BUDGET = Setting.floatSetting(
        "index.lucene_rust.knn.recall_check_budget",
        0.05f,
        0f,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /** {@code value} as a float in {@code (0, 1]} (or {@code 0} when {@code zero} allows it). */
    static float unitFloat(String key, String value, boolean zero) {
        float f;
        try {
            f = Float.parseFloat(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Failed to parse value [" + value + "] for setting [" + key + "]", e);
        }
        if ((f > 0f && f <= 1f) || (zero && f == 0f)) {
            return f;
        }
        throw new IllegalArgumentException(
            "Failed to parse value [" + value + "] for setting [" + key + "] must be " + (zero ? "0 or " : "") + "in (0, 1]"
        );
    }

    /**
     * The least vectors a partition of a sorted index needs for its IVF clusters to be its own: at
     * flush and merge, each partition of the index sort's first field (a tenant id, say) with at
     * least this many vectors in the segment is clustered on its own, and the smaller ones are
     * clustered together, so a k-NN search filtered to one partition probes only the clusters that
     * hold it. On 8M vectors at 20000, tenants of 400k and 1.5M vectors searched about 4x and 1.5x
     * faster at the same or higher recall, while unfiltered searches took 2.3x as long. {@code
     * 0} (the default) clusters each field as a whole, like stock; no effect without an index sort, on
     * fields other than {@code method: ivf}, or on segments holding nested documents (whose blocks
     * keep whole-field clusters). Segments read like any other IVF segment.
     */
    public static final Setting<Integer> IVF_PARTITION_CLUSTERS = Setting.intSetting(
        "index.lucene_rust.knn.ivf_partition_clusters",
        0,
        0,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * The field whose value is a document's partition, as a partitioned table's partition key: every
     * segment of the shard holds the documents of one partition (a keyword's value, or a numeric's or
     * date's), and merges never mix partitions, so each partition's documents stay together on disk.
     * Documents without a value, and delete tombstones, share one default partition. Empty (the
     * default) partitions nothing. Segments record their partition the way OpenSearch's context-aware
     * segments do, so OpenSearch's own engine reads and merges them alike.
     */
    public static final Setting<String> PARTITION_FIELD = Setting.simpleString(
        "index.lucene_rust.partition_field",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * Test-only: a document with a {@code __lucene_rust_panic} field makes the writer panic inside
     * its lock, for the "a panic fails exactly one shard" check. Never set on a real index.
     */
    public static final Setting<Boolean> FAULT_INJECTION = Setting.boolSetting(
        "index.lucene_rust.engine.fault_injection",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /**
     * The {@code index.codec} values the Rust writer writes, and the stored fields each takes: the
     * default codec's LZ4, {@code best_compression}'s DEFLATE, and the custom-codecs plugin's
     * {@code zstd} and {@code zstd_no_dict}, in the plugin's own format and under its codec names, so
     * either engine opens the other's segments.
     */
    private static final Map<String, Integer> CODECS = Map.of(
        "default",
        NativeBridge.STORED_FIELDS_BEST_SPEED,
        "lucene_default",
        NativeBridge.STORED_FIELDS_BEST_SPEED,
        "best_compression",
        NativeBridge.STORED_FIELDS_BEST_COMPRESSION,
        "zstd",
        NativeBridge.STORED_FIELDS_ZSTD,
        "zstd_no_dict",
        NativeBridge.STORED_FIELDS_ZSTD_NO_DICT
    );

    /** {@code NativeLuceneIndexWriterFactory}'s {@code opensearch.shuffle_forced_merge}, read as it reads it. */
    private static final boolean SHUFFLE_FORCED_MERGE = org.opensearch.common.Booleans.parseBoolean(
        System.getProperty("opensearch.shuffle_forced_merge", Boolean.TRUE.toString())
    );

    /** Why an index with a fuzzy-set {@code _id} postings format cannot be served. */
    public static final String FUZZY_SET_REFUSED = "the Rust engine does not support index.optimize_doc_id_lookup.fuzzy_set.enabled";

    /**
     * Whether an index with these settings is served by the Rust engine: it asked for it (or for
     * realtime search, which only it serves), or it inherits the node default and none of its static settings sends it to OpenSearch's engine.
     */
    public static boolean servedByRust(org.opensearch.common.settings.Settings settings) {
        if (ENGINE_ENABLED.get(settings) == false) {
            return false;
        }
        return ENGINE_ENABLED.exists(settings)
            || REALTIME_SEARCH.get(settings)
            || (CODECS.containsKey(EngineConfig.INDEX_CODEC_SETTING.get(settings))
                && (org.opensearch.index.IndexSortConfig.INDEX_SORT_FIELD_SETTING.exists(settings) == false
                    || unsupportedSort(settings) == null));
    }

    /** The circuit breaker the plugin registers for the writers' buffers. */
    public static final String BREAKER_NAME = "lucene_rust_writer";

    private static volatile CircuitBreaker breaker;

    private RustEngineSupport() {}

    /** Set once, by the plugin, when OpenSearch hands it the breaker it registered. */
    public static void setBreaker(CircuitBreaker b) {
        breaker = b;
    }

    /** The breaker the plugin registered, or null before OpenSearch hands it over. */
    public static CircuitBreaker breaker() {
        return breaker;
    }

    /** Throws when the Rust writer cannot produce what this index is configured for. */
    static void checkSupported(EngineConfig config) {
        String reason = unsupported(config.getIndexSettings(), config.getIndexSort() != null);
        if (reason != null) {
            throw new IllegalArgumentException(reason);
        }
        reason = unsupportedPolicy(config);
        if (reason != null) {
            throw new IllegalArgumentException(reason);
        }
        // A segment-replication primary needs RustIndexerFactory in its shard (see there).
        if (config.getIndexSettings().isSegRepLocalEnabled() && RustIndexerFactory.building() == false) {
            throw new IllegalArgumentException(
                "the Rust engine for "
                    + config.getShardId()
                    + " was not built through RustIndexerFactory, so it cannot be a segment-replication primary; "
                    + "use index.replication.type: DOCUMENT"
            );
        }
    }

    /**
     * Why the Rust engine cannot run under this engine config's {@code PrimaryOperationPolicy}, or
     * {@code null} when it can. A policy other than the default one comes from an engine plugin (one
     * per index) and plans primary operations with OpenSearch's own strategy classes, whose
     * factories the engine's copies cannot reach, so the derived engine plans as primary directly.
     */
    static String unsupportedPolicy(EngineConfig config) {
        if (config.getPrimaryOperationPolicy() instanceof DefaultPrimaryOperationPolicy) {
            return null;
        }
        return "the Rust engine supports the default primary operation policy only, and "
            + config.getShardId()
            + " has ["
            + config.getPrimaryOperationPolicy()
            + "]";
    }

    /** The {@code NativeBridge.STORED_FIELDS_*} the Rust writer writes for {@code settings}. */
    static int storedFields(IndexSettings settings) {
        if (PAGED_ZSTD_STORED_FIELDS.get(settings.getSettings())) {
            return NativeBridge.STORED_FIELDS_ZSTD_PAGED;
        }
        return CODECS.getOrDefault(settings.getValue(EngineConfig.INDEX_CODEC_SETTING), NativeBridge.STORED_FIELDS_BEST_SPEED);
    }

    /**
     * The zstd level: {@code index.codec.compression_level}, which OpenSearch only accepts with a zstd
     * {@code index.codec}; paged zstd under any other codec takes zstd's default, 3.
     */
    static int compressionLevel(IndexSettings settings) {
        int stored = CODECS.getOrDefault(settings.getValue(EngineConfig.INDEX_CODEC_SETTING), NativeBridge.STORED_FIELDS_BEST_SPEED);
        if (PAGED_ZSTD_STORED_FIELDS.get(settings.getSettings())
            && stored != NativeBridge.STORED_FIELDS_ZSTD
            && stored != NativeBridge.STORED_FIELDS_ZSTD_NO_DICT) {
            return 3;
        }
        return settings.getValue(EngineConfig.INDEX_CODEC_COMPRESSION_LEVEL_SETTING);
    }

    /**
     * Why the Rust writer cannot serve an index configured like this, or {@code null} when it can.
     * An index that asks for the Rust engine gets this as its creation error; one that only
     * inherits {@link #ENGINE_DEFAULT} is served by OpenSearch's own engine instead.
     */
    public static String unsupported(IndexSettings settings, boolean indexSorted) {
        String codec = settings.getValue(EngineConfig.INDEX_CODEC_SETTING);
        if (CODECS.containsKey(codec) == false) {
            return "the Rust engine writes the codecs " + new java.util.TreeSet<>(CODECS.keySet()) + " only, and index.codec is [" + codec + "]";
        }
        if ((CODECS.get(codec) != NativeBridge.STORED_FIELDS_BEST_SPEED || PAGED_ZSTD_STORED_FIELDS.get(settings.getSettings()))
            && org.lucenerust.opensearch.sparse.SparseSettings.SPARSE_INDEX.get(settings.getSettings())) {
            if (PAGED_ZSTD_STORED_FIELDS.get(settings.getSettings())) {
                return "the Rust engine writes an index.sparse index with the default stored fields only, and "
                    + PAGED_ZSTD_STORED_FIELDS.getKey()
                    + " is set";
            }
            return "the Rust engine writes an index.sparse index with the default codec only, and index.codec is [" + codec + "]";
        }
        if (indexSorted && unsupportedSort(settings.getSettings()) != null) {
            return unsupportedSort(settings.getSettings());
        }
        // Partitions are the index sort's first field; the setting is final, so a missing sort is
        // refused rather than left to do nothing.
        if (indexSorted == false && IVF_PARTITION_CLUSTERS.get(settings.getSettings()) > 0) {
            return IVF_PARTITION_CLUSTERS.getKey() + " needs index.sort.field: partitions are the index sort's first field";
        }
        // The codec then gives _id a FuzzyFilterPostingsFormat (a bloom filter beside the terms),
        // which the Rust writer does not lay out.
        if (settings.isEnableFuzzySetForDocId()) {
            return FUZZY_SET_REFUSED;
        }
        if (settings.isContextAwareEnabled()) {
            return "the Rust engine does not support context-aware segments";
        }
        // Remote-backed storage uploads segments through RemoteStoreRefreshListener, which asks for an
        // InternalEngine by class; segment replication between nodes is served through
        // RustIndexerFactory.
        if (settings.isRemoteStoreEnabled() || settings.isAssignedOnRemoteNode()) {
            return "the Rust engine does not support remote-backed storage";
        }
        // The engine writes the shard's files itself: a store type that changes the bytes on disk
        // (another plugin's encryption, say) would be bypassed. `rust_encrypted` is the encrypted one
        // it writes through.
        String storeType = settings.getValue(org.opensearch.index.IndexModule.INDEX_STORE_TYPE_SETTING);
        if (org.lucenerust.opensearch.store.NativeStores.STORE_TYPES.contains(storeType) == false) {
            return "the Rust engine cannot write through index.store.type [" + storeType + "]; use [rust_encrypted] for encryption at rest";
        }
        return null;
    }

    private static volatile boolean refuseKnnVector;

    /**
     * The stock k-NN plugin serves {@code knn_vector} on this node: its fields (its codec's formats)
     * are left to OpenSearch's engine.
     */
    public static void refuseKnnVectorFields() {
        refuseKnnVector = true;
    }

    /**
     * The first field of {@code mapping} (a mapping's source, as parsed) whose documents the Rust
     * writer would refuse -- completion fields (their postings format), term vectors and, beside the
     * stock k-NN plugin, its {@code knn_vector} fields, in its properties or dynamic templates -- or
     * {@code null}. An index that only inherits {@link #ENGINE_DEFAULT} and is created
     * with such a mapping is served by OpenSearch's engine; one that asked for the Rust engine
     * refuses those documents one by one.
     */
    public static String unsupportedField(Map<String, Object> mapping) {
        if (mapping.get("dynamic_templates") instanceof List<?> templates) {
            for (Object t : templates) {
                if (t instanceof Map<?, ?> named) {
                    for (Map.Entry<?, ?> e : named.entrySet()) {
                        if (e.getValue() instanceof Map<?, ?> template && template.get("mapping") instanceof Map<?, ?> m) {
                            @SuppressWarnings("unchecked")
                            String found = unsupportedField(Map.of("properties", Map.of("*", (Map<String, Object>) m)));
                            if (found != null) {
                                return "dynamic template " + e.getKey() + ": " + found;
                            }
                        }
                    }
                }
            }
        }
        Object properties = mapping.get("properties");
        if (properties instanceof Map<?, ?> props) {
            for (Map.Entry<?, ?> e : props.entrySet()) {
                if (e.getValue() instanceof Map<?, ?> field) {
                    Object type = field.get("type");
                    if ("completion".equals(type) || (refuseKnnVector && "knn_vector".equals(type))) {
                        return e.getKey() + " (" + type + ")";
                    }
                    Object tv = field.get("term_vector");
                    if (tv != null && "no".equals(tv) == false) {
                        return e.getKey() + " (term_vector)";
                    }
                    @SuppressWarnings("unchecked")
                    String nested = unsupportedField((Map<String, Object>) field);
                    if (nested != null) {
                        return e.getKey() + "." + nested;
                    }
                    if (field.get("fields") instanceof Map<?, ?> multi) {
                        @SuppressWarnings("unchecked")
                        String sub = unsupportedField(Map.of("properties", (Map<String, Object>) multi));
                        if (sub != null) {
                            return e.getKey() + "." + sub;
                        }
                    }
                }
            }
        }
        return null;
    }

    /**
     * Whether the index is a k-NN one ({@code index.knn}) whose {@code knn_vector} fields this node
     * serves: its segments are written under the k-NN codec's name, which reads their derived source.
     */
    static boolean knnIndex(EngineConfig config) {
        return refuseKnnVector == false
            && org.lucenerust.opensearch.knn.KnnSettings.KNN_INDEX.get(config.getIndexSettings().getSettings());
    }

    /** {@link #PARTITION_FIELD}, or null when it names none. */
    static String partitionField(org.opensearch.common.settings.Settings settings) {
        String field = PARTITION_FIELD.get(settings);
        return field.isEmpty() ? null : field;
    }

    /**
     * Why the Rust writer cannot sort an index with these settings, or {@code null} when it can.
     * Before 3.2.0 OpenSearch sets no parent field under an index sort (see {@link #parentField}),
     * which the writer needs to keep nested blocks whole; a search of the buffers reads documents
     * in arrival order, which a sorted flush renumbers.
     */
    static String unsupportedSort(org.opensearch.common.settings.Settings settings) {
        if (org.opensearch.cluster.metadata.IndexMetadata.SETTING_INDEX_VERSION_CREATED.get(settings)
            .before(org.opensearch.Version.V_3_2_0)) {
            return "the Rust engine supports index sorting on indices created on 3.2.0 or later";
        }
        if (REALTIME_SEARCH.get(settings)) {
            return "index.lucene_rust.realtime_search does not support index sorting";
        }
        return null;
    }

    /**
     * The parent field {@code NativeLuceneIndexWriterFactory} sets under an index sort, which marks
     * each nested block's last document ({@link #unsupportedSort} refuses the indices it does not
     * set it on).
     */
    static String parentField(EngineConfig config) {
        return config.getIndexSort() != null ? org.opensearch.common.lucene.Lucene.PARENT_FIELD : null;
    }

    static RustIndexWriter openWriter(
        EngineConfig config,
        Store store,
        IndexDeletionPolicy deletionPolicy,
        LongSupplier minRetainedSeqNo,
        String softDeletesField,
        int maxDocs
    ) throws IOException {
        Codec codec = config.getCodec();
        bindStoreMode(config, store);
        RustIndexWriter writer = RustIndexWriter.open(
            store.directory(),
            indexPath(store),
            config.getIndexingBufferSize().getMbFrac(),
            FAULT_INJECTION.get(config.getIndexSettings().getSettings()),
            org.lucenerust.opensearch.sparse.SparseSettings.SPARSE_INDEX.get(config.getIndexSettings().getSettings()),
            REALTIME_SEARCH.get(config.getIndexSettings().getSettings()),
            ADAPTIVE_DOC_VALUES.get(config.getIndexSettings().getSettings()),
            ZSTD_TERMS.get(config.getIndexSettings().getSettings()),
            MERGE_LZ4_HIGH_COMPRESSION.get(config.getIndexSettings().getSettings()),
            partitionField(config.getIndexSettings().getSettings()),
            knnIndex(config),
            config.getIndexSort(),
            parentField(config),
            maxDocs,
            storedFields(config.getIndexSettings()),
            compressionLevel(config.getIndexSettings()),
            config.getAnalyzer(),
            config.getSimilarity(),
            config.getIndexSettings().getIndexVersionCreated().luceneVersion.major,
            softDeletesField,
            (field, schema) -> unsupportedFormat(codec, field, schema),
            deletionPolicy,
            minRetainedSeqNo,
            breaker,
            codec
        );
        // NativeLuceneIndexWriterFactory's merge-on-commit/refresh settings.
        IndexSettings indexSettings = config.getIndexSettings();
        long wait = indexSettings.isMergeOnFlushEnabled() ? indexSettings.getMaxFullFlushMergeWaitTime().millis() : 0;
        boolean success = false;
        try {
            writer.setLiveStaleness(REALTIME_MAX_STALENESS.get(indexSettings.getSettings()));
            writer.setSeqNoLayout(
                SeqNoSettings.docValuesOnly(indexSettings.getSettings()),
                SeqNoSettings.sentinels(indexSettings.getSettings())
            );
            refreshMergePolicy(writer, config);
            int ivfPartitionClusters = IVF_PARTITION_CLUSTERS.get(indexSettings.getSettings());
            if (ivfPartitionClusters > 0) {
                writer.setIvfPartitionClusters(ivfPartitionClusters);
            }
            writer.setMergeOnFlush(Math.max(wait, 0), wait > 0 && indexSettings.getMergeOnFlushPolicy().isPresent());
            refreshMergeScheduler(writer, config);
            // NativeLuceneIndexWriterFactory: merged segments are warmed (pre-copied to replicas) on
            // segment-replication indices.
            if (indexSettings.isDocumentReplication() == false
                && (indexSettings.isSegRepLocalEnabled() || indexSettings.isRemoteStoreEnabled())
                && config.getIndexReaderWarmer() != null) {
                writer.setMergedSegmentWarmer(config.getIndexReaderWarmer());
            }
            success = true;
        } finally {
            if (success == false) {
                writer.close();
            }
        }
        return writer;
    }

    /**
     * {@code OpenSearchConcurrentMergeScheduler.refreshConfig} for the Rust writer's merge scheduler:
     * the shard's current {@code index.merge.scheduler.*} settings.
     */
    static void refreshMergeScheduler(DocumentIndexWriter writer, EngineConfig config) {
        refreshMergePolicy(writer, config);
        if (writer instanceof RustIndexWriter rust) {
            IndexSettings indexSettings = config.getIndexSettings();
            MergeSchedulerConfig merges = indexSettings.getMergeSchedulerConfig();
            merges.updateMaxForceMergeMBPerSec(indexSettings);
            try {
                rust.setMergeScheduler(
                    merges.getMaxMergeCount(),
                    merges.getMaxThreadCount(),
                    merges.isAutoThrottle(),
                    merges.getMaxForceMergeMBPerSec()
                );
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }

    /**
     * The shard's merge policy for the Rust writer, under its wrappers: the {@code TieredMergePolicy}
     * or {@code LogByteSizeMergePolicy} OpenSearch's provider keeps (and updates in place when the
     * settings change), or {@code NoMergePolicy} when merges are off.
     */
    static void refreshMergePolicy(DocumentIndexWriter writer, EngineConfig config) {
        if (writer instanceof RustIndexWriter rust) {
            MergePolicy policy = config.getMergePolicy();
            while (policy instanceof FilterMergePolicy filter) {
                policy = filter.unwrap();
            }
            try {
                if (policy instanceof TieredMergePolicy tiered) {
                    rust.setMergePolicy(tiered);
                    rust.setPluggableMergePolicy(null);
                } else {
                    // log_byte_size, or index.merge.enabled: false. No OpenSearch setting yields any
                    // other policy; one a test passes in directly merges with the tiered defaults.
                    rust.setPluggableMergePolicy(policy);
                }
                // NativeLuceneIndexWriterFactory's setUseCompoundFile and setCheckPendingFlushUpdate,
                // the policy's compound-file ratio and cap, and its ShuffleForcedMergePolicy wrapper.
                rust.setOptions(
                    config.useCompoundFile(),
                    policy.getNoCFSRatio(),
                    policy.getMaxCFSSegmentSizeMB(),
                    config.getIndexSettings().isCheckPendingFlushEnabled(),
                    SHUFFLE_FORCED_MERGE
                );
                rust.setBpReorder(bpReorder(config.getIndexSettings()));
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }

    /**
     * The shard's {@code index.lucene_rust.bp_reorder.*}. A sparse index has its own segment layout,
     * which BP reordering does not support, so it is never reordered.
     */
    static BpReorderSettings.Values bpReorder(IndexSettings indexSettings) {
        BpReorderSettings.Values values = BpReorderSettings.Values.of(indexSettings.getSettings());
        if (values.enabled() && org.lucenerust.opensearch.sparse.SparseSettings.SPARSE_INDEX.get(indexSettings.getSettings())) {
            return BpReorderSettings.Values.of(org.opensearch.common.settings.Settings.EMPTY);
        }
        return values;
    }

    /**
     * Where the store's files are: the Rust writer reads and writes them itself, so the store's
     * directory must be a filesystem one underneath its wrappers.
     */
    static java.nio.file.Path indexPath(Store store) {
        return org.lucenerust.opensearch.store.NativeStores.path(store.directory());
    }

    /**
     * The per-field formats OpenSearch's codec picks for a field (a {@code completion} field gets its
     * own postings format, for instance) must be the ones the Rust writer lays out.
     */
    private static String unsupportedFormat(Codec codec, String field, FieldSchema schema) {
        if (codec instanceof org.lucenerust.opensearch.sparse.SparseCodec sparse) {
            codec = sparse.delegate();
        }
        if (schema.indexOptions != IndexOptions.NONE && codec.postingsFormat() instanceof PerFieldPostingsFormat p) {
            String name = p.getPostingsFormatForField(field).getName();
            if (name.equals("Lucene104") == false) {
                return "field [" + field + "] uses postings format [" + name + "], which the Rust engine does not write";
            }
        }
        if (schema.docValuesType != DocValuesType.NONE && codec.docValuesFormat() instanceof PerFieldDocValuesFormat d) {
            String name = d.getDocValuesFormatForField(field).getName();
            if (name.equals("Lucene90") == false) {
                return "field [" + field + "] uses doc values format [" + name + "], which the Rust engine does not write";
            }
        }
        return null;
    }
}
