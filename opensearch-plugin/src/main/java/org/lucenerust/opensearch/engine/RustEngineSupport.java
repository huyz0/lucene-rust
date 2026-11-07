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
     * Whether an index with these settings is served by the Rust engine: it asked for it, or it
     * inherits the node default and none of its static settings sends it to OpenSearch's engine.
     */
    public static boolean servedByRust(org.opensearch.common.settings.Settings settings) {
        if (ENGINE_ENABLED.get(settings) == false) {
            return false;
        }
        return ENGINE_ENABLED.exists(settings)
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

    /**
     * The first field of {@code mapping} (a mapping's source, as parsed) whose documents the Rust
     * writer would refuse -- completion fields (their postings format), term vectors, vector
     * fields, in its properties or dynamic templates -- or {@code null}. An index that only inherits {@link #ENGINE_DEFAULT} and is created
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
                    if ("completion".equals(type) || "knn_vector".equals(type)) {
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
        RustIndexWriter writer = RustIndexWriter.open(
            store.directory(),
            indexPath(store),
            config.getIndexingBufferSize().getMbFrac(),
            FAULT_INJECTION.get(config.getIndexSettings().getSettings()),
            org.lucenerust.opensearch.sparse.SparseSettings.SPARSE_INDEX.get(config.getIndexSettings().getSettings()),
            REALTIME_SEARCH.get(config.getIndexSettings().getSettings()),
            ADAPTIVE_DOC_VALUES.get(config.getIndexSettings().getSettings()),
            MERGE_LZ4_HIGH_COMPRESSION.get(config.getIndexSettings().getSettings()),
            partitionField(config.getIndexSettings().getSettings()),
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
            refreshMergePolicy(writer, config);
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
