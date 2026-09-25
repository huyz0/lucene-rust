/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.perfield.PerFieldDocValuesFormat;
import org.apache.lucene.codecs.perfield.PerFieldPostingsFormat;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexDeletionPolicy;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.store.Store;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
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
     * Test-only: a document with a {@code __lucene_rust_panic} field makes the writer panic inside
     * its lock, for the "a panic fails exactly one shard" check. Never set on a real index.
     */
    public static final Setting<Boolean> FAULT_INJECTION = Setting.boolSetting(
        "index.lucene_rust.engine.fault_injection",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    /** The codecs whose formats are the ones the Rust writer lays out. */
    private static final Set<String> CODECS = Set.of("default", "lucene_default");

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
    }

    /**
     * Why the Rust writer cannot serve an index configured like this, or {@code null} when it can.
     * An index that asks for the Rust engine gets this as its creation error; one that only
     * inherits {@link #ENGINE_DEFAULT} is served by OpenSearch's own engine instead.
     */
    public static String unsupported(IndexSettings settings, boolean indexSorted) {
        String codec = settings.getValue(EngineConfig.INDEX_CODEC_SETTING);
        if (CODECS.contains(codec) == false) {
            return "the Rust engine writes the default codec only, and index.codec is [" + codec + "]";
        }
        if (indexSorted) {
            return "the Rust engine does not support index sorting";
        }
        if (settings.isContextAwareEnabled()) {
            return "the Rust engine does not support context-aware segments";
        }
        // OpenSearch 3.8's segment-replication source (CopyState) reads the primary's last refreshed
        // checkpoint through EngineBackedIndexer, which answers only for an InternalEngine -- and
        // InternalEngine.lastRefreshedCheckpoint() is final. A plugin engine cannot be a segment
        // replication primary on this version; document replication is supported. (A segment
        // replication replica never reaches this: it runs NRTReplicationEngine.)
        if (settings.isSegRepEnabledOrRemoteNode()) {
            return "the Rust engine cannot be a segment-replication primary on OpenSearch 3.8 "
                + "(EngineBackedIndexer.lastRefreshedCheckpoint answers only for InternalEngine); "
                + "use index.replication.type: DOCUMENT";
        }
        return null;
    }

    /**
     * The first field of {@code mapping} (a mapping's source, as parsed) whose documents the Rust
     * writer would refuse -- completion fields (their postings format), term vectors, vector
     * fields -- or {@code null}. An index that only inherits {@link #ENGINE_DEFAULT} and is created
     * with such a mapping is served by OpenSearch's engine; one that asked for the Rust engine
     * refuses those documents one by one.
     */
    public static String unsupportedField(Map<String, Object> mapping) {
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

    static RustIndexWriter openWriter(
        EngineConfig config,
        Store store,
        IndexDeletionPolicy deletionPolicy,
        LongSupplier minRetainedSeqNo,
        String softDeletesField,
        int maxDocs
    ) throws IOException {
        Codec codec = config.getCodec();
        return RustIndexWriter.open(
            store.directory(),
            indexPath(store),
            config.getIndexingBufferSize().getMbFrac(),
            FAULT_INJECTION.get(config.getIndexSettings().getSettings()),
            maxDocs,
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
    }

    /**
     * Where the store's files are: the Rust writer reads and writes them itself, so the store's
     * directory must be a filesystem one underneath its wrappers.
     */
    static java.nio.file.Path indexPath(Store store) {
        org.apache.lucene.store.Directory d = org.apache.lucene.store.FilterDirectory.unwrap(store.directory());
        if (d instanceof org.apache.lucene.store.FSDirectory fs) {
            return fs.getDirectory();
        }
        throw new IllegalArgumentException("the Rust engine needs a filesystem directory, and the store's is a [" + d.getClass().getName() + "]");
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
