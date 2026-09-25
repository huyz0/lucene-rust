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
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.store.Store;

import java.io.IOException;
import java.util.Set;
import java.util.function.LongSupplier;

/**
 * What {@link RustEngine} adds to the {@code InternalEngine} it is derived from: opening the Rust
 * writer for a shard, and refusing, at engine creation, the index configurations the Rust writer
 * cannot produce -- so an unsupported index fails to start rather than indexing something else.
 */
public final class RustEngineSupport {
    /** Serves the index's primaries (and document-replication replicas) with {@link RustEngine}. */
    public static final Setting<Boolean> ENGINE_ENABLED = Setting.boolSetting(
        "index.lucene_rust.engine",
        false,
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
        String codec = config.getIndexSettings().getValue(EngineConfig.INDEX_CODEC_SETTING);
        if (CODECS.contains(codec) == false) {
            throw new IllegalArgumentException(
                "the Rust engine writes the default codec only, and index.codec is [" + codec + "]"
            );
        }
        if (config.getIndexSort() != null) {
            throw new IllegalArgumentException("the Rust engine does not support index sorting");
        }
        if (config.getIndexSettings().isContextAwareEnabled()) {
            throw new IllegalArgumentException("the Rust engine does not support context-aware segments");
        }
    }

    static RustIndexWriter openWriter(
        EngineConfig config,
        Store store,
        IndexDeletionPolicy deletionPolicy,
        LongSupplier minRetainedSeqNo,
        String softDeletesField
    ) throws IOException {
        Codec codec = config.getCodec();
        return RustIndexWriter.open(
            store.directory(),
            store.shardPath().resolveIndex(),
            config.getIndexingBufferSize().getMbFrac(),
            config.getIndexSettings().getValue(FAULT_INJECTION),
            config.getAnalyzer(),
            config.getSimilarity(),
            config.getIndexSettings().getIndexVersionCreated().luceneVersion.major,
            softDeletesField,
            (field, schema) -> unsupportedFormat(codec, field, schema),
            deletionPolicy,
            minRetainedSeqNo,
            breaker
        );
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
