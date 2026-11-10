/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The native surface of {@code liblucene_ffi}, reached through the Foreign Function & Memory API:
 * each method is a downcall to one of the library's C-ABI functions ({@code jvm_reader.rs}, {@code
 * engine_writer.rs}, and {@code ffm_bridge.rs} for what those do not offer).
 *
 * <p>Every call returns a status code ({@link #OK} on success) and never throws: results come
 * back through caller-allocated arrays, and a failure's message through {@link #lastError()}, which
 * is thread-local on the Rust side and so must be read on the thread that saw the failure. A Rust
 * panic is caught before it reaches the JVM and reported as {@link #PANIC}. An argument this class
 * refuses before calling native code (a null array, a short one) reports through the same slot.
 *
 * <p>Arguments and results cross in native memory: each call copies its arrays into a per-thread
 * scratch buffer and the results back out. The heap arrays are never handed to native code
 * directly, which only a {@code critical} downcall allows -- and a critical downcall holds off
 * every safepoint in the JVM, garbage collection included, for as long as it runs, which for a
 * search is not a bounded time.
 */
public final class NativeBridge {
    /** The contract version this jar was built against; {@code JVM_ABI_VERSION} in {@code jvm_reader.rs}. */
    public static final int EXPECTED_ABI_VERSION = 108;

    public static final int OK = 0;
    public static final int INVALID_HANDLE = 3;
    public static final int IO = 4;
    public static final int DECODE = 5;
    public static final int BUFFER_TOO_SMALL = 8;
    public static final int PANIC = 9;
    public static final int INVALID_ARGUMENT = 10;

    /** Query blob tags ({@code decode_query} in {@code jvm_reader.rs}). */
    public static final byte QUERY_TERM = 0;
    public static final byte QUERY_BOOLEAN = 1;
    public static final byte QUERY_TREE = 2;

    private NativeBridge() {}

    // ---- Downcall handles -------------------------------------------------------------

    /** A downcall handle for {@code symbol} in the library {@link NativeLibrary#load} opened. */
    @SuppressWarnings("restricted") // a downcall is the whole point; NativeLibrary documents the flag
    private static MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = NativeLibrary.lookup()
            .find(symbol)
            .orElseThrow(() -> new UnsatisfiedLinkError("liblucene_ffi has no symbol [" + symbol + "]"));
        return Linker.nativeLinker().downcallHandle(address, descriptor);
    }

    /**
     * The handshake, bound apart from everything else: a library from another build may lack the
     * rest of the symbols, and must still get as far as saying which ABI it speaks.
     */
    private static final class Abi {
        static final MethodHandle VERSION = bind("ffi_jvm_abi_version", FunctionDescriptor.of(JAVA_INT));
    }

    /**
     * The downcall handles, bound on first use -- after {@link NativeLibrary#load} has opened the
     * library and checked its ABI version. A missing symbol is an {@link UnsatisfiedLinkError}.
     * Held in {@code static final} fields so the JIT sees each as a constant and compiles the call
     * site into a direct native call.
     */
    private static final class H {
        static final MethodHandle LAST_ERROR = bind("ffi_jvm_last_error", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        static final MethodHandle SET_LAST_ERROR = bind("ffi_jvm_set_last_error", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        static final MethodHandle FREE_BYTES = bind("ffi_jvm_free_bytes", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
        static final MethodHandle OPEN_READER = bind(
            "ffi_open_jvm_reader",
            FunctionDescriptor.of(
                JAVA_INT,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle SEARCH = bind(
            "ffi_jvm_reader_search",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle READER_KNN = bind(
            "ffi_jvm_reader_knn",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_KNN = bind(
            "ffi_live_view_knn",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle PARENT_CACHE_STATS = bind("ffi_jvm_parent_cache_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle DOC_FREQ = bind(
            "ffi_jvm_reader_doc_freq",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle SEARCH_SORTED = bind(
            "ffi_jvm_reader_search_sorted_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle AGGREGATE = bind(
            "ffi_jvm_reader_aggregate_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle AGGREGATE_TREE = bind(
            "ffi_jvm_reader_aggregate_tree_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_HYBRID = bind(
            "ffi_jvm_reader_search_hybrid_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_COLLAPSED = bind(
            "ffi_jvm_reader_search_collapsed_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle RESCORE = bind(
            "ffi_jvm_reader_rescore",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_INT,
                JAVA_FLOAT,
                JAVA_FLOAT,
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                JAVA_LONG
            )
        );
        static final MethodHandle DOCUMENT = bind(
            "ffi_jvm_reader_document_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle DOC_VALUES = bind(
            "ffi_jvm_reader_doc_values_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle DERIVED_SOURCE = bind(
            "ffi_jvm_reader_derived_source_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle PREFETCH_DOCUMENTS = bind(
            "ffi_jvm_reader_prefetch_documents",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle COUNT_TERMINATES = bind(
            "ffi_jvm_reader_count_terminates",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle CLOSE_READER = bind("ffi_close_jvm_reader", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ADD_DATA_PACK_DIR = bind(
            "ffi_add_data_pack_dir",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle STORE_SET_KEY_UPCALL = bind("ffi_store_set_key_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle STORE_BIND_ENCRYPTED = bind(
            "ffi_store_bind_encrypted",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT)
        );
        static final MethodHandle STORE_UNBIND = bind("ffi_store_unbind", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        static final MethodHandle STORE_IS_FIPS = bind("ffi_store_is_fips", FunctionDescriptor.of(JAVA_INT));
        static final MethodHandle STORE_SET_LAZY_DECRYPT = bind(
            "ffi_store_set_lazy_decrypt",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle STORE_LAZY_DECRYPT_ACTIVE = bind("ffi_store_lazy_decrypt_active", FunctionDescriptor.of(JAVA_INT));
        static final MethodHandle STORE_BIND_PAGE_CACHE = bind(
            "ffi_store_bind_page_cache",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT)
        );
        static final MethodHandle STORE_UNBIND_PAGE_CACHE = bind(
            "ffi_store_unbind_page_cache",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle STORE_PAGE_CACHE_AVAILABLE = bind("ffi_store_page_cache_available", FunctionDescriptor.of(JAVA_INT));
        static final MethodHandle STORE_SET_PAGE_CACHE_BUDGET = bind(
            "ffi_store_set_page_cache_budget",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle STORE_PAGE_CACHE_RESIDENT = bind("ffi_store_page_cache_resident_bytes", FunctionDescriptor.of(JAVA_LONG));
        static final MethodHandle ANALYZE = bind(
            "ffi_jvm_analyze_alloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );

        static final MethodHandle WRITER_OPEN = bind(
            "ffi_engine_writer_open_codec",
            FunctionDescriptor.of(
                JAVA_INT, ADDRESS, JAVA_LONG, JAVA_DOUBLE, JAVA_BYTE, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_BYTE,
                JAVA_INT, ADDRESS
            )
        );
        static final MethodHandle WRITER_REGISTER_FIELD = bind(
            "ffi_engine_writer_register_field",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_APPLY = bind(
            "ffi_engine_writer_apply",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_PREPARE = bind(
            "ffi_engine_prepare",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_APPLY_PREPARED = bind(
            "ffi_engine_writer_apply_prepared",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle PREPARED_FREE = bind("ffi_engine_prepared_free", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ANALYSIS_REGISTER = bind(
            "ffi_jvm_analysis_register",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_COMMIT = bind(
            "ffi_engine_writer_commit",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle SET_COMMIT_DATA_UPCALL = bind("ffi_engine_set_commit_data_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle WRITER_SET_MERGE_SCHEDULER = bind(
            "ffi_engine_writer_set_merge_scheduler",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_BYTE, JAVA_DOUBLE)
        );
        static final MethodHandle WRITER_SET_MERGE_POLICY = bind(
            "ffi_engine_writer_set_merge_policy",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_DOUBLE, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE)
        );
        static final MethodHandle WRITER_SET_OPTIONS = bind(
            "ffi_engine_writer_set_options",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE, JAVA_DOUBLE, JAVA_LONG, JAVA_BYTE, JAVA_BYTE)
        );
        static final MethodHandle WRITER_COMMIT_GENERATIONS = bind(
            "ffi_engine_writer_commit_generations",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_DELETE_COMMITS = bind(
            "ffi_engine_writer_delete_commits",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_HOLD_COMMIT = bind(
            "ffi_engine_writer_hold_commit",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_NRT_SNAPSHOT = bind(
            "ffi_engine_writer_nrt_snapshot",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, ADDRESS)
        );
        static final MethodHandle WRITER_RELEASE_HOLD = bind(
            "ffi_engine_writer_release_hold",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_RETENTION = bind(
            "ffi_engine_writer_set_retention",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_MERGE_ON_FLUSH = bind(
            "ffi_engine_writer_set_merge_on_flush",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_BYTE)
        );
        static final MethodHandle WRITER_SET_IVF_PARTITION_CLUSTERS = bind(
            "ffi_engine_writer_set_ivf_partition_clusters",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_BP_REORDER = bind(
            "ffi_engine_writer_set_bp_reorder",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                JAVA_BYTE,
                JAVA_INT,
                JAVA_FLOAT,
                JAVA_INT,
                JAVA_INT,
                JAVA_DOUBLE,
                JAVA_INT,
                JAVA_FLOAT,
                ADDRESS,
                JAVA_LONG
            )
        );
        static final MethodHandle WRITER_FORCE_MERGE = bind(
            "ffi_engine_writer_force_merge",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle WRITER_SET_PLUGGABLE_MERGE_POLICY = bind(
            "ffi_engine_writer_set_pluggable_merge_policy",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_DOUBLE, JAVA_LONG)
        );
        static final MethodHandle SET_WARM_UPCALL = bind("ffi_engine_set_warm_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle WRITER_SET_MERGED_SEGMENT_WARMER = bind(
            "ffi_engine_writer_set_merged_segment_warmer",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE)
        );
        static final MethodHandle SET_MERGE_UPCALL = bind("ffi_engine_set_merge_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle WRITER_MERGE_STATE = bind(
            "ffi_engine_writer_merge_state",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle WRITER_STATS = bind(
            "ffi_engine_writer_stats",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_CLOSE = bind("ffi_engine_writer_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ZSTD_DICTIONARY_OPEN = bind(
            "ffi_zstd_dictionary_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ZSTD_DICTIONARY_CLOSE = bind("ffi_zstd_dictionary_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ML_RUNTIME_INIT = bind("ffi_ml_runtime_init", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT));
        static final MethodHandle ML_TEXT_EMBEDDING_OPEN = bind(
            "ffi_ml_text_embedding_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle ML_TEXT_EMBEDDING_EMBED = bind(
            "ffi_ml_text_embedding_embed_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_TOKENIZE_OPEN = bind(
            "ffi_ml_sparse_tokenize_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_TOKENIZE = bind(
            "ffi_ml_sparse_tokenize_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_ENCODING_OPEN = bind(
            "ffi_ml_sparse_encoding_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_SPARSE_ENCODING = bind(
            "ffi_ml_sparse_encoding_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_TEXT_SIMILARITY_OPEN = bind(
            "ffi_ml_text_similarity_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ML_TEXT_SIMILARITY = bind(
            "ffi_ml_text_similarity_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ML_MODEL_CLOSE = bind("ffi_ml_model_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ML_TORCH_INIT = bind(
            "ffi_ml_torch_init",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT)
        );
        static final MethodHandle ML_TORCH_OPEN = bind(
            "ffi_ml_torch_open",
            FunctionDescriptor.of(JAVA_INT, JAVA_BYTE, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle ML_TORCH_TEXT_EMBEDDING = bind(
            "ffi_ml_torch_text_embedding_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle ZSTD_PAGED_DECOMPRESS = bind(
            "ffi_zstd_paged_decompress",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle LIVE_VIEW_OPEN = bind(
            "ffi_engine_writer_live_view_open",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_CLOSE = bind("ffi_live_view_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle LIVE_VIEW_READER = bind(
            "ffi_live_view_reader",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_LUCENE = bind(
            "ffi_live_view_lucene",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle WRITER_SET_LIVE_STALENESS = bind(
            "ffi_engine_writer_set_live_staleness",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_LIVE_REFRESH_SHARED = bind(
            "ffi_engine_writer_live_refresh_shared",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle WRITER_RELEASE_LIVE = bind(
            "ffi_engine_writer_release_live",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle LIVE_DOC_VALUES = bind(
            "ffi_live_view_doc_values_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_DOCUMENT = bind(
            "ffi_live_view_document_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_POINT_BOUNDS = bind(
            "ffi_live_view_point_bounds_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_LIVE = bind(
            "ffi_jvm_reader_search_live",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle ZSTD_TERMS_DICTIONARY_OPEN = bind(
            "ffi_zstd_terms_dictionary_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ZSTD_TERMS_DICTIONARY_CLOSE = bind(
            "ffi_zstd_terms_dictionary_close",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle ZSTD_TERMS_DECOMPRESS = bind(
            "ffi_zstd_terms_decompress",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
    }

    /** A {@code MethodHandle.invokeExact} failure: only an error the JVM raised, never a status. */
    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof Error e) {
            throw e;
        }
        if (t instanceof RuntimeException e) {
            throw e;
        }
        throw new IllegalStateException("lucene-rust: downcall failed", t);
    }

    // ---- Per-thread native scratch memory ---------------------------------------------

    /**
     * A bump allocator over native memory, one per thread, reset at the start of each call. What a
     * call allocates stays valid until that call returns. Each thread that calls in keeps at most
     * {@link #RETAINED} bytes of it between calls, outside the heap and any circuit breaker; a
     * buffer larger than that (a big document, a reader with many deleted documents) comes from an
     * arena closed when the call returns.
     */
    private static final class Scratch implements AutoCloseable {
        private static final long INITIAL = 64 * 1024;
        private static final long RETAINED = 256 * 1024;
        private static final ThreadLocal<Scratch> CURRENT = ThreadLocal.withInitial(Scratch::new);

        static {
            // Every `usize` crosses as a `long`, and an owned buffer's slot is a pointer then a
            // `long`: both assume 64-bit pointers, which every platform with a library has.
            if (ADDRESS.byteSize() != Long.BYTES) {
                throw new UnsatisfiedLinkError("lucene-rust needs 64-bit pointers, this JVM has " + ADDRESS.byteSize() * 8);
            }
        }

        private MemorySegment segment = Arena.ofAuto().allocate(INITIAL, 16);
        private long used;
        private Arena oversized;
        /** Open calls on this thread: 0 idle, 1 in a call, more while a call reports a refusal. */
        private int depth;

        /**
         * The thread's scratch, reset unless a call on this thread already has it open -- as when a
         * call records a refusal through {@link #fail} -- in which case the inner use allocates past
         * what the outer one holds, which stays valid.
         */
        static Scratch open() {
            Scratch s = CURRENT.get();
            if (s.depth++ == 0) {
                s.used = 0;
            }
            return s;
        }

        @Override
        public void close() {
            if (--depth > 0) {
                return;
            }
            if (oversized != null) {
                oversized.close();
                oversized = null;
            }
        }

        MemorySegment alloc(long bytes) {
            long size = Math.max(8, (bytes + 7) & ~7L);
            if (size > RETAINED) {
                if (oversized == null) {
                    oversized = Arena.ofConfined();
                }
                return oversized.allocate(size, 16).asSlice(0, bytes);
            }
            if (used + size > segment.byteSize()) {
                // Earlier allocations of this call keep the old segment reachable, so it stays
                // valid until they are done with.
                segment = Arena.ofAuto().allocate(Math.min(RETAINED, Math.max(segment.byteSize() * 2, size)), 16);
                used = 0;
            }
            MemorySegment s = segment.asSlice(used, bytes);
            used += size;
            return s;
        }

        MemorySegment bytes(byte[] a) {
            return bytes(a, a.length);
        }

        MemorySegment bytes(byte[] a, int len) {
            MemorySegment s = alloc(len);
            MemorySegment.copy(a, 0, s, JAVA_BYTE, 0, len);
            return s;
        }

        MemorySegment longs(long[] a) {
            MemorySegment s = alloc((long) a.length * Long.BYTES);
            MemorySegment.copy(a, 0, s, JAVA_LONG, 0, a.length);
            return s;
        }

        MemorySegment ints(int[] a) {
            MemorySegment s = alloc((long) a.length * Integer.BYTES);
            MemorySegment.copy(a, 0, s, JAVA_INT, 0, a.length);
            return s;
        }

        /**
         * Room for {@code n} values of {@code layout}. Never the null address, even for none: the
         * native side refuses a null output pointer whatever its length.
         */
        MemorySegment room(MemoryLayout layout, long n) {
            return alloc(n * layout.byteSize());
        }
    }

    /**
     * Takes an owned buffer {@code (ptr, len)} out of {@code slot}: copies it and frees it. Null
     * when it is too large for a Java array (the buffer is freed all the same); {@link #tooLarge}
     * turns that into a status.
     */
    @SuppressWarnings("restricted") // the native side wrote exactly `len` bytes at `ptr`
    private static byte[] take(MemorySegment slot) {
        MemorySegment ptr = slot.get(ADDRESS, 0);
        long len = slot.get(JAVA_LONG, ADDRESS.byteSize());
        if (ptr.equals(MemorySegment.NULL)) {
            return new byte[0];
        }
        try {
            if (len > MAX_ARRAY) {
                return null;
            }
            byte[] out = new byte[(int) len];
            MemorySegment.copy(ptr.reinterpret(len), JAVA_BYTE, 0, out, 0, out.length);
            return out;
        } finally {
            try {
                H.FREE_BYTES.invokeExact(ptr, len);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
    }

    /** Reads native bytes lent for the call only: they are freed once it returns, so it keeps no view of them. */
    @FunctionalInterface
    public interface Borrowed {
        void read(MemorySegment bytes) throws IOException;
    }

    /** The owned buffer an FFI call left in {@code slot}, as a segment: empty for a null pointer. */
    @SuppressWarnings("restricted") // the native side wrote exactly `len` bytes at `ptr`
    private static MemorySegment owned(MemorySegment slot) {
        MemorySegment ptr = slot.get(ADDRESS, 0);
        long len = slot.get(JAVA_LONG, ADDRESS.byteSize());
        return ptr.equals(MemorySegment.NULL) ? MemorySegment.NULL : ptr.reinterpret(len);
    }

    /**
     * Lends {@code bytes}, an owned buffer from {@link #owned}, to {@code use}, then frees it. The
     * view {@code use} gets is closed first, so a read through one kept past the call fails instead
     * of touching freed memory.
     */
    @SuppressWarnings("restricted") // the view covers exactly the buffer the native side handed over
    private static void lend(MemorySegment bytes, Borrowed use) throws IOException {
        try (Arena call = Arena.ofConfined()) {
            use.read(bytes.reinterpret(call, null));
        } finally {
            if (bytes.equals(MemorySegment.NULL) == false) {
                try {
                    H.FREE_BYTES.invokeExact(bytes, bytes.byteSize());
                } catch (Throwable t) {
                    throw rethrow(t);
                }
            }
        }
    }

    /** Calls {@code call}, a per-document FFI fn that leaves an owned buffer, and lends the buffer to {@code use}. */
    private static int lendOwned(MethodHandle call, long handle, int segment, int doc, Borrowed use) throws IOException {
        if (use == null) {
            return nullArgument("use");
        }
        MemorySegment bytes;
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) call.invokeExact(handle, segment, doc, ptrOf(result), lenOf(result));
            if (rc != OK) {
                return rc;
            }
            bytes = owned(result);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        lend(bytes, use);
        return OK;
    }

    /** The largest {@code byte[]} every JVM allocates. */
    private static final long MAX_ARRAY = Integer.MAX_VALUE - 8;

    private static int tooLarge(String what) {
        return fail(BUFFER_TOO_SMALL, what + " is larger than a Java array holds");
    }

    /** A slot for an owned buffer's {@code (ptr, len)} pair. */
    private static MemorySegment ownedSlot(Scratch s) {
        return s.alloc(ADDRESS.byteSize() + Long.BYTES);
    }

    private static MemorySegment ptrOf(MemorySegment slot) {
        return slot;
    }

    private static MemorySegment lenOf(MemorySegment slot) {
        return slot.asSlice(ADDRESS.byteSize(), Long.BYTES);
    }

    /** Records {@code message} as this thread's last error and returns {@code status}. */
    private static int fail(int status, String message) {
        byte[] utf8 = message.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            // Fails only for a null message, which this is not.
            int ignored = (int) H.SET_LAST_ERROR.invokeExact(s.bytes(utf8), (long) utf8.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        return status;
    }

    private static int nullArgument(String what) {
        return fail(INVALID_ARGUMENT, what + " is null");
    }

    // ---- The reader ---------------------------------------------------------------------

    public static int abiVersion() {
        try {
            return (int) Abi.VERSION.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The native parent bit-set cache of {@code nested} queries: {@code [entries, bytes]}, or
     * {@code [-1, -1]} if the call fails.
     */
    public static long[] parentCacheStats() {
        try (Scratch s = Scratch.open()) {
            MemorySegment out = s.alloc(2 * Long.BYTES);
            int rc = (int) H.PARENT_CACHE_STATS.invokeExact(out);
            return rc == OK ? new long[] { out.get(JAVA_LONG, 0), out.get(JAVA_LONG, Long.BYTES) } : new long[] { -1, -1 };
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static String lastError() {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.LAST_ERROR.invokeExact(ptrOf(slot), lenOf(slot));
            if (rc != OK) {
                return null;
            }
            byte[] message = take(slot);
            return message == null ? "" : new String(message, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a reader over the segments listed in {@code segmentInfos} (bytes written by {@code
     * SegmentInfos.write(IndexOutput)} at {@code generation}), checking that segment {@code i} has
     * {@code maxDocs[i]} documents and masking it with {@code liveDocs[i]} ({@code
     * FixedBitSet.getBits()} words, exactly {@code ceil(maxDoc / 64)} of them; null for no
     * deletions; a null array for none anywhere). {@code previous} is an open handle whose unchanged
     * segments are reused, or 0. The new handle is written to {@code outHandle[0]}; it is immutable,
     * so searches on it never wait on each other or on opens and closes.
     */
    public static int openReader(
        byte[] indexPathUtf8,
        byte[] segmentInfos,
        long generation,
        long previous,
        int[] maxDocs,
        long[][] liveDocs,
        long[] outHandle
    ) {
        if (indexPathUtf8 == null) {
            return nullArgument("path");
        }
        if (segmentInfos == null) {
            return nullArgument("infos");
        }
        if (maxDocs == null) {
            return nullArgument("maxDocs");
        }
        int segments = maxDocs.length;
        if (liveDocs != null && liveDocs.length != segments) {
            return fail(INVALID_ARGUMENT, "liveDocs has " + liveDocs.length + " entries for " + segments + " segments");
        }
        try (Scratch s = Scratch.open()) {
            // `liveDocs[i]`'s words, one after another, and how many each segment has.
            long[] counts = new long[segments];
            long words = 0;
            if (liveDocs != null) {
                for (int i = 0; i < segments; i++) {
                    if (liveDocs[i] != null) {
                        counts[i] = liveDocs[i].length;
                        words += liveDocs[i].length;
                    }
                }
            }
            MemorySegment wordsSeg = s.room(JAVA_LONG, words);
            long at = 0;
            if (liveDocs != null) {
                for (long[] w : liveDocs) {
                    if (w != null) {
                        MemorySegment.copy(w, 0, wordsSeg, JAVA_LONG, at * Long.BYTES, w.length);
                        at += w.length;
                    }
                }
            }
            MemorySegment out = s.alloc(Long.BYTES);
            int rc = (int) H.OPEN_READER.invokeExact(
                s.bytes(indexPathUtf8),
                (long) indexPathUtf8.length,
                s.bytes(segmentInfos),
                (long) segmentInfos.length,
                generation,
                previous,
                s.ints(maxDocs),
                (long) segments,
                wordsSeg,
                s.longs(counts),
                out
            );
            if (rc == OK) {
                long handle = out.get(JAVA_LONG, 0);
                if (outHandle == null || outHandle.length < 1) {
                    // The caller will never see this handle, so nothing would close it.
                    closeReader(handle);
                    return fail(INVALID_ARGUMENT, "outHandle holds no slot");
                }
                outHandle[0] = handle;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code IndexReader.docFreq(new Term(field, term))} over the reader's segments, into
     * {@code out[0]}: the total-hits shortcut of a term query, counted by the native term
     * dictionaries the search reads next.
     */
    public static int docFreq(long handle, byte[] field, byte[] term, long[] out) {
        if (field == null) {
            return nullArgument("field");
        }
        if (term == null) {
            return nullArgument("term");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment n = s.alloc(Long.BYTES);
            int rc = docFreq(s, handle, field, term, n);
            if (rc == OK) {
                out[0] = n.get(JAVA_LONG, 0);
            }
            return rc;
        }
    }

    private static int docFreq(Scratch s, long handle, byte[] field, byte[] term, MemorySegment out) {
        try {
            return (int) H.DOC_FREQ.invokeExact(handle, s.bytes(field), (long) field.length, s.bytes(term), (long) term.length, out);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob. {@code outCounts[0]} receives the number of hits written; {@code
     * outCounts[1]} the total hits, exact below {@code countLimit} and otherwise a lower bound of at
     * least {@code countLimit}, with {@code outCounts[2]} 1 in that case -- Lucene's {@code
     * totalHitsThreshold}. A {@code countLimit} of 0 counts nothing ({@code outCounts[1]} is -1);
     * {@link Long#MAX_VALUE} counts exactly.
     */
    public static int search(
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        int refused = checkSearch(query, topN, outDocs, outScores, outCounts, 3);
        if (refused != OK) {
            return refused;
        }
        try (Scratch s = Scratch.open()) {
            return searchInto(s, handle, query, topN, countLimit, outDocs, outScores, outCounts);
        }
    }

    /** {@link #search}'s argument checks, in the order the native code made them. */
    private static int checkSearch(byte[] query, int topN, int[] outDocs, float[] outScores, long[] outCounts, int countSlots) {
        if (topN < 0) {
            return fail(INVALID_ARGUMENT, "topN " + topN + " is negative");
        }
        if (query == null) {
            return nullArgument("query");
        }
        if (topN > 0) {
            if (outDocs == null) {
                return nullArgument("outDocs");
            }
            if (outScores == null) {
                return nullArgument("outScores");
            }
            int room = Math.min(outDocs.length, outScores.length);
            if (room < topN) {
                return fail(BUFFER_TOO_SMALL, "output arrays hold " + room + " hits, topN is " + topN);
            }
        }
        if (outCounts == null || outCounts.length < countSlots) {
            return fail(INVALID_ARGUMENT, "outCounts holds fewer than " + countSlots + " slots");
        }
        return OK;
    }

    private static int searchInto(
        Scratch s,
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        MemorySegment docs = s.room(JAVA_INT, topN);
        MemorySegment scores = s.room(JAVA_FLOAT, topN);
        MemorySegment counts = s.alloc(3 * Long.BYTES);
        int rc;
        try {
            rc = (int) H.SEARCH.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                (long) topN,
                countLimit,
                docs,
                scores,
                (long) topN,
                counts,
                counts.asSlice(Long.BYTES),
                counts.asSlice(2 * Long.BYTES)
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
        if (rc != OK) {
            return rc;
        }
        int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
        MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
        MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, n);
        outCounts[0] = n;
        outCounts[1] = counts.get(JAVA_LONG, Long.BYTES);
        outCounts[2] = counts.get(JAVA_BOOLEAN, 2 * Long.BYTES) ? 1 : 0;
        return OK;
    }

    /**
     * The reader's top hits for a KNN request ({@code jvm_knn.rs}): global doc ids into {@code
     * outDocs}, scores into {@code outScores}, best first, at most {@code room} of them; returns the
     * status, the hit count in {@code outCount[0]}. An expanded nested search can find more than
     * {@code room}: the status is then {@link #BUFFER_TOO_SMALL} and {@code outCount[0]} the room it
     * needs.
     */
    public static int readerKnn(long handle, byte[] request, int room, int[] outDocs, float[] outScores, int[] outCount) {
        return knn(handle, 0, false, request, room, outDocs, outScores, outCount);
    }

    /**
     * {@link #readerKnn} over the reader at {@code handle} and the writer's buffers of {@code view}
     * ({@code ffi_live_view_knn}): doc ids numbered as {@link #liveViewReader}'s handles number them.
     */
    public static int liveViewKnn(long handle, long view, byte[] request, int room, int[] outDocs, float[] outScores, int[] outCount) {
        return knn(handle, view, true, request, room, outDocs, outScores, outCount);
    }

    private static int knn(
        long handle,
        long view,
        boolean live,
        byte[] request,
        int room,
        int[] outDocs,
        float[] outScores,
        int[] outCount
    ) {
        if (request == null) {
            return nullArgument("request");
        }
        if (room <= 0 || outDocs == null || outScores == null || Math.min(outDocs.length, outScores.length) < room) {
            return fail(BUFFER_TOO_SMALL, "output arrays hold fewer than " + room + " hits");
        }
        if (outCount == null || outCount.length < 1) {
            return fail(INVALID_ARGUMENT, "outCount holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, room);
            MemorySegment scores = s.room(JAVA_FLOAT, room);
            MemorySegment count = s.alloc(Long.BYTES);
            count.set(JAVA_LONG, 0, 0L);
            int rc;
            try {
                rc = live
                    ? (int) H.LIVE_VIEW_KNN.invokeExact(handle, view, s.bytes(request), (long) request.length, docs, scores, (long) room, count)
                    : (int) H.READER_KNN.invokeExact(handle, s.bytes(request), (long) request.length, docs, scores, (long) room, count);
            } catch (Throwable t) {
                throw rethrow(t);
            }
            long n = count.get(JAVA_LONG, 0);
            if (rc == BUFFER_TOO_SMALL) {
                outCount[0] = n > room && n <= Integer.MAX_VALUE - 8 ? (int) n : 0;
                return rc;
            }
            if (rc != OK) {
                return rc;
            }
            int hits = Math.toIntExact(n);
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, hits);
            MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, hits);
            outCount[0] = hits;
            return OK;
        }
    }

    /**
     * {@link #search} and {@link #docFreq} of {@code field:term} together -- a term query's
     * total-hits shortcut and its hits: {@code outCounts} as for {@link #search} (all 0 when {@code
     * topN} is 0, which searches nothing), with a fourth slot receiving the document frequency.
     */
    public static int searchDocFreq(
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        byte[] field,
        byte[] term,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        if (field == null) {
            return nullArgument("field");
        }
        if (term == null) {
            return nullArgument("term");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment df = s.alloc(Long.BYTES);
            int rc = docFreq(s, handle, field, term, df);
            if (rc != OK) {
                return rc;
            }
            if (topN == 0) {
                if (outCounts == null || outCounts.length < 4) {
                    return fail(INVALID_ARGUMENT, "outCounts holds fewer than 4 slots");
                }
                outCounts[0] = 0;
                outCounts[1] = 0;
                outCounts[2] = 0;
            } else {
                int refused = checkSearch(query, topN, outDocs, outScores, outCounts, 4);
                if (refused != OK) {
                    return refused;
                }
                rc = searchInto(s, handle, query, topN, countLimit, outDocs, outScores, outCounts);
                if (rc != OK) {
                    return rc;
                }
            }
            outCounts[3] = df.get(JAVA_LONG, 0);
            return OK;
        }
    }

    /**
     * Whether a concurrent {@code size: 0} search's count stops early: {@code spec} is the limit
     * ({@code int}), each segment's iterate flag ({@code int} count, one byte each: 1 where Lucene's
     * {@code Weight.count} gave -1) and the slices ({@link NativeAggregations#writeSlices});
     * {@code out[0]} receives 1 or 0.
     */
    public static int countTerminates(long handle, byte[] query, byte[] spec, long[] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (spec == null) {
            return nullArgument("spec");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment terminated = s.alloc(1);
            int rc = (int) H.COUNT_TERMINATES.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(spec),
                (long) spec.length,
                terminated
            );
            if (rc == OK) {
                out[0] = terminated.get(JAVA_BOOLEAN, 0) ? 1 : 0;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob sorted by a sort blob ({@link SortEncoder}): {@code outDocs} receives the
     * hits' global doc ids, best first, and {@code outValues} their sort values, one {@code long}
     * per key per hit in key order ({@link SortEncoder#value} turns them back). A keyword key's
     * terms come back in a new array stored in {@code outTerms[0]} (see {@link SortEncoder#hits});
     * {@code outCounts} as for {@link #search}, a fourth slot holding the tracked max score's
     * float bits ({@code NaN} untracked) and a fifth 1 when the sort blob's {@code terminate_after}
     * ended the search early; {@code topN} must be at least 1.
     */
    public static int searchSorted(
        long handle,
        byte[] query,
        byte[] sort,
        int topN,
        long countLimit,
        int[] outDocs,
        long[] outValues,
        long[] outCounts,
        byte[][] outTerms
    ) {
        if (topN < 0) {
            return fail(INVALID_ARGUMENT, "topN " + topN + " is negative");
        }
        if (query == null) {
            return nullArgument("query");
        }
        if (sort == null) {
            return nullArgument("sort");
        }
        if (outDocs == null) {
            return nullArgument("outDocs");
        }
        if (outValues == null) {
            return nullArgument("outValues");
        }
        int keys = sort.length == 0 ? 0 : sort[0] & 0xff;
        long wantValues = (long) topN * keys;
        if (outDocs.length < topN || outValues.length < wantValues) {
            return fail(
                BUFFER_TOO_SMALL,
                "output arrays hold " + outDocs.length + " hits and " + outValues.length + " values, topN is " + topN + " with " + keys + " keys"
            );
        }
        if (topN == 0) {
            return fail(INVALID_ARGUMENT, "searchSorted: topN must be at least 1");
        }
        if (outCounts == null || outCounts.length < 5) {
            return fail(INVALID_ARGUMENT, "outCounts holds fewer than 5 slots");
        }
        if (outTerms == null || outTerms.length < 1) {
            return fail(INVALID_ARGUMENT, "outTerms holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, topN);
            MemorySegment values = s.room(JAVA_LONG, wantValues);
            MemorySegment counts = s.alloc(6 * Long.BYTES);
            MemorySegment terms = ownedSlot(s);
            int rc = (int) H.SEARCH_SORTED.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(sort),
                (long) sort.length,
                (long) topN,
                countLimit,
                docs,
                (long) topN,
                values,
                wantValues,
                counts,
                ptrOf(terms),
                lenOf(terms)
            );
            if (rc != OK) {
                return rc;
            }
            int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
            MemorySegment.copy(values, JAVA_LONG, 0, outValues, 0, n * keys);
            MemorySegment.copy(counts, JAVA_LONG, 0, outCounts, 0, 5);
            byte[] termBytes = take(terms);
            if (termBytes == null) {
                return tooLarge("the sort terms");
            }
            if (counts.get(JAVA_LONG, 5 * Long.BYTES) != 0) {
                outTerms[0] = termBytes;
            }
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob's matches through the numeric metrics of a metrics blob ({@link
     * NativeAggregations.Plan#blob}): per slice (one when the blob names none), per field, {@code
     * outCounts} receives its value count and {@code outValues} {@link NativeAggregations#VALUES}
     * doubles -- the compensated sum and its delta, the minimum and maximum over every value, and
     * over each document's first and last; and in {@code outTerms[0]} the {@code terms} results:
     * per slice, per terms aggregation, the other-doc count ({@code long}), the bucket count ({@code
     * int}) and per bucket, by term, its doc count ({@code long}) and term ({@code int} length,
     * bytes), little-endian. With a positive {@code countLimit}, {@code outTotal} receives the {@code
     * size: 0} search's total and whether it is a lower bound, as {@link #search} counts them, from
     * the matches the aggregations visited -- left alone when some segment's matches were not
     * visited behind a {@code min_score} (its aggregations answered from points). When counted,
     * every segment's matches were visited and {@code outSegCounts} (may be null) holds a slot per
     * segment, those get each segment's live match count -- what the count collector iterates
     * there -- and {@code outTotal[2]} (when {@code outTotal} has a third slot) becomes 1, else 0.
     */
    public static int aggregate(
        long handle,
        byte[] query,
        byte[] aggs,
        long[] outCounts,
        double[] outValues,
        byte[][] outTerms,
        long countLimit,
        long[] outTotal,
        long[] outSegCounts
    ) {
        if (query == null) {
            return nullArgument("query");
        }
        if (aggs == null) {
            return nullArgument("aggs");
        }
        if (outCounts == null) {
            return nullArgument("outCounts");
        }
        if (outValues == null) {
            return nullArgument("outValues");
        }
        if (outTerms == null || outTerms.length < 1) {
            return fail(INVALID_ARGUMENT, "outTerms holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            // The output arrays' current contents go in and all of them come back out, so the
            // slots past the fields the blob names keep what they held.
            MemorySegment counts = s.longs(outCounts);
            MemorySegment values = s.room(JAVA_DOUBLE, outValues.length);
            if (outValues.length > 0) {
                MemorySegment.copy(outValues, 0, values, JAVA_DOUBLE, 0, outValues.length);
            }
            MemorySegment total = s.alloc(4 * Long.BYTES);
            total.set(JAVA_LONG, 0, 0);
            int segCap = outSegCounts == null ? 0 : outSegCounts.length;
            MemorySegment segCounts = s.room(JAVA_LONG, segCap);
            MemorySegment terms = ownedSlot(s);
            int rc = (int) H.AGGREGATE.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(aggs),
                (long) aggs.length,
                countLimit,
                counts,
                (long) outCounts.length,
                values,
                (long) outValues.length,
                total,
                segCounts,
                (long) segCap,
                ptrOf(terms),
                lenOf(terms)
            );
            // Taken (and so freed) first: every return below would otherwise leak it.
            byte[] termBytes = take(terms);
            if (total.get(JAVA_LONG, 0) != 0) {
                if (outTotal == null || outTotal.length < 2) {
                    return fail(INVALID_ARGUMENT, "outTotal holds fewer than 2 slots");
                }
                outTotal[0] = total.get(JAVA_LONG, Long.BYTES);
                outTotal[1] = total.get(JAVA_LONG, 2 * Long.BYTES);
                boolean perSegment = total.get(JAVA_LONG, 3 * Long.BYTES) != 0;
                if (outTotal.length > 2) {
                    outTotal[2] = perSegment ? 1 : 0;
                }
                if (perSegment) {
                    MemorySegment.copy(segCounts, JAVA_LONG, 0, outSegCounts, 0, segCap);
                }
            }
            if (rc != OK) {
                return rc;
            }
            MemorySegment.copy(counts, JAVA_LONG, 0, outCounts, 0, outCounts.length);
            MemorySegment.copy(values, JAVA_DOUBLE, 0, outValues, 0, outValues.length);
            if (termBytes == null) {
                return tooLarge("the terms aggregations' results");
            }
            outTerms[0] = termBytes;
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob's matches through an aggregation tree blob ({@link
     * NativeAggregationTree.Tree#blob}): the encoded shard results in {@code out[0]} (the layout
     * {@code jvm_aggs.rs} documents), which {@link NativeAggregationTree.Tree#build} reads.
     */
    public static int aggregateTree(long handle, byte[] query, byte[] tree, byte[][] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (tree == null) {
            return nullArgument("tree");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.AGGREGATE_TREE.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(tree),
                (long) tree.length,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the aggregation results");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a hybrid query's shard phase ({@link NativeHybrid}): the encoded slices' answers
     * ({@code search_hybrid_blobs} in {@code jvm_reader.rs}) in {@code out[0]}.
     */
    public static int searchHybrid(long handle, byte[] blob, byte[][] out) {
        if (blob == null) {
            return nullArgument("blob");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.SEARCH_HYBRID.invokeExact(handle, s.bytes(blob), (long) blob.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the hybrid results");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob collapsed ({@link NativeCollapse}): the sort blob's keys ({@link
     * SortEncoder}) on the collapse blob's field, {@code topN} groups per slice; the encoded
     * groups ({@code search_collapsed_blobs} in {@code jvm_reader.rs}) in {@code out[0]}.
     */
    public static int searchCollapsed(long handle, byte[] query, byte[] sort, byte[] collapse, int topN, byte[][] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (sort == null) {
            return nullArgument("sort");
        }
        if (collapse == null) {
            return nullArgument("collapse");
        }
        if (topN < 1) {
            return fail(INVALID_ARGUMENT, "searchCollapsed: topN must be at least 1");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.SEARCH_COLLAPSED.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(sort),
                (long) sort.length,
                s.bytes(collapse),
                (long) collapse.length,
                (long) topN,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the collapsed hits");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Rescores {@code docs}/{@code scores} (a search's hits, best first, global ids) in place as
     * OpenSearch's {@code QueryRescorer} does: the query blob's query over the top {@code window},
     * {@code mode} the {@code QueryRescoreMode} ordinal ({@code rescore_hits} in {@code
     * jvm_reader.rs}).
     */
    public static int rescore(
        long handle,
        byte[] query,
        int window,
        float queryWeight,
        float rescoreQueryWeight,
        int mode,
        int[] docs,
        float[] scores
    ) {
        if (query == null) {
            return nullArgument("query");
        }
        if (docs == null || scores == null || docs.length != scores.length) {
            return fail(INVALID_ARGUMENT, "rescore: as many scores as documents");
        }
        int n = docs.length;
        try (Scratch s = Scratch.open()) {
            MemorySegment d = s.room(JAVA_INT, Math.max(1, n));
            MemorySegment sc = s.room(JAVA_FLOAT, Math.max(1, n));
            MemorySegment.copy(docs, 0, d, JAVA_INT, 0, n);
            MemorySegment.copy(scores, 0, sc, JAVA_FLOAT, 0, n);
            int rc = (int) H.RESCORE.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                window,
                queryWeight,
                rescoreQueryWeight,
                mode,
                d,
                sc,
                (long) n
            );
            if (rc == OK) {
                MemorySegment.copy(d, JAVA_INT, 0, docs, 0, n);
                MemorySegment.copy(sc, JAVA_FLOAT, 0, scores, 0, n);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Segment {@code segment}'s document {@code doc} (segment-local): its stored fields, in stored
     * order, encoded as {@code jvm_fetch.rs} documents ({@link NativeStoredFieldsReader} reads them),
     * lent to {@code use} in native memory and freed when it returns.
     */
    public static int document(long handle, int segment, int doc, Borrowed use) throws IOException {
        return lendOwned(H.DOCUMENT, handle, segment, doc, use);
    }


    /**
     * Segment {@code segment}'s document {@code doc} (segment-local): its values of the fields
     * numbered {@code fields}, encoded as {@code jvm_fetch.rs}'s doc-values blob, in {@code out[0]}
     * ({@link NativeDocValues} reads them).
     */
    public static int docValues(long handle, int segment, int doc, int[] fields, byte[][] out) {
        if (fields == null) {
            return nullArgument("fields");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.DOC_VALUES.invokeExact(handle, segment, doc, s.ints(fields), (long) fields.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the doc values");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The {@code _source} of segment {@code segment}'s document {@code doc} with its k-NN vectors put
     * back, lent to {@code use}: the value a {@code DERIVED_SOURCE} field of {@link #document} leaves out.
     */
    public static int derivedSource(long handle, int segment, int doc, Borrowed use) throws IOException {
        return lendOwned(H.DERIVED_SOURCE, handle, segment, doc, use);
    }

    /**
     * {@code StoredFields.prefetch} for the documents {@code docs} (segment-local) of segment {@code
     * segment}: their stored-fields chunks fetched together where the segment is read by range, so
     * the fetch phase's reads one at a time find them local. Nothing to do for files on local disk.
     */
    public static int prefetchDocuments(long handle, int segment, int[] docs) {
        if (docs == null) {
            return nullArgument("docs");
        }
        try (Scratch s = Scratch.open()) {
            return (int) H.PREFETCH_DOCUMENTS.invokeExact(handle, segment, s.ints(docs), (long) docs.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- The encrypted store (store.rs) ---------------------------------------------

    /** The key upcall's operations and statuses ({@code KEY_OP_*}, {@code KEY_UPCALL_*} in {@code store.rs}). */
    public static final int KEY_OP_GENERATE = 1;
    public static final int KEY_OP_UNWRAP = 2;
    public static final int KEY_UPCALL_OK = 0;
    public static final int KEY_UPCALL_BUFFER_TOO_SMALL = 1;
    public static final int KEY_UPCALL_FAILED = 2;

    /**
     * Registers {@code upcall} (a stub with {@code store.rs}'s {@code KeyUpcall} signature, alive for
     * the life of the JVM) as the library's key provider callback. Only the first registration counts.
     */
    public static int storeSetKeyUpcall(MemorySegment upcall) {
        try {
            return (int) H.STORE_SET_KEY_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Binds the shard directory {@code path} to the encrypted store: the engine's writers and readers
     * opened there from now on use the JVM key provider {@code provider} through the key upcall.
     */
    public static int storeBindEncrypted(String path, long provider, int chunkSize) {
        if (path == null) {
            return nullArgument("path");
        }
        byte[] utf8 = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.STORE_BIND_ENCRYPTED.invokeExact(s.bytes(utf8), (long) utf8.length, provider, chunkSize);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Removes {@code path}'s store binding; directories the engine already opened keep theirs. */
    public static int storeUnbind(String path) {
        if (path == null) {
            return nullArgument("path");
        }
        byte[] utf8 = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.STORE_UNBIND.invokeExact(s.bytes(utf8), (long) utf8.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Whether the library's encryption runs in AWS-LC's FIPS 140-3 module in FIPS mode. */
    public static boolean storeIsFips() {
        try {
            return (int) H.STORE_IS_FIPS.invokeExact() == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Turns demand decryption of encrypted shards on or off for directories the engine opens from now
     * on, keeping at most {@code budgetBytes} of decrypted pages resident (0 keeps the current budget).
     */
    public static int storeSetLazyDecrypt(boolean enabled, long budgetBytes) {
        try {
            return (int) H.STORE_SET_LAZY_DECRYPT.invokeExact(enabled ? 1 : 0, budgetBytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Whether encrypted shards the engine opens now are demand-decrypted (on, and supported here). */
    public static boolean storeLazyDecryptActive() {
        try {
            return (int) H.STORE_LAZY_DECRYPT_ACTIVE.invokeExact() == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Binds the shard directory {@code path} to the page cache: the engine's writers and readers opened
     * there from now on read its files in pages of {@code pageSize} bytes (0: the default) within the
     * node's page cache budget; with {@code vectorsOnly}, only the vector formats' files, the rest
     * staying mapped. Nothing is bound where {@link #storePageCacheAvailable} is false.
     */
    public static int storeBindPageCache(String path, int pageSize, boolean vectorsOnly) {
        if (path == null) {
            return nullArgument("path");
        }
        byte[] utf8 = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.STORE_BIND_PAGE_CACHE.invokeExact(s.bytes(utf8), (long) utf8.length, pageSize, vectorsOnly ? 1 : 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Removes {@code path}'s page-cache binding; directories already open keep theirs. */
    public static int storeUnbindPageCache(String path) {
        if (path == null) {
            return nullArgument("path");
        }
        byte[] utf8 = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.STORE_UNBIND_PAGE_CACHE.invokeExact(s.bytes(utf8), (long) utf8.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Whether shards can be read through the page cache here ({@code userfaultfd} is permitted). */
    public static boolean storePageCacheAvailable() {
        try {
            return (int) H.STORE_PAGE_CACHE_AVAILABLE.invokeExact() == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Sets the bytes the page cache keeps resident across every shard bound to it (0 keeps it). */
    public static int storeSetPageCacheBudget(long budgetBytes) {
        try {
            return (int) H.STORE_SET_PAGE_CACHE_BUDGET.invokeExact(budgetBytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Bytes the page cache holds resident now, across every bound shard. */
    public static long storePageCacheResidentBytes() {
        try {
            return (long) H.STORE_PAGE_CACHE_RESIDENT.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Adds a directory Rust searches for the analysis modules' data packs ({@code analysis_modules.rs}). */
    public static int addDataPackDir(String dir) {
        if (dir == null) {
            return nullArgument("dir");
        }
        byte[] utf8 = dir.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.ADD_DATA_PACK_DIR.invokeExact(s.bytes(utf8), (long) utf8.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code text} analyzed as field {@code field} by the analyzer {@code spec} describes ({@code
     * jvm_analysis.rs}; {@link AnalyzerSpec} writes the spec), its tokens in {@code out[0]}.
     */
    public static int analyze(byte[] spec, byte[] field, byte[] text, byte[][] out) {
        if (spec == null || field == null || text == null) {
            return nullArgument(spec == null ? "spec" : field == null ? "field" : "text");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.ANALYZE.invokeExact(
                s.bytes(spec),
                (long) spec.length,
                s.bytes(field),
                (long) field.length,
                s.bytes(text),
                (long) text.length,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the tokens");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static int closeReader(long handle) {
        try {
            return (int) H.CLOSE_READER.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- The engine writer (engine_writer.rs). ----

    /** Indices into {@link #writerStats}' output. */
    public static final int STAT_RAM_BYTES = 0;
    public static final int STAT_FLUSHING_BYTES = 1;
    public static final int STAT_PENDING_DOCS = 2;
    public static final int STAT_UNCOMMITTED = 3;
    public static final int STAT_GENERATION = 4;
    public static final int STAT_SEGMENTS = 5;
    public static final int STAT_COUNT = 6;

    /** {@link #writerOpen}'s flags. */
    public static final int OPEN_FAULT_INJECTION = 1;
    public static final int OPEN_SPARSE = 2;
    public static final int OPEN_LIVE_SEARCH = 4;
    public static final int OPEN_MERGE_LZ4_HC = 8;
    public static final int OPEN_ZSTD_TERMS = 16;
    public static final int OPEN_ADAPTIVE_DOC_VALUES = 32;
    public static final int OPEN_PARTITIONED = 64;
    public static final int OPEN_KNN = 128;

    /** {@link #writerOpen}'s stored fields, one per {@code index.codec} the Rust writer writes. */
    public static final int STORED_FIELDS_BEST_SPEED = 0;
    public static final int STORED_FIELDS_BEST_COMPRESSION = 1;
    public static final int STORED_FIELDS_ZSTD = 2;
    public static final int STORED_FIELDS_ZSTD_NO_DICT = 3;
    /** The Rust engine's paged zstd stored fields ({@code index.lucene_rust.stored_fields.paged_zstd}). */
    public static final int STORED_FIELDS_ZSTD_PAGED = 4;

    // ---- ml-commons local models (ml.rs). ----

    /**
     * Loads the ONNX Runtime library at {@code path}, with {@code threads} intra-op threads shared by
     * every model (0: one per core). Later calls keep the first library.
     */
    public static int mlRuntimeInit(String path, int threads) {
        byte[] p = path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.ML_RUNTIME_INIT.invokeExact(s.bytes(p), (long) p.length, threads);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens the ONNX text-embedding model at {@code modelPath} with the {@code tokenizer.json} bytes,
     * ml-commons' {@code model_type} and {@code pooling_mode}, and whether results are normalized;
     * its handle goes to {@code outHandle[0]}.
     */
    public static int mlTextEmbeddingOpen(
        String modelPath,
        byte[] tokenizer,
        String modelType,
        String pooling,
        boolean normalize,
        long[] outHandle
    ) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] type = modelType.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] mode = pooling.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_TEXT_EMBEDDING_OPEN.invokeExact(
                s.bytes(model),
                (long) model.length,
                s.bytes(tokenizer),
                (long) tokenizer.length,
                s.bytes(type),
                (long) type.length,
                s.bytes(mode),
                (long) mode.length,
                (byte) (normalize ? 1 : 0),
                handle
            );
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Embeds {@code texts} (each a little-endian {@code u32} byte length then UTF-8) with model {@code
     * handle}: per text a {@code u32} rank, the dimensions, then the {@code f32} values, all little
     * endian. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlTextEmbeddingEmbed(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TEXT_EMBEDDING_EMBED.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a sparse tokenize model from its {@code tokenizer.json} bytes and its {@code idf.json}
     * bytes ({@code null} when it has none); its handle goes to {@code outHandle[0]}.
     */
    public static int mlSparseTokenizeOpen(byte[] tokenizer, byte[] idf, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            MemorySegment idfBytes = idf == null ? MemorySegment.NULL : s.bytes(idf);
            long idfLength = idf == null ? 0L : idf.length;
            int rc = (int) H.ML_SPARSE_TOKENIZE_OPEN.invokeExact(s.bytes(tokenizer), (long) tokenizer.length, idfBytes, idfLength, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The token weights of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) from sparse
     * tokenize model {@code handle}: per text a {@code u32} count, then per token its {@code u32} id,
     * {@code u32} byte length, UTF-8 token and {@code f32} weight, all little endian. {@code null} on
     * failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlSparseTokenize(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_SPARSE_TOKENIZE.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens the ONNX sparse encoding model ({@code sparse} true) or cross-encoder at {@code modelPath}
     * with the {@code tokenizer.json} bytes; its handle goes to {@code outHandle[0]}.
     */
    public static int mlOnnxModelOpen(boolean sparse, String modelPath, byte[] tokenizer, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        MethodHandle open = sparse ? H.ML_SPARSE_ENCODING_OPEN : H.ML_TEXT_SIMILARITY_OPEN;
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) open.invokeExact(s.bytes(model), (long) model.length, s.bytes(tokenizer), (long) tokenizer.length, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The sparse vectors of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) from sparse
     * encoding model {@code handle}, keyed by token id when {@code byId}: per text a {@code u32}
     * output count, then per output its {@code u32} name length, UTF-8 name and its weights as {@link
     * #mlSparseTokenize} gives a text's. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlSparseEncode(long handle, byte[] texts, boolean byId) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_SPARSE_ENCODING.invokeExact(
                handle,
                s.bytes(texts),
                (long) texts.length,
                (byte) (byId ? 1 : 0),
                ptrOf(slot),
                lenOf(slot)
            );
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Scores each of {@code texts} (encoded as for {@link #mlTextEmbeddingEmbed}) against {@code query}
     * with cross-encoder {@code handle}: per text a {@code u32} output count, then each output as
     * {@link #mlTextEmbeddingEmbed} gives a vector. {@code null} on failure, with the reason in {@link
     * #lastError}.
     */
    public static byte[] mlTextSimilarity(long handle, String query, byte[] texts) {
        byte[] q = query.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TEXT_SIMILARITY.invokeExact(
                handle,
                s.bytes(q),
                (long) q.length,
                s.bytes(texts),
                (long) texts.length,
                ptrOf(slot),
                lenOf(slot)
            );
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Loads libtorch from directory {@code dir} and the first TorchScript shim of {@code shims}
     * (encoded as texts are for {@link #mlTextEmbeddingEmbed}) that runs against it, with {@code
     * threads} intra-op threads shared by every model (0: one per physical core). Later calls keep
     * the first.
     */
    public static int mlTorchInit(String dir, byte[] shims, int threads) {
        byte[] d = dir.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.ML_TORCH_INIT.invokeExact(s.bytes(d), (long) d.length, s.bytes(shims), (long) shims.length, threads);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@link #mlTorchOpen} functions: a {@code sentence_transformers} text embedding model. */
    public static final int TORCH_SENTENCE_TRANSFORMERS = 0;
    /** A {@code huggingface_transformers} text embedding model, pooled natively. */
    public static final int TORCH_HUGGING_FACE = 1;
    /** A sparse encoding model, read with {@link #mlSparseEncode}. */
    public static final int TORCH_SPARSE_ENCODING = 2;
    /** A cross-encoder, read with {@link #mlTextSimilarity}. */
    public static final int TORCH_TEXT_SIMILARITY = 3;

    /**
     * Opens the TorchScript model at {@code modelPath} with the {@code tokenizer.json} bytes for
     * {@code function} (one of the {@code TORCH_*} constants); a Hugging Face text embedding model is
     * pooled by the {@code pooling} mode name and normalized when {@code normalize}. Its handle goes to
     * {@code outHandle[0]}. {@link #mlTorchInit} must have succeeded.
     */
    public static int mlTorchOpen(int function, String modelPath, byte[] tokenizer, String pooling, boolean normalize, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        byte[] model = modelPath.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] mode = pooling.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ML_TORCH_OPEN.invokeExact(
                (byte) function,
                s.bytes(model),
                (long) model.length,
                s.bytes(tokenizer),
                (long) tokenizer.length,
                s.bytes(mode),
                (long) mode.length,
                (byte) (normalize ? 1 : 0),
                handle
            );
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The outputs of TorchScript text embedding model {@code handle} for each of {@code texts}
     * (encoded as for {@link #mlTextEmbeddingEmbed}): per text a {@code u32} output count, then per
     * output its {@code u32} name length and UTF-8 name, {@code u32} dict number (0 outside a dict),
     * {@code u8} data type (0 {@code FLOAT32}, 1 {@code INT64}), {@code u32} rank and dimensions, and
     * the values, all little endian. {@code null} on failure, with the reason in {@link #lastError}.
     */
    public static byte[] mlTorchEmbed(long handle, byte[] texts) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.ML_TORCH_TEXT_EMBEDDING.invokeExact(handle, s.bytes(texts), (long) texts.length, ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Releases a handle from any of the {@code ml*Open} calls. */
    public static int mlModelClose(long handle) {
        try {
            return (int) H.ML_MODEL_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- Paged zstd stored fields (zstd_dictionary.rs). ----

    /** Prepares a segment's zstd dictionary; its handle into {@code outHandle[0]}. */
    public static int zstdDictionaryOpen(byte[] dictionary, long[] outHandle) {
        if (dictionary == null) {
            return nullArgument("dictionary");
        }
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ZSTD_DICTIONARY_OPEN.invokeExact(s.bytes(dictionary), (long) dictionary.length, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Releases a handle from {@link #zstdDictionaryOpen}. */
    public static int zstdDictionaryClose(long handle) {
        try {
            return (int) H.ZSTD_DICTIONARY_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Decodes the zstd frame {@code src[0..srcLen)}, compressed against dictionary {@code handle} (0:
     * none), into exactly {@code dst[0..dstLen)}.
     */
    public static int zstdPagedDecompress(long handle, byte[] src, int srcLen, byte[] dst, int dstLen) {
        if (src == null || dst == null) {
            return nullArgument(src == null ? "src" : "dst");
        }
        if (srcLen < 0 || srcLen > src.length || dstLen < 0 || dstLen > dst.length) {
            return fail(INVALID_ARGUMENT, "src or dst is shorter than its length");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment in = s.bytes(src, srcLen);
            MemorySegment out = s.alloc(dstLen);
            int rc = (int) H.ZSTD_PAGED_DECOMPRESS.invokeExact(handle, in, (long) srcLen, out, (long) dstLen);
            if (rc == OK) {
                MemorySegment.copy(out, JAVA_BYTE, 0, dst, 0, dstLen);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a shard's writer over the index at {@code path}, which must already hold a commit. Commit
     * points are dropped only through {@link #writerDeleteCommits}. {@code faultInjection} allows the
     * test-only panic operation; {@code sparse} writes the segments of an {@code index.sparse} index,
     * with neural-search's codec and clustered postings; {@code liveSearch} keeps what a search of the
     * buffered documents reads ({@link #liveViewOpen}); {@code adaptiveDocValues} writes doc values in
     * the {@code RustAdaptive} format; {@code zstdTerms} writes every field's terms in the {@code
     * RustZstd104} postings format, with zstd term blocks; {@code mergeLz4Hc} has merges write {@code BEST_SPEED} stored
     * fields with the high-compression LZ4 match finder (stock-readable); with {@code partitioned} every segment holds one partition
     * (the value of the index's partition field) and records it, and merges never mix partitions;
     * {@code knn} writes a k-NN index's segments under the k-NN codec's name, listing the fields
     * whose vectors {@code _source} leaves out (derived source).
     * {@code indexSort} is the index sort as a {@code .si} encodes it (null or empty for none), and
     * {@code parentFieldUtf8} the field marking each block's last document (null or empty for none).
     * {@code maxDocs} is {@code IndexWriter.getActualMaxDocs()}, which Lucene's tests lower.
     * {@code storedFields} is one of the {@code STORED_FIELDS_*} codes, the zstd ones written at
     * {@code compressionLevel}.
     */
    public static int writerOpen(
        byte[] indexPathUtf8,
        double ramBufferMb,
        boolean faultInjection,
        boolean sparse,
        boolean liveSearch,
        boolean adaptiveDocValues,
        boolean zstdTerms,
        boolean mergeLz4Hc,
        boolean partitioned,
        boolean knn,
        byte[] indexSort,
        byte[] parentFieldUtf8,
        int maxDocs,
        int storedFields,
        int compressionLevel,
        long[] outHandle
    ) {
        if (indexPathUtf8 == null) {
            return nullArgument("path");
        }
        byte[] sort = indexSort == null ? new byte[0] : indexSort;
        byte[] parent = parentFieldUtf8 == null ? new byte[0] : parentFieldUtf8;
        try (Scratch s = Scratch.open()) {
            MemorySegment out = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_OPEN.invokeExact(
                s.bytes(indexPathUtf8),
                (long) indexPathUtf8.length,
                ramBufferMb,
                (byte) ((faultInjection ? OPEN_FAULT_INJECTION : 0) | (sparse ? OPEN_SPARSE : 0)
                    | (liveSearch ? OPEN_LIVE_SEARCH : 0) | (adaptiveDocValues ? OPEN_ADAPTIVE_DOC_VALUES : 0)
                    | (zstdTerms ? OPEN_ZSTD_TERMS : 0) | (mergeLz4Hc ? OPEN_MERGE_LZ4_HC : 0)
                    | (partitioned ? OPEN_PARTITIONED : 0) | (knn ? OPEN_KNN : 0)),
                s.bytes(sort),
                (long) sort.length,
                s.bytes(parent),
                (long) parent.length,
                maxDocs,
                (byte) storedFields,
                compressionLevel,
                out
            );
            if (rc == OK) {
                long handle = out.get(JAVA_LONG, 0);
                if (outHandle == null || outHandle.length < 1) {
                    writerClose(handle);
                    return fail(INVALID_ARGUMENT, "outHandle holds no slot");
                }
                outHandle[0] = handle;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Registers a field spec (see {@code decode_field}); its global number goes to {@code outNumber[0]}. */
    public static int writerRegisterField(long handle, byte[] spec, int[] outNumber) {
        if (spec == null) {
            return nullArgument("spec");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment number = s.alloc(Integer.BYTES);
            int rc = (int) H.WRITER_REGISTER_FIELD.invokeExact(handle, s.bytes(spec), (long) spec.length, number);
            if (rc == OK) {
                if (outNumber == null || outNumber.length < 1) {
                    return fail(INVALID_ARGUMENT, "outNumber holds no slot");
                }
                outNumber[0] = number.get(JAVA_INT, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Applies the operation in {@code op[0..len)} (see {@code decode_op}; a negative {@code len} is
     * all of {@code op}). {@link #INVALID_ARGUMENT} and {@link #DECODE} refuse the document and change
     * nothing; any other failure is tragic.
     */
    public static int writerApply(long handle, byte[] op, int len) {
        if (op == null) {
            return nullArgument("op");
        }
        int n = len < 0 ? op.length : len;
        if (n > op.length) {
            return fail(INVALID_ARGUMENT, "op: length " + n + " exceeds the array's " + op.length);
        }
        try (Scratch s = Scratch.open()) {
            return (int) H.WRITER_APPLY.invokeExact(handle, s.bytes(op, n), (long) n);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Decodes {@code op} -- analyzing the fields it leaves to Rust -- outside any writer's lock;
     * {@code outId[0]} names it for {@link #writerApplyPrepared} or {@link #preparedFree}.
     */
    public static int writerPrepare(byte[] op, int len, long[] outId) {
        if (op == null) {
            return nullArgument("op");
        }
        if (outId == null || outId.length < 1) {
            return fail(INVALID_ARGUMENT, "outId holds no slot");
        }
        int n = len < 0 ? op.length : len;
        if (n > op.length) {
            return fail(INVALID_ARGUMENT, "op: length " + n + " exceeds the array's " + op.length);
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment id = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_PREPARE.invokeExact(s.bytes(op, n), (long) n, id);
            if (rc == OK) {
                outId[0] = id.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Applies (and forgets) the operation {@link #writerPrepare} decoded. */
    public static int writerApplyPrepared(long handle, long id) {
        try {
            return (int) H.WRITER_APPLY_PREPARED.invokeExact(handle, id);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Forgets a prepared operation that will not be applied. */
    public static int preparedFree(long id) {
        try {
            return (int) H.PREPARED_FREE.invokeExact(id);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Registers an analyzer for indexing ({@code jvm_analysis::register}: its spec and the
     * similarity's name), its id in {@code outId[0]}.
     */
    public static int analysisRegister(byte[] spec, String similarity, int[] outId) {
        if (spec == null || similarity == null) {
            return nullArgument(spec == null ? "spec" : "similarity");
        }
        if (outId == null || outId.length < 1) {
            return fail(INVALID_ARGUMENT, "outId holds no slot");
        }
        byte[] sim = similarity.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] blob = new byte[8 + spec.length + sim.length];
        java.nio.ByteBuffer b = java.nio.ByteBuffer.wrap(blob).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.putInt(spec.length).put(spec).putInt(sim.length).put(sim);
        try (Scratch s = Scratch.open()) {
            MemorySegment id = s.alloc(Integer.BYTES);
            int rc = (int) H.ANALYSIS_REGISTER.invokeExact(s.bytes(blob), (long) blob.length, id);
            if (rc == OK) {
                outId[0] = id.get(JAVA_INT, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Commits, and writes the newest generation. Once the buffers holding earlier operations are
     * flushed, the native writer asks the commit-data upcall ({@link #setCommitDataUpcall}) for the
     * user data of {@code context}, on this thread; other threads go on indexing meanwhile.
     */
    public static int writerCommit(long handle, long context, long[] outGeneration) {
        if (outGeneration == null || outGeneration.length < 1) {
            return fail(INVALID_ARGUMENT, "outGeneration holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment generation = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_COMMIT.invokeExact(handle, context, generation);
            if (rc == OK) {
                outGeneration[0] = generation.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The commit-data upcall's statuses ({@code COMMIT_DATA_*} in {@code engine_writer.rs}). */
    public static final int COMMIT_DATA_OK = 0;
    public static final int COMMIT_DATA_BUFFER_TOO_SMALL = 1;
    public static final int COMMIT_DATA_FAILED = 2;

    /** Registers the commit-data upcall {@link #writerCommit} reads a commit's user data through. */
    public static int setCommitDataUpcall(MemorySegment upcall) {
        try {
            return (int) H.SET_COMMIT_DATA_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code index.merge.scheduler.*}: merges at once, threads running big merges, the auto IO
     * throttle, and forced merges' MB/s.
     */
    public static int writerSetMergeScheduler(
        long handle,
        int maxMergeCount,
        int maxThreadCount,
        boolean autoThrottle,
        double maxForceMergeMbPerSec
    ) {
        try {
            return (int) H.WRITER_SET_MERGE_SCHEDULER.invokeExact(
                handle,
                maxMergeCount,
                maxThreadCount,
                (byte) (autoThrottle ? 1 : 0),
                maxForceMergeMbPerSec
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code index.merge.policy.*}: the writer's {@code TieredMergePolicy} settings, sizes in bytes and
     * deletes as percentages.
     */
    public static int writerSetMergePolicy(
        long handle,
        int maxMergeAtOnce,
        double segmentsPerTier,
        long maxMergedSegmentBytes,
        long floorSegmentBytes,
        double deletesPctAllowed,
        double forceMergeDeletesPctAllowed
    ) {
        try {
            return (int) H.WRITER_SET_MERGE_POLICY.invokeExact(
                handle,
                maxMergeAtOnce,
                segmentsPerTier,
                maxMergedSegmentBytes,
                floorSegmentBytes,
                deletesPctAllowed,
                forceMergeDeletesPctAllowed
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The generations of every live commit point, oldest first; their number goes to {@code outLen[0]}. */
    public static int writerCommitGenerations(long handle, long[] out, int[] outLen) {
        if (out == null) {
            return nullArgument("out");
        }
        if (outLen == null || outLen.length < 1) {
            return fail(INVALID_ARGUMENT, "outLen holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment gens = s.room(JAVA_LONG, out.length);
            MemorySegment n = s.alloc(Long.BYTES);
            n.set(JAVA_LONG, 0, 0);
            int rc = (int) H.WRITER_COMMIT_GENERATIONS.invokeExact(handle, gens, (long) out.length, n);
            long count = n.get(JAVA_LONG, 0);
            outLen[0] = (int) Math.min(count, Integer.MAX_VALUE);
            if (rc == OK) {
                MemorySegment.copy(gens, JAVA_LONG, 0, out, 0, Math.toIntExact(count));
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Drops the named commit points (never the newest). */
    public static int writerDeleteCommits(long handle, long[] generations) {
        if (generations == null) {
            return nullArgument("generations");
        }
        try (Scratch s = Scratch.open()) {
            return (int) H.WRITER_DELETE_COMMITS.invokeExact(handle, s.longs(generations), (long) generations.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Flushes the writer's buffers and puts its live segment list, uncommitted, in {@code
     * bytes[0]} as {@code segments_N} bytes; its files stay pinned under {@code out[0]} until
     * {@link #writerReleaseHold}, and {@code out[1]} is the last commit's generation, which the
     * bytes' header names. Returns the status.
     */
    public static int writerNrtSnapshot(long handle, long[] out, byte[][] bytes) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            MemorySegment hold = s.alloc(Long.BYTES);
            MemorySegment gen = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_NRT_SNAPSHOT.invokeExact(handle, ptrOf(slot), lenOf(slot), hold, gen);
            if (rc != OK) {
                return rc;
            }
            long id = hold.get(JAVA_LONG, 0);
            byte[] infos = take(slot);
            if (infos == null) {
                writerReleaseHold(handle, id);
                return tooLarge("the segment list");
            }
            out[0] = id;
            out[1] = gen.get(JAVA_LONG, 0);
            bytes[0] = infos;
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Pins a commit's segment files for a reader; the hold id goes to {@code outHold[0]}. */
    public static int writerHoldCommit(long handle, long generation, long[] outHold) {
        try (Scratch s = Scratch.open()) {
            MemorySegment hold = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_HOLD_COMMIT.invokeExact(handle, generation, hold);
            if (rc == OK) {
                long id = hold.get(JAVA_LONG, 0);
                if (outHold == null || outHold.length < 1) {
                    writerReleaseHold(handle, id);
                    return fail(INVALID_ARGUMENT, "outHold holds no slot");
                }
                outHold[0] = id;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static int writerReleaseHold(long handle, long hold) {
        try {
            return (int) H.WRITER_RELEASE_HOLD.invokeExact(handle, hold);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Soft-deleted documents with {@code _seq_no} below {@code minRetainedSeqNo} are dropped by merges. */
    public static int writerSetRetention(long handle, boolean enabled, long minRetainedSeqNo) {
        try {
            return (int) H.WRITER_SET_RETENTION.invokeExact(handle, (byte) (enabled ? 1 : 0), minRetainedSeqNo);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The writer options beside its merge policy: compound flushed segments, the merge policy's
     * {@code noCFSRatio} and {@code maxCFSSegmentSize} (bytes) for merged ones, whether indexing threads
     * help with pending flushes, and whether forced merges interleave their sources.
     */
    public static int writerSetOptions(
        long handle,
        boolean useCompoundFile,
        double noCfsRatio,
        long maxCfsSegmentBytes,
        boolean checkPendingFlush,
        boolean shuffleForcedMerges
    ) {
        try {
            return (int) H.WRITER_SET_OPTIONS.invokeExact(
                handle,
                (byte) (useCompoundFile ? 1 : 0),
                noCfsRatio,
                maxCfsSegmentBytes,
                (byte) (checkPendingFlush ? 1 : 0),
                (byte) (shuffleForcedMerges ? 1 : 0)
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * IVF fields' vectors clustered per partition of the index sort's first field: each partition of
     * at least {@code minVectors} vectors on its own, the smaller ones pooled; {@code 0} turns it off.
     */
    public static int writerSetIvfPartitionClusters(long handle, long minVectors) {
        try {
            return (int) H.WRITER_SET_IVF_PARTITION_CLUSTERS.invokeExact(handle, minVectors);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Merge-on-commit/refresh: commits and refreshes wait up to {@code waitMillis} for the merges of the
     * segments they publish ({@code 0} turns them off), picked by {@code MergeOnFlushMergePolicy} when
     * {@code mergeOnFlushPolicy}.
     */
    public static int writerSetMergeOnFlush(long handle, long waitMillis, boolean mergeOnFlushPolicy) {
        try {
            return (int) H.WRITER_SET_MERGE_ON_FLUSH.invokeExact(handle, waitMillis, (byte) (mergeOnFlushPolicy ? 1 : 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code BPReorderingMergePolicy} around the shard's merge policy, or off when {@code enabled} is
     * false: merges of at least {@code max(minMergeDocs, minMergeRatio * biggest segment)} documents,
     * and every forced merge, renumber their documents with {@code BPIndexReorderer}. {@code fields}
     * empty means every indexed field.
     */
    public static int writerSetBpReorder(
        long handle,
        boolean enabled,
        int minDocFreq,
        float maxDocFreq,
        int minPartitionSize,
        int maxIters,
        double ramBudgetMb,
        int minMergeDocs,
        float minMergeRatio,
        java.util.List<String> fields
    ) {
        byte[] names = String.join("\0", fields).getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.WRITER_SET_BP_REORDER.invokeExact(
                handle,
                (byte) (enabled ? 1 : 0),
                minDocFreq,
                maxDocFreq,
                minPartitionSize,
                maxIters,
                ramBudgetMb,
                minMergeDocs,
                minMergeRatio,
                s.bytes(names),
                (long) names.length
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code forceMerge(maxSegments)}, or {@code forceMergeDeletes()}; writes the newest generation. */
    public static int writerForceMerge(long handle, int maxSegments, boolean onlyDeletes, long[] outGeneration) {
        try (Scratch s = Scratch.open()) {
            MemorySegment generation = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_FORCE_MERGE.invokeExact(handle, maxSegments, (byte) (onlyDeletes ? 1 : 0), generation);
            if (rc == OK) {
                if (outGeneration == null || outGeneration.length < 1) {
                    return fail(INVALID_ARGUMENT, "outGeneration holds no slot");
                }
                outGeneration[0] = generation.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Fills {@code out} with the {@code STAT_*} counters, as many as it holds: the writer reads only
     * those, so a one-element array asks for the RAM count alone, which is cheap while other threads
     * index.
     */
    public static int writerStats(long handle, long[] out) {
        if (out == null) {
            return nullArgument("out");
        }
        int n = Math.min(out.length, STAT_COUNT);
        try (Scratch s = Scratch.open()) {
            MemorySegment stats = s.alloc((long) STAT_COUNT * Long.BYTES);
            int rc = (int) H.WRITER_STATS.invokeExact(handle, stats, (long) n);
            if (rc == OK) {
                MemorySegment.copy(stats, JAVA_LONG, 0, out, 0, n);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Merge policy kinds for {@link #writerSetPluggableMergePolicy} ({@code MERGE_POLICY_*} in {@code engine_writer.rs}). */
    public static final int MERGE_POLICY_TIERED = 0;
    public static final int MERGE_POLICY_LOG_BYTE_SIZE = 1;
    public static final int MERGE_POLICY_NONE = 2;

    /**
     * The shard's merge policy when it is not tiered: {@link #MERGE_POLICY_LOG_BYTE_SIZE} with its
     * merge factor, merged-size bounds in bytes (smallest, largest, largest when forced), most merged
     * documents and compound-file ratio and cap in bytes; {@link #MERGE_POLICY_NONE} (the rest
     * ignored) for no merges; {@link #MERGE_POLICY_TIERED} back to the tiered policy.
     */
    public static int writerSetPluggableMergePolicy(
        long handle,
        int kind,
        int mergeFactor,
        long minMergeBytes,
        long maxMergeBytes,
        long maxForcedMergeBytes,
        int maxMergeDocs,
        double noCfsRatio,
        long maxCfsSegmentBytes
    ) {
        try {
            return (int) H.WRITER_SET_PLUGGABLE_MERGE_POLICY.invokeExact(
                handle,
                kind,
                mergeFactor,
                minMergeBytes,
                maxMergeBytes,
                maxForcedMergeBytes,
                maxMergeDocs,
                noCfsRatio,
                maxCfsSegmentBytes
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The warm upcall's answers: {@code WARM_OK} in {@code engine_writer.rs}, and any other value fails the merge. */
    public static final int WARM_OK = 0;
    public static final int WARM_FAILED = 1;

    /** Registers the upcall that warms the merged segments of writers given a warmer. */
    public static int setWarmUpcall(MemorySegment upcall) {
        try {
            return (int) H.SET_WARM_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code IndexWriterConfig.setMergedSegmentWarmer}: whether merged segments go to the warm upcall. */
    public static int writerSetMergedSegmentWarmer(long handle, boolean on) {
        try {
            return (int) H.WRITER_SET_MERGED_SEGMENT_WARMER.invokeExact(handle, (byte) (on ? 1 : 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The merge upcall's events and failure kinds ({@code MERGE_*} in {@code engine_writer.rs}). */
    public static final int MERGE_EVENT_BEFORE = 0;
    public static final int MERGE_EVENT_AFTER = 1;
    public static final int MERGE_OK = 0;
    public static final int MERGE_FAILED_IO = 1;
    public static final int MERGE_FAILED_OTHER = 2;

    /**
     * Registers the merge upcall that hears every merge of writers opened afterwards, before and after
     * it runs; those writers' merge schedulers no longer stall indexing threads.
     */
    public static int setMergeUpcall(MemorySegment upcall) {
        try {
            return (int) H.SET_MERGE_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Whether merges are waiting for a merge thread ({@code IndexWriter.hasPendingMerges}) into
     * {@code outPending[0]}, and the merge scheduler's IO rate in MB/s into {@code outMbPerSec[0]}.
     * Safe to call from the merge upcall.
     */
    public static int writerMergeState(long handle, boolean[] outPending, double[] outMbPerSec) {
        if (outPending == null || outPending.length < 1) {
            return nullArgument("outPending");
        }
        if (outMbPerSec == null || outMbPerSec.length < 1) {
            return nullArgument("outMbPerSec");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment pending = s.alloc(1);
            MemorySegment rate = s.alloc(Double.BYTES);
            int rc = (int) H.WRITER_MERGE_STATE.invokeExact(handle, pending, rate);
            if (rc == OK) {
                outPending[0] = pending.get(JAVA_BYTE, 0) != 0;
                outMbPerSec[0] = rate.get(JAVA_DOUBLE, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Closes without committing (Lucene's {@code rollback()}). */
    public static int writerClose(long handle) {
        try {
            return (int) H.WRITER_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- Live search of the writer's buffers (live.rs). ----

    /**
     * Opens a view of the documents buffered in writer {@code writer} that a reader of commit
     * {@code commitGeneration} lacks: its handle into {@code outView[0]}, its blob ({@code published:
     * i64}, {@code count: i32}, then per buffer {@code maxDoc: i32}, {@code deleted: i32} and that many
     * deleted documents, little-endian) into {@code outBlob[0]}. {@link #INVALID_ARGUMENT} when the
     * buffers cannot be read for that commit; the caller then searches the reader alone.
     */
    public static int liveViewOpen(long writer, long commitGeneration, long[] outView, byte[][] outBlob) {
        if (outView == null || outView.length < 1 || outBlob == null || outBlob.length < 1) {
            return fail(INVALID_ARGUMENT, "outView and outBlob need a slot each");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment view = s.alloc(Long.BYTES);
            MemorySegment blob = ownedSlot(s);
            int rc = (int) H.LIVE_VIEW_OPEN.invokeExact(writer, commitGeneration, view, ptrOf(blob), lenOf(blob));
            if (rc == OK) {
                long id = view.get(JAVA_LONG, 0);
                byte[] bytes = take(blob);
                if (bytes == null) {
                    liveViewClose(id);
                    return tooLarge("the live view");
                }
                outView[0] = id;
                outBlob[0] = bytes;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * What a search Lucene answers reads {@code view}'s buffers through, over the reader at {@code
     * handle} (the view's base), into {@code out[0]}: the reader's live documents as of the view, and
     * each leaf of the view as a segment with its postings ({@code ffi_live_view_lucene} gives the
     * layout). A buffer flushed since the view opened is waited for up to {@code waitMillis};
     * {@link #INVALID_ARGUMENT} when it has not published by then.
     */
    public static int liveViewLucene(long handle, long view, long waitMillis, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out needs a slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment blob = ownedSlot(s);
            int rc = (int) H.LIVE_VIEW_LUCENE.invokeExact(handle, view, waitMillis, ptrOf(blob), lenOf(blob));
            if (rc == OK) {
                byte[] bytes = take(blob);
                if (bytes == null) {
                    return tooLarge("the buffered segments");
                }
                out[0] = bytes;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * A reader handle over the reader at {@code handle} and {@code view}'s buffers as segments after
     * its own, one per buffer, into {@code out[0]}: every native search of it reads the buffered
     * documents as the leaves of the view number them. It belongs to the view, which closes it; the
     * caller must not. {@link #INVALID_ARGUMENT} when the buffers cannot be written (one was taken
     * for a flush since the view opened); the caller then searches the reader alone.
     * <p>
     * With {@code query} (a query blob), {@code tree} (an aggregation tree blob, or null) and
     * {@code extra} (the other queries the search runs, a rescore's, or null), the buffers are
     * written without their postings and the handle holds only the terms that request reads: it
     * serves searches of that request alone, much cheaper to build. With a null {@code query}, or a
     * request whose terms cannot be named up front, it holds every term.
     */
    public static int liveViewReader(long handle, long view, byte[] query, byte[] tree, byte[][] extra, long[] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out needs a slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment id = s.alloc(Long.BYTES);
            MemorySegment q = query == null || query.length == 0 ? MemorySegment.NULL : s.bytes(query);
            MemorySegment t = query == null || tree == null || tree.length == 0 ? MemorySegment.NULL : s.bytes(tree);
            byte[] listed = query == null ? null : lengthPrefixed(extra);
            MemorySegment e = listed == null ? MemorySegment.NULL : s.bytes(listed);
            int rc = (int) H.LIVE_VIEW_READER.invokeExact(
                handle,
                view,
                q,
                q == MemorySegment.NULL ? 0L : (long) query.length,
                t,
                t == MemorySegment.NULL ? 0L : (long) tree.length,
                e,
                e == MemorySegment.NULL ? 0L : (long) listed.length,
                id
            );
            if (rc == OK) {
                out[0] = id.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code blobs} one after another, each after its length as a little-endian int; null when none. */
    private static byte[] lengthPrefixed(byte[][] blobs) {
        if (blobs == null || blobs.length == 0) {
            return null;
        }
        int size = 0;
        for (byte[] b : blobs) {
            size = Math.addExact(size, Math.addExact(Integer.BYTES, b.length));
        }
        java.nio.ByteBuffer out = java.nio.ByteBuffer.allocate(size).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (byte[] b : blobs) {
            out.putInt(b.length).put(b);
        }
        return out.array();
    }

    /** Closes a view: its buffers may go once nothing else reads them. */
    public static int liveViewClose(long view) {
        try {
            return (int) H.LIVE_VIEW_CLOSE.invokeExact(view);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Lets searches of the writer's buffers share a view up to {@code millis} milliseconds old,
     * which the writer opens ahead of them; 0 gives each search a view of every write before it.
     */
    public static int writerSetLiveStaleness(long writer, long millis) {
        try {
            return (int) H.WRITER_SET_LIVE_STALENESS.invokeExact(writer, millis);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Makes every later search of the writer's buffers read a view opened from now on. */
    public static int writerLiveRefreshShared(long writer) {
        try {
            return (int) H.WRITER_LIVE_REFRESH_SHARED.invokeExact(writer);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Lets go of the buffers a reader of commit {@code generation} already has, once no view reads them. */
    public static int writerReleaseLive(long writer, long generation) {
        try {
            return (int) H.WRITER_RELEASE_LIVE.invokeExact(writer, generation);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The bounds of field number {@code field}'s points in buffer {@code buffer} of {@code view}
     * into {@code out[0]}: the point count (little-endian long), then the packed least values, then
     * the packed greatest; empty when the buffer holds no point of the field.
     */
    public static int livePointBounds(long view, int buffer, int field, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_POINT_BOUNDS.invokeExact(view, buffer, field, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the bounds");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Document {@code doc} of buffer {@code buffer} of view {@code view}: its doc values of the
     * fields numbered {@code fields}, encoded as {@link #docValues} encodes a segment's, the terms
     * of each field numbered {@code 0..n} for that document alone, in {@code out[0]}.
     */
    public static int liveDocValues(long view, int buffer, int doc, int[] fields, byte[][] out) {
        if (fields == null) {
            return nullArgument("fields");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_DOC_VALUES.invokeExact(view, buffer, doc, s.ints(fields), (long) fields.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the doc values");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Document {@code doc} of buffer {@code buffer} of {@code view}: its stored fields encoded as
     * {@link #document} encodes them, into {@code out[0]}.
     */
    public static int liveDocument(long view, int buffer, int doc, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_DOCUMENT.invokeExact(view, buffer, doc, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the document");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@link #search} over reader {@code handle} and the buffers of {@code view} together, a
     * buffered hit numbered after the reader's {@code maxDoc}. {@code outCounts} as for {@link
     * #search}, with a fourth slot 1 when the buffers could answer the query and 0 when they could
     * not, in which case nothing else is written.
     */
    public static int searchLive(
        long handle,
        long view,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        int refused = checkSearch(query, topN, outDocs, outScores, outCounts, 4);
        if (refused != OK) {
            return refused;
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, topN);
            MemorySegment scores = s.room(JAVA_FLOAT, topN);
            MemorySegment counts = s.alloc(4 * Long.BYTES);
            int rc = (int) H.SEARCH_LIVE.invokeExact(
                handle,
                view,
                s.bytes(query),
                (long) query.length,
                (long) topN,
                countLimit,
                docs,
                scores,
                (long) topN,
                counts,
                counts.asSlice(Long.BYTES),
                counts.asSlice(2 * Long.BYTES),
                counts.asSlice(3 * Long.BYTES)
            );
            if (rc != OK) {
                return rc;
            }
            boolean supported = counts.get(JAVA_BOOLEAN, 3 * Long.BYTES);
            outCounts[3] = supported ? 1 : 0;
            if (supported == false) {
                return OK;
            }
            int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
            MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, n);
            outCounts[0] = n;
            outCounts[1] = counts.get(JAVA_LONG, Long.BYTES);
            outCounts[2] = counts.get(JAVA_BOOLEAN, 2 * Long.BYTES) ? 1 : 0;
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- RustZstd104 terms dictionaries (zstd_terms.rs). ----

    /** Prepares a field's zstd term-block dictionary; its handle into {@code outHandle[0]}. */
    public static int zstdTermsDictionaryOpen(byte[] dictionary, long[] outHandle) {
        if (dictionary == null) {
            return nullArgument("dictionary");
        }
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.ZSTD_TERMS_DICTIONARY_OPEN.invokeExact(s.bytes(dictionary), (long) dictionary.length, handle);
            if (rc == OK) {
                outHandle[0] = handle.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Releases a handle from {@link #zstdTermsDictionaryOpen}. */
    public static int zstdTermsDictionaryClose(long handle) {
        try {
            return (int) H.ZSTD_TERMS_DICTIONARY_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Decodes the bare zstd frame {@code src[0..srcLen)} of a zstd term block, compressed against
     * dictionary {@code handle} (0: none), into exactly {@code dst[0..dstLen)}.
     */
    public static int zstdTermsDecompress(long handle, byte[] src, int srcLen, byte[] dst, int dstLen) {
        if (src == null || dst == null) {
            return nullArgument(src == null ? "src" : "dst");
        }
        if (srcLen < 0 || srcLen > src.length || dstLen < 0 || dstLen > dst.length) {
            return fail(INVALID_ARGUMENT, "src or dst is shorter than its length");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment in = s.bytes(src, srcLen);
            MemorySegment out = s.alloc(dstLen);
            int rc = (int) H.ZSTD_TERMS_DECOMPRESS.invokeExact(handle, in, (long) srcLen, out, (long) dstLen);
            if (rc == OK) {
                MemorySegment.copy(out, JAVA_BYTE, 0, dst, 0, dstLen);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
