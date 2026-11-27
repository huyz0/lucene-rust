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
import static org.lucenerust.opensearch.NativeBridge.*;
import static org.lucenerust.opensearch.NativeCall.*;
import static org.lucenerust.opensearch.NativeSearch.*;
import static org.lucenerust.opensearch.NativeStore.*;
import static org.lucenerust.opensearch.NativeAnalysis.*;
import static org.lucenerust.opensearch.NativeMl.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The engine writer downcalls: open, apply, commit, merge and retention; see {@link NativeBridge}. */
public final class NativeWriter {
    private NativeWriter() {}

    private static final class H {
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
        static final MethodHandle INGEST_REGISTER = bind(
            "ffi_jvm_ingest_register",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle INGEST_PARSE = bindCritical(
            "ffi_jvm_ingest_parse",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle INGEST_PREPARE = bindCritical(
            "ffi_jvm_ingest_prepare",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_APPLY_INGESTED = bind(
            "ffi_engine_writer_apply_ingested",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_BYTE, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle INGEST_VERIFY_HELD = bind(
            "ffi_jvm_ingest_verify_held",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle INGEST_VERIFY = bind(
            "ffi_jvm_ingest_verify",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
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
        static final MethodHandle WRITER_SET_HOT_TIER = bind(
            "ffi_engine_writer_set_hot_tier",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_IVF_PARTITION_CLUSTERS = bind(
            "ffi_engine_writer_set_ivf_partition_clusters",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_INDEX_FORMAT = bind(
            "ffi_engine_writer_set_index_format",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT)
        );
        static final MethodHandle WRITER_SET_MERGE_LZ4_HC_NON_FORCE = bind(
            "ffi_engine_writer_set_merge_lz4_hc_non_force",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE)
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
        static final MethodHandle WRITER_UPGRADE = bind(
            "ffi_engine_writer_upgrade",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_BYTE, ADDRESS, ADDRESS)
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
        static final MethodHandle WRITER_SET_ADAPTIVE_AUTO = bind(
            "ffi_engine_writer_set_adaptive_auto",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle WRITER_ADAPTIVE_CHOICE = bind(
            "ffi_engine_writer_adaptive_choice",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS)
        );
    }

    // ---- The engine writer (writer.rs). ----

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

    /** The Rust engine's paged zstd stored fields ({@code index.codec.stored_fields.page_aligned}). */
    public static final int STORED_FIELDS_ZSTD_PAGED = 4;

    /**
     * Opens a shard's writer over the index at {@code path}, which must already hold a commit. Commit
     * points are dropped only through {@link #writerDeleteCommits}. {@code faultInjection} allows the
     * test-only panic operation; {@code sparse} writes the segments of an {@code index.sparse} index,
     * with neural-search's codec and clustered postings; {@code liveSearch} keeps what a search of the
     * buffered documents reads ({@link #liveViewOpen}); {@code adaptiveDocValues} writes doc values in
     * the {@code Adaptive90} format; {@code zstdTerms} writes every field's terms in the {@code
     * ZstdTerms104} postings format, with zstd term blocks; {@code mergeLz4Hc} has merges write {@code BEST_SPEED} stored
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

    /** Registers a mapping's field plan for {@link #ingestParse}, its id in {@code outId[0]}. */
    public static int ingestRegister(byte[] plan, int[] outId) {
        try (Scratch s = Scratch.open()) {
            MemorySegment id = s.alloc(Integer.BYTES);
            int rc = (int) H.INGEST_REGISTER.invokeExact(s.bytes(plan), (long) plan.length, id);
            if (rc == OK) {
                outId[0] = id.get(JAVA_INT, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Reads {@code src[off, off + len)} against a registered plan into {@code out}: 0 when the
     * document is refused, else the fragment's length, which is not written when it exceeds {@code out}.
     */
    public static int ingestParse(int plan, byte[] src, int off, int len, byte[] out) {
        try (Scratch s = Scratch.open()) {
            MemorySegment n = s.alloc(Long.BYTES);
            int rc = (int) H.INGEST_PARSE.invokeExact(
                plan,
                MemorySegment.ofArray(src).asSlice(off, len),
                (long) len,
                MemorySegment.ofArray(out),
                (long) out.length,
                n
            );
            return rc != OK ? 0 : (int) n.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Reads {@code src[off, off + len)} against a registered plan into the documents of the source
     * whose encoded {@code _id} is {@code id}, held for this thread's {@link #writerApplyIngested}.
     * {@code out} receives {@code [token, documents, weight]} and then the plan index of each nested
     * child's object; the token is 0 when the document is refused (the document mapper takes it), as
     * when it has more children than {@code out} has room for.
     */
    public static int ingestPrepare(int plan, byte[] src, int off, int len, byte[] id, long[] out) {
        try {
            return (int) H.INGEST_PREPARE.invokeExact(
                plan,
                MemorySegment.ofArray(src).asSlice(off, len),
                (long) len,
                MemorySegment.ofArray(id),
                (long) id.length,
                MemorySegment.ofArray(out),
                (long) out.length
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static final java.util.concurrent.ConcurrentHashMap<String, MemorySegment> SOFT_FIELD_NAMES = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Applies the document {@link #ingestPrepare} held on this thread under {@code token}, with
     * the values the engine assigned it: an add, or ({@code softField} not null) the replacement of
     * the live documents of its id, whose {@code softField} becomes {@code softValue}. Status
     * {@link NativeBridge#INVALID_HANDLE} means the thread no longer holds that document.
     */
    public static int writerApplyIngested(long handle, long token, String softField, long softValue, long seqNo, long primaryTerm, long version) {
        try {
            MemorySegment name = MemorySegment.NULL;
            long length = 0;
            if (softField != null) {
                name = SOFT_FIELD_NAMES.computeIfAbsent(softField, f -> {
                    byte[] b = f.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                    return java.lang.foreign.Arena.global().allocateFrom(JAVA_BYTE, b);
                });
                length = name.byteSize();
            }
            return (int) H.WRITER_APPLY_INGESTED.invokeExact(
                handle,
                token,
                (byte) (softField == null ? 0 : 1),
                name,
                length,
                softValue,
                seqNo,
                primaryTerm,
                version
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Compares the document {@link #ingestPrepare} held, given the engine's values, with the mapper's operation blob; null when they agree. */
    public static String ingestVerifyHeld(long token, long seqNo, long primaryTerm, long version, byte[] expected) {
        try (Scratch s = Scratch.open()) {
            MemorySegment msg = s.alloc(4096);
            MemorySegment n = s.alloc(Long.BYTES);
            int rc = (int) H.INGEST_VERIFY_HELD.invokeExact(token, seqNo, primaryTerm, version, s.bytes(expected), (long) expected.length, msg, 4096L, n);
            if (rc != OK) {
                return "verify failed with status " + rc;
            }
            int length = (int) n.get(JAVA_LONG, 0);
            if (length == 0) {
                return null;
            }
            byte[] text = new byte[Math.min(length, 4096)];
            MemorySegment.copy(msg, JAVA_BYTE, 0, text, 0, text.length);
            return new String(text, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Compares two single-document operation blobs as decoded documents; null when they agree. */
    public static String ingestVerify(byte[] expected, byte[] actual) {
        try (Scratch s = Scratch.open()) {
            MemorySegment msg = s.alloc(4096);
            MemorySegment n = s.alloc(Long.BYTES);
            int rc = (int) H.INGEST_VERIFY.invokeExact(s.bytes(expected), (long) expected.length, s.bytes(actual), (long) actual.length, msg, 4096L, n);
            if (rc != OK) {
                return "verify failed with status " + rc;
            }
            int length = (int) n.get(JAVA_LONG, 0);
            if (length == 0) {
                return null;
            }
            byte[] text = new byte[Math.min(length, 4096)];
            MemorySegment.copy(msg, JAVA_BYTE, 0, text, 0, text.length);
            return new String(text, java.nio.charset.StandardCharsets.UTF_8);
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
     * Registers an analyzer for indexing ({@code analysis::register}: its spec and the
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

    /** The commit-data upcall's statuses ({@code COMMIT_DATA_*} in {@code writer.rs}). */
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

    /**
     * Soft-deleted documents with {@code _seq_no} below {@code minRetainedSeqNo} are dropped by
     * merges; with {@code seqNoSentinels}, merges also write {@code UNASSIGNED_SEQ_NO} and {@code
     * UNASSIGNED_PRIMARY_TERM} for every document below it.
     */
    public static int writerSetRetention(long handle, boolean enabled, boolean seqNoSentinels, long minRetainedSeqNo) {
        try {
            byte flags = (byte) ((enabled ? 1 : 0) | (seqNoSentinels ? 2 : 0));
            return (int) H.WRITER_SET_RETENTION.invokeExact(handle, flags, minRetainedSeqNo);
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
     * The hot tier ({@code index.hot_tier.max_memory}, {@code index.hot_tier.min_update_rate}): the writer
     * holds the latest update of a row updated more than {@code minUpdatesPerSecond} times a second, within
     * {@code maxBytes} ({@code 0} leaves it off), and applies it at the next refresh, flush or live view.
     * {@code nodeMaxBytes} ({@code indices.hot_tier.max_memory}, {@code 0} for none) bounds all the node's
     * hot tiers together.
     */
    public static int writerSetHotTier(long handle, long maxBytes, double minUpdatesPerSecond, long nodeMaxBytes) {
        try {
            return (int) H.WRITER_SET_HOT_TIER.invokeExact(handle, maxBytes, minUpdatesPerSecond, nodeMaxBytes);
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

    /** The Lucene format generation written from now on: {@code 105} or {@code 106}. */
    public static int writerSetIndexFormat(long handle, int format) {
        try {
            return (int) H.WRITER_SET_INDEX_FORMAT.invokeExact(handle, format);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code index.codec.stored_fields.recompress_on_merge: non_force}: with recompression on,
     * forced merges (force merge, upgrade) copy stored-field chunks instead of recompressing them.
     */
    public static int writerSetMergeLz4HcNonForce(long handle, boolean nonForce) {
        try {
            return (int) H.WRITER_SET_MERGE_LZ4_HC_NON_FORCE.invokeExact(handle, (byte) (nonForce ? 1 : 0));
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

    /**
     * {@code ffi_engine_writer_upgrade}: rewrites every committed segment an older Lucene than
     * {@code major.minor} wrote (an older major only, when {@code onlyAncient}), each on its own.
     * {@code out[0]} is the newest commit's generation, {@code out[1]} the segments rewritten.
     */
    public static int writerUpgrade(long handle, int major, int minor, boolean onlyAncient, long[] out) {
        try (Scratch s = Scratch.open()) {
            MemorySegment generation = s.alloc(Long.BYTES);
            MemorySegment upgraded = s.alloc(Integer.BYTES);
            int rc = (int) H.WRITER_UPGRADE.invokeExact(handle, major, minor, (byte) (onlyAncient ? 1 : 0), generation, upgraded);
            if (rc == OK) {
                if (out == null || out.length < 2) {
                    return fail(INVALID_ARGUMENT, "out holds fewer than two slots");
                }
                out[0] = generation.get(JAVA_LONG, 0);
                out[1] = upgraded.get(JAVA_INT, 0);
            }
            return rc;
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

    /** Merge policy kinds for {@link #writerSetPluggableMergePolicy} ({@code MERGE_POLICY_*} in {@code writer.rs}). */
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

    /** The warm upcall's answers: {@code WARM_OK} in {@code writer.rs}, and any other value fails the merge. */
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

    /** The merge upcall's events and failure kinds ({@code MERGE_*} in {@code writer.rs}). */
    public static final int MERGE_EVENT_BEFORE = 0;

    public static final int MERGE_EVENT_AFTER = 1;

    public static final int MERGE_OK = 0;

    public static final int MERGE_FAILED_IO = 1;

    public static final int MERGE_FAILED_OTHER = 2;

    /**
     * Registers the merge upcall that hears every merge of writers opened afterwards, before and after
     * it runs; those writers' merge schedulers do not stall indexing threads.
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

    /**
     * {@code index.codec.adaptive_doc_values: auto}: the writer chooses its doc-values format from
     * its first documents, or keeps the choice its commit holds. Call before the first document.
     */
    public static int writerSetAdaptiveAuto(long writer) {
        try {
            return (int) H.WRITER_SET_ADAPTIVE_AUTO.invokeExact(writer);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code auto}'s choice into {@code out[0]}: 1 adaptive, 0 stock, -1 not made yet. */
    public static int writerAdaptiveChoice(long writer, int[] out) {
        if (out == null || out.length < 1) {
            return nullArgument("out");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = s.alloc(Integer.BYTES);
            int rc = (int) H.WRITER_ADAPTIVE_CHOICE.invokeExact(writer, slot);
            if (rc == OK) {
                out[0] = slot.get(JAVA_INT, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
