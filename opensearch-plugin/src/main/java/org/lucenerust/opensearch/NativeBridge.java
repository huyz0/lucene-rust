/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

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
    public static final int EXPECTED_ABI_VERSION = 31;

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
                ADDRESS
            )
        );
        static final MethodHandle AGGREGATE_TREE = bind(
            "ffi_jvm_reader_aggregate_tree_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle DOCUMENT = bind(
            "ffi_jvm_reader_document_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle COUNT_TERMINATES = bind(
            "ffi_jvm_reader_count_terminates",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle CLOSE_READER = bind("ffi_close_jvm_reader", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));

        static final MethodHandle WRITER_OPEN = bind(
            "ffi_engine_writer_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_DOUBLE, JAVA_BYTE, JAVA_INT, ADDRESS)
        );
        static final MethodHandle WRITER_REGISTER_FIELD = bind(
            "ffi_engine_writer_register_field",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle WRITER_APPLY = bind(
            "ffi_engine_writer_apply",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_COMMIT = bind(
            "ffi_engine_writer_commit",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
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
        static final MethodHandle WRITER_RELEASE_HOLD = bind(
            "ffi_engine_writer_release_hold",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_SET_RETENTION = bind(
            "ffi_engine_writer_set_retention",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE, JAVA_LONG)
        );
        static final MethodHandle WRITER_FORCE_MERGE = bind(
            "ffi_engine_writer_force_merge",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_BYTE, ADDRESS)
        );
        static final MethodHandle WRITER_STATS = bind(
            "ffi_engine_writer_stats",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle WRITER_CLOSE = bind("ffi_engine_writer_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
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
     * visited behind a {@code min_score} (its aggregations answered from points).
     */
    public static int aggregate(
        long handle,
        byte[] query,
        byte[] aggs,
        long[] outCounts,
        double[] outValues,
        byte[][] outTerms,
        long countLimit,
        long[] outTotal
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
            MemorySegment total = s.alloc(3 * Long.BYTES);
            total.set(JAVA_LONG, 0, 0);
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
     * Segment {@code segment}'s document {@code doc} (segment-local): its stored fields, in stored
     * order, encoded as {@code jvm_fetch.rs} documents, in {@code out[0]} ({@link
     * NativeStoredFieldsReader} reads them).
     */
    public static int document(long handle, int segment, int doc, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.DOCUMENT.invokeExact(handle, segment, doc, ptrOf(result), lenOf(result));
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

    public static int closeReader(long handle) {
        try {
            return (int) H.CLOSE_READER.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- The engine writer (engine_writer.rs, M5). ----

    /** Indices into {@link #writerStats}' output. */
    public static final int STAT_RAM_BYTES = 0;
    public static final int STAT_PENDING_DOCS = 1;
    public static final int STAT_UNCOMMITTED = 2;
    public static final int STAT_GENERATION = 3;
    public static final int STAT_SEGMENTS = 4;
    public static final int STAT_COUNT = 5;

    /**
     * Opens a shard's writer over the index at {@code path}, which must already hold a commit. Commit
     * points are dropped only through {@link #writerDeleteCommits}. {@code faultInjection} allows the
     * test-only panic operation. {@code maxDocs} is {@code IndexWriter.getActualMaxDocs()}, which
     * Lucene's tests lower.
     */
    public static int writerOpen(byte[] indexPathUtf8, double ramBufferMb, boolean faultInjection, int maxDocs, long[] outHandle) {
        if (indexPathUtf8 == null) {
            return nullArgument("path");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment out = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_OPEN.invokeExact(
                s.bytes(indexPathUtf8),
                (long) indexPathUtf8.length,
                ramBufferMb,
                (byte) (faultInjection ? 1 : 0),
                maxDocs,
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

    /** Commits with {@code userData} (see {@code decode_user_data}), runs merges, writes the newest generation. */
    public static int writerCommit(long handle, byte[] userData, long[] outGeneration) {
        if (userData == null) {
            return nullArgument("userData");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment generation = s.alloc(Long.BYTES);
            int rc = (int) H.WRITER_COMMIT.invokeExact(handle, s.bytes(userData), (long) userData.length, generation);
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

    /** Fills {@code out} with the {@code STAT_*} counters (as many as it holds). */
    public static int writerStats(long handle, long[] out) {
        try (Scratch s = Scratch.open()) {
            MemorySegment stats = s.alloc((long) STAT_COUNT * Long.BYTES);
            int rc = (int) H.WRITER_STATS.invokeExact(handle, stats, (long) STAT_COUNT);
            if (rc == OK) {
                if (out == null) {
                    return nullArgument("out");
                }
                MemorySegment.copy(stats, JAVA_LONG, 0, out, 0, Math.min(out.length, STAT_COUNT));
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
}
