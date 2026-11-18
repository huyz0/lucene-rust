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
import static org.lucenerust.opensearch.NativeCall.*;

/**
 * The native surface of {@code liblucene_ffi}, reached through the Foreign Function & Memory API:
 * each method is a downcall to one of the library's C-ABI functions ({@code reader_search.rs}, {@code
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
    /** The contract version this jar was built against; {@code JVM_ABI_VERSION} in {@code abi.rs}. */
    public static final int EXPECTED_ABI_VERSION = 198;

    public static final int OK = 0;

    public static final int INVALID_HANDLE = 3;

    public static final int IO = 4;

    public static final int DECODE = 5;

    public static final int BUFFER_TOO_SMALL = 8;

    public static final int PANIC = 9;

    public static final int INVALID_ARGUMENT = 10;

    /** The request outgrew the memory it was granted, or the node is over its off-heap bound. */
    public static final int MEMORY_LIMIT = 12;

    /** The request was cancelled while the native call ran ({@link #memoryScopeCancel}). */
    public static final int CANCELLED = 13;

    /** An aggregation made more buckets than the request allows; the last error is the count. */
    public static final int TOO_MANY_BUCKETS = 14;

    /** Query blob tags ({@code decode_query} in {@code query_blob.rs}). */
    public static final byte QUERY_TERM = 0;

    public static final byte QUERY_BOOLEAN = 1;

    public static final byte QUERY_TREE = 2;

    private NativeBridge() {}

    /**
     * The handshake, bound apart from everything else: a library from another build may lack the
     * rest of the symbols, and must still get as far as saying which ABI it speaks.
     */
    private static final class Abi {
        static final MethodHandle VERSION = bind("ffi_jvm_abi_version", FunctionDescriptor.of(JAVA_INT));
    }

    /** Reads native bytes lent for the call only: they are freed once it returns, so it keeps no view of them. */
    @FunctionalInterface
    public interface Borrowed {
        void read(MemorySegment bytes) throws IOException;
    }

    /** Clears this thread's last error, so that a later one is a later call's. */
    public static void clearLastError() {
        fail(OK, "");
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

    /**
     * What each recall-targeted k-NN tuner on this node measured, as a JSON object keyed by {@code
     * <index uuid>/<shard>/<field>}; null when the native call fails.
     */
    public static String knnRecallStats() {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            int rc = (int) H.KNN_RECALL_STATS.invokeExact(ptrOf(slot), lenOf(slot));
            if (rc != OK) {
                return null;
            }
            byte[] json = take(slot);
            return json == null ? "{}" : new String(json, java.nio.charset.StandardCharsets.UTF_8);
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

    private static final class H {
        static final MethodHandle LAST_ERROR = bind("ffi_jvm_last_error", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        static final MethodHandle PARENT_CACHE_STATS = bind("ffi_jvm_parent_cache_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle KNN_RECALL_STATS = bind("ffi_knn_recall_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    }
}
