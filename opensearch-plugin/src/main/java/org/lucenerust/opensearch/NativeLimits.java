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
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The memory accounting downcalls: request scopes, process limit and cache budgets; see {@link NativeBridge}. */
public final class NativeLimits {
    private NativeLimits() {}

    private static final class H {
        static final MethodHandle MEMORY_SET_UPCALL = bind("ffi_memory_set_upcall", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        static final MethodHandle MEMORY_SCOPE_BEGIN = bind(
            "ffi_memory_scope_begin",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle MEMORY_SCOPE_END = bind("ffi_memory_scope_end", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        static final MethodHandle MEMORY_SCOPE_CANCEL = bind("ffi_memory_scope_cancel", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle MEMORY_SCOPE_SET_DEADLINE = bind(
            "ffi_memory_scope_set_deadline",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle MEMORY_SET_PROCESS_LIMIT = bind(
            "ffi_memory_set_process_limit",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle MEMORY_STATS = bind("ffi_memory_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        static final MethodHandle QUERY_CACHE_SET_BUDGET = bind("ffi_query_cache_set_budget", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
    }

    // ---- Memory accounting (memory.rs) ----------------------------------------------------

    /** Registers the upcall a request's scope asks for more memory through ({@code MemoryUpcall}). */
    public static int memorySetUpcall(MemorySegment upcall) {
        try {
            return (int) H.MEMORY_SET_UPCALL.invokeExact(upcall);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a memory scope on this thread for the request {@code token}: what the next native calls
     * on it allocate is charged to it, at most {@code hardLimit} bytes (0: only the grants bound it).
     */
    public static int memoryScopeBegin(long token, long hardLimit, byte[] label) {
        try (Scratch s = Scratch.open()) {
            return (int) H.MEMORY_SCOPE_BEGIN.invokeExact(token, hardLimit, s.bytes(label), (long) label.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Closes this thread's innermost memory scope: {@code out} gets the bytes granted, the most held,
     * what was still held and 1 when its deadline passed while it collected, as far as it reaches.
     */
    public static int memoryScopeEnd(long[] out) {
        try (Scratch s = Scratch.open()) {
            int n = out == null ? 0 : out.length;
            MemorySegment values = s.alloc(Math.max(1, n) * (long) Long.BYTES);
            int rc = (int) H.MEMORY_SCOPE_END.invokeExact(values, (long) n);
            if (rc == OK) {
                for (int i = 0; i < n; i++) {
                    out[i] = values.getAtIndex(JAVA_LONG, i);
                }
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Stops the request {@code token} at its next checkpoint, from any thread. */
    public static int memoryScopeCancel(long token) {
        try {
            return (int) H.MEMORY_SCOPE_CANCEL.invokeExact(token);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Gives the request {@code token} a deadline {@code nanos} from now (0 or less: passed): once it
     * passes, its native calls collect no more and answer with what they collected.
     */
    public static int memoryScopeSetDeadline(long token, long nanos) {
        try {
            return (int) H.MEMORY_SCOPE_SET_DEADLINE.invokeExact(token, nanos);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The bytes every segment's native query cache may hold together. */
    public static int queryCacheSetBudget(long bytes) {
        try {
            return (int) H.QUERY_CACHE_SET_BUDGET.invokeExact(bytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The node's bound on what the library holds, in bytes; 0 removes it. */
    public static int memorySetProcessLimit(long bytes) {
        try {
            return (int) H.MEMORY_SET_PROCESS_LIMIT.invokeExact(bytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The accounting's counters into {@code out}: bytes held, the most held, the bound, requests
     * stopped by their grant, requests stopped by the bound, requests cancelled; then the native
     * query caches' bytes, budget and evictions, decoded sort columns' bytes, the global ordinals'
     * bytes, and requests that stopped collecting at their deadline.
     */
    public static int memoryStats(long[] out) {
        if (out == null) {
            return nullArgument("out");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment values = s.alloc(Math.max(1, out.length) * (long) Long.BYTES);
            int rc = (int) H.MEMORY_STATS.invokeExact(values, (long) out.length);
            if (rc == OK) {
                for (int i = 0; i < out.length; i++) {
                    out[i] = values.getAtIndex(JAVA_LONG, i);
                }
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
