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
import static org.lucenerust.opensearch.NativeAnalysis.*;
import static org.lucenerust.opensearch.NativeMl.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The store downcalls: encrypted and paged-cache stores, key upcalls, cache budgets and data packs; see {@link NativeBridge}. */
public final class NativeStore {
    private NativeStore() {}

    private static final class H {
        static final MethodHandle AGG_CACHE_SET_BUDGET = bind(
            "ffi_agg_segment_cache_set_budget",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle AGG_CACHE_STATS = bind(
            "ffi_agg_segment_cache_stats",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle STORED_SOURCE = bind(
            "ffi_jvm_reader_stored_source_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle STORED_VALUE = bind(
            "ffi_jvm_reader_stored_value_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
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
        static final MethodHandle STORE_SET_READ_ADVICE = bind(
            "ffi_store_set_read_advice",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)
        );
        static final MethodHandle STORE_PAGE_CACHE_RESIDENT = bind("ffi_store_page_cache_resident_bytes", FunctionDescriptor.of(JAVA_LONG));
        static final MethodHandle STORE_PAGE_CACHE_STATS = bind("ffi_store_page_cache_stats", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    }

    /**
     * {@link #derivedSource} as stored: each k-NN vector's masked value (the number {@code 1}) where
     * the vector would go, for a fetch whose {@code _source} filter drops every vector field.
     */
    public static int storedSource(long handle, int segment, int doc, Borrowed use) throws IOException {
        return lendOwned(H.STORED_SOURCE, handle, segment, doc, use);
    }

    /**
     * The value of the field at position {@code ordinal} of segment {@code segment}'s document {@code
     * doc}, as raw bytes (UTF-8 for a string), lent to {@code use}: the value a {@code DEFERRED} field
     * of {@link #document} leaves out.
     */
    public static int storedValue(long handle, int segment, int doc, int ordinal, Borrowed use) throws IOException {
        if (use == null) {
            return nullArgument("use");
        }
        MemorySegment bytes;
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.STORED_VALUE.invokeExact(handle, segment, doc, ordinal, ptrOf(result), lenOf(result));
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
     * Whether the engine reads encrypted shards through the page cache, decrypting each chunk once as
     * it is read in ({@code userfaultfd} is available); otherwise it decrypts each file whole.
     */
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

    /**
     * Sets the bytes the aggregation segment cache keeps parts in across the node; 0 turns it off
     * and drops every part.
     */
    public static int aggCacheSetBudget(long budgetBytes) {
        try {
            return (int) H.AGG_CACHE_SET_BUDGET.invokeExact(budgetBytes);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Sets how mapped shard files are read, by extension: {@code spec} is comma-separated
     * {@code ext:advice} entries ({@code normal}, {@code random}, {@code sequential},
     * {@code willneed}), {@code *} for every other extension, empty for the defaults.
     */
    public static int storeSetReadAdvice(String spec, boolean warmRandom) {
        byte[] utf8 = (spec == null ? "" : spec).getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            return (int) H.STORE_SET_READ_ADVICE.invokeExact(s.bytes(utf8), (long) utf8.length, warmRandom ? 1 : 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** The counters {@link #aggCacheStats} reads ({@code AGG_CACHE_STATS} in {@code jvm_aggs.rs}). */
    public static final int AGG_CACHE_STATS = 6;

    /**
     * The aggregation segment cache's counters into {@code out} ({@link #AGG_CACHE_STATS} slots):
     * hits, misses, evictions, entries, bytes, budget.
     */
    public static int aggCacheStats(long[] out) {
        if (out == null || out.length < AGG_CACHE_STATS) {
            return fail(INVALID_ARGUMENT, "out holds fewer than " + AGG_CACHE_STATS + " slots");
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(JAVA_LONG, AGG_CACHE_STATS);
            int rc = (int) H.AGG_CACHE_STATS.invokeExact(buf, (long) AGG_CACHE_STATS);
            if (rc == OK) {
                MemorySegment.copy(buf, JAVA_LONG, 0, out, 0, AGG_CACHE_STATS);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** How many values {@link #storePageCacheStats} returns. */
    public static final int PAGE_CACHE_STATS = 8;

    /**
     * The page cache's numbers: {@code [budget, resident bytes, bytes of files read whole at open,
     * faults, pages read again after eviction, evictions, pages that could not be read, nanoseconds
     * serving faults]}; every value -1 if the call fails.
     */
    public static long[] storePageCacheStats() {
        try (Scratch s = Scratch.open()) {
            MemorySegment out = s.alloc(PAGE_CACHE_STATS * Long.BYTES);
            int rc = (int) H.STORE_PAGE_CACHE_STATS.invokeExact(out);
            long[] values = new long[PAGE_CACHE_STATS];
            for (int i = 0; i < PAGE_CACHE_STATS; i++) {
                values[i] = rc == OK ? out.get(JAVA_LONG, (long) i * Long.BYTES) : -1;
            }
            return values;
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
}
