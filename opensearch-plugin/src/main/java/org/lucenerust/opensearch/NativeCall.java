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

/** What the downcall classes share: the handle binding, the per-thread scratch memory and the result helpers. */
final class NativeCall {
    private NativeCall() {}

    static final class H {
        static final MethodHandle SET_LAST_ERROR = bind("ffi_jvm_set_last_error", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        static final MethodHandle FREE_BYTES = bind("ffi_jvm_free_bytes", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
    }

    // ---- Downcall handles -------------------------------------------------------------

    /** A downcall handle for {@code symbol} in the library {@link NativeLibrary#load} opened. */
    @SuppressWarnings("restricted") // a downcall is the whole point; NativeLibrary documents the flag
    static MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = NativeLibrary.lookup()
            .find(symbol)
            .orElseThrow(() -> new UnsatisfiedLinkError("liblucene_ffi has no symbol [" + symbol + "]"));
        return Linker.nativeLinker().downcallHandle(address, descriptor);
    }

    /** A {@code MethodHandle.invokeExact} failure: only an error the JVM raised, never a status. */
    static RuntimeException rethrow(Throwable t) {
        if (t instanceof Error e) {
            throw e;
        }
        if (t instanceof RuntimeException e) {
            throw e;
        }
        throw new IllegalStateException("accelerated-search: downcall failed", t);
    }

    // ---- Per-thread native scratch memory ---------------------------------------------

    /**
     * A bump allocator over native memory, one per thread, reset at the start of each call. What a
     * call allocates stays valid until that call returns. Each thread that calls in keeps at most
     * {@link #RETAINED} bytes of it between calls, outside the heap and any circuit breaker; a
     * buffer larger than that (a big document, a reader with many deleted documents) comes from an
     * arena closed when the call returns.
     */
    static final class Scratch implements AutoCloseable {
        private static final long INITIAL = 64 * 1024;
        private static final long RETAINED = 256 * 1024;
        private static final ThreadLocal<Scratch> CURRENT = ThreadLocal.withInitial(Scratch::new);

        static {
            // Every `usize` crosses as a `long`, and an owned buffer's slot is a pointer then a
            // `long`: both assume 64-bit pointers, which every platform with a library has.
            if (ADDRESS.byteSize() != Long.BYTES) {
                throw new UnsatisfiedLinkError("accelerated-search needs 64-bit pointers, this JVM has " + ADDRESS.byteSize() * 8);
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
    static byte[] take(MemorySegment slot) {
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

    /** The owned buffer an FFI call left in {@code slot}, as a segment: empty for a null pointer. */
    @SuppressWarnings("restricted") // the native side wrote exactly `len` bytes at `ptr`
    static MemorySegment owned(MemorySegment slot) {
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
    static void lend(MemorySegment bytes, NativeBridge.Borrowed use) throws IOException {
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
    static int lendOwned(MethodHandle call, long handle, int segment, int doc, NativeBridge.Borrowed use) throws IOException {
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
    static final long MAX_ARRAY = Integer.MAX_VALUE - 8;

    static int tooLarge(String what) {
        return fail(BUFFER_TOO_SMALL, what + " is larger than a Java array holds");
    }

    /** A slot for an owned buffer's {@code (ptr, len)} pair. */
    static MemorySegment ownedSlot(Scratch s) {
        return s.alloc(ADDRESS.byteSize() + Long.BYTES);
    }

    static MemorySegment ptrOf(MemorySegment slot) {
        return slot;
    }

    static MemorySegment lenOf(MemorySegment slot) {
        return slot.asSlice(ADDRESS.byteSize(), Long.BYTES);
    }

    /** Records {@code message} as this thread's last error and returns {@code status}. */
    static int fail(int status, String message) {
        byte[] utf8 = message.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try (Scratch s = Scratch.open()) {
            // Fails only for a null message, which this is not.
            int ignored = (int) H.SET_LAST_ERROR.invokeExact(s.bytes(utf8), (long) utf8.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        return status;
    }

    static int nullArgument(String what) {
        return fail(INVALID_ARGUMENT, what + " is null");
    }
}
