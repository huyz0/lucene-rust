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
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The sparse-vector writer downcalls; see {@link NativeBridge}. */
public final class NativeSparse {
    private NativeSparse() {}

    private static final class H {
        static final MethodHandle SPARSE_CLUSTERER_OPEN = bind(
            "ffi_sparse_clusterer_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS)
        );
        static final MethodHandle SPARSE_CLUSTERER_ADD = bind(
            "ffi_sparse_clusterer_add",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle SPARSE_CLUSTERER_CLOSE = bind("ffi_sparse_clusterer_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle SPARSE_POSTINGS_WRITE = bind(
            "ffi_sparse_postings_write_alloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS)
        );
    }

    /**
     * Opens a clusterer for {@code sparse_vector} field {@code number} ({@code name}) of a segment of
     * {@code maxDoc} documents, its attributes encoded as a little-endian {@code u32} count then per
     * entry a {@code u32} length and UTF-8 key, a {@code u32} length and UTF-8 value; its handle goes
     * to {@code outHandle[0]}.
     */
    public static int sparseClustererOpen(byte[] name, int number, byte[] attributes, int maxDoc, long[] outHandle) {
        if (outHandle == null || outHandle.length < 1) {
            return fail(INVALID_ARGUMENT, "outHandle holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment handle = s.alloc(Long.BYTES);
            int rc = (int) H.SPARSE_CLUSTERER_OPEN.invokeExact(
                s.bytes(name),
                (long) name.length,
                number,
                s.bytes(attributes),
                (long) attributes.length,
                maxDoc,
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

    /** Adds {@code doc}'s forward-index value, {@code length} bytes of {@code bytes} from {@code offset}. */
    public static int sparseClustererAdd(long handle, int doc, byte[] bytes, int offset, int length) {
        try (Scratch s = Scratch.open()) {
            MemorySegment value = s.alloc(length);
            MemorySegment.copy(bytes, offset, value, JAVA_BYTE, 0, length);
            return (int) H.SPARSE_CLUSTERER_ADD.invokeExact(handle, doc, value, (long) length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Releases a clusterer {@link #sparsePostingsWrite} did not take. */
    public static int sparseClustererClose(long handle) {
        try {
            return (int) H.SPARSE_CLUSTERER_CLOSE.invokeExact(handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Clusters the fields of {@code handles} (in field-name order, each released whatever the
     * outcome) and encodes the segment's {@code .sit} and {@code .sip}: the {@code .sit} length as
     * a little-endian {@code u64}, then {@code .sit}, then {@code .sip}. {@code null} on failure,
     * with the reason in {@link #lastError}.
     */
    public static byte[] sparsePostingsWrite(long[] handles, byte[] segmentId) {
        try (Scratch s = Scratch.open()) {
            MemorySegment slot = ownedSlot(s);
            MemorySegment hs = handles.length == 0 ? s.alloc(Long.BYTES) : s.longs(handles);
            int rc = (int) H.SPARSE_POSTINGS_WRITE.invokeExact(hs, (long) handles.length, s.bytes(segmentId), ptrOf(slot), lenOf(slot));
            return rc == OK ? take(slot) : null;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
