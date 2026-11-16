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
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The zstd dictionary and paged stored-fields downcalls; see {@link NativeBridge}. */
public final class NativeZstd {
    private NativeZstd() {}

    private static final class H {
        static final MethodHandle ZSTD_DICTIONARY_OPEN = bind(
            "ffi_zstd_dictionary_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle ZSTD_DICTIONARY_CLOSE = bind("ffi_zstd_dictionary_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle ZSTD_PAGED_DECOMPRESS = bind(
            "ffi_zstd_paged_decompress",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG)
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

    // ---- ZstdTerms104 terms dictionaries (zstd_terms.rs). ----

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
