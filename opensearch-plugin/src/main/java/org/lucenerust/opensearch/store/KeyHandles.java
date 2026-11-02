/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.lucenerust.opensearch.NativeBridge;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The key sources the native engine may call, by handle, and the one upcall it calls them through
 * ({@code KeyUpcall} in {@code store.rs}). A handle stays valid until {@link #release}; the native
 * side caches what it unwrapped, so a released handle only fails new key operations.
 *
 * <p>The upcall never throws: an exception crossing an upcall would end the JVM. Every failure
 * becomes {@link NativeBridge#KEY_UPCALL_FAILED} with its message in the output buffer.
 */
public final class KeyHandles {
    private static final Map<Long, KeySource> SOURCES = new ConcurrentHashMap<>();
    private static final AtomicLong NEXT = new AtomicLong(1);
    private static volatile boolean installed;

    private KeyHandles() {}

    /** Registers the key upcall with the native library, once per JVM. */
    @SuppressWarnings("restricted") // an upcall stub is the point
    public static synchronized void install() {
        if (installed) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    KeyHandles.class,
                    "upcall",
                    MethodType.methodType(
                        int.class,
                        long.class,
                        int.class,
                        MemorySegment.class,
                        long.class,
                        MemorySegment.class,
                        long.class,
                        MemorySegment.class
                    )
                );
            FunctionDescriptor fd = FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS);
            MemorySegment stub = Linker.nativeLinker().upcallStub(target, fd, Arena.global());
            int rc = NativeBridge.storeSetKeyUpcall(stub);
            if (rc != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the key upcall failed: " + NativeBridge.lastError());
            }
            installed = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Makes {@code source} callable from native code; the handle goes to {@code ffi_store_bind_encrypted}. */
    public static long register(KeySource source) {
        long h = NEXT.getAndIncrement();
        SOURCES.put(h, source);
        return h;
    }

    public static void release(long handle) {
        SOURCES.remove(handle);
    }

    static int registered() {
        return SOURCES.size();
    }

    /** The native engine's key operation; see {@code store.rs}'s {@code KeyUpcall}. */
    @SuppressWarnings("restricted") // the segments are the native caller's, sized by its contract
    static int upcall(long provider, int op, MemorySegment input, long inputLen, MemorySegment out, long outCap, MemorySegment outLen) {
        MemorySegment lenSlot = outLen.reinterpret(Long.BYTES);
        MemorySegment outBuf = out.reinterpret(outCap);
        byte[] result;
        int status;
        try {
            byte[] in = inputLen == 0 ? new byte[0] : input.reinterpret(inputLen).toArray(JAVA_BYTE);
            result = call(provider, op, in);
            status = NativeBridge.KEY_UPCALL_OK;
        } catch (Throwable t) {
            String message = t.getMessage() == null ? t.toString() : t.getMessage();
            byte[] utf8 = message.getBytes(StandardCharsets.UTF_8);
            result = utf8.length <= outCap ? utf8 : java.util.Arrays.copyOf(utf8, (int) outCap);
            status = NativeBridge.KEY_UPCALL_FAILED;
        }
        lenSlot.set(JAVA_LONG, 0, result.length);
        if (result.length > outCap) {
            return NativeBridge.KEY_UPCALL_BUFFER_TOO_SMALL;
        }
        MemorySegment.copy(result, 0, outBuf, JAVA_BYTE, 0, result.length);
        java.util.Arrays.fill(result, (byte) 0);
        return status;
    }

    static byte[] call(long provider, int op, byte[] in) throws java.io.IOException {
        KeySource source = SOURCES.get(provider);
        if (source == null) {
            throw new java.io.IOException("key provider handle " + provider + " is closed");
        }
        switch (op) {
            case NativeBridge.KEY_OP_GENERATE -> {
                KeySource.DataKey k = source.generate();
                byte[] id = k.keyId().getBytes(StandardCharsets.UTF_8);
                byte[] out = new byte[2 + id.length + k.raw().length + k.wrapped().length];
                out[0] = (byte) (id.length >>> 8);
                out[1] = (byte) id.length;
                System.arraycopy(id, 0, out, 2, id.length);
                System.arraycopy(k.raw(), 0, out, 2 + id.length, k.raw().length);
                System.arraycopy(k.wrapped(), 0, out, 2 + id.length + k.raw().length, k.wrapped().length);
                return out;
            }
            case NativeBridge.KEY_OP_UNWRAP -> {
                if (in.length < 2) {
                    throw new java.io.IOException("malformed unwrap request");
                }
                int idLen = ((in[0] & 0xFF) << 8) | (in[1] & 0xFF);
                if (in.length < 2 + idLen) {
                    throw new java.io.IOException("malformed unwrap request");
                }
                String keyId = new String(in, 2, idLen, StandardCharsets.UTF_8);
                byte[] wrapped = java.util.Arrays.copyOfRange(in, 2 + idLen, in.length);
                return source.unwrap(keyId, wrapped).clone();
            }
            default -> throw new java.io.IOException("unknown key operation " + op);
        }
    }
}
