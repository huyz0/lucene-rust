/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;

import java.nio.charset.StandardCharsets;

/**
 * A growable little-endian byte buffer: the wire format {@code engine_writer.rs} decodes with its
 * {@code Cursor}. Not thread-safe; each operation builds its own.
 */
final class Blob {
    private byte[] buf;
    private int len;

    Blob(int capacity) {
        buf = new byte[Math.max(16, capacity)];
    }

    byte[] array() {
        return buf;
    }

    int length() {
        return len;
    }

    byte[] toArray() {
        return ArrayUtil.copyOfSubArray(buf, 0, len);
    }

    private void ensure(int more) {
        if (len + more > buf.length) {
            buf = ArrayUtil.grow(buf, len + more);
        }
    }

    Blob u8(int v) {
        ensure(1);
        buf[len++] = (byte) v;
        return this;
    }

    Blob bool(boolean v) {
        return u8(v ? 1 : 0);
    }

    Blob i32(int v) {
        ensure(4);
        buf[len++] = (byte) v;
        buf[len++] = (byte) (v >>> 8);
        buf[len++] = (byte) (v >>> 16);
        buf[len++] = (byte) (v >>> 24);
        return this;
    }

    Blob i64(long v) {
        i32((int) v);
        return i32((int) (v >>> 32));
    }

    /** Reserves an {@code i32} to be filled in later with {@link #patchI32}. */
    int reserveI32() {
        int at = len;
        i32(0);
        return at;
    }

    void patchI32(int at, int v) {
        buf[at] = (byte) v;
        buf[at + 1] = (byte) (v >>> 8);
        buf[at + 2] = (byte) (v >>> 16);
        buf[at + 3] = (byte) (v >>> 24);
    }

    Blob bytes(byte[] b, int off, int n) {
        i32(n);
        ensure(n);
        System.arraycopy(b, off, buf, len, n);
        len += n;
        return this;
    }

    Blob bytes(BytesRef b) {
        return bytes(b.bytes, b.offset, b.length);
    }

    Blob string(String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        return bytes(b, 0, b.length);
    }
}
