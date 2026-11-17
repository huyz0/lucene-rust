/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Builds the bytes of a query blob by hand, as the native decoder reads them: little-endian
 * integers and floats, and length-prefixed byte strings. The golden tests write the expected blob
 * with this, with the node kind numbers as literals, so a renumbered or reordered field in an
 * encoder fails them (the constants check only pairs the numbers between Java and Rust).
 */
final class WireWriter {
    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    WireWriter u8(int v) {
        out.write(v);
        return this;
    }

    WireWriter i32(int v) {
        for (int i = 0; i < 4; i++) {
            out.write(v >>> (8 * i));
        }
        return this;
    }

    WireWriter i64(long v) {
        i32((int) v);
        return i32((int) (v >>> 32));
    }

    WireWriter f32(float v) {
        return i32(Float.floatToIntBits(v));
    }

    /** A length-prefixed byte string. */
    WireWriter bytes(byte[] b) {
        i32(b.length);
        out.writeBytes(b);
        return this;
    }

    /** Bytes written as they are, such as a child node. */
    WireWriter raw(byte[] b) {
        out.writeBytes(b);
        return this;
    }

    WireWriter str(String s) {
        return bytes(s.getBytes(StandardCharsets.UTF_8));
    }

    byte[] build() {
        return out.toByteArray();
    }
}
