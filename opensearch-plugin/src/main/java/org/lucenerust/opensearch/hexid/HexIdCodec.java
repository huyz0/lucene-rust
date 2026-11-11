/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.hexid;

import org.apache.lucene.util.BytesRef;

import java.nio.charset.StandardCharsets;

/**
 * The term a {@code hex_id} value is indexed and doc-valued as, and back.
 *
 * <p>A <em>canonical</em> value -- a non-empty, even-length string of lowercase hex digits, which
 * is what trace, span and request ids are -- is its bytes: 16 bytes for a 32-character trace id
 * instead of 32. Every other value (uppercase or mixed-case hex, odd length, a dash, any other
 * character, the empty string) is {@link #MARKER}, {@link #TEXT}, then its UTF-8 bytes, so it
 * round-trips exactly and matches exactly what a {@code keyword} field would. A canonical value
 * whose first byte is {@link #MARKER} is {@link #MARKER}, {@link #HEX}, then its bytes, so a term
 * that starts with the marker always carries the kind byte after it.
 *
 * <p>The encoding keeps the order of canonical values: their terms sort as their strings do (two
 * hex digits per byte, digits below letters), and those whose first byte is {@code ff} sort last,
 * as {@code "ff..."} does. Non-canonical values sort after every canonical one, in their own
 * string order, where a {@code keyword} would interleave them.
 */
public final class HexIdCodec {
    /** The first byte of a term that is not a canonical value's bare bytes. */
    static final byte MARKER = (byte) 0xFF;
    /** After {@link #MARKER}: a canonical value whose bytes start with {@link #MARKER}. */
    static final byte HEX = 0x00;
    /** After {@link #MARKER}: any other value, as UTF-8. */
    static final byte TEXT = 0x01;

    private static final char[] DIGITS = "0123456789abcdef".toCharArray();

    private HexIdCodec() {}

    /** Whether {@code s} is stored as its bytes: non-empty, even length, lowercase hex digits. */
    public static boolean isCanonical(CharSequence s) {
        int n = s.length();
        if (n == 0 || (n & 1) != 0) {
            return false;
        }
        for (int i = 0; i < n; i++) {
            if (nibble(s.charAt(i)) < 0) {
                return false;
            }
        }
        return true;
    }

    /** The term {@code value} is indexed as. */
    public static BytesRef encode(String value) {
        if (isCanonical(value)) {
            int n = value.length() / 2;
            boolean escape = nibble(value.charAt(0)) == 0xF && nibble(value.charAt(1)) == 0xF;
            int off = escape ? 2 : 0;
            byte[] out = new byte[off + n];
            if (escape) {
                out[0] = MARKER;
                out[1] = HEX;
            }
            for (int i = 0; i < n; i++) {
                out[off + i] = (byte) ((nibble(value.charAt(2 * i)) << 4) | nibble(value.charAt(2 * i + 1)));
            }
            return new BytesRef(out);
        }
        // UTF-8 as a keyword's term is (an unpaired surrogate as U+FFFD).
        BytesRef utf8 = new BytesRef(value);
        byte[] out = new byte[2 + utf8.length];
        out[0] = MARKER;
        out[1] = TEXT;
        System.arraycopy(utf8.bytes, utf8.offset, out, 2, utf8.length);
        return new BytesRef(out);
    }

    /** The value {@code term} was encoded from. */
    public static String decode(BytesRef term) {
        int off = term.offset;
        int len = term.length;
        if (len >= 2 && term.bytes[off] == MARKER) {
            if (term.bytes[off + 1] == TEXT) {
                return new String(term.bytes, off + 2, len - 2, StandardCharsets.UTF_8);
            }
            if (term.bytes[off + 1] == HEX) {
                off += 2;
                len -= 2;
            }
        }
        char[] out = new char[2 * len];
        for (int i = 0; i < len; i++) {
            int b = term.bytes[off + i] & 0xFF;
            out[2 * i] = DIGITS[b >>> 4];
            out[2 * i + 1] = DIGITS[b & 0xF];
        }
        return new String(out);
    }

    /** {@code c}'s value as a lowercase hex digit, or -1. */
    private static int nibble(char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    }
}
