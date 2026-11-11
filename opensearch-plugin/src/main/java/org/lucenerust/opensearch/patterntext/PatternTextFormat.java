/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.patterntext;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@code pattern_text} message's template and variables, and their stored bytes: {@code
 * crates/lucene-pattern-text/src/format.rs}, byte for byte.
 *
 * <p>A message splits on single spaces. A token holding an ASCII digit or {@link #VARIABLE}
 * (U+0002) is a variable; the template is the message with each variable replaced by {@link
 * #VARIABLE}, so its U+0002 characters are exactly its slots. Empty tokens (runs of spaces, edge
 * spaces) stay in the template, so {@link #join} rebuilds the message exactly. The variables are
 * stored as, per slot in order, a vInt byte length and the token's UTF-8 bytes.
 */
public final class PatternTextFormat {
    /** The character that stands for one variable in a template. */
    public static final char VARIABLE = '\u0002';

    private PatternTextFormat() {}

    /** A message split into its template and variables. */
    public record Split(String template, List<String> args) {}

    static boolean isVariable(String message, int from, int to) {
        for (int i = from; i < to; i++) {
            char c = message.charAt(i);
            if ((c >= '0' && c <= '9') || c == VARIABLE) {
                return true;
            }
        }
        return false;
    }

    /** {@code message} split on single spaces into its template and variables. */
    public static Split split(String message) {
        StringBuilder template = new StringBuilder(message.length());
        List<String> args = new ArrayList<>();
        int start = 0;
        while (true) {
            int end = message.indexOf(' ', start);
            int to = end < 0 ? message.length() : end;
            if (isVariable(message, start, to)) {
                template.append(VARIABLE);
                args.add(message.substring(start, to));
            } else {
                template.append(message, start, to);
            }
            if (end < 0) {
                break;
            }
            template.append(' ');
            start = end + 1;
        }
        return new Split(template.toString(), args);
    }

    /**
     * The message {@code template} and {@code args} were split from.
     *
     * @throws IllegalArgumentException when the number of variables is not the template's number of
     *     slots
     */
    public static String join(String template, List<String> args) {
        int extra = 0;
        for (String a : args) {
            extra += a.length();
        }
        StringBuilder out = new StringBuilder(template.length() + extra);
        int next = 0;
        for (int i = 0; i < template.length(); i++) {
            char c = template.charAt(i);
            if (c == VARIABLE) {
                if (next == args.size()) {
                    throw new IllegalArgumentException("corrupt pattern_text value: fewer variables than template slots");
                }
                out.append(args.get(next++));
            } else {
                out.append(c);
            }
        }
        if (next != args.size()) {
            throw new IllegalArgumentException("corrupt pattern_text value: more variables than template slots");
        }
        return out.toString();
    }

    /** The stored bytes of {@code args}: per variable, a vInt length and its UTF-8 bytes. */
    public static byte[] encodeArgs(List<String> args) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (String a : args) {
            byte[] b = a.getBytes(StandardCharsets.UTF_8);
            int n = b.length;
            while ((n & ~0x7f) != 0) {
                out.write((n & 0x7f) | 0x80);
                n >>>= 7;
            }
            out.write(n);
            out.write(b, 0, b.length);
        }
        return out.toByteArray();
    }

    /**
     * The variables {@link #encodeArgs} wrote into {@code bytes[offset, offset + length)}.
     *
     * @throws IllegalArgumentException for a truncated length or value
     */
    public static List<String> decodeArgs(byte[] bytes, int offset, int length) {
        List<String> out = new ArrayList<>();
        int pos = offset;
        int end = offset + length;
        while (pos < end) {
            int len = 0;
            int shift = 0;
            while (true) {
                if (pos >= end) {
                    throw new IllegalArgumentException("corrupt pattern_text value: truncated length");
                }
                if (shift > 28) {
                    throw new IllegalArgumentException("corrupt pattern_text value: length too long");
                }
                int b = bytes[pos++] & 0xff;
                len |= (b & 0x7f) << shift;
                if ((b & 0x80) == 0) {
                    break;
                }
                shift += 7;
            }
            if (len < 0 || len > end - pos) {
                throw new IllegalArgumentException("corrupt pattern_text value: truncated value");
            }
            out.add(new String(bytes, pos, len, StandardCharsets.UTF_8));
            pos += len;
        }
        return out;
    }
}
