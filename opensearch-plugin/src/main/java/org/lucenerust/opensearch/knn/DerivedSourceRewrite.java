/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A JSON {@code _source} with the values at some field paths replaced, written as the parsed-map
 * round trip ({@code XContentHelper.convertToMap}, {@code XContentMapValues.transform},
 * {@code XContentBuilder.map}) writes it, without the map: tokens are copied from the parser to the builder,
 * so a document costs its output rather than a boxed value per number. {@link #rewrite} returns null
 * for what it does not prove it reads the same way (a key with a dot in it, a masked field that is not an
 * array or null, fewer replacements than fields), and the caller takes the map route.
 */
final class DerivedSourceRewrite {
    private DerivedSourceRewrite() {}

    /** What replaces a field's value: its next vector, or one constant. */
    interface Replacement {
        /** The value for the next occurrence of {@code path}, or null when there is none left. */
        Object next(String path);
    }

    /**
     * {@code source} with each of {@code paths}' arrays replaced; null when this cannot say what the map route would write.
     * {@code reorder} is for {@code XContentMapValues.transform} on a copy, which copies each map it walks through into a
     * {@code HashMap} (the root, and the objects on the way to a replaced field), so these write their members in that order.
     */
    static BytesReference rewrite(BytesReference source, Iterable<String> paths, boolean reorder, Replacement replacement) {
        Scratch out = new Scratch(source.length() + 64);
        try (
            XContentParser parser = XContentType.JSON.xContent()
                .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, source.streamInput());
            XContentBuilder builder = XContentType.JSON.contentBuilder(out)
        ) {
            if (parser.nextToken() != XContentParser.Token.START_OBJECT) {
                return null;
            }
            Walk walk = new Walk(parser, builder, paths, replacement, out, reorder);
            if (walk.object(new StringBuilder()) == false) {
                return null;
            }
            if (parser.nextToken() != null) {
                return null;
            }
            builder.flush();
            return new org.opensearch.core.common.bytes.BytesArray(out.buf, 0, out.len);
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    /** The output, its bytes open to {@link Walk#reorder}. */
    private static final class Scratch extends OutputStream {
        byte[] buf;
        int len;

        Scratch(int capacity) {
            buf = new byte[Math.max(64, capacity)];
        }

        private void ensure(int more) {
            if (len + more > buf.length) {
                buf = java.util.Arrays.copyOf(buf, Math.max(buf.length * 2, len + more));
            }
        }

        @Override
        public void write(int b) {
            ensure(1);
            buf[len++] = (byte) b;
        }

        @Override
        public void write(byte[] b, int off, int n) {
            ensure(n);
            System.arraycopy(b, off, buf, len, n);
            len += n;
        }
    }

    private static final class Walk {
        final Scratch out;
        final boolean reorder;
        byte[] swap = new byte[0];
        final XContentParser parser;
        final XContentBuilder builder;
        final List<String> paths;
        final Replacement replacement;

        Walk(XContentParser parser, XContentBuilder builder, Iterable<String> paths, Replacement replacement, Scratch out, boolean reorder) {
            this.out = out;
            this.reorder = reorder;
            this.parser = parser;
            this.builder = builder;
            this.paths = new java.util.ArrayList<>();
            paths.forEach(this.paths::add);
            this.replacement = replacement;
        }

        /** The object the parser is at the start of, to its end. */
        boolean object(StringBuilder path) throws IOException {
            int prefix = path.length();
            builder.startObject();
            String[] names = reorder ? new String[8] : null;
            int[] spans = reorder ? new int[16] : null;
            int members = 0;
            int regionStart = 0;
            if (reorder) {
                builder.flush();
                regionStart = out.len;
            }
            while (parser.nextToken() == XContentParser.Token.FIELD_NAME) {
                String name = parser.currentName();
                if (name.indexOf('.') >= 0) {
                    return false;
                }
                if (reorder) {
                    if (members == names.length) {
                        names = java.util.Arrays.copyOf(names, members * 2);
                        spans = java.util.Arrays.copyOf(spans, members * 4);
                    }
                    builder.flush();
                    names[members] = name;
                    spans[2 * members] = out.len;
                }
                builder.field(name);
                path.setLength(prefix);
                if (prefix > 0) {
                    path.append('.');
                }
                path.append(name);
                XContentParser.Token token = parser.nextToken();
                if (matches(path)) {
                    if (token == XContentParser.Token.VALUE_NULL) {
                        builder.nullValue();
                    } else {
                        if (token != XContentParser.Token.START_ARRAY) {
                            return false;
                        }
                        parser.skipChildren();
                        Object value = replacement.next(path.toString());
                        if (value == null) {
                            return false;
                        }
                        builder.value(value);
                    }
                } else if (isAncestor(path) && (token == XContentParser.Token.START_OBJECT || token == XContentParser.Token.START_ARRAY)) {
                    if (value(path) == false) {
                        return false;
                    }
                } else {
                    builder.copyCurrentStructure(parser);
                }
                if (reorder) {
                    builder.flush();
                    spans[2 * members + 1] = out.len;
                    members++;
                }
            }
            if (reorder) {
                builder.flush();
                reorder(regionStart, out.len, names, spans, members);
            }
            builder.endObject();
            path.setLength(prefix);
            return true;
        }

        /**
         * The members' bytes, written in source order at {@code [from, to)} (each after the first with its comma in
         * front), put in the order {@code new HashMap<>(map)} of the members iterates in.
         */
        private void reorder(int from, int to, String[] names, int[] spans, int n) {
            if (n < 2) {
                return;
            }
            // The copy constructor itself, not its sizing arithmetic, which differs between JDKs.
            Map<String, Integer> inSource = new java.util.LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                inSource.put(names[i], i);
            }
            Map<String, Integer> order = new HashMap<>(inSource);
            if (swap.length < to - from) {
                swap = new byte[Math.max(to - from, swap.length * 2)];
            }
            int p = 0;
            for (Integer member : order.values()) {
                int i = member;
                int s = i == 0 ? spans[0] : spans[2 * i] + 1;
                int e = spans[2 * i + 1];
                if (p > 0) {
                    swap[p++] = ',';
                }
                System.arraycopy(out.buf, s, swap, p, e - s);
                p += e - s;
            }
            System.arraycopy(swap, 0, out.buf, from, p);
        }

        /** The object or array the parser is at the start of, at {@code path}. */
        boolean value(StringBuilder path) throws IOException {
            if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                return object(path);
            }
            builder.startArray();
            while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                XContentParser.Token token = parser.currentToken();
                if (token == XContentParser.Token.START_OBJECT || token == XContentParser.Token.START_ARRAY) {
                    if (value(path) == false) {
                        return false;
                    }
                } else {
                    builder.copyCurrentStructure(parser);
                }
            }
            builder.endArray();
            return true;
        }

        boolean matches(StringBuilder path) {
            for (int i = 0; i < paths.size(); i++) {
                if (paths.get(i).contentEquals(path)) {
                    return true;
                }
            }
            return false;
        }

        boolean isAncestor(StringBuilder path) {
            for (int i = 0; i < paths.size(); i++) {
                String p = paths.get(i);
                int n = path.length();
                if (p.length() > n && p.charAt(n) == '.') {
                    int k = 0;
                    while (k < n && p.charAt(k) == path.charAt(k)) {
                        k++;
                    }
                    if (k == n) {
                        return true;
                    }
                }
            }
            return false;
        }
    }
}
