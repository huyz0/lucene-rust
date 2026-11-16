/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.highlight.InvalidTokenOffsetsException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * What OpenSearch's highlighters throw, thrown from what the native call reports: the class name
 * and message of the exception Java's highlighter would throw on the hit ({@code throws} in
 * {@code jvm_highlight.rs}), so the request fails as it does in OpenSearch without running
 * OpenSearch's highlighter to find out.
 */
final class HighlightErrors {
    private HighlightErrors() {}

    /** Reads {@code class} and {@code message} (each {@code len: i32}, UTF-8) and throws the exception they name. */
    static IOException thrown(ByteBuffer b, boolean wrapsTokenOffsets) throws IOException {
        String cls = string(b);
        String message = string(b);
        switch (cls) {
            case "java.lang.IllegalArgumentException":
                throw new IllegalArgumentException(message);
            case "java.lang.StringIndexOutOfBoundsException":
                throw new StringIndexOutOfBoundsException(message);
            case "java.lang.ArrayIndexOutOfBoundsException":
                throw new ArrayIndexOutOfBoundsException(message);
            case "java.lang.RuntimeException":
                throw new RuntimeException(message);
            case "java.io.IOException":
                throw new IOException(message);
            case "org.apache.lucene.search.IndexSearcher$TooManyClauses":
                throw new IndexSearcher.TooManyClauses(message);
            case "org.apache.lucene.search.highlight.InvalidTokenOffsetsException":
                // PlainHighlighter: throw new IllegalArgumentException(e)
                if (wrapsTokenOffsets) {
                    throw new IllegalArgumentException(new InvalidTokenOffsetsException(message));
                }
                throw new RuntimeException(new InvalidTokenOffsetsException(message));
            default:
                throw new IllegalStateException("lucene-rust: highlighting threw " + cls + ": " + message);
        }
    }

    private static String string(ByteBuffer b) {
        byte[] bytes = new byte[b.getInt()];
        b.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
