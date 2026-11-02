/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A Rust tokenizer, by its SPI name and factory arguments, run from Java: the plugin's
 * {@code rust_*} tokenizers ({@link RustAnalysis}). Java calls the same Rust code the native
 * index path runs, so a field indexed natively and the query text OpenSearch analyzes in Java
 * split alike, and {@link AnalyzerSpec} maps the tokenizer to its SPI name and arguments as they
 * are, so the chain around it runs natively too.
 */
public final class RustTokenizer extends Tokenizer {
    /** The plugin's name for it ({@code rust_thai}), for errors. */
    final String name;
    /** The Rust factory's SPI name ({@code thai}). */
    final String spi;
    /** The Rust factory's arguments. */
    final Map<String, String> args;
    /** A spec of this tokenizer alone ({@code jvm_analysis::build}'s layout). */
    private final byte[] spec;

    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
    private final PositionIncrementAttribute increment = addAttribute(PositionIncrementAttribute.class);
    private final TypeAttribute type = addAttribute(TypeAttribute.class);
    private List<AnalyzerSpec.Token> tokens = List.of();
    private int next;
    private int finalOffset;

    /** {@code spec} must be {@link #spec}{@code (spi, args)}; build it once per factory. */
    RustTokenizer(String name, String spi, Map<String, String> args, byte[] spec) {
        this.name = name;
        this.spi = spi;
        this.args = args;
        this.spec = spec;
    }

    RustTokenizer(String name, String spi, Map<String, String> args) {
        this(name, spi, args, spec(spi, args));
    }

    @Override
    public void reset() throws IOException {
        super.reset();
        StringBuilder sb = new StringBuilder();
        char[] buf = new char[4096];
        for (int n; (n = input.read(buf)) != -1;) {
            sb.append(buf, 0, n);
        }
        String text = wellFormed(sb);
        AnalyzerSpec.Tokens got = AnalyzerSpec.rust(spec, "", text);
        if (got == null) {
            throw new IllegalStateException(name + ": Rust analysis failed: " + NativeBridge.lastError());
        }
        tokens = got.tokens();
        finalOffset = correctOffset(got.endOffset());
        next = 0;
    }

    @Override
    public boolean incrementToken() {
        if (next >= tokens.size()) {
            return false;
        }
        AnalyzerSpec.Token t = tokens.get(next++);
        clearAttributes();
        term.setEmpty().append(t.term().utf8ToString());
        offset.setOffset(correctOffset(t.startOffset()), correctOffset(t.endOffset()));
        increment.setPositionIncrement(t.positionIncrement());
        type.setType(t.type());
        return true;
    }

    @Override
    public void end() throws IOException {
        super.end();
        offset.setOffset(finalOffset, finalOffset);
    }

    @Override
    public void close() throws IOException {
        super.close();
        tokens = List.of();
    }

    /**
     * The text with each unpaired surrogate replaced by U+FFFD, one unit for one: UTF-8 cannot carry
     * it to Rust, and offsets stay those of the original.
     */
    static String wellFormed(CharSequence s) {
        StringBuilder out = null;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            boolean lone;
            if (Character.isHighSurrogate(c)) {
                lone = i + 1 == s.length() || Character.isLowSurrogate(s.charAt(i + 1)) == false;
                if (lone == false) {
                    if (out != null) out.append(c).append(s.charAt(i + 1));
                    i++;
                    continue;
                }
            } else {
                lone = Character.isLowSurrogate(c);
            }
            if (lone && out == null) {
                out = new StringBuilder(s.length()).append(s, 0, i);
            }
            if (out != null) out.append(lone ? '�' : c);
        }
        return out == null ? s.toString() : out.toString();
    }

    /** A spec of the Rust tokenizer {@code spi} with {@code args}, alone. */
    static byte[] spec(String spi, Map<String, String> args) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(AnalyzerSpec.VERSION);
        writeInt(out, 0); // position increment gap
        writeInt(out, 1); // offset gap
        writeInt(out, 0); // char filters
        writeBytes(out, spi);
        writeInt(out, args.size());
        for (Map.Entry<String, String> e : args.entrySet()) {
            writeBytes(out, e.getKey());
            writeBytes(out, e.getValue());
        }
        writeInt(out, 0); // token filters
        writeInt(out, 0); // resources
        return out.toByteArray();
    }

    /** {@code args} as an unmodifiable map in its own order. */
    static Map<String, String> args(Map<String, String> args) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(args));
    }

    private static void writeBytes(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        writeInt(out, b.length);
        out.write(b, 0, b.length);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        byte[] b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
        out.write(b, 0, 4);
    }
}
