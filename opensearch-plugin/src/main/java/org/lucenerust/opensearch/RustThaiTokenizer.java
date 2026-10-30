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
import java.util.List;

/**
 * Lucene's {@code ThaiTokenizer} as the Rust port runs it: the JDK's sentence loop, with words
 * from ICU's Unicode-licensed Thai dictionary instead of the JDK's (whose word list cannot be
 * shipped in Rust). The Java side calls the same Rust code, so a field indexed natively and the
 * query text OpenSearch analyzes in Java split Thai the same way. Thai runs may split differently
 * from Lucene's {@code thai} tokenizer; other text tokenizes as Lucene's does.
 */
public final class RustThaiTokenizer extends Tokenizer {
    /** A spec of the Rust {@code thai} tokenizer alone ({@code jvm_analysis::build}'s layout). */
    static final byte[] SPEC = spec();

    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
    private final PositionIncrementAttribute increment = addAttribute(PositionIncrementAttribute.class);
    private final TypeAttribute type = addAttribute(TypeAttribute.class);
    private List<AnalyzerSpec.Token> tokens = List.of();
    private int next;
    private int finalOffset;

    @Override
    public void reset() throws IOException {
        super.reset();
        StringBuilder sb = new StringBuilder();
        char[] buf = new char[4096];
        for (int n; (n = input.read(buf)) != -1;) {
            sb.append(buf, 0, n);
        }
        String text = wellFormed(sb);
        AnalyzerSpec.Tokens got = AnalyzerSpec.rust(SPEC, "", text);
        if (got == null) {
            throw new IllegalStateException("rust_thai: Rust analysis failed: " + NativeBridge.lastError());
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

    private static byte[] spec() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(AnalyzerSpec.VERSION);
        writeInt(out, 0); // position increment gap
        writeInt(out, 1); // offset gap
        writeInt(out, 0); // char filters
        byte[] name = "thai".getBytes(StandardCharsets.UTF_8);
        writeInt(out, name.length);
        out.write(name, 0, name.length);
        writeInt(out, 0); // arguments
        writeInt(out, 0); // token filters
        writeInt(out, 0); // resources
        return out.toByteArray();
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        byte[] b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
        out.write(b, 0, 4);
    }
}
