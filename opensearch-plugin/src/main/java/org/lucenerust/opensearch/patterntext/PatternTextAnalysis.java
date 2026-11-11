/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.patterntext;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** What a {@code pattern_text} field's analyzer allows its phrases to skip. */
public final class PatternTextAnalysis {
    private PatternTextAnalysis() {}

    /** Log-shaped messages the analyzer is probed on. */
    static final List<String> PROBES = List.of(
        "Failed password for invalid user admin from 10.0.0.1 port 22 ssh2",
        "dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.19.102:54106",
        "  two  spaces and-hyphens under_scores CamelCase x.y.z ",
        "Connection closed by 10.0.0.1 [preauth] (code=255) \"quoted words\" end."
    );

    /** The filter classes whose rules can join words across a space. */
    private static final List<String> JOINING = List.of("Synonym", "CommonGrams", "Shingle", "WordDelimiter", "Concatenate", "Fingerprint");

    /** Whether {@code analyzer}'s chain for {@code field} holds a filter of {@link #JOINING}. */
    static boolean joinsByRule(Analyzer analyzer, String field) throws IOException {
        java.lang.reflect.Field input;
        try {
            input = TokenFilter.class.getDeclaredField("input");
            input.setAccessible(true);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return true;
        }
        try (TokenStream ts = analyzer.tokenStream(field, "")) {
            TokenStream s = ts;
            while (s instanceof TokenFilter f) {
                String name = f.getClass().getSimpleName();
                for (String j : JOINING) {
                    if (name.contains(j)) {
                        return true;
                    }
                }
                try {
                    s = (TokenStream) input.get(f);
                } catch (ReflectiveOperationException | RuntimeException e) {
                    return true;
                }
            }
        }
        return false;
    }

    /** One token: its position (from 0) and term. */
    record Token(int position, String term) {}

    static List<Token> tokens(Analyzer analyzer, String field, String text) throws IOException {
        List<Token> out = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream(field, text)) {
            CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
            PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
            PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
            ts.reset();
            int pos = -1;
            while (ts.incrementToken()) {
                if (len.getPositionLength() > 1) {
                    throw new IOException("a token spans several positions");
                }
                pos += inc.getPositionIncrement();
                out.add(new Token(pos, term.toString()));
            }
            ts.end();
        }
        return out;
    }

    /** Whether {@code want}'s tokens all appear in {@code have} at one start, at their relative positions. */
    static boolean foundIn(List<Token> want, List<Token> have) {
        if (want.isEmpty()) {
            return true;
        }
        int base = want.get(0).position();
        for (Token start : have) {
            if (start.term().equals(want.get(0).term()) == false) {
                continue;
            }
            boolean all = true;
            for (Token w : want) {
                int at = start.position() + w.position() - base;
                boolean found = false;
                for (Token h : have) {
                    if (h.position() == at && h.term().equals(w.term())) {
                        found = true;
                        break;
                    }
                }
                if (found == false) {
                    all = false;
                    break;
                }
            }
            if (all) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether, on every probe, the tokens of every run of whole space-separated words appear among
     * the whole message's at the same relative positions: the condition under which a phrase found
     * in a template's static text is in every message of that template. True of the standard,
     * whitespace, simple and pattern tokenizers with per-token filters; false of the keyword
     * tokenizer, or a filter that joins words across a space. The probes cannot know a rule's words,
     * so a chain holding a filter whose rules join words (synonyms, common grams, shingles, word
     * delimiters) is refused outright, as is a token spanning several positions.
     */
    public static boolean splitsOnSpaces(Analyzer analyzer, String field) {
        try {
            if (joinsByRule(analyzer, field)) {
                return false;
            }
            for (String probe : PROBES) {
                List<Token> whole = tokens(analyzer, field, probe);
                String[] words = probe.split(" ", -1);
                for (int i = 0; i < words.length; i++) {
                    StringBuilder run = new StringBuilder();
                    for (int j = i; j < words.length; j++) {
                        if (j > i) {
                            run.append(' ');
                        }
                        run.append(words[j]);
                        if (foundIn(tokens(analyzer, field, run.toString()), whole) == false) {
                            return false;
                        }
                    }
                }
            }
            return true;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }
}
