/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.ar.ArabicAnalyzer;
import org.apache.lucene.analysis.bg.BulgarianAnalyzer;
import org.apache.lucene.analysis.br.BrazilianAnalyzer;
import org.apache.lucene.analysis.cjk.CJKAnalyzer;
import org.apache.lucene.analysis.classic.ClassicAnalyzer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.UnicodeWhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.cz.CzechAnalyzer;
import org.apache.lucene.analysis.de.GermanAnalyzer;
import org.apache.lucene.analysis.el.GreekAnalyzer;
import org.apache.lucene.analysis.email.UAX29URLEmailAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.es.SpanishAnalyzer;
import org.apache.lucene.analysis.fa.PersianAnalyzer;
import org.apache.lucene.analysis.fr.FrenchAnalyzer;
import org.apache.lucene.analysis.hi.HindiAnalyzer;
import org.apache.lucene.analysis.it.ItalianAnalyzer;
import org.apache.lucene.analysis.pt.PortugueseAnalyzer;
import org.apache.lucene.analysis.ru.RussianAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.th.ThaiAnalyzer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;

/**
 * The analysis bridge ({@link AnalyzerSpec}, {@code jvm_analysis.rs}): Lucene's built-in analyzers
 * and custom chains over every mapped component, described from their objects and run in Rust;
 * every token's every attribute must match Lucene's, on the probes and on random text.
 */
final class NativeSelfTestAnalysis {
    static int described;
    static int compared;

    private NativeSelfTestAnalysis() {}

    private static java.nio.file.Path resources;

    /** A custom analyzer builder reading the word lists and rules below. */
    private static CustomAnalyzer.Builder b() throws java.io.IOException {
        if (resources == null) {
            resources = java.nio.file.Files.createTempDirectory("analysis");
            Map<String, String> files = Map.of(
                "mapping.txt", "\"ß\" => \"ss\"\n\"&\" => \" and \"\n\"ﬁ\" => \"fi\"\n\"\\u00e9\" => \"e\"\n",
                "types.txt", "<NUM>\n<APOSTROPHE>\n",
                "keep.txt", "the\nquick\nfox's\njumps\nbanana\ncafé\n",
                "articles.txt", "l\nd\nqu\n",
                "common.txt", "the\nof\nand\nto\n",
                "wdtypes.txt", "- => ALPHA\n"
            );
            for (Map.Entry<String, String> f : files.entrySet()) {
                java.nio.file.Files.writeString(resources.resolve(f.getKey()), f.getValue());
            }
        }
        return CustomAnalyzer.builder(resources);
    }

    /** Analyzers that must be described, by name. */
    static Map<String, Analyzer> analyzers() throws Exception {
        Map<String, Analyzer> m = new LinkedHashMap<>();
        m.put("standard", new StandardAnalyzer());
        StandardAnalyzer shortTokens = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
        shortTokens.setMaxTokenLength(7);
        m.put("standard english stop, max 7", shortTokens);
        m.put("simple", new SimpleAnalyzer());
        m.put("whitespace", new WhitespaceAnalyzer());
        m.put("unicode whitespace", new UnicodeWhitespaceAnalyzer());
        m.put("keyword", new KeywordAnalyzer());
        m.put("stop", new StopAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET));
        m.put("classic", new ClassicAnalyzer());
        m.put("uax29 url email", new UAX29URLEmailAnalyzer());
        m.put("english", new EnglishAnalyzer());
        m.put("english with keywords", new EnglishAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, new CharArraySet(List.of("running", "Jumps"), true)));
        m.put("german", new GermanAnalyzer());
        m.put("french", new FrenchAnalyzer());
        m.put("spanish", new SpanishAnalyzer());
        m.put("italian", new ItalianAnalyzer());
        m.put("portuguese", new PortugueseAnalyzer());
        m.put("brazilian", new BrazilianAnalyzer());
        m.put("russian", new RussianAnalyzer());
        m.put("czech", new CzechAnalyzer());
        m.put("greek", new GreekAnalyzer());
        m.put("bulgarian", new BulgarianAnalyzer());
        m.put("arabic", new ArabicAnalyzer());
        m.put("hindi", new HindiAnalyzer());
        m.put("cjk", new CJKAnalyzer());
        m.put(
            "html, mapping, standard, folding, stop, snowball, shingles",
            b()
                .addCharFilter("htmlStrip", "escapedTags", "b,script")
                .addCharFilter("mapping", "mapping", "mapping.txt")
                .withTokenizer("standard", "maxTokenLength", "20")
                .addTokenFilter("lowercase")
                .addTokenFilter("asciiFolding", "preserveOriginal", "true")
                .addTokenFilter("stop", "ignoreCase", "true")
                .addTokenFilter("snowballPorter", "language", "English")
                .addTokenFilter("shingle", "maxShingleSize", "3", "outputUnigrams", "true", "tokenSeparator", "_")
                .build()
        );
        m.put(
            "whitespace, word delimiter graph, flatten",
            b()
                .withTokenizer("whitespace")
                .addTokenFilter("wordDelimiterGraph", "catenateAll", "1", "preserveOriginal", "1", "splitOnNumerics", "0")
                .addTokenFilter("flattenGraph")
                .addTokenFilter("lowercase")
                .build()
        );
        m.put(
            "whitespace, word delimiter",
            b()
                .withTokenizer("whitespace", "maxTokenLen", "30")
                .addTokenFilter("wordDelimiter", "catenateWords", "1", "catenateNumbers", "1", "stemEnglishPossessive", "0")
                .build()
        );
        m.put(
            "edge ngram tokenizer",
            b().withTokenizer("edgeNGram", "minGramSize", "2", "maxGramSize", "5").addTokenFilter("uppercase").build()
        );
        m.put("ngram tokenizer", b().withTokenizer("nGram", "minGramSize", "1", "maxGramSize", "3").build());
        m.put(
            "letter, ngram and edge ngram filters",
            b()
                .withTokenizer("letter")
                .addTokenFilter("edgeNGram", "minGramSize", "1", "maxGramSize", "4", "preserveOriginal", "true")
                .addTokenFilter("nGram", "minGramSize", "2", "maxGramSize", "3")
                .addTokenFilter("removeDuplicates")
                .build()
        );
        m.put(
            "path hierarchy",
            b().withTokenizer("pathHierarchy", "delimiter", "/", "replace", "|", "skip", "1").build()
        );
        m.put(
            "reverse path hierarchy",
            b().withTokenizer("pathHierarchy", "delimiter", "/", "reverse", "true").addTokenFilter("reverseString").build()
        );
        m.put(
            "pattern tokenizer, pattern replace",
            b()
                .addCharFilter("patternReplace", "pattern", "(\\d+)-(\\d+)", "replacement", "$1$2")
                .withTokenizer("pattern", "pattern", "[\\s,;]+")
                .addTokenFilter("patternReplace", "pattern", "a+", "replacement", "A", "replace", "first")
                .addTokenFilter("trim")
                .build()
        );
        m.put(
            "pattern tokenizer group",
            b().withTokenizer("pattern", "pattern", "'([^']+)'|(\\w+)", "group", "0").build()
        );
        m.put(
            "standard, cjk width and bigram, truncate, length, limit",
            b()
                .withTokenizer("standard")
                .addTokenFilter("cjkWidth")
                .addTokenFilter("cjkBigram", "hangul", "false", "outputUnigrams", "true")
                .addTokenFilter("truncate", "prefixLength", "6")
                .addTokenFilter("length", "min", "2", "max", "5")
                .addTokenFilter("limitTokenCount", "maxTokenCount", "12", "consumeAllTokens", "true")
                .build()
        );
        m.put(
            "classic tokenizer and filter, keep words, types",
            b()
                .withTokenizer("classic")
                .addTokenFilter("classic")
                .addTokenFilter("type", "types", "types.txt", "useWhitelist", "false")
                .addTokenFilter("keepWord", "words", "keep.txt", "ignoreCase", "true")
                .build()
        );
        m.put(
            "uax29, elision, common grams, kstem",
            b()
                .withTokenizer("uax29UrlEmail", "maxTokenLength", "40")
                .addTokenFilter("elision", "articles", "articles.txt", "ignoreCase", "true")
                .addTokenFilter("lowercase")
                .addTokenFilter("commonGrams", "words", "common.txt")
                .addTokenFilter("kStem")
                .build()
        );
        m.put(
            "keyword repeat, porter, remove duplicates",
            b()
                .withTokenizer("standard")
                .addTokenFilter("lowercase")
                .addTokenFilter("keywordRepeat")
                .addTokenFilter("porterStem")
                .addTokenFilter("removeDuplicates")
                .build()
        );
        m.put(
            "language filters",
            b()
                .withTokenizer("standard")
                .addTokenFilter("decimalDigit")
                .addTokenFilter("apostrophe")
                .addTokenFilter("turkishLowercase")
                .addTokenFilter("germanNormalization")
                .addTokenFilter("scandinavianFolding")
                .addTokenFilter("arabicNormalization")
                .addTokenFilter("indicNormalization")
                .addTokenFilter("frenchLightStem")
                .addTokenFilter("englishPossessive")
                .addTokenFilter("englishMinimalStem")
                .build()
        );
        m.put("keyword tokenizer, ascii folding", b().withTokenizer("keyword").addTokenFilter("asciiFolding").build());
        return m;
    }

    /** Analyzers that must stay in Java, with the reason they must give. */
    static Map<String, Map.Entry<Analyzer, String>> refused() throws Exception {
        Map<String, Map.Entry<Analyzer, String>> m = new LinkedHashMap<>();
        // The Rust port has no dictionary segmentation for Thai.
        m.put("thai", Map.entry(new ThaiAnalyzer(), "analysis_ThaiTokenizer"));
        // PersianCharFilter is not mapped.
        m.put("persian", Map.entry(new PersianAnalyzer(), "analysis_PersianCharFilter"));
        m.put(
            "word delimiter with a type table",
            Map.entry(
                b().withTokenizer("whitespace").addTokenFilter("wordDelimiterGraph", "types", "wdtypes.txt").build(),
                "analysis_word_delimiter_types"
            )
        );
        return m;
    }

    private static final String[] WORDS = {
        "the", "The", "QUICK", "fox's", "jumps", "running", "Jumps", "e-mail", "Wi-Fi", "PowerShot500", "3.14", "1,234", "42-7",
        "l'avion", "d'Artagnan", "O'Neil", "café", "naïve", "Straße", "İstanbul", "ΣΊΣΥΦΟΣ", "Москва", "東京都", "서울", "カタカナ",
        "ﾊﾝｶｸ", "ＦＵＬＬ", "ภาษาไทย", "العربية", "हिन्दी", "😀", "👍🏽", "ab😀cd", "<b>", "</b>", "<p>", "&amp;", "&eacute;", "<br/>",
        "<script>", "</script>", "https://example.org/a?b=c", "user@example.com", "/usr/local/lib", "a/b/c/d", "'quoted'", "aaa",
        "banana", "of", "and", "to", "is", ";", ",", ".", "!", "?", "-", "_", "#tag", "@who", "x", "supercalifragilistic", "٣٤٥",
    };

    private static final String[] SPACES = { " ", " ", " ", "  ", "\t", "\n", " ", "　", "", ", " };

    static String randomText(Random r) {
        StringBuilder sb = new StringBuilder();
        int n = r.nextInt(40);
        for (int i = 0; i < n; i++) {
            sb.append(WORDS[r.nextInt(WORDS.length)]).append(SPACES[r.nextInt(SPACES.length)]);
        }
        return sb.toString();
    }

    /** Where two token streams part, for a failure message. */
    static String difference(AnalyzerSpec.Tokens java, AnalyzerSpec.Tokens rust) {
        if (rust == null) {
            return ": Rust failed: " + NativeBridge.lastError();
        }
        int n = Math.min(java.tokens().size(), rust.tokens().size());
        for (int i = 0; i < n; i++) {
            if (java.tokens().get(i).equals(rust.tokens().get(i)) == false) {
                return "\n  token " + i + ": java " + show(java.tokens().get(i)) + "\n  token " + i + ": rust " + show(rust.tokens().get(i));
            }
        }
        if (java.tokens().size() != rust.tokens().size()) {
            return "\n  java has " + java.tokens().size() + " tokens, rust " + rust.tokens().size()
                + (java.tokens().size() > n ? ", next java " + show(java.tokens().get(n)) : ", next rust " + show(rust.tokens().get(n)));
        }
        return "\n  end: java " + java.endOffset() + "/" + java.endPositionIncrement() + ", rust " + rust.endOffset() + "/" + rust.endPositionIncrement();
    }

    private static String show(AnalyzerSpec.Token t) {
        return t.term().utf8ToString() + " " + t;
    }

    static void run(Random r, BiConsumer<Boolean, String> check) throws Exception {
        for (Map.Entry<String, Analyzer> e : analyzers().entrySet()) {
            String name = e.getKey();
            Analyzer a = e.getValue();
            AnalyzerSpec.Spec spec = AnalyzerSpec.describeUnchecked(a, "body");
            check.accept(spec.ok(), "analyzer described: " + name + " (" + spec.reason() + ")");
            if (spec.ok() == false) continue;
            described++;
            List<String> texts = new ArrayList<>(AnalyzerSpec.PROBES);
            for (int i = 0; i < 60; i++) texts.add(randomText(r));
            for (String text : texts) {
                AnalyzerSpec.Tokens java = AnalyzerSpec.java(a, "body", text);
                AnalyzerSpec.Tokens rust = AnalyzerSpec.rust(spec.blob(), "body", text);
                boolean same = java.equals(rust);
                check.accept(same, "analysis matches Lucene: " + name + " on [" + text + "]" + (same ? "" : difference(java, rust)));
                compared++;
                if (same == false) break;
            }
            check.accept(AnalyzerSpec.describe(a, "body").ok(), "analyzer passes its probe check: " + name);
        }
        for (Map.Entry<String, Map.Entry<Analyzer, String>> e : refused().entrySet()) {
            AnalyzerSpec.Spec spec = AnalyzerSpec.describe(e.getValue().getKey(), "body");
            check.accept(
                spec.ok() == false && e.getValue().getValue().equals(spec.reason()),
                "analyzer left in Java: " + e.getKey() + " (" + spec.reason() + ")"
            );
        }
        check.accept(AnalyzerSpec.encodable("a😀b"), "a surrogate pair is encodable");
        check.accept(AnalyzerSpec.encodable("a\ud83db") == false, "a lone surrogate is not");
    }
}
