/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.ar.ArabicAnalyzer;
import org.apache.lucene.analysis.bg.BulgarianAnalyzer;
import org.apache.lucene.analysis.br.BrazilianAnalyzer;
import org.apache.lucene.analysis.cjk.CJKAnalyzer;
import org.apache.lucene.analysis.classic.ClassicAnalyzer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
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
import org.apache.lucene.analysis.hunspell.Dictionary;
import org.apache.lucene.analysis.hunspell.HunspellStemFilter;
import org.apache.lucene.analysis.it.ItalianAnalyzer;
import org.apache.lucene.analysis.pt.PortugueseAnalyzer;
import org.apache.lucene.analysis.ru.RussianAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.analysis.th.ThaiAnalyzer;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
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
                "wdtypes.txt", "- => ALPHA\n",
                "synonyms.txt", "quick, fast, speedy\njumps => leaps, hops over\nthe fox => the vixen\nbanana, ba\\,na\ne mail, email\ncafé => coffee shop\nwi fi, wifi, wireless\n",
                "override.txt", "running\trun\njumps\tjump\ncafé\tcoffee\nbanana\tfruit thing\n"
            );
            for (Map.Entry<String, String> f : files.entrySet()) {
                java.nio.file.Files.writeString(resources.resolve(f.getKey()), f.getValue());
            }
        }
        return CustomAnalyzer.builder(resources);
    }

    // The e2e runs' dictionary (opensearch-plugin/e2e/hunspell/en_US), inline.
    private static final String EN_AFF = "SET UTF-8\nTRY esianrtolcdugmphbyfvkwz\n\nPFX U Y 1\nPFX U 0 un .\n\n"
        + "SFX S Y 2\nSFX S 0 s [^sxz]\nSFX S 0 es [sxz]\n\nSFX D Y 2\nSFX D 0 d e\nSFX D 0 ed [^e]\n\n"
        + "SFX G Y 2\nSFX G e ing e\nSFX G 0 ing [^e]\n\nSFX N Y 2\nSFX N 0 ning n\nSFX N 0 ner/S n\n";
    private static final String EN_DIC = "14\nthe\nquick\nbrown\nfox/S\njump/SDG\nover\nlazy\ndog/S\ncat/S\nsleep/SG\n"
        + "run/N\nran st:run\nhouse/SDG\nhappy/U\n";
    private static final String EXTRA_DIC = "3\nwhile\nalpha/S\nrunner/S\n";
    // ISO-8859-1 with a one-letter UTF-8 flag (fixtures/corpus/hunspell/latin1.*).
    private static final String LATIN1_AFF = "SET ISO8859-1\nFLAG UTF-8\nTRY e\n\nSFX \u00c3\u00a9 Y 1\nSFX \u00c3\u00a9 0 \u00e9 .\n";
    private static final String LATIN1_DIC = "2\ncaf/\u00c3\u00a9\nna\u00efve\n";

    /** A dictionary parsed from these bytes, its files recorded for {@link AnalyzerSpec} unless {@code record} is false. */
    private static Dictionary dictionary(byte[] aff, List<byte[]> dics, boolean ignoreCase, boolean record) throws Exception {
        List<InputStream> streams = new ArrayList<>();
        for (byte[] d : dics) {
            streams.add(new ByteArrayInputStream(d));
        }
        Dictionary d = new Dictionary(new ByteBuffersDirectory(), "hunspell", new ByteArrayInputStream(aff), streams, ignoreCase);
        if (record) {
            AnalyzerSpec.hunspellFiles(d, new AnalyzerSpec.HunspellFiles(aff, dics));
        }
        return d;
    }

    /** {@code tokenizer}, optionally lowercased, then hunspell over {@code d}. */
    private static Analyzer hunspell(boolean whitespace, boolean lowercase, Dictionary d, boolean dedup, boolean longestOnly) {
        return new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String field) {
                Tokenizer t = whitespace ? new WhitespaceTokenizer() : new StandardTokenizer();
                TokenStream s = lowercase ? new LowerCaseFilter(t) : t;
                return new TokenStreamComponents(t, new HunspellStemFilter(s, d, dedup, longestOnly));
            }
        };
    }

    private static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** Analyzers that must be described, by name. */
    static Map<String, Analyzer> analyzers() throws Exception {
        Map<String, Analyzer> m = new LinkedHashMap<>();
        m.put("standard", new StandardAnalyzer());
        m.put("rust_thai", new RustThaiAnalysis.RustThaiAnalyzer(ThaiAnalyzer.getDefaultStopSet()));
        m.put("rust_thai, html strip, no stop words", new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String fieldName) {
                return new TokenStreamComponents(new RustThaiTokenizer());
            }

            @Override
            protected java.io.Reader initReader(String fieldName, java.io.Reader reader) {
                return new org.apache.lucene.analysis.charfilter.HTMLStripCharFilter(reader);
            }
        });
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
            "synonym graph, flatten",
            b()
                .withTokenizer("standard")
                .addTokenFilter("lowercase")
                .addTokenFilter("synonymGraph", "synonyms", "synonyms.txt", "expand", "true")
                .addTokenFilter("flattenGraph")
                .build()
        );
        m.put(
            "synonym graph without expand, ignore case",
            b()
                .withTokenizer("whitespace")
                .addTokenFilter("synonymGraph", "synonyms", "synonyms.txt", "expand", "false", "ignoreCase", "true")
                .build()
        );
        m.put(
            "synonym (old filter), stemmer override",
            b()
                .withTokenizer("classic")
                .addTokenFilter("lowercase")
                .addTokenFilter("stemmerOverride", "dictionary", "override.txt", "ignoreCase", "true")
                .addTokenFilter("synonym", "synonyms", "synonyms.txt")
                .addTokenFilter("porterStem")
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
        List<byte[]> en = List.of(utf8(EN_DIC), utf8(EXTRA_DIC));
        m.put("hunspell (two .dic files, stem exception)", hunspell(false, true, dictionary(utf8(EN_AFF), en, false, true), true, false));
        m.put(
            "hunspell ignore case, longest only, no dedup",
            hunspell(true, false, dictionary(utf8(EN_AFF), en, true, true), false, true)
        );
        m.put(
            "hunspell ISO-8859-1",
            hunspell(
                false,
                false,
                dictionary(
                    LATIN1_AFF.getBytes(StandardCharsets.ISO_8859_1),
                    List.of(LATIN1_DIC.getBytes(StandardCharsets.ISO_8859_1)),
                    false,
                    true
                ),
                true,
                false
            )
        );
        return m;
    }

    /** Analyzers that must stay in Java, with the reason they must give. */
    static Map<String, Map.Entry<Analyzer, String>> refused() throws Exception {
        Map<String, Map.Entry<Analyzer, String>> m = new LinkedHashMap<>();
        // The Rust port has no dictionary segmentation for Thai.
        m.put("thai", Map.entry(new ThaiAnalyzer(), "analysis_ThaiTokenizer"));
        // A dictionary with no known files (a plugin's own, say).
        m.put(
            "hunspell without its files",
            Map.entry(hunspell(false, true, dictionary(utf8(EN_AFF), List.of(utf8(EN_DIC)), false, false), true, false), "analysis_hunspell")
        );
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
        "banana", "fox", "fast", "wifi", "runners", "Running", "houses", "housing", "ran", "unhappy", "Jumping", "dogs", "caf", "alphas", "of", "and", "to", "is", ";", ",", ".", "!", "?", "-", "_", "#tag", "@who", "x", "supercalifragilistic", "٣٤٥",
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

    /** Thai sentences, for how often rust_thai's words are Lucene's (the JDK dictionary's). */
    static final String[] THAI = {
        "การที่ได้ต้องแสดงว่างานดี",
        "ภาษาไทยง่ายนิดเดียว",
        "ประเทศไทยมีประชากรประมาณหกสิบหกล้านคน",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "ฉันชอบกินข้าวผัดกับไข่ดาว",
        "นักเรียนกำลังอ่านหนังสือในห้องสมุด",
        "รถไฟฟ้าวิ่งเร็วมากในตอนเช้า",
        "โปรแกรมคอมพิวเตอร์ทำงานผิดพลาด",
        "เขาไปเที่ยวทะเลเมื่อวานนี้",
        "ราคาน้ำมันขึ้นอีกแล้ว ๑๒๓ บาท",
    };

    /** Lucene's ThaiAnalyzer and rust_thai side by side: Lucene's test words, and the agreement. */
    static void thai(BiConsumer<Boolean, String> check) throws Exception {
        Analyzer rust = new RustThaiAnalysis.RustThaiAnalyzer(ThaiAnalyzer.getDefaultStopSet());
        // TestThaiAnalyzer.testOffsets and testStopWords: the JDK's and ICU's words agree here.
        List<String> want = List.of("แสดง", "งาน", "ดี");
        List<String> got = AnalyzerSpec.java(rust, "body", THAI[0]).tokens().stream().map(t -> t.term().utf8ToString()).toList();
        check.accept(want.equals(got), "rust_thai splits Lucene's test sentence as Lucene does: " + got);
        // The tokenizers alone: the same text kept (stop words aside, which drop different words
        // when the splits differ), split the same way or not.
        Analyzer luceneWords = new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String fieldName) {
                return new TokenStreamComponents(new org.apache.lucene.analysis.th.ThaiTokenizer());
            }
        };
        Analyzer rustWords = new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String fieldName) {
                return new TokenStreamComponents(new RustThaiTokenizer());
            }
        };
        int same = 0;
        int total = 0;
        for (String text : THAI) {
            List<String> j = AnalyzerSpec.java(luceneWords, "body", text).tokens().stream().map(t -> t.term().utf8ToString()).toList();
            List<String> r = AnalyzerSpec.java(rustWords, "body", text).tokens().stream().map(t -> t.term().utf8ToString()).toList();
            check.accept(String.join("", j).equals(String.join("", r)), "rust_thai keeps the text Lucene keeps: " + j + " vs " + r);
            total++;
            if (j.equals(r)) same++;
        }
        System.out.printf("NativeSelfTest: rust_thai splits %d of %d Thai sentences as Lucene's ThaiAnalyzer does%n", same, total);
        check.accept("ab\ufffdc".equals(RustThaiTokenizer.wellFormed("ab\ud800c")), "a lone surrogate becomes U+FFFD");
        check.accept("a😀b".equals(RustThaiTokenizer.wellFormed("a😀b")), "a surrogate pair stays");
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
        thai(check);
        check.accept(AnalyzerSpec.encodable("a😀b"), "a surrogate pair is encodable");
        check.accept(AnalyzerSpec.encodable("a\ud83db") == false, "a lone surrogate is not");
    }
}
