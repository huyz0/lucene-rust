/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.CharFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.miscellaneous.StemmerOverrideFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.analysis.tokenattributes.FlagsAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.IntsRefFSTEnum;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Describes a Lucene {@link Analyzer} as the spec {@code jvm_analysis.rs} builds its own copy from:
 * the char filters, tokenizer and token filters of the chain the analyzer really builds for a
 * field, each by its Lucene SPI name and the factory arguments that rebuild it, recovered from the
 * objects themselves. This covers Lucene's own analyzers ({@code standard}, the language
 * analyzers) and OpenSearch's {@code custom} ones alike, since both end in Lucene objects.
 *
 * <p>Only exact classes in {@link #MAPPERS} are described; anything else (an OpenSearch class, a
 * subclass such as the {@code ngram} tokenizer with {@code token_chars}, a setting with no factory
 * argument) leaves the analyzer in Java with a reason. A description is then checked against the
 * analyzer itself over {@link #PROBES}: every attribute of every token, and the end state, must
 * match, or the analyzer stays in Java ({@code analysis_mismatch}).
 */
public final class AnalyzerSpec {
    /** The layout version {@code jvm_analysis.rs} expects. */
    static final int VERSION = 1;

    /** A description, or why there is none. */
    public record Spec(byte[] blob, String reason) {
        public boolean ok() {
            return blob != null;
        }
    }

    /** One token, with every attribute the Rust side reports. */
    public record Token(
        BytesRef term,
        int positionIncrement,
        int positionLength,
        int startOffset,
        int endOffset,
        String type,
        int flags,
        boolean keyword,
        int termFrequency,
        BytesRef payload
    ) {}

    /** A stream's tokens and what {@code end()} left. */
    public record Tokens(List<Token> tokens, int endOffset, int endPositionIncrement) {}

    /** Texts every new description is checked on: scripts, markup, punctuation, long tokens. */
    static final List<String> PROBES = List.of(
        "The QUICK brown fox's jumps over the lazy dogs, isn't it? 3.14 x-ray Wi-Fi PowerShot500 e-mail@example.com https://example.org/a/b?c=d",
        "<p>Caf&eacute; na&iuml;ve r&eacute;sum&eacute; &amp; <b>bold</b> <script>x()</script> <br/>&#x263a; tags</p>",
        "Ünïcödé façade Straße ΣΊΣΥΦΟΣ İstanbul Iğdır l'avion d'Artagnan O'Neil \u00a0nbsp\u3000ideographic\ttab\nline",
        "東京都に住んでいます。서울특별시 中华人民共和国 カタカナ ﾊﾝｶｸ ＦＵＬＬＷＩＤＴＨ ภาษาไทยง่ายนิดเดียว العربية فارسی हिन्दी বাংলা",
        "/usr/local/lib/opensearch c:\\windows\\path 2024-10-30T04:29:41Z 1,234,567.89 #hashtag @mention 😀👍🏽 ab😀cd",
        "supercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocious end",
        "running runs ran easily fairly generously happiness the of and a an to in is it that was for on are as with",
        "",
        "   "
    );

    private AnalyzerSpec() {}

    /**
     * {@code analyzer}'s chain for {@code field}, described and checked against the analyzer on the
     * probes; the Rust copy is built (and kept) on the way.
     */
    public static Spec describe(Analyzer analyzer, String field) {
        Spec spec = describeUnchecked(analyzer, field);
        if (spec.ok() == false) {
            return spec;
        }
        try {
            for (String probe : PROBES) {
                Tokens java = java(analyzer, field, probe);
                Tokens rust = rust(spec.blob(), field, probe);
                if (rust == null) {
                    return new Spec(null, "analysis_spec");
                }
                if (java.equals(rust) == false) {
                    return new Spec(null, "analysis_mismatch");
                }
            }
        } catch (IOException | RuntimeException e) {
            return new Spec(null, "analysis_mismatch");
        }
        return spec;
    }

    /** {@link #describe} without the probe check, for the self-test that runs its own. */
    public static Spec describeUnchecked(Analyzer analyzer, String field) {
        List<Object> filters = new ArrayList<>();
        List<Object> charFilters = new ArrayList<>();
        Object tokenizer;
        try (TokenStream ts = analyzer.tokenStream(field, "")) {
            TokenStream s = ts;
            while (s instanceof TokenFilter f) {
                filters.add(0, f);
                s = (TokenStream) read(f, TokenFilter.class, "input");
            }
            if (s instanceof Tokenizer == false) {
                return new Spec(null, "analysis_source");
            }
            tokenizer = s;
            // The reader setReader handed over, before reset() makes it the input: the char
            // filters initReader wrapped around the text.
            Object r = read(s, Tokenizer.class, "inputPending");
            while (r instanceof CharFilter cf) {
                charFilters.add(0, cf);
                r = read(cf, CharFilter.class, "input");
            }
            ts.reset();
            while (ts.incrementToken()) {}
            ts.end();
        } catch (ReflectionMiss e) {
            return new Spec(null, "analysis_reflect");
        } catch (IOException | RuntimeException e) {
            return new Spec(null, "analysis_reflect");
        }
        Writer w = new Writer();
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(VERSION);
            writeInt(out, analyzer.getPositionIncrementGap(field));
            writeInt(out, analyzer.getOffsetGap(field));
            writeInt(out, charFilters.size());
            for (Object cf : charFilters) {
                String reason = w.component(out, cf);
                if (reason != null) return new Spec(null, reason);
            }
            String reason = w.component(out, tokenizer);
            if (reason != null) return new Spec(null, reason);
            writeInt(out, filters.size());
            for (Object f : filters) {
                reason = w.component(out, f);
                if (reason != null) return new Spec(null, reason);
            }
            writeInt(out, w.resources.size());
            for (Map.Entry<String, String> e : w.resources.entrySet()) {
                writeBytes(out, e.getKey().getBytes(StandardCharsets.UTF_8));
                writeBytes(out, e.getValue().getBytes(StandardCharsets.UTF_8));
            }
            return new Spec(out.toByteArray(), null);
        } catch (ReflectionMiss e) {
            return new Spec(null, "analysis_reflect");
        } catch (RuntimeException e) {
            return new Spec(null, "analysis_reflect");
        }
    }

    /** Whether {@code text} survives UTF-8 unchanged: no unpaired surrogate. */
    public static boolean encodable(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (i + 1 == text.length() || Character.isLowSurrogate(text.charAt(i + 1)) == false) return false;
                i++;
            } else if (Character.isLowSurrogate(c)) {
                return false;
            }
        }
        return true;
    }

    /** Lucene's own tokens for {@code text}, as a consumer drains them. */
    public static Tokens java(Analyzer analyzer, String field, String text) throws IOException {
        List<Token> tokens = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream(field, text)) {
            TermToBytesRefAttribute term = ts.getAttribute(TermToBytesRefAttribute.class);
            PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
            PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
            OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
            TypeAttribute type = ts.addAttribute(TypeAttribute.class);
            FlagsAttribute flags = ts.addAttribute(FlagsAttribute.class);
            KeywordAttribute kw = ts.addAttribute(KeywordAttribute.class);
            TermFrequencyAttribute tf = ts.addAttribute(TermFrequencyAttribute.class);
            PayloadAttribute payload = ts.addAttribute(PayloadAttribute.class);
            ts.reset();
            while (ts.incrementToken()) {
                BytesRef p = payload.getPayload();
                tokens.add(
                    new Token(
                        BytesRef.deepCopyOf(term.getBytesRef()),
                        inc.getPositionIncrement(),
                        len.getPositionLength(),
                        off.startOffset(),
                        off.endOffset(),
                        type.type(),
                        flags.getFlags(),
                        kw.isKeyword(),
                        tf.getTermFrequency(),
                        p == null ? null : BytesRef.deepCopyOf(p)
                    )
                );
            }
            ts.end();
            return new Tokens(tokens, off.endOffset(), inc.getPositionIncrement());
        }
    }

    /** The Rust copy's tokens for {@code text}, or null when it failed (the reason is logged natively). */
    public static Tokens rust(byte[] spec, String field, String text) {
        byte[][] out = new byte[1][];
        int rc = NativeBridge.analyze(
            spec,
            field.getBytes(StandardCharsets.UTF_8),
            text.getBytes(StandardCharsets.UTF_8),
            out
        );
        if (rc != 0) {
            return null;
        }
        return decode(out[0]);
    }

    /** {@code jvm_analysis::analyze}'s layout. */
    static Tokens decode(byte[] blob) {
        ByteBuffer b = ByteBuffer.wrap(blob == null ? new byte[8] : blob).order(ByteOrder.LITTLE_ENDIAN);
        int n = blob == null ? 0 : b.getInt();
        List<Token> tokens = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            BytesRef term = bytes(b);
            int inc = b.getInt();
            int len = b.getInt();
            int start = b.getInt();
            int end = b.getInt();
            String type = bytes(b).utf8ToString();
            int flags = b.getInt();
            boolean kw = b.get() != 0;
            int tf = b.getInt();
            int pl = b.getInt();
            BytesRef payload = null;
            if (pl >= 0) {
                byte[] p = new byte[pl];
                b.get(p);
                payload = new BytesRef(p);
            }
            tokens.add(new Token(term, inc, len, start, end, type, flags, kw, tf, payload));
        }
        return new Tokens(tokens, b.getInt(), b.getInt());
    }

    private static BytesRef bytes(ByteBuffer b) {
        byte[] v = new byte[b.getInt()];
        b.get(v);
        return new BytesRef(v);
    }

    // ---- Reflection ----

    /** A field the description needs could not be read. */
    private static final class ReflectionMiss extends RuntimeException {
        ReflectionMiss(String what) {
            super(what, null, false, false);
        }
    }

    private static Object read(Object o, Class<?> declaring, String name) {
        Field f = Reflect.declared(declaring, name);
        if (f == null) throw new ReflectionMiss(declaring.getName() + "." + name);
        try {
            return f.get(o);
        } catch (IllegalAccessException e) {
            throw new ReflectionMiss(declaring.getName() + "." + name);
        }
    }

    private static Object read(Object o, String name) {
        Field f = Reflect.inHierarchy(o.getClass(), name);
        if (f == null) throw new ReflectionMiss(o.getClass().getName() + "." + name);
        try {
            return f.get(o);
        } catch (IllegalAccessException e) {
            throw new ReflectionMiss(o.getClass().getName() + "." + name);
        }
    }

    private static int readInt(Object o, String name) {
        return (Integer) read(o, name);
    }

    private static boolean readBool(Object o, String name) {
        return (Boolean) read(o, name);
    }

    private static String flag(boolean b) {
        return b ? "true" : "false";
    }

    // ---- The mapping ----

    /** One component's SPI name and arguments, filled in by a {@link Mapper}. */
    static final class Component {
        String name;
        final Map<String, String> args = new LinkedHashMap<>();
        final Writer writer;

        Component(Writer writer) {
            this.writer = writer;
        }

        Component arg(String key, String value) {
            args.put(key, value);
            return this;
        }

        Component arg(String key, int value) {
            return arg(key, Integer.toString(value));
        }

        /**
         * A word list argument: the set's words as an inline resource, and its {@code ignoreCase}.
         * Null when a word cannot survive {@code getWordSet}'s line format.
         */
        String words(String key, CharArraySet set, String ignoreCaseKey) {
            StringBuilder sb = new StringBuilder();
            for (Object o : set) {
                String word = new String((char[]) o);
                if (word.isEmpty()
                    || word.startsWith("#")
                    || word.indexOf('\n') >= 0
                    || word.indexOf('\r') >= 0
                    || word.charAt(0) <= ' '
                    || word.charAt(word.length() - 1) <= ' '
                    || word.startsWith("\uFEFF")) {
                    return "analysis_words";
                }
                sb.append(word).append('\n');
            }
            arg(key, writer.resource(sb.toString()));
            if (ignoreCaseKey != null) {
                arg(ignoreCaseKey, flag(ignoreCase(set)));
            }
            return null;
        }
    }

    /**
     * A synonym filter's map, entry by entry ({@code lucene_rust.synonym_entries}): each input with
     * its {@code keepOrig} and outputs in order, as the FST holds them, so the Rust map is this one.
     */
    private static String synonyms(Object o, Component c, String name) {
        SynonymMap map = (SynonymMap) read(o, "synonyms");
        StringBuilder sb = new StringBuilder();
        try {
            // The FST's inputs are code points (FST.INPUT_TYPE.BYTE4).
            IntsRefFSTEnum<BytesRef> e = new IntsRefFSTEnum<>(map.fst);
            IntsRefFSTEnum.InputOutput<BytesRef> io;
            ByteArrayDataInput in = new ByteArrayDataInput();
            BytesRef word = new BytesRef();
            while ((io = e.next()) != null) {
                in.reset(io.output.bytes, io.output.offset, io.output.length);
                int code = in.readVInt();
                sb.append((code & 1) == 0 ? '1' : '0').append('\t');
                hex(sb, new BytesRef(new String(io.input.ints, io.input.offset, io.input.length)));
                for (int i = code >>> 1; i > 0; i--) {
                    map.words.get(in.readVInt(), word);
                    sb.append('\t');
                    hex(sb, word);
                }
                sb.append('\n');
            }
        } catch (IOException | RuntimeException ex) {
            return "analysis_synonyms";
        }
        c.name = name;
        c.arg("synonyms", c.writer.resource(sb.toString()))
            .arg("format", "lucene_rust.synonym_entries")
            .arg("ignoreCase", flag(readBool(o, "ignoreCase")));
        return null;
    }

    private static void hex(StringBuilder sb, BytesRef b) {
        for (int i = 0; i < b.length; i++) {
            int v = b.bytes[b.offset + i] & 0xFF;
            sb.append(Character.forDigit(v >>> 4, 16)).append(Character.forDigit(v & 15, 16));
        }
    }

    private static boolean ignoreCase(CharArraySet set) {
        return readBool(read(set, CharArraySet.class, "map"), "ignoreCase");
    }

    /** Fills in a component, or answers why it cannot. */
    @FunctionalInterface
    interface Mapper {
        String map(Object component, Component out);
    }

    /** Collects the inline resources while components are written. */
    static final class Writer {
        final Map<String, String> resources = new LinkedHashMap<>();

        String resource(String content) {
            String name = "r" + resources.size() + ".txt";
            resources.put(name, content);
            return name;
        }

        String component(ByteArrayOutputStream out, Object o) {
            Mapper m = MAPPERS.get(o.getClass().getName());
            if (m == null) {
                m = noArgs(o.getClass());
                if (m == null) {
                    return "analysis_" + o.getClass().getSimpleName();
                }
            }
            Component c = new Component(this);
            String reason = m.map(o, c);
            if (reason != null) {
                return reason;
            }
            writeBytes(out, c.name.getBytes(StandardCharsets.UTF_8));
            writeInt(out, c.args.size());
            for (Map.Entry<String, String> e : c.args.entrySet()) {
                writeBytes(out, e.getKey().getBytes(StandardCharsets.UTF_8));
                writeBytes(out, e.getValue().getBytes(StandardCharsets.UTF_8));
            }
            return null;
        }
    }

    private static final String A = "org.apache.lucene.analysis.";

    /**
     * Components with no settings at all: their SPI name is their factory's ({@code <class>Factory}
     * beside them, Lucene's naming for every one of these).
     */
    private static final Set<String> NO_ARGS = Set.of(
        A + "LowerCaseFilter",
        A + "core.LowerCaseFilter",
        A + "core.UpperCaseFilter",
        A + "core.DecimalDigitFilter",
        A + "core.FlattenGraphFilter",
        A + "en.PorterStemFilter",
        A + "en.KStemFilter",
        A + "en.EnglishPossessiveFilter",
        A + "en.EnglishMinimalStemFilter",
        A + "miscellaneous.TrimFilter",
        A + "miscellaneous.RemoveDuplicatesTokenFilter",
        A + "miscellaneous.KeywordRepeatFilter",
        A + "miscellaneous.ScandinavianFoldingFilter",
        A + "miscellaneous.HyphenatedWordsFilter",
        A + "cjk.CJKWidthFilter",
        A + "classic.ClassicFilter",
        A + "tr.ApostropheFilter",
        A + "tr.TurkishLowerCaseFilter",
        A + "el.GreekLowerCaseFilter",
        A + "el.GreekStemFilter",
        A + "ga.IrishLowerCaseFilter",
        A + "de.GermanNormalizationFilter",
        A + "de.GermanStemFilter",
        A + "de.GermanLightStemFilter",
        A + "de.GermanMinimalStemFilter",
        A + "fr.FrenchLightStemFilter",
        A + "fr.FrenchMinimalStemFilter",
        A + "es.SpanishLightStemFilter",
        A + "es.SpanishPluralStemFilter",
        A + "pt.PortugueseLightStemFilter",
        A + "pt.PortugueseMinimalStemFilter",
        A + "pt.PortugueseStemFilter",
        A + "it.ItalianLightStemFilter",
        A + "ru.RussianLightStemFilter",
        A + "ar.ArabicNormalizationFilter",
        A + "ar.ArabicStemFilter",
        A + "fa.PersianNormalizationFilter",
        A + "hi.HindiNormalizationFilter",
        A + "hi.HindiStemFilter",
        A + "in.IndicNormalizationFilter",
        A + "bn.BengaliNormalizationFilter",
        A + "bn.BengaliStemFilter",
        A + "ckb.SoraniNormalizationFilter",
        A + "ckb.SoraniStemFilter",
        A + "cz.CzechStemFilter",
        A + "br.BrazilianStemFilter",
        A + "lv.LatvianStemFilter",
        A + "bg.BulgarianStemFilter",
        A + "gl.GalicianStemFilter",
        A + "gl.GalicianMinimalStemFilter",
        A + "sv.SwedishLightStemFilter",
        A + "fi.FinnishLightStemFilter",
        A + "hu.HungarianLightStemFilter",
        A + "core.KeywordTokenizer"
    );

    private static Mapper noArgs(Class<?> c) {
        if (NO_ARGS.contains(c.getName()) == false) {
            return null;
        }
        String spi = spiName(c);
        if (spi == null) {
            return null;
        }
        return (o, out) -> {
            out.name = spi;
            return null;
        };
    }

    private static String spiName(Class<?> c) {
        String name = c.getName();
        // The core.* subclasses of the moved classes share their factory.
        if (name.equals(A + "LowerCaseFilter")) name = A + "core.LowerCaseFilter";
        try {
            Class<?> factory = Class.forName(name + "Factory", false, c.getClassLoader());
            return (String) factory.getField("NAME").get(null);
        } catch (ReflectiveOperationException | RuntimeException | LinkageError e) {
            return null;
        }
    }

    private static final Map<String, Mapper> MAPPERS = new LinkedHashMap<>();

    private static void map(String className, Mapper m) {
        MAPPERS.put(A + className, m);
    }

    /** A pattern with no flags Lucene's factories cannot set. */
    private static String plainPattern(Pattern p) {
        return p.flags() == 0 ? p.pattern() : null;
    }

    static {
        // ---- Tokenizers ----
        map("standard.StandardTokenizer", (o, c) -> {
            c.name = "standard";
            c.arg("maxTokenLength", ((org.apache.lucene.analysis.standard.StandardTokenizer) o).getMaxTokenLength());
            return null;
        });
        map("classic.ClassicTokenizer", (o, c) -> {
            c.name = "classic";
            c.arg("maxTokenLength", ((org.apache.lucene.analysis.classic.ClassicTokenizer) o).getMaxTokenLength());
            return null;
        });
        map("email.UAX29URLEmailTokenizer", (o, c) -> {
            c.name = "uax29UrlEmail";
            c.arg("maxTokenLength", ((org.apache.lucene.analysis.email.UAX29URLEmailTokenizer) o).getMaxTokenLength());
            return null;
        });
        map("core.WhitespaceTokenizer", (o, c) -> {
            c.name = "whitespace";
            c.arg("rule", "java").arg("maxTokenLen", readInt(o, "maxTokenLen"));
            return null;
        });
        map("core.UnicodeWhitespaceTokenizer", (o, c) -> {
            c.name = "whitespace";
            c.arg("rule", "unicode").arg("maxTokenLen", readInt(o, "maxTokenLen"));
            return null;
        });
        map("core.LetterTokenizer", (o, c) -> {
            c.name = "letter";
            c.arg("maxTokenLen", readInt(o, "maxTokenLen"));
            return null;
        });
        Mapper ngramTokenizer = (o, c) -> {
            boolean edges = readBool(o, "edgesOnly");
            c.name = edges ? "edgeNGram" : "nGram";
            c.arg("minGramSize", readInt(o, "minGram")).arg("maxGramSize", readInt(o, "maxGram"));
            return null;
        };
        map("ngram.NGramTokenizer", ngramTokenizer);
        map("ngram.EdgeNGramTokenizer", ngramTokenizer);
        map("path.PathHierarchyTokenizer", (o, c) -> {
            c.name = "pathHierarchy";
            c.arg("delimiter", String.valueOf((char) read(o, "delimiter")))
                .arg("replace", String.valueOf((char) read(o, "replacement")))
                .arg("skip", readInt(o, "skip"));
            return null;
        });
        map("path.ReversePathHierarchyTokenizer", (o, c) -> {
            c.name = "pathHierarchy";
            c.arg("delimiter", String.valueOf((char) read(o, "delimiter")))
                .arg("replace", String.valueOf((char) read(o, "replacement")))
                .arg("skip", readInt(o, "skip"))
                .arg("reverse", "true");
            return null;
        });
        map("pattern.PatternTokenizer", (o, c) -> {
            String p = plainPattern(((Matcher) read(o, "matcher")).pattern());
            if (p == null) return "analysis_pattern_flags";
            c.name = "pattern";
            c.arg("pattern", p).arg("group", readInt(o, "group"));
            return null;
        });

        // ---- Char filters ----
        map("charfilter.HTMLStripCharFilter", (o, c) -> {
            c.name = "htmlStrip";
            // The constructor keeps BR, SCRIPT and STYLE as flags, the other tags in the set.
            List<String> tags = new ArrayList<>();
            if (readBool(o, "escapeBR")) tags.add("br");
            if (readBool(o, "escapeSCRIPT")) tags.add("script");
            if (readBool(o, "escapeSTYLE")) tags.add("style");
            CharArraySet set = (CharArraySet) read(o, "escapedTags");
            if (set != null) {
                for (Object t : set) {
                    String tag = new String((char[]) t);
                    if (tag.isEmpty() || tag.chars().anyMatch(ch -> ch == ',' || Character.isWhitespace(ch))) {
                        return "analysis_words";
                    }
                    tags.add(tag);
                }
            }
            if (tags.isEmpty() == false) {
                c.arg("escapedTags", String.join(",", tags));
            }
            return null;
        });
        map("pattern.PatternReplaceCharFilter", (o, c) -> {
            String p = plainPattern((Pattern) read(o, "pattern"));
            if (p == null) return "analysis_pattern_flags";
            c.name = "patternReplace";
            c.arg("pattern", p).arg("replacement", (String) read(o, "replacement"));
            return null;
        });
        map("charfilter.MappingCharFilter", (o, c) -> {
            @SuppressWarnings("unchecked")
            FST<CharsRef> fst = (FST<CharsRef>) read(o, "map");
            StringBuilder rules = new StringBuilder();
            if (fst != null) {
                try {
                    IntsRefFSTEnum<CharsRef> e = new IntsRefFSTEnum<>(fst);
                    IntsRefFSTEnum.InputOutput<CharsRef> io;
                    while ((io = e.next()) != null) {
                        rules.append('"');
                        IntsRef in = io.input;
                        for (int i = 0; i < in.length; i++) escape(rules, (char) in.ints[in.offset + i]);
                        rules.append("\" => \"");
                        CharsRef outc = io.output;
                        for (int i = 0; i < outc.length; i++) escape(rules, outc.chars[outc.offset + i]);
                        rules.append("\"\n");
                    }
                } catch (IOException ex) {
                    return "analysis_reflect";
                }
            }
            c.name = "mapping";
            c.arg("mapping", c.writer.resource(rules.toString()));
            return null;
        });

        // ---- Token filters ----
        Mapper stop = (o, c) -> {
            c.name = "stop";
            return c.words("words", (CharArraySet) read(o, "stopWords"), "ignoreCase");
        };
        map("StopFilter", stop);
        map("core.StopFilter", stop);
        map("miscellaneous.ASCIIFoldingFilter", (o, c) -> {
            c.name = "asciiFolding";
            c.arg("preserveOriginal", flag(readBool(o, "preserveOriginal")));
            return null;
        });
        map("snowball.SnowballFilter", (o, c) -> {
            String stemmer = read(o, "stemmer").getClass().getSimpleName();
            if (stemmer.endsWith("Stemmer") == false) return "analysis_snowball";
            c.name = "snowballPorter";
            c.arg("language", stemmer.substring(0, stemmer.length() - "Stemmer".length()));
            return null;
        });
        map("miscellaneous.SetKeywordMarkerFilter", (o, c) -> {
            c.name = "keywordMarker";
            return c.words("protected", (CharArraySet) read(o, "keywordSet"), "ignoreCase");
        });
        map("miscellaneous.LengthFilter", (o, c) -> {
            c.name = "length";
            c.arg("min", readInt(o, "min")).arg("max", readInt(o, "max"));
            return null;
        });
        map("miscellaneous.TruncateTokenFilter", (o, c) -> {
            c.name = "truncate";
            c.arg(readBool(o, "useCodePoints") ? "truncateAfterCodePoints" : "truncateAfterChars", readInt(o, "truncateAfter"));
            return null;
        });
        map("reverse.ReverseStringFilter", (o, c) -> {
            if ((char) read(o, "marker") != (char) read(null, o.getClass(), "NOMARKER")) {
                return "analysis_reverse_marker";
            }
            c.name = "reverseString";
            return null;
        });
        map("shingle.ShingleFilter", (o, c) -> {
            if ("shingle".equals(read(o, "tokenType")) == false) return "analysis_shingle_type";
            c.name = "shingle";
            c.arg("minShingleSize", readInt(o, "minShingleSize"))
                .arg("maxShingleSize", readInt(o, "maxShingleSize"))
                .arg("outputUnigrams", flag(readBool(o, "outputUnigrams")))
                .arg("outputUnigramsIfNoShingles", flag(readBool(o, "outputUnigramsIfNoShingles")))
                .arg("tokenSeparator", (String) read(o, "tokenSeparator"))
                .arg("fillerToken", new String((char[]) read(o, "fillerToken")));
            return null;
        });
        Mapper ngramFilter = (o, c) -> {
            c.name = o.getClass().getSimpleName().startsWith("Edge") ? "edgeNGram" : "nGram";
            c.arg("minGramSize", readInt(o, "minGram"))
                .arg("maxGramSize", readInt(o, "maxGram"))
                .arg("preserveOriginal", flag(readBool(o, "preserveOriginal")));
            return null;
        };
        map("ngram.NGramTokenFilter", ngramFilter);
        map("ngram.EdgeNGramTokenFilter", ngramFilter);
        map("util.ElisionFilter", (o, c) -> {
            c.name = "elision";
            return c.words("articles", (CharArraySet) read(o, "articles"), "ignoreCase");
        });
        map("miscellaneous.LimitTokenCountFilter", (o, c) -> {
            c.name = "limitTokenCount";
            c.arg("maxTokenCount", readInt(o, "maxTokenCount")).arg("consumeAllTokens", flag(readBool(o, "consumeAllTokens")));
            return null;
        });
        map("cjk.CJKBigramFilter", (o, c) -> {
            Object no = read(null, org.apache.lucene.analysis.cjk.CJKBigramFilter.class, "NO");
            c.name = "cjkBigram";
            c.arg("han", flag(read(o, "doHan") != no))
                .arg("hiragana", flag(read(o, "doHiragana") != no))
                .arg("katakana", flag(read(o, "doKatakana") != no))
                .arg("hangul", flag(read(o, "doHangul") != no))
                .arg("outputUnigrams", flag(readBool(o, "outputUnigrams")));
            return null;
        });
        map("synonym.SynonymGraphFilter", (o, c) -> synonyms(o, c, "synonymGraph"));
        map("synonym.SynonymFilter", (o, c) -> synonyms(o, c, "synonym"));
        map("miscellaneous.StemmerOverrideFilter", (o, c) -> {
            StemmerOverrideFilter.StemmerOverrideMap m = (StemmerOverrideFilter.StemmerOverrideMap) read(o, "stemmerOverrideMap");
            @SuppressWarnings("unchecked")
            FST<BytesRef> fst = (FST<BytesRef>) read(m, "fst");
            StringBuilder lines = new StringBuilder();
            if (fst != null) {
                try {
                    IntsRefFSTEnum<BytesRef> e = new IntsRefFSTEnum<>(fst);
                    IntsRefFSTEnum.InputOutput<BytesRef> io;
                    while ((io = e.next()) != null) {
                        String word = new String(io.input.ints, io.input.offset, io.input.length);
                        String stem = io.output.utf8ToString();
                        // getLines trims each line and skips comments; split("\t", 2) takes the first tab.
                        if (word.isEmpty()
                            || word.startsWith("#")
                            || word.startsWith("\uFEFF")
                            || word.charAt(0) <= ' '
                            || word.indexOf('\t') >= 0
                            || (stem.isEmpty() == false && stem.charAt(stem.length() - 1) <= ' ')
                            || (word + stem).indexOf('\n') >= 0
                            || (word + stem).indexOf('\r') >= 0) {
                            return "analysis_stemmer_override";
                        }
                        lines.append(word).append('\t').append(stem).append('\n');
                    }
                } catch (IOException ex) {
                    return "analysis_reflect";
                }
            }
            c.name = "stemmerOverride";
            c.arg("dictionary", c.writer.resource(lines.toString())).arg("ignoreCase", flag(readBool(m, "ignoreCase")));
            return null;
        });
        map("miscellaneous.KeepWordFilter", (o, c) -> {
            c.name = "keepWord";
            return c.words("words", (CharArraySet) read(o, "words"), "ignoreCase");
        });
        map("core.TypeTokenFilter", (o, c) -> {
            @SuppressWarnings("unchecked")
            Set<String> types = (Set<String>) read(o, "stopTypes");
            StringBuilder sb = new StringBuilder();
            for (String t : types) {
                if (t.isEmpty() || t.startsWith("#") || t.indexOf('\n') >= 0 || t.indexOf('\r') >= 0 || t.charAt(0) <= ' ' || t.charAt(t.length() - 1) <= ' ') {
                    return "analysis_words";
                }
                sb.append(t).append('\n');
            }
            c.name = "type";
            c.arg("types", c.writer.resource(sb.toString())).arg("useWhitelist", flag(readBool(o, "useWhiteList")));
            return null;
        });
        map("pattern.PatternReplaceFilter", (o, c) -> {
            String p = plainPattern(((Matcher) read(o, "m")).pattern());
            if (p == null) return "analysis_pattern_flags";
            c.name = "patternReplace";
            c.arg("pattern", p)
                .arg("replacement", (String) read(o, "replacement"))
                .arg("replace", readBool(o, "all") ? "all" : "first");
            return null;
        });
        map("commongrams.CommonGramsFilter", (o, c) -> {
            c.name = "commonGrams";
            return c.words("words", (CharArraySet) read(o, "commonWords"), "ignoreCase");
        });
        map("id.IndonesianStemFilter", (o, c) -> {
            c.name = "indonesianStem";
            c.arg("stemDerivational", flag(readBool(o, "stemDerivational")));
            return null;
        });
        Mapper wordDelimiter = (o, c) -> {
            boolean graph = o.getClass().getSimpleName().equals("WordDelimiterGraphFilter");
            Object iterator = read(o, "iterator");
            byte[] table = (byte[]) read(iterator, "charTypeTable");
            byte[] defaults = (byte[]) read(null, iterator.getClass(), "DEFAULT_WORD_DELIM_TABLE");
            if (table != defaults && java.util.Arrays.equals(table, defaults) == false) {
                return "analysis_word_delimiter_types";
            }
            int flags = readInt(o, "flags");
            c.name = graph ? "wordDelimiterGraph" : "wordDelimiter";
            String[] names = {
                "generateWordParts",
                "generateNumberParts",
                "catenateWords",
                "catenateNumbers",
                "catenateAll",
                "preserveOriginal",
                "splitOnCaseChange",
                "splitOnNumerics",
                "stemEnglishPossessive",
                "ignoreKeywords" };
            int known = graph ? 10 : 9;
            if ((flags & ~((1 << known) - 1)) != 0) return "analysis_word_delimiter_flags";
            for (int i = 0; i < known; i++) {
                c.arg(names[i], (flags & (1 << i)) != 0 ? 1 : 0);
            }
            if (graph) {
                c.arg("adjustOffsets", flag(readBool(o, "adjustInternalOffsets")));
            }
            CharArraySet prot = (CharArraySet) read(o, "protWords");
            if (prot != null) {
                return c.words("protected", prot, null);
            }
            return null;
        };
        map("miscellaneous.WordDelimiterGraphFilter", wordDelimiter);
        map("miscellaneous.WordDelimiterFilter", wordDelimiter);
    }

    private static void escape(StringBuilder sb, char ch) {
        sb.append(String.format(java.util.Locale.ROOT, "\\u%04x", (int) ch));
    }

    // ---- Encoding ----

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.write(b, 0, b.length);
    }
}
