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
import org.apache.lucene.analysis.hunspell.Dictionary;
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
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.WeakHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Describes a Lucene {@link Analyzer} as the spec {@code analysis.rs} builds its own copy from:
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
    /** The layout version {@code analysis.rs} expects. */
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

    /**
     * The described analyzers, by instance (not by {@code equals}: OpenSearch's {@code NamedAnalyzer}
     * compares by name alone, so two indices' {@code default} analyzers would share a spec) and
     * field, each described once.
     */
    private static final org.apache.lucene.util.WeakIdentityMap<Analyzer, java.util.Map<String, Spec>> CACHED =
        org.apache.lucene.util.WeakIdentityMap.newHashMap();

    /** {@link #describe}, kept for the life of {@code analyzer}. */
    public static Spec cached(Analyzer analyzer, String field) {
        java.util.Map<String, Spec> byField;
        synchronized (CACHED) {
            byField = CACHED.get(analyzer);
            if (byField == null) {
                byField = new java.util.concurrent.ConcurrentHashMap<>();
                CACHED.put(analyzer, byField);
            }
        }
        return byField.computeIfAbsent(field, f -> describe(analyzer, f));
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
        Writer w = new Writer(analyzer, field);
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
            for (Map.Entry<String, byte[]> e : w.resources.entrySet()) {
                writeBytes(out, e.getKey().getBytes(StandardCharsets.UTF_8));
                writeBytes(out, e.getValue());
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

    /** {@code analysis::analyze}'s layout. */
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

    /** The bytes a hunspell {@link Dictionary} was parsed from: its affix file and its dictionary files, in load order. */
    public record HunspellFiles(byte[] affix, List<byte[]> dictionaries) {}

    /** Each dictionary's files, by identity ({@link Dictionary} keeps the default equals). */
    private static final Map<Dictionary, HunspellFiles> HUNSPELL = Collections.synchronizedMap(new WeakHashMap<>());

    /** Records the files {@code d} was parsed from, for a dictionary no OpenSearch factory leads to (the self-test's). */
    public static void hunspellFiles(Dictionary d, HunspellFiles files) {
        HUNSPELL.put(d, files);
    }

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
         */
        String words(String key, CharArraySet set, String ignoreCaseKey) {
            List<String> all = new ArrayList<>();
            for (Object o : set) {
                all.add(new String((char[]) o));
            }
            String text = lines(all);
            arg(key, writer.resource(text));
            if (ignoreCaseKey != null) {
                arg(ignoreCaseKey, flag(ignoreCase(set)));
            }
            return null;
        }
    }

    /**
     * A synonym filter's map, entry by entry ({@code accelerated.synonym_entries}): each input with
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
            .arg("format", "accelerated.synonym_entries")
            .arg("ignoreCase", flag(readBool(o, "ignoreCase")));
        return null;
    }

    /**
     * The {@code types} rules that make the factory build {@code table}: one per char whose type is
     * not its default, and one for the last char so the table is as long. Null when the table could
     * not have come from a factory (shorter than the default, or a type no rule names).
     */
    private static String typeRules(byte[] table, int defaultLength) {
        if (table.length < defaultLength) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < table.length; i++) {
            byte def = org.apache.lucene.analysis.miscellaneous.WordDelimiterIterator.getType(i);
            if (table[i] != def || i == table.length - 1 && table.length > defaultLength) {
                switch (table[i]) {
                    case 1, 2, 3, 4, 7, 8 -> {}
                    default -> {
                        return null;
                    }
                }
                String name = switch (table[i]) {
                    case 1 -> "LOWER";
                    case 2 -> "UPPER";
                    case 3 -> "ALPHA";
                    case 4 -> "DIGIT";
                    case 7 -> "ALPHANUM";
                    default -> "SUBWORD_DELIM";
                };
                sb.append(String.format("\\u%04x => %s\n", i, name));
            }
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    /** A comma-separated argument value, or null when an item would not survive {@code getSet}'s split. */
    private static String items(Collection<String> words) {
        StringJoiner j = new StringJoiner(",");
        for (String w : words) {
            if (w.isEmpty() || w.indexOf(',') >= 0 || w.chars().anyMatch(Character::isWhitespace)) {
                return null;
            }
            j.add(w);
        }
        return j.toString();
    }

    /**
     * Words one per line, as {@code getWordSet} reads them back. When one cannot survive that
     * format (empty, a comment, a line break, or edge whitespace) the whole list is written in the
     * native exact format instead: a marker line, then each word with {@code %}, line feed and
     * carriage return escaped, taken as written.
     */
    private static String lines(Collection<String> words) {
        StringBuilder plain = new StringBuilder();
        boolean exact = false;
        for (String t : words) {
            if (t.isEmpty() || t.startsWith("#") || t.startsWith("\uFEFF") || t.indexOf('\n') >= 0 || t.indexOf('\r') >= 0
                || t.charAt(0) <= ' ' || t.charAt(t.length() - 1) <= ' ') {
                exact = true;
                break;
            }
            plain.append(t).append('\n');
        }
        if (exact == false) {
            return plain.toString();
        }
        StringBuilder sb = new StringBuilder("\u0001exact\n");
        for (String t : words) {
            sb.append(t.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")).append('\n');
        }
        return sb.toString();
    }

    /** The bundled dictionary {@code className.getInstance()}, from {@code near}'s class loader. */
    private static Object bundled(Object near, String className) {
        try {
            Class<?> c = Class.forName(className, false, near.getClass().getClassLoader());
            return c.getMethod("getInstance").invoke(null);
        } catch (ReflectiveOperationException | RuntimeException | LinkageError e) {
            throw new ReflectionMiss(className);
        }
    }

    /**
     * Whether a kuromoji or nori lattice ({@code pkg} {@code ja} or {@code ko}) runs on the module's
     * bundled dictionaries, the only ones the Rust data pack holds.
     */
    private static boolean bundledDictionaries(Object viterbi, String pkg) {
        String d = A + pkg + ".dict.";
        return read(viterbi, "dictionary") == bundled(viterbi, d + "TokenInfoDictionary")
            && read(viterbi, "unkDictionary") == bundled(viterbi, d + "UnknownDictionary")
            && read(viterbi, "costs") == bundled(viterbi, d + "ConnectionCosts");
    }

    /** A kuromoji or nori user dictionary's surface forms, each with its ordinal, in FST order. */
    private static List<Map.Entry<String, Integer>> userEntries(Object dictionary) throws IOException {
        @SuppressWarnings("unchecked")
        FST<Long> fst = (FST<Long>) read(read(dictionary, "fst"), "fst");
        List<Map.Entry<String, Integer>> out = new ArrayList<>();
        IntsRefFSTEnum<Long> e = new IntsRefFSTEnum<>(fst);
        IntsRefFSTEnum.InputOutput<Long> io;
        while ((io = e.next()) != null) {
            // BYTE2 input: one UTF-16 unit per label.
            StringBuilder sb = new StringBuilder(io.input.length);
            for (int i = 0; i < io.input.length; i++) {
                sb.append((char) io.input.ints[io.input.offset + i]);
            }
            out.add(Map.entry(sb.toString(), io.output.intValue()));
        }
        return out;
    }

    /** A user dictionary field that its line format keeps as is: no separator, quote, or whitespace. */
    private static boolean plainField(String s) {
        if (s.isEmpty()) return false;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch <= ' ' || ch == ',' || ch == '"' || ch == '#' || Character.isWhitespace(ch) || Character.isSpaceChar(ch)) {
                return false;
            }
        }
        return true;
    }

    /**
     * A kuromoji {@code UserDictionary} as the CSV it is read from ({@code surface,segments,readings,pos}),
     * or null when an entry cannot be written back exactly.
     */
    private static String japaneseUserDictionary(Object dictionary) throws IOException {
        int[][] segmentations = (int[][]) read(dictionary, "segmentations");
        String[] data = (String[]) read(read(dictionary, "morphAtts"), "data");
        int firstWordId = (Integer) read(null, dictionary.getClass(), "CUSTOM_DICTIONARY_WORD_ID_OFFSET");
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> entry : userEntries(dictionary)) {
            String surface = entry.getKey();
            int[] s = segmentations[entry.getValue()];
            if (plainField(surface) == false || s.length < 2) return null;
            StringJoiner segments = new StringJoiner(" ");
            StringJoiner readings = new StringJoiner(" ");
            String pos = null;
            int at = 0;
            for (int i = 1; i < s.length; i++) {
                if (s[i] <= 0 || s[i] > surface.length() - at) return null;
                segments.add(surface.substring(at, at + s[i]));
                at += s[i];
                String[] feature = data[s[0] - firstWordId + i - 1].split("\u0000", -1);
                if (feature.length != 2 || plainField(feature[0]) == false || plainField(feature[1]) == false) return null;
                if (pos != null && pos.equals(feature[1]) == false) return null;
                pos = feature[1];
                readings.add(feature[0]);
            }
            if (at != surface.length()) return null;
            sb.append(surface).append(',').append(segments).append(',').append(readings).append(',').append(pos).append('\n');
        }
        return sb.toString();
    }

    /**
     * A nori {@code UserDictionary} as the lines it is read from ({@code token [segment...]}), or null
     * when an entry cannot be written back exactly. Only segment lengths are kept, so segments are
     * written as consecutive slices of the token; the entry's right id, which the line's last
     * character decides, must come out the same.
     */
    private static String koreanUserDictionary(Object dictionary, Object charDef) throws ReflectiveOperationException, IOException {
        Object morph = read(dictionary, "morphAtts");
        int[][] segmentations = (int[][]) read(morph, "segmentations");
        short[] rightIds = (short[]) read(morph, "rightIds");
        Class<?> d = dictionary.getClass();
        short plain = (Short) read(null, d, "RIGHT_ID");
        short coda = (Short) read(null, d, "RIGHT_ID_T");
        short noCoda = (Short) read(null, d, "RIGHT_ID_F");
        Method isHangul = charDef.getClass().getMethod("isHangul", char.class);
        Method hasCoda = charDef.getClass().getMethod("hasCoda", char.class);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> entry : userEntries(dictionary)) {
            String token = entry.getKey();
            int ord = entry.getValue();
            if (plainField(token) == false) return null;
            StringBuilder line = new StringBuilder(token);
            int[] s = segmentations[ord];
            if (s != null) {
                if (s.length == 0) return null;
                int at = 0;
                for (int len : s) {
                    if (len <= 0 || len > token.length() - at) return null;
                    line.append(' ').append(token, at, at + len);
                    at += len;
                }
            }
            char last = line.charAt(line.length() - 1);
            short rightId = (Boolean) isHangul.invoke(charDef, last) ? ((Boolean) hasCoda.invoke(charDef, last) ? coda : noCoda) : plain;
            if (rightIds[ord] != rightId) return null;
            sb.append(line).append('\n');
        }
        return sb.toString();
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
        final Map<String, byte[]> resources = new LinkedHashMap<>();
        final Analyzer analyzer;
        final String field;

        Writer(Analyzer analyzer, String field) {
            this.analyzer = analyzer;
            this.field = field;
        }

        String resource(String content) {
            return resource(content.getBytes(StandardCharsets.UTF_8), ".txt");
        }

        String resource(byte[] content, String extension) {
            String name = "r" + resources.size() + extension;
            resources.put(name, content);
            return name;
        }

        /** The files {@code d} was built from: registered, or found through the analyzer's OpenSearch factories. */
        HunspellFiles hunspell(Dictionary d) {
            HunspellFiles f = HUNSPELL.get(d);
            if (f == null) {
                f = HunspellSources.find(analyzer, field, d);
                if (f != null) {
                    HUNSPELL.put(d, f);
                }
            }
            return f;
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
        A + "core.KeywordTokenizer",
        A + "cjk.CJKWidthCharFilter",
        A + "ja.JapaneseBaseFormFilter",
        A + "ja.JapaneseHiraganaUppercaseFilter",
        A + "ja.JapaneseKatakanaUppercaseFilter",
        A + "ja.JapaneseNumberFilter",
        A + "ko.KoreanReadingFormFilter",
        A + "ko.KoreanNumberFilter",
        A + "es.SpanishMinimalStemFilter",
        A + "fa.PersianStemFilter",
        A + "fa.PersianCharFilter",
        A + "no.NorwegianNormalizationFilter",
        A + "ro.RomanianNormalizationFilter",
        A + "te.TeluguNormalizationFilter",
        A + "te.TeluguStemFilter",
        A + "sv.SwedishMinimalStemFilter",
        A + "miscellaneous.ScandinavianNormalizationFilter",
        A + "miscellaneous.FixBrokenOffsetsFilter",
        A + "payloads.TokenOffsetPayloadTokenFilter",
        A + "payloads.TypeAsPayloadTokenFilter"
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
        int f = p.flags();
        String embedded = "";
        String body = p.pattern();
        if ((f & Pattern.LITERAL) != 0) {
            body = Pattern.quote(body);
            f &= ~Pattern.LITERAL;
        }
        if ((f & Pattern.CANON_EQ) != 0) {
            return null;
        }
        int[] bits = { Pattern.CASE_INSENSITIVE, Pattern.UNIX_LINES, Pattern.MULTILINE, Pattern.DOTALL, Pattern.UNICODE_CASE, Pattern.COMMENTS, Pattern.UNICODE_CHARACTER_CLASS };
        String letters = "idmsuxU";
        for (int i = 0; i < bits.length; i++) {
            if ((f & bits[i]) != 0) {
                embedded += letters.charAt(i);
                f &= ~bits[i];
            }
        }
        if (f != 0) {
            return null;
        }
        return embedded.isEmpty() ? body : "(?" + embedded + ")" + body;
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
        // Lucene's ThaiTokenizer cuts Thai with the JDK's dictionary: the native one asks this JVM's
        // break iterator for the boundaries, so the words are the same ones.
        map("th.ThaiTokenizer", (o, c) -> {
            ThaiBreak.install();
            c.name = "thai";
            c.arg("jvmBreaker", "true");
            return null;
        });
        // The plugin's rust_* tokenizers (RustAnalysis) run Rust on the Java side too, so the chain
        // runs natively as is: the Rust factory by its SPI name and arguments.
        MAPPERS.put(RustTokenizer.class.getName(), (o, c) -> {
            RustTokenizer t = (RustTokenizer) o;
            c.name = t.spi;
            t.args.forEach(c::arg);
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
        map("hunspell.HunspellStemFilter", (o, c) -> {
            Dictionary d = (Dictionary) read(read(o, "stemmer"), "dictionary");
            HunspellFiles files = c.writer.hunspell(d);
            if (files == null) {
                return "analysis_hunspell";
            }
            StringJoiner dics = new StringJoiner(",");
            for (byte[] dic : files.dictionaries()) {
                dics.add(c.writer.resource(dic, ".dic"));
            }
            c.name = "hunspellStem";
            c.arg("dictionary", dics.toString())
                .arg("affix", c.writer.resource(files.affix(), ".aff"))
                .arg("ignoreCase", flag(readBool(d, "ignoreCase")))
                .arg("longestOnly", flag(readBool(o, "longestOnly")));
            // OpenSearch's dedup setting; Lucene's factory always dedups.
            if (readBool(o, "dedup") == false) {
                c.arg("dedup", "false");
            }
            return null;
        });
        map("miscellaneous.KeepWordFilter", (o, c) -> {
            c.name = "keepWord";
            return c.words("words", (CharArraySet) read(o, "words"), "ignoreCase");
        });
        map("core.TypeTokenFilter", (o, c) -> {
            @SuppressWarnings("unchecked")
            String types = lines((Set<String>) read(o, "stopTypes"));
            c.name = "type";
            c.arg("types", c.writer.resource(types)).arg("useWhitelist", flag(readBool(o, "useWhiteList")));
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
            String types = null;
            if (table != defaults && java.util.Arrays.equals(table, defaults) == false) {
                types = typeRules(table, defaults.length);
                if (types == null) return "analysis_word_delimiter_types";
            }
            int flags = readInt(o, "flags");
            c.name = graph ? "wordDelimiterGraph" : "wordDelimiter";
            if (types != null) {
                c.arg("types", c.writer.resource(types));
            }
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

        // ---- more of analysis-common ----
        map("boost.DelimitedBoostTokenFilter", (o, c) -> {
            c.name = "delimitedBoost";
            c.arg("delimiter", String.valueOf((char) read(o, "delimiter")));
            return null;
        });
        map("miscellaneous.DelimitedTermFrequencyTokenFilter", (o, c) -> {
            c.name = "delimitedTermFrequency";
            c.arg("delimiter", String.valueOf((char) read(o, "delimiter")));
            return null;
        });
        map("miscellaneous.DropIfFlaggedFilter", (o, c) -> {
            c.name = "dropIfFlagged";
            c.arg("dropFlags", readInt(o, "dropFlags"));
            return null;
        });
        map("miscellaneous.CodepointCountFilter", (o, c) -> {
            c.name = "codepointCount";
            c.arg("min", readInt(o, "min")).arg("max", readInt(o, "max"));
            return null;
        });
        map("miscellaneous.FingerprintFilter", (o, c) -> {
            c.name = "fingerprint";
            c.arg("maxOutputTokenSize", readInt(o, "maxOutputTokenSize")).arg("separator", String.valueOf((char) read(o, "separator")));
            return null;
        });
        map("miscellaneous.LimitTokenOffsetFilter", (o, c) -> {
            c.name = "limitTokenOffset";
            c.arg("maxStartOffset", readInt(o, "maxStartOffset")).arg("consumeAllTokens", flag(readBool(o, "consumeAllTokens")));
            return null;
        });
        map("miscellaneous.LimitTokenPositionFilter", (o, c) -> {
            c.name = "limitTokenPosition";
            c.arg("maxTokenPosition", readInt(o, "maxTokenPosition")).arg("consumeAllTokens", flag(readBool(o, "consumeAllTokens")));
            return null;
        });
        map("minhash.MinHashFilter", (o, c) -> {
            c.name = "minHash";
            c.arg("hashCount", readInt(o, "hashCount"))
                .arg("bucketCount", readInt(o, "bucketCount"))
                .arg("hashSetSize", readInt(o, "hashSetSize"))
                .arg("withRotation", flag(readBool(o, "withRotation")));
            return null;
        });
        map("shingle.FixedShingleFilter", (o, c) -> {
            c.name = "fixedShingle";
            c.arg("shingleSize", readInt(o, "shingleSize"))
                .arg("tokenSeparator", (String) read(o, "tokenSeparator"))
                .arg("fillerToken", (String) read(o, "fillerToken"));
            return null;
        });
        map("miscellaneous.TypeAsSynonymFilter", (o, c) -> {
            c.name = "typeAsSynonym";
            String prefix = (String) read(o, "prefix");
            if (prefix != null) {
                c.arg("prefix", prefix);
            }
            @SuppressWarnings("unchecked")
            Set<String> ignore = (Set<String>) read(o, "ignore");
            if (ignore != null && ignore.isEmpty() == false) {
                String items = items(ignore);
                if (items == null) return "analysis_words";
                c.arg("ignore", items);
            }
            c.arg("synFlagsMask", readInt(o, "synFlagsMask"));
            return null;
        });
        map("miscellaneous.CapitalizationFilter", (o, c) -> {
            c.name = "capitalization";
            CharArraySet keep = (CharArraySet) read(o, "keep");
            if (keep != null) {
                List<String> words = new ArrayList<>();
                for (Object w : keep) {
                    words.add(new String((char[]) w));
                }
                String items = items(words);
                if (items == null) return "analysis_words";
                c.arg("keep", items).arg("keepIgnoreCase", flag(ignoreCase(keep)));
            }
            @SuppressWarnings("unchecked")
            Collection<char[]> okPrefix = (Collection<char[]>) read(o, "okPrefix");
            if (okPrefix != null) {
                List<String> prefixes = new ArrayList<>();
                for (char[] w : okPrefix) {
                    prefixes.add(new String(w));
                }
                String items = items(prefixes);
                if (items == null) return "analysis_words";
                c.arg("okPrefix", items);
            }
            c.arg("onlyFirstWord", flag(readBool(o, "onlyFirstWord")))
                .arg("forceFirstLetter", flag(readBool(o, "forceFirstLetter")))
                .arg("minWordLength", readInt(o, "minWordLength"))
                .arg("maxWordCount", readInt(o, "maxWordCount"))
                .arg("maxTokenLength", readInt(o, "maxTokenLength"));
            return null;
        });
        map("compound.DictionaryCompoundWordTokenFilter", (o, c) -> {
            c.name = "dictionaryCompoundWord";
            c.arg("minWordSize", readInt(o, "minWordSize"))
                .arg("minSubwordSize", readInt(o, "minSubwordSize"))
                .arg("maxSubwordSize", readInt(o, "maxSubwordSize"))
                .arg("onlyLongestMatchIgnoreSubwords", flag(readBool(o, "onlyLongestMatch")));
            return c.words("dictionary", (CharArraySet) read(o, "dictionary"), null);
        });
        map("pattern.PatternCaptureGroupTokenFilter", (o, c) -> {
            Matcher[] matchers = (Matcher[]) read(o, "matchers");
            c.name = "patternCaptureGroup";
            for (int i = 0; i < matchers.length; i++) {
                String p = plainPattern(matchers[i].pattern());
                if (p == null) return "analysis_pattern_flags";
                c.arg(i == 0 ? "pattern" : "pattern" + i, p);
            }
            c.arg("preserve_original", flag(readBool(o, "preserveOriginal")));
            return null;
        });
        map("payloads.DelimitedPayloadTokenFilter", (o, c) -> {
            String encoder = read(o, "encoder").getClass().getSimpleName();
            c.name = "delimitedPayload";
            c.arg("delimiter", String.valueOf((char) read(o, "delimiter")));
            switch (encoder) {
                case "FloatEncoder" -> c.arg("encoder", "float");
                case "IntegerEncoder" -> c.arg("encoder", "integer");
                case "IdentityEncoder" -> {
                    if (StandardCharsets.UTF_8.equals(read(read(o, "encoder"), "charset")) == false) {
                        throw new ReflectionMiss("a payload encoder no factory builds");
                    }
                    c.arg("encoder", "identity");
                }
                default -> {
                    throw new ReflectionMiss("a payload encoder no factory builds");
                }
            }
            return null;
        });
        map("payloads.NumericPayloadTokenFilter", (o, c) -> {
            BytesRef payload = (BytesRef) read(o, "thePayload");
            c.name = "numericPayload";
            c.arg("typeMatch", (String) read(o, "typeMatch"))
                .arg("payload", Float.toString(org.apache.lucene.analysis.payloads.PayloadHelper.decodeFloat(payload.bytes, payload.offset)));
            return null;
        });
        Mapper norwegian = (o, c) -> {
            Object stemmer = read(o, "stemmer");
            boolean bokmaal = readBool(stemmer, "useBokmaal");
            boolean nynorsk = readBool(stemmer, "useNynorsk");
            if (bokmaal == false && nynorsk == false) throw new ReflectionMiss("a Norwegian stemmer no variant names");
            c.name = o.getClass().getSimpleName().contains("Minimal") ? "norwegianMinimalStem" : "norwegianLightStem";
            c.arg("variant", bokmaal && nynorsk ? "no" : bokmaal ? "nb" : "nn");
            return null;
        };
        map("no.NorwegianLightStemFilter", norwegian);
        map("no.NorwegianMinimalStemFilter", norwegian);

        // ---- kuromoji and nori (OpenSearch's analysis-kuromoji and analysis-nori) ----
        map("ja.JapaneseTokenizer", (o, c) -> {
            Object v = read(o, "viterbi");
            if (bundledDictionaries(v, "ja") == false) return "analysis_japanese_dictionary";
            boolean search = readBool(v, "searchMode");
            c.name = "japanese";
            c.arg("mode", readBool(v, "extendedMode") ? "EXTENDED" : search ? "SEARCH" : "NORMAL")
                .arg("discardPunctuation", flag(readBool(v, "discardPunctuation")));
            if (search) {
                c.arg("discardCompoundToken", flag(readBool(v, "outputCompounds") == false));
            }
            int nBestCost = readInt(v, "nBestCost");
            if (nBestCost != 0) {
                c.arg("nBestCost", nBestCost);
            }
            Object user = read(v, "userDictionary");
            if (user != null) {
                String text;
                try {
                    text = japaneseUserDictionary(user);
                } catch (IOException e) {
                    return "analysis_reflect";
                }
                if (text == null) return "analysis_japanese_user_dictionary";
                c.arg("userDictionary", c.writer.resource(text));
            }
            return null;
        });
        map("ja.JapanesePartOfSpeechStopFilter", (o, c) -> {
            @SuppressWarnings("unchecked")
            String tags = lines((Set<String>) read(o, "stopTags"));
            c.name = "japanesePartOfSpeechStop";
            c.arg("tags", c.writer.resource(tags));
            return null;
        });
        map("ja.JapaneseReadingFormFilter", (o, c) -> {
            c.name = "japaneseReadingForm";
            c.arg("useRomaji", flag(readBool(o, "useRomaji")));
            return null;
        });
        map("ja.JapaneseKatakanaStemFilter", (o, c) -> {
            c.name = "japaneseKatakanaStem";
            c.arg("minimumLength", readInt(o, "minimumKatakanaLength"));
            return null;
        });
        map("ja.JapaneseCompletionFilter", (o, c) -> {
            c.name = "japaneseCompletion";
            c.arg("mode", ((Enum<?>) read(read(o, "tokenGenerator"), "mode")).name());
            return null;
        });
        map("ja.JapaneseIterationMarkCharFilter", (o, c) -> {
            c.name = "japaneseIterationMark";
            c.arg("normalizeKanji", flag(readBool(o, "normalizeKanji"))).arg("normalizeKana", flag(readBool(o, "normalizeKana")));
            return null;
        });
        map("ko.KoreanTokenizer", (o, c) -> {
            Object v = read(o, "viterbi");
            if (bundledDictionaries(v, "ko") == false) return "analysis_korean_dictionary";
            c.name = "korean";
            c.arg("decompoundMode", ((Enum<?>) read(v, "mode")).name())
                .arg("outputUnknownUnigrams", flag(readBool(v, "outputUnknownUnigrams")))
                .arg("discardPunctuation", flag(readBool(v, "discardPunctuation")));
            Object user = read(v, "userDictionary");
            if (user != null) {
                String text;
                try {
                    text = koreanUserDictionary(user, read(v, "characterDefinition"));
                } catch (IOException | ReflectiveOperationException e) {
                    return "analysis_reflect";
                }
                if (text == null) return "analysis_korean_user_dictionary";
                c.arg("userDictionary", c.writer.resource(text));
            }
            return null;
        });
        map("ko.KoreanPartOfSpeechStopFilter", (o, c) -> {
            Set<?> tags = (Set<?>) read(o, "stopTags");
            // An empty argument means the default tags, so none is spelled "-".
            StringJoiner names = new StringJoiner(",");
            if (tags.isEmpty()) names.add("-");
            for (Object t : tags) {
                names.add(((Enum<?>) t).name());
            }
            c.name = "koreanPartOfSpeechStop";
            c.arg("tags", names.toString());
            return null;
        });
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
