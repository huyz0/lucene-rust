import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.AutomatonToTokenStream;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.CharFilter;
import org.apache.lucene.analysis.FilteringTokenFilter;
import org.apache.lucene.analysis.GraphTokenFilter;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.TokenStreamToAutomaton;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.Transition;

/**
 * Differential fixture for M7's analysis port ({@code crates/lucene-analysis/tests/
 * standard_tokenizer_fixtures.rs}): real Lucene 10.5.0 {@code StandardTokenizer}, {@code
 * StandardAnalyzer}, a custom chain built from core APIs only (a {@code CharFilter}, a {@code
 * TokenFilter}, a {@code FilteringTokenFilter}, a case-insensitive {@code StopFilter}), {@code
 * Analyzer.normalize}, a {@code GraphTokenFilter} subclass, and the two automaton converters.
 *
 * <p>Writes {@code standard_tokenizer/}:
 *
 * <ul>
 *   <li>{@code cmap_runs.txt}: {@code StandardTokenizerImpl.zzCMap} (private, by reflection) over
 *       every code point 0..0x10FFFF, run-length encoded as {@code start class} lines -- the
 *       character-class half of the scanner, checked exhaustively.
 *   <li>{@code cases.txt}: {@code T|name|text} lines give each text, then one {@code
 *       name|config|tokens|end} line per analysis of it. Texts and terms are UTF-8 hex (so no
 *       escaping question can arise), a term equal to its offsets' slice of the text is "=" (which
 *       keeps the file small); a token is {@code term,start,end,posInc,posLen,type}; {@code end}
 *       is the offset and increment {@code end()} left. {@code N|config|text|bytes} lines are
 *       {@code Analyzer.normalize}. Configs: {@code tok:N} (bare tokenizer, maxTokenLength N), {@code std:N}
 *       (StandardAnalyzer, no stopwords), {@code stop} (StandardAnalyzer with stopwords), {@code
 *       chain} (the custom chain below); normalize configs are {@code norm-std}/{@code
 *       norm-chain}.
 *   <li>{@code graph.txt}: canned token graphs (a spec the Rust side rebuilds) through a {@code
 *       GraphTokenFilter} subclass, {@code TokenStreamToAutomaton} (three settings) and {@code
 *       AutomatonToTokenStream}.
 * </ul>
 *
 * <p>Inputs are deterministic (fixed seeds), so the files are byte-identical run to run.
 */
public class GenStandardTokenizer {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("standard_tokenizer");
    Files.createDirectories(out);
    writeCmapRuns(out.resolve("cmap_runs.txt"));

    StringBuilder cases = new StringBuilder();
    List<String[]> texts = corpus();
    for (String[] t : texts) {
      String name = t[0];
      String text = t[1];
      cases.append("T|").append(name).append('|').append(hex(new BytesRef(text))).append('\n');
      cases.append(line(name, "tok:255", tokenize(text, 255, text)));
      // conformance inputs are a few chars: maxTokenLength 5 adds little.
      if (!conformance(name)) {
        cases.append(line(name, "tok:5", tokenize(text, 5, text)));
      }
    }
    // A handful of lengths around the limits, on the texts that have long runs.
    for (String[] t : texts) {
      if (!t[0].startsWith("long")) continue;
      for (int max : new int[] {1, 2, 3, 4, 254, 256, 1024}) {
        cases.append(line(t[0], "tok:" + max, tokenize(t[1], max, t[1])));
      }
    }
    try (Analyzer std = new StandardAnalyzer();
        Analyzer std5 = new StandardAnalyzer();
        Analyzer stop = new StandardAnalyzer(new CharArraySet(STOP, false));
        Analyzer chain = new ChainAnalyzer()) {
      ((StandardAnalyzer) std5).setMaxTokenLength(5);
      for (String[] t : texts) {
        // conformance inputs test the tokenizer; the analyzers run
        // over everything else.
        if (conformance(t[0])) continue;
        cases.append(line(t[0], "std:255", analyze(std, t[1], t[1])));
        cases.append(line(t[0], "chain", analyze(chain, t[1], null)));
        // the random strings run the two analyzer variants below only as
        // far as the tokenizer runs already cover them.
        if (t[0].startsWith("rand")) continue;
        cases.append(line(t[0], "std:5", analyze(std5, t[1], t[1])));
        cases.append(line(t[0], "stop", analyze(stop, t[1], t[1])));
      }
      for (String s : NORMALIZE) {
        cases.append(normLine("norm-std", s, std.normalize("f", s)));
        cases.append(normLine("norm-chain", s, chain.normalize("f", s)));
      }
    }
    Files.writeString(out.resolve("cases.txt"), cases.toString(), StandardCharsets.UTF_8);

    StringBuilder graph = new StringBuilder();
    for (String spec : GRAPHS) {
      graph.append("graph|").append(spec).append('|');
      graph.append(tokensOf(new PathsFilter(new CannedStream(spec), 3), null)).append('\n');
      String[] settings = {"default", "nopreserve", "finalhole-unicode"};
      for (String setting : settings) {
        TokenStreamToAutomaton ts2a = new TokenStreamToAutomaton();
        if (setting.equals("nopreserve")) ts2a.setPreservePositionIncrements(false);
        if (setting.equals("finalhole-unicode")) {
          ts2a.setFinalOffsetGapAsHole(true);
          ts2a.setUnicodeArcs(true);
        }
        Automaton a;
        try (TokenStream in = new CannedStream(spec)) {
          a = ts2a.toAutomaton(in);
        }
        graph.append("ts2a-").append(setting).append('|').append(spec).append('|');
        graph.append(dump(a)).append('\n');
        if (setting.equals("default")) {
          graph.append("a2ts|").append(spec).append('|');
          graph.append(tokensOf(AutomatonToTokenStream.toTokenStream(a), null)).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("graph.txt"), graph.toString(), StandardCharsets.UTF_8);
  }

  /** The tokenizer-only inputs: UAX#29 conformance data and the pairs/triples built like it. */
  static boolean conformance(String name) {
    return name.startsWith("pair") || name.startsWith("tri") || name.startsWith("uax29-");
  }

  static final List<String> STOP = Arrays.asList("the", "a", "of", "and", "is", "und", "le");

  static final String[] NORMALIZE = {
    "Hello", "ÉCOLE-Straße", "İSTANBUL", "ΟΔΟΣ", "", "日本", "A_B-C", "😀X"
  };

  // ------------------------------------------------------------------ cmap

  static void writeCmapRuns(Path path) throws Exception {
    Class<?> impl = Class.forName("org.apache.lucene.analysis.standard.StandardTokenizerImpl");
    Method cmap = impl.getDeclaredMethod("zzCMap", int.class);
    cmap.setAccessible(true);
    StringBuilder sb = new StringBuilder();
    int prev = -1;
    for (int cp = 0; cp <= 0x10FFFF; cp++) {
      int c = (Integer) cmap.invoke(null, cp);
      if (c != prev) {
        sb.append(Integer.toHexString(cp)).append(' ').append(c).append('\n');
        prev = c;
      }
    }
    Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
  }

  // ---------------------------------------------------------------- corpus

  static List<String[]> corpus() {
    List<String[]> out = new ArrayList<>();
    String[][] fixed = {
      {"en", "The quick brown fox can't jump 32.3 feet, right? U.S.A. O'Neil's 1,000,000 3.14"},
      {"urls", "visit https://www.example.com/path?q=1&r=2 or mail john.doe@example.co.uk now"},
      {"hosts", "foo.bar.com 192.168.0.1 wi-fi e-mail AT&T C++ C# .NET node.js"},
      {"cjk", "我是中国人。日本語のテキスト、カタカナとひらがな。한국어 텍스트입니다. 𠀀𠀁 ｱｲｳ"},
      {"thai", "ภาษาไทยไม่มีช่องว่างระหว่างคำ ລາວ ພາສາ မြန်မာ ខ្មែរ"},
      {"hangul-jamo", "한국 각 한글"},
      {"kana", "ひらがな カタカナ ｶﾀｶﾅ ヿ ゟ 〱〲 ー 日本ー"},
      {"emoji", "I ❤️ 🍕! 👨‍👩‍👧‍👦 family 🏳️‍🌈 flag 🇺🇸🇫🇷 1️⃣ #️⃣ 👍🏽 ☺ ☺️ 😀😃"},
      {"emoji-zwj", "a👩‍💻b 👩‍❤️‍💋‍👨 🧑🏿‍🤝‍🧑🏻 ‍‍x ‍😀"},
      {"numbers", "1 22 333 4,4 5.5 6_6 ٠١٢ ０１２ 1st 2nd 3.4.5 $100 -7 +8 10:30 1e10"},
      {"mixed", "Ça va? Élève naïve, Straße, İstanbul; ΟΔΟΣ ὁδός — Москва; עברית ש\"ס ג'ירפה"},
      {"arabic", "العربية ١٢٣ كتاب‎ فارسی"},
      {"indic", "हिन्दी भाषा বাংলা தமிழ்"},
      {"punct", "... --- !!! ??? ((())) [x] {y} <z> \"q\" 'r' `s`"},
      {"underscore", "snake_case __init__ a_1 _ __ _x x_"},
      {"combining", "café é̂ ́x a⃝"},
      {"format", "soft­hyphen zero​width a⁠b ﻿bom"},
      {"whitespace", "\t\n\r\u000B\f  　x y z"},
      {"empty", ""},
      {"spaces", "   "},
      {"single", "x"},
      {"long-ascii", "a".repeat(300) + " " + "b".repeat(1100) + " " + "c".repeat(255) + " d"},
      {"long-digits", "1".repeat(260) + " 2"},
      {"long-cjk", "日".repeat(20) + "カ".repeat(300)},
      {"long-thai", "ก".repeat(280) + " x"},
      {"long-supp", "a".repeat(4) + "𝒜" + "b".repeat(10) + " " + "𝒜".repeat(200)},
      {"long-supp2", "a".repeat(254) + "𝒜" + "c".repeat(3)},
      {"long-emoji", "😀".repeat(150) + " " + "👍🏽".repeat(70)},
      {"long-mixed", "x".repeat(253) + "_" + "9".repeat(5) + "." + "z".repeat(3)},
    };
    for (String[] f : fixed) out.add(f);

    // UAX#29 conformance-style pairs and triples: one or two sample
    // characters per Word_Break (and emoji / script) class, every ordered
    // pair with and without an intervening U+0308, as Unicode's
    // WordBreakTest.txt is built.
    String[] samples = {
      "\r", "\n", "\u000B", "̀", "‍", "🇦", "­", "ァ", "א", "A",
      "é", "'", "\"", ".", "‘", ":", "·", ",", ";", "0", "٠", "_", " ",
      " ", "　", "一", "𠀀", "ぁ", "가", "ᄀ", "ᅡ", "ᆨ",
      "ก", "ั", "ກ", "က", "ក", "😀", "❤", "️",
      "🏻", "⃣", "#", "$", "؀", "１", "々", "〱", "ｱ",
      "׳", "๐", "©", "⌚", "👍", "Ä"
    };
    int n = 0;
    for (String a : samples) {
      for (String b : samples) {
        out.add(new String[] {"pair" + n, a + b});
        out.add(new String[] {"pair" + n + "x", a + "̈" + b});
        n++;
      }
    }
    String[] tri = {"a", "5", "א", "ァ", ".", "'", ":", ",", "\"", "_", "‍", "ก"};
    n = 0;
    for (String a : tri) for (String b : tri) for (String c : tri) {
      out.add(new String[] {"tri" + n++, a + b + c});
    }

    // Unicode's own conformance data, as Lucene's test framework ships it
    // (WordBreakTestUnicode_12_1_0, EmojiTokenizationTestUnicode_12_1): the
    // inputs, extracted by crates/lucene-analysis/tools/extract_uax29_inputs.py.
    // The expected tokens are what StandardTokenizer produces below.
    Path uax29 = Path.of("fixtures/src/uax29-inputs.txt");
    try {
      for (String l : Files.readAllLines(uax29, StandardCharsets.UTF_8)) {
        if (l.startsWith("#")) continue;
        String[] kv = l.split("\t");
        byte[] b = new byte[kv[1].length() / 2];
        for (int i = 0; i < b.length; i++) {
          b[i] = (byte) Integer.parseInt(kv[1].substring(2 * i, 2 * i + 2), 16);
        }
        out.add(new String[] {"uax29-" + kv[0], new String(b, StandardCharsets.UTF_8)});
      }
    } catch (IOException e) {
      throw new IllegalStateException("run from the repository root: " + uax29, e);
    }

    // Random strings over a weighted mix of scripts, joiners and emoji.
    Random r = new Random(0x5eed_1234L);
    int[][] pools = {
      {'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {' ', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'},
      {0x00C0, 0x024F}, {0x0300, 0x036F}, {0x0370, 0x03FF}, {0x0400, 0x04FF}, {0x05D0, 0x05F4},
      {0x0600, 0x06FF}, {0x0900, 0x097F}, {0x0E00, 0x0E7F}, {0x0E80, 0x0EFF}, {0x1000, 0x109F},
      {0x1780, 0x17FF}, {0x1100, 0x11FF}, {0xAC00, 0xD7A3}, {0x3040, 0x309F}, {0x30A0, 0x30FF},
      {0xFF61, 0xFF9F}, {0x4E00, 0x9FFF}, {0x20000, 0x2A6DF}, {0x1F300, 0x1FAFF}, {0x2600, 0x27BF},
      {0x1F1E6, 0x1F1FF}, {0x1F3FB, 0x1F3FF}, {0x200B, 0x200F}, {0x2000, 0x206F}, {0xFE00, 0xFE0F},
      {0xFF10, 0xFF19}, {0x0660, 0x0669}, {0x3000, 0x303F}, {0xE0020, 0xE007F}, {0x0, 0xFFFF},
      {0x10000, 0x10FFFF}
    };
    for (int i = 0; i < 800; i++) {
      int len = 1 + r.nextInt(i % 10 == 0 ? 300 : 30);
      StringBuilder sb = new StringBuilder();
      while (sb.length() < len) {
        int[] p = pools[r.nextInt(pools.length)];
        int cp = p[0] + r.nextInt(p[1] - p[0] + 1);
        if (cp >= 0xD800 && cp <= 0xDFFF) continue; // no lone surrogates
        sb.appendCodePoint(cp);
        if (r.nextInt(5) == 0) sb.append(' ');
      }
      out.add(new String[] {"rand" + i, sb.toString()});
    }

    // One long multilingual document: many refills of the 255-char buffer.
    StringBuilder doc = new StringBuilder();
    Random r2 = new Random(7);
    for (int i = 0; i < 150; i++) {
      doc.append(fixed[r2.nextInt(18)][1]).append(i % 7 == 0 ? "\n" : " ");
    }
    out.add(new String[] {"document", doc.toString()});
    return out;
  }

  // --------------------------------------------------------------- analyse

  static String tokenize(String text, int max, String source) throws IOException {
    try (StandardTokenizer t = new StandardTokenizer()) {
      t.setMaxTokenLength(max);
      t.setReader(new StringReader(text));
      return tokensOf(t, source);
    }
  }

  static String analyze(Analyzer a, String text, String source) throws IOException {
    try (TokenStream ts = a.tokenStream("f", text)) {
      return tokensOf(ts, source);
    }
  }

  /**
   * reset, every token, end -- and the caller closes. A term equal to its offsets' slice of
   * {@code source} (when given) is written as "=", which keeps the fixture small.
   */
  static String tokensOf(TokenStream ts, String source) throws IOException {
    TermToBytesRefAttribute term = ts.addAttribute(TermToBytesRefAttribute.class);
    OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
    PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
    PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
    TypeAttribute type = ts.addAttribute(TypeAttribute.class);
    StringBuilder sb = new StringBuilder();
    ts.reset();
    boolean first = true;
    while (ts.incrementToken()) {
      if (!first) sb.append(';');
      first = false;
      String h = hex(term.getBytesRef());
      if (source != null
          && off.endOffset() <= source.length()
          && h.equals(hex(new BytesRef(source.substring(off.startOffset(), off.endOffset()))))) {
        h = "=";
      }
      sb.append(h).append(',');
      sb.append(off.startOffset()).append(',').append(off.endOffset()).append(',');
      sb.append(inc.getPositionIncrement()).append(',').append(len.getPositionLength()).append(',');
      sb.append(type.type());
    }
    ts.end();
    sb.append('|').append(off.endOffset()).append(',').append(inc.getPositionIncrement());
    return sb.toString();
  }

  static String line(String name, String config, String tokens) {
    return name + "|" + config + "|" + tokens + "\n";
  }

  static String normLine(String config, String text, BytesRef normalized) {
    return "N|" + config + "|" + hex(new BytesRef(text)) + "|" + hex(normalized) + "\n";
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) {
      sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    }
    return sb.toString();
  }

  static String dump(Automaton a) {
    StringBuilder sb = new StringBuilder();
    sb.append(a.getNumStates());
    for (int s = 0; s < a.getNumStates(); s++) {
      sb.append(';').append(a.isAccept(s) ? "A" : "");
      Transition[] ts = a.getSortedTransitions()[s];
      for (Transition t : ts) {
        sb.append(' ').append(t.dest).append(':').append(t.min).append('-').append(t.max);
      }
    }
    return sb.toString();
  }

  // --------------------------------------------------- the custom chain

  /**
   * Deletes '-', expands 'ß' to "ss", maps '_' to ' '. {@code correct} maps an output offset
   * through a per-output-char table of input offsets (the Rust test builds the same table).
   */
  static final class MapCharFilter extends CharFilter {
    private String output;
    private int[] map;
    private int inLen;
    private int pos;

    MapCharFilter(Reader in) {
      super(in);
    }

    private void fill() throws IOException {
      StringBuilder src = new StringBuilder();
      char[] buf = new char[64];
      int n;
      while ((n = input.read(buf, 0, buf.length)) != -1) src.append(buf, 0, n);
      inLen = src.length();
      StringBuilder o = new StringBuilder();
      List<Integer> m = new ArrayList<>();
      for (int i = 0; i < src.length(); i++) {
        char c = src.charAt(i);
        if (c == '-') continue;
        if (c == 'ß') {
          o.append("ss");
          m.add(i);
          m.add(i);
        } else {
          o.append(c == '_' ? ' ' : c);
          m.add(i);
        }
      }
      m.add(inLen);
      output = o.toString();
      map = m.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public int read(char[] cbuf, int off, int len) throws IOException {
      if (output == null) fill();
      if (pos >= output.length()) return -1;
      int n = Math.min(len, output.length() - pos);
      output.getChars(pos, pos + n, cbuf, off);
      pos += n;
      return n;
    }

    @Override
    protected int correct(int currentOff) {
      if (currentOff < map.length) return map[currentOff];
      return currentOff + (inLen - output.length());
    }
  }

  /** Prefixes a {@code <NUM>} token's term with "n:" and retypes it "number". */
  static final class NumberTagFilter extends TokenFilter {
    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final TypeAttribute type = addAttribute(TypeAttribute.class);

    NumberTagFilter(TokenStream in) {
      super(in);
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (!input.incrementToken()) return false;
      if (type.type().equals(StandardTokenizer.TOKEN_TYPES[StandardTokenizer.NUM])) {
        String t = term.toString();
        term.setEmpty().append("n:").append(t);
        type.setType("number");
      }
      return true;
    }
  }

  /** Drops single-char (one UTF-16 unit) terms. */
  static final class MinLengthFilter extends FilteringTokenFilter {
    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);

    MinLengthFilter(TokenStream in) {
      super(in);
    }

    @Override
    protected boolean accept() {
      return term.length() > 1;
    }
  }

  /**
   * MapCharFilter -> StandardTokenizer(maxTokenLength 10) -> NumberTagFilter -> StopFilter
   * (ignoreCase, before lowercasing) -> MinLengthFilter -> LowerCaseFilter; normalize is
   * LowerCaseFilter, initReaderForNormalization the char filter too.
   */
  static final class ChainAnalyzer extends Analyzer {
    @Override
    protected Reader initReader(String fieldName, Reader reader) {
      return new MapCharFilter(reader);
    }

    @Override
    protected Reader initReaderForNormalization(String fieldName, Reader reader) {
      return new MapCharFilter(reader);
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      StandardTokenizer src = new StandardTokenizer();
      src.setMaxTokenLength(10);
      TokenStream ts = new NumberTagFilter(src);
      ts = new StopFilter(ts, new CharArraySet(Arrays.asList("THE", "und", "Le"), true));
      ts = new MinLengthFilter(ts);
      ts = new LowerCaseFilter(ts);
      return new TokenStreamComponents(src, ts);
    }

    @Override
    protected TokenStream normalize(String fieldName, TokenStream in) {
      return new LowerCaseFilter(in);
    }
  }

  // ------------------------------------------------------------- graphs

  /** Canned graphs: {@code term/posInc/posLen/start/end} tokens, then {@code end/posInc/offset}. */
  static final String[] GRAPHS = {
    "fast/1/1/0/4 wi/1/1/5/7 wifi/0/2/5/10 fi/1/1/8/10 network/1/1/11/18 end/0/18",
    "the/1/1/0/3 x/2/1/8/9 y/1/1/10/11 end/2/20",
    "ny/1/3/0/8 new/0/1/0/3 york/1/1/4/8 city/1/1/9/13 nyc/0/2/4/13 is/1/1/14/16 end/0/16",
    "a/1/1/0/1 b/0/1/0/1 c/0/1/0/1 d/1/1/2/3 e/0/1/2/3 f/1/1/4/5 end/0/5",
    "é/1/1/0/1 日本/1/1/2/4 😀/1/1/5/7 end/0/9",
    "solo/3/1/4/8 end/0/8",
    "end/0/0",
  };

  static final class CannedStream extends TokenStream {
    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final PositionIncrementAttribute inc = addAttribute(PositionIncrementAttribute.class);
    private final PositionLengthAttribute len = addAttribute(PositionLengthAttribute.class);
    private final OffsetAttribute off = addAttribute(OffsetAttribute.class);
    private final String[][] toks;
    private final int endInc;
    private final int endOff;
    private int upto;

    CannedStream(String spec) {
      String[] parts = spec.split(" ");
      toks = new String[parts.length - 1][];
      for (int i = 0; i < parts.length - 1; i++) toks[i] = parts[i].split("/");
      String[] e = parts[parts.length - 1].split("/");
      endInc = Integer.parseInt(e[1]);
      endOff = Integer.parseInt(e[2]);
    }

    @Override
    public boolean incrementToken() {
      clearAttributes();
      if (upto == toks.length) return false;
      String[] t = toks[upto++];
      term.append(t[0]);
      inc.setPositionIncrement(Integer.parseInt(t[1]));
      len.setPositionLength(Integer.parseInt(t[2]));
      off.setOffset(Integer.parseInt(t[3]), Integer.parseInt(t[4]));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      upto = 0;
    }

    @Override
    public void end() throws IOException {
      super.end();
      inc.setPositionIncrement(endInc);
      off.setOffset(endOff, endOff);
    }
  }

  /**
   * For each base token, every path of up to {@code depth} tokens through the graph, joined by
   * '_', each emitted with the base token's other attributes.
   */
  static final class PathsFilter extends GraphTokenFilter {
    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final int depth;
    private final List<String> pending = new ArrayList<>();
    private State baseState;

    PathsFilter(TokenStream in, int depth) {
      super(in);
      this.depth = depth;
    }

    @Override
    public boolean incrementToken() throws IOException {
      while (pending.isEmpty()) {
        if (!incrementBaseToken()) return false;
        baseState = captureState();
        do {
          StringBuilder path = new StringBuilder(term);
          int n = 1;
          while (n < depth && incrementGraphToken()) {
            path.append('_').append(term);
            n++;
          }
          pending.add(path.toString());
        } while (incrementGraph());
      }
      restoreState(baseState);
      term.setEmpty().append(pending.remove(0));
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      pending.clear();
    }
  }
}
