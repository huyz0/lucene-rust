import java.io.StringReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.ko.KoreanAnalyzer;
import org.apache.lucene.analysis.ko.KoreanPartOfSpeechStopFilter;
import org.apache.lucene.analysis.ko.KoreanTokenizer;
import org.apache.lucene.analysis.ko.KoreanTokenizer.DecompoundMode;
import org.apache.lucene.analysis.ko.POS;
import org.apache.lucene.analysis.ko.dict.ConnectionCosts;
import org.apache.lucene.analysis.ko.dict.KoMorphData;
import org.apache.lucene.analysis.ko.dict.TokenInfoDictionary;
import org.apache.lucene.analysis.ko.dict.UserDictionary;
import org.apache.lucene.analysis.morph.GraphvizFormatter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.IntsRefFSTEnum;

/**
 * M12 T12.1: Lucene's analysis-nori module -- KoreanTokenizer over the mecab-ko-dic dictionary.
 *
 * <p>Writes {@code analysis_nori/}:
 *
 * <ul>
 *   <li>{@code tok_<config>.tsv}: {@code corpus/analysis-korean.txt} and a generated stress corpus
 *       ({@code stress.txt}, seeded) through {@link KoreanTokenizer} in every decompound mode,
 *       with and without punctuation and unknown unigrams and the user dictionary {@code
 *       corpus/analysis-korean-userdict.txt}: rows {@code T line term start end posInc posLen},
 *       then, for three configurations, every Nori attribute's {@code reflectWith} values (their
 *       keys head the file: {@code K key...}); {@code E}/{@code X} rows as Kuromoji's.
 *   <li>{@code sweep.tsv}: the hostile sweep -- a seeded corpus of 400 lines from {@link
 *       HostileText}'s pools ({@code hostile.txt}) through every combination of decompound mode,
 *       unknown unigrams, punctuation and user dictionary, 24 configurations, rows with every
 *       attribute summarised per line as {@code config line rows fnv1a64}.
 *   <li>{@code graphviz_<config>.txt}: the {@link GraphvizFormatter} lattice of each corpus line.
 *   <li>{@code dictionary.tsv}: every 80th surface of the system dictionary's FST with each of its
 *       words' ids, connection ids, cost, POS type, left and right POS, reading and morphemes.
 *   <li>{@code userdict_lookup.tsv}: {@code UserDictionary.lookup} over every corpus line.
 *   <li>{@code chains/<name>.tsv}: {@code KoreanAnalyzer} configurations and every configuration
 *       of {@code corpus/analysis-nori.conf} through {@code CustomAnalyzer}, as Kuromoji's.
 * </ul>
 *
 * Runs with lucene-analysis-nori on its own classpath ({@code generator_classpath} in {@code
 * scripts/gen-fixtures.sh}). Deterministic. Read by {@code
 * crates/lucene-analysis-nori/tests/nori_fixtures.rs}.
 */
public class GenAnalysisNori {

  static String esc(String s) {
    return s == null ? "null" : AnalysisRows.esc(s);
  }

  /**
   * The Nori attributes' reflected values, tab-separated; their keys are the file's first row.
   */
  static String reflect(TokenStream ts, StringBuilder keys) {
    StringBuilder b = new StringBuilder();
    StringBuilder k = new StringBuilder();
    ts.reflectWith(
        (attClass, key, value) -> {
          if (attClass.getName().startsWith("org.apache.lucene.analysis.ko.")) {
            k.append('\t').append(attClass.getSimpleName()).append('#').append(key);
            b.append('\t').append(esc(value == null ? null : value.toString()));
          }
        });
    if (keys.length() == 0) keys.append("K").append(k).append('\n');
    return b.toString();
  }
  static String rows(Supplier<Tokenizer> tok, List<String> lines, boolean attributes) {
    StringBuilder m = new StringBuilder();
    StringBuilder keys = new StringBuilder();
    Tokenizer ts = tok.get();
    CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
    OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
    PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
    PositionLengthAttribute len = ts.addAttribute(PositionLengthAttribute.class);
    for (int ln = 0; ln < lines.size(); ln++) {
      try {
        ts.setReader(new StringReader(lines.get(ln)));
        ts.reset();
        while (ts.incrementToken()) {
          m.append("T\t").append(ln).append('\t').append(esc(term.toString()))
              .append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
              .append('\t').append(inc.getPositionIncrement()).append('\t').append(len.getPositionLength())
              .append(attributes ? reflect(ts, keys) : "").append('\n');
        }
        ts.end();
        m.append("E\t").append(ln).append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
            .append('\t').append(inc.getPositionIncrement()).append('\n');
      } catch (Exception ex) {
        m.append("X\t").append(ln).append('\t').append(ex.getClass().getSimpleName()).append('\n');
      } finally {
        try {
          ts.close();
        } catch (Exception ignored) {
          // as the Rust side
        }
      }
    }
    return keys.toString() + m;
  }

  /** The system dictionary's FST, through reflection (TokenInfoFST keeps it protected). */
  @SuppressWarnings("unchecked")
  static FST<Long> systemFst() throws Exception {
    Field f = org.apache.lucene.analysis.morph.TokenInfoFST.class.getDeclaredField("fst");
    f.setAccessible(true);
    return (FST<Long>) f.get(TokenInfoDictionary.getInstance().getFST());
  }

  static String utf16(IntsRef in) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < in.length; i++) b.append((char) in.ints[in.offset + i]);
    return b.toString();
  }

  static List<String> surfaces() throws Exception {
    List<String> out = new ArrayList<>();
    IntsRefFSTEnum<Long> e = new IntsRefFSTEnum<>(systemFst());
    IntsRefFSTEnum.InputOutput<Long> io;
    while ((io = e.next()) != null) out.add(utf16(io.input));
    return out;
  }

  static String tag(POS.Tag t) {
    return t == null ? "null" : t.name();
  }

  static String morphemes(KoMorphData.Morpheme[] ms) {
    if (ms == null) return "null";
    StringBuilder b = new StringBuilder();
    for (KoMorphData.Morpheme m : ms) {
      if (b.length() > 0) b.append('+');
      b.append(esc(m.surfaceForm())).append('/').append(m.posTag().name());
    }
    return b.toString();
  }

  static void dictionary(Path out, List<String> surfaces) throws Exception {
    TokenInfoDictionary dict = TokenInfoDictionary.getInstance();
    FST<Long> fst = systemFst();
    StringBuilder m = new StringBuilder();
    IntsRef ref = new IntsRef();
    KoMorphData md = dict.getMorphAttributes();
    for (int s = 0; s < surfaces.size(); s += 80) {
      String surface = surfaces.get(s);
      IntsRefBuilder key = new IntsRefBuilder();
      for (int i = 0; i < surface.length(); i++) key.append(surface.charAt(i));
      Long output = org.apache.lucene.util.fst.Util.get(fst, key.get());
      dict.lookupWordIds(output.intValue(), ref);
      char[] chars = surface.toCharArray();
      for (int i = 0; i < ref.length; i++) {
        int id = ref.ints[ref.offset + i];
        m.append(esc(surface)).append('\t').append(output).append('\t').append(id)
            .append('\t').append(md.getLeftId(id)).append('\t').append(md.getRightId(id))
            .append('\t').append(md.getWordCost(id))
            .append('\t').append(md.getPOSType(id).name())
            .append('\t').append(tag(md.getLeftPOS(id)))
            .append('\t').append(tag(md.getRightPOS(id)))
            .append('\t').append(esc(md.getReading(id)))
            .append('\t').append(morphemes(md.getMorphemes(id, chars, 0, chars.length)))
            .append('\n');
      }
    }
    Files.writeString(out.resolve("dictionary.tsv"), m.toString(), StandardCharsets.UTF_8);
  }

  /** Random runs of dictionary surfaces and characters, seeded. */
  static List<String> stress(List<String> surfaces) {
    Random r = new Random(20261008L);
    String extra = ".,!?·ㆍ「」()~ \u3000\t0123456789０１ABCabcéÉ\u0301\u0300ㄱㅏ한漢字かカ😀𠮷-";
    int[] cps = extra.codePoints().toArray();
    List<String> lines = new ArrayList<>();
    for (int n = 0; n < 300; n++) {
      StringBuilder b = new StringBuilder();
      int words = 1 + r.nextInt(n < 290 ? 20 : 300);
      for (int w = 0; w < words; w++) {
        int k = r.nextInt(10);
        if (k < 6) {
          b.append(surfaces.get(r.nextInt(surfaces.size())));
        } else if (k < 7) {
          b.append(' ');
        } else if (k < 9) {
          b.appendCodePoint(cps[r.nextInt(cps.length)]);
        } else {
          // Hangul syllables and Hanja runs
          int base = r.nextBoolean() ? 0xAC00 : 0x4E00;
          int span = base == 0xAC00 ? 11172 : 0x5000;
          for (int c = r.nextInt(6); c >= 0; c--) b.append((char) (base + r.nextInt(span)));
        }
      }
      lines.add(b.toString());
    }
    // Long runs without a frontier force the 1024-position backtrace.
    lines.add("가".repeat(3000));
    lines.add("漢".repeat(1500) + "字".repeat(1500));
    lines.add("ABC".repeat(700) + " 오늘 날씨");
    return lines;
  }

  static String normalized(Analyzer a, List<String> lines) {
    StringBuilder m = new StringBuilder();
    for (int ln = 0; ln < lines.size(); ln++) {
      m.append("N\t").append(ln).append('\t');
      try {
        BytesRef n = a.normalize("f", lines.get(ln));
        m.append(AnalysisRows.hex(n));
      } catch (Exception e) {
        m.append("X:").append(e.getClass().getSimpleName());
      }
      m.append('\n');
    }
    return m.toString();
  }

  static void chains(Path out, UserDictionary user, String corpusDir) throws Exception {
    Files.createDirectories(out);
    List<String> lines = new ArrayList<>(AnalysisRows.corpus("analysis-korean.txt"));
    Map<String, Supplier<Analyzer>> direct = new LinkedHashMap<>();
    direct.put("analyzer_default", KoreanAnalyzer::new);
    direct.put("analyzer_user_mixed", () -> new KoreanAnalyzer(user, DecompoundMode.MIXED,
        EnumSet.of(POS.Tag.JKS, POS.Tag.JKO, POS.Tag.EF, POS.Tag.SF), false));
    direct.put("analyzer_none_unigrams", () -> new KoreanAnalyzer(null, DecompoundMode.NONE,
        KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS, true));
    for (Map.Entry<String, Supplier<Analyzer>> e : direct.entrySet()) {
      try (Analyzer a = e.getValue().get()) {
        Files.writeString(out.resolve(e.getKey() + ".tsv"), AnalysisRows.rows(a, lines) + normalized(a, lines),
            StandardCharsets.UTF_8);
      }
    }
    for (String config : Files.readAllLines(Path.of(corpusDir, "analysis-nori.conf"), StandardCharsets.UTF_8)) {
      if (config.isEmpty() || config.startsWith("#")) continue;
      String[] fields = config.split("\t", -1);
      StringBuilder m = new StringBuilder();
      CustomAnalyzer a;
      try {
        a = GenAnalysisFactories.build(GenAnalysisFactories.parse(fields), Path.of(corpusDir, "analysis-factories"));
      } catch (Exception | Error e) {
        m.append("B\t").append(e.getClass().getSimpleName()).append('\t')
            .append(AnalysisRows.esc(GenAnalysisFactories.stable(e))).append('\n');
        Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
        continue;
      }
      try (a) {
        m.append("S\t").append(a.toString().replaceAll("@[0-9a-f]+", "")).append('\n');
        m.append(AnalysisRows.rows(a, lines)).append(normalized(a, lines));
      }
      Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
    }
  }

  /** The configurations whose rows carry every attribute (the rest: term, offsets, positions). */
  static final List<String> FULL = List.of("discard", "mixed_punct", "user_mixed_punct");

  static KoreanTokenizer tok(UserDictionary user, DecompoundMode mode, boolean unigrams, boolean discardPunct) {
    return new KoreanTokenizer(TokenStream.DEFAULT_TOKEN_ATTRIBUTE_FACTORY, user, mode, unigrams, discardPunct);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_nori");
    Files.createDirectories(out);
    List<String> corpus = AnalysisRows.corpus("analysis-korean.txt");
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String userText = Files.readString(Path.of(corpusDir, "analysis-korean-userdict.txt"), StandardCharsets.UTF_8);
    UserDictionary user = UserDictionary.open(new StringReader(userText));

    List<String> surfaces = surfaces();
    dictionary(out, surfaces);
    List<String> stress = stress(surfaces);
    Files.writeString(out.resolve("stress.txt"), String.join("\n", stress) + "\n", StandardCharsets.UTF_8);

    List<String> both = new ArrayList<>(corpus);
    both.addAll(stress);
    Map<String, Supplier<Tokenizer>> configs = new LinkedHashMap<>();
    configs.put("discard", () -> tok(null, DecompoundMode.DISCARD, false, true));
    configs.put("none", () -> tok(null, DecompoundMode.NONE, false, true));
    configs.put("mixed", () -> tok(null, DecompoundMode.MIXED, false, true));
    configs.put("mixed_punct", () -> tok(null, DecompoundMode.MIXED, false, false));
    configs.put("discard_unigrams_punct", () -> tok(null, DecompoundMode.DISCARD, true, false));
    configs.put("user_discard", () -> tok(user, DecompoundMode.DISCARD, false, true));
    configs.put("user_mixed_punct", () -> tok(user, DecompoundMode.MIXED, false, false));
    configs.put("user_none_unigrams", () -> tok(user, DecompoundMode.NONE, true, true));
    for (Map.Entry<String, Supplier<Tokenizer>> e : configs.entrySet()) {
      boolean attributes = FULL.contains(e.getKey());
      Files.writeString(out.resolve("tok_" + e.getKey() + ".tsv"), rows(e.getValue(), both, attributes), StandardCharsets.UTF_8);
    }

    // The hostile sweep: every mode combination over a seeded hostile corpus,
    // every attribute, one digest per line (HostileText).
    List<String> hostile = HostileText.lines(20261009L, 400);
    Files.writeString(out.resolve("hostile.txt"), String.join("\n", hostile) + "\n", StandardCharsets.UTF_8);
    StringBuilder sweep = new StringBuilder();
    for (DecompoundMode mode : DecompoundMode.values()) {
      for (boolean unigrams : new boolean[] {false, true}) {
        for (boolean punct : new boolean[] {true, false}) {
          for (UserDictionary ud : new UserDictionary[] {null, user}) {
            String name = mode.name().toLowerCase() + "_un" + (unigrams ? 1 : 0) + "_dp" + (punct ? 1 : 0)
                + "_ud" + (ud == null ? 0 : 1);
            sweep.append(HostileText.digests(name, rows(() -> tok(ud, mode, unigrams, punct), hostile, true), hostile.size()));
          }
        }
      }
    }
    Files.writeString(out.resolve("sweep.tsv"), sweep.toString(), StandardCharsets.UTF_8);

    Object[][] dots = {{"discard", DecompoundMode.DISCARD, null}, {"user_mixed", DecompoundMode.MIXED, user}};
    for (Object[] d : dots) {
      StringBuilder m = new StringBuilder();
      for (int ln = 0; ln < corpus.size(); ln++) {
        KoreanTokenizer t = tok((UserDictionary) d[2], (DecompoundMode) d[1], false, false);
        GraphvizFormatter<KoMorphData> g = new GraphvizFormatter<>(ConnectionCosts.getInstance());
        t.setGraphvizFormatter(g);
        t.setReader(new StringReader(corpus.get(ln)));
        t.reset();
        while (t.incrementToken()) {}
        t.end();
        t.close();
        m.append("== ").append(ln).append('\n').append(g.finish()).append('\n');
      }
      Files.writeString(out.resolve("graphviz_" + d[0] + ".txt"), m.toString(), StandardCharsets.UTF_8);
    }

    chains(out.resolve("chains"), user, corpusDir);

    StringBuilder lu = new StringBuilder();
    for (String line : corpus) {
      char[] c = line.toCharArray();
      lu.append(esc(line));
      for (int id : user.lookup(c, 0, c.length)) lu.append('\t').append(id);
      lu.append('\n');
    }
    Files.writeString(out.resolve("userdict_lookup.tsv"), lu.toString(), StandardCharsets.UTF_8);
  }
}
