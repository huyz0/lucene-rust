import java.io.StringReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.analysis.ja.JapaneseCompletionAnalyzer;
import org.apache.lucene.analysis.ja.JapaneseCompletionFilter;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.ja.JapaneseTokenizer;
import org.apache.lucene.analysis.ja.JapaneseTokenizer.Mode;
import org.apache.lucene.analysis.ja.dict.ConnectionCosts;
import org.apache.lucene.analysis.ja.dict.ToStringUtil;
import org.apache.lucene.analysis.ja.dict.TokenInfoDictionary;
import org.apache.lucene.analysis.ja.dict.UserDictionary;
import org.apache.lucene.analysis.ja.tokenattributes.BaseFormAttribute;
import org.apache.lucene.analysis.ja.tokenattributes.InflectionAttribute;
import org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttribute;
import org.apache.lucene.analysis.ja.tokenattributes.ReadingAttribute;
import org.apache.lucene.analysis.morph.GraphvizFormatter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.IntsRefFSTEnum;

/**
 * M12 T12.1: Lucene's analysis-kuromoji module -- JapaneseTokenizer over the IPADIC dictionary.
 *
 * <p>Writes {@code analysis_kuromoji/}:
 *
 * <ul>
 *   <li>{@code tok_<config>.tsv}: {@code corpus/analysis-japanese.txt} and a generated stress corpus
 *       ({@code stress.txt}, random runs of dictionary surfaces, kana, kanji, digits and
 *       punctuation, seeded) through {@link JapaneseTokenizer} in every mode, with and without
 *       punctuation and compound tokens, n-best costs and the user dictionary {@code
 *       corpus/analysis-japanese-userdict.txt}. Rows are {@code T line term start end posInc posLen}
 *       then, for four configurations, every Kuromoji attribute's {@code reflectWith} values (their keys
 *       head the file: {@code K key...}), {@code E line
 *       finalStart finalEnd finalPosInc} after each line, {@code X line Exception} on failure.
 *   <li>{@code sweep.tsv}: the hostile sweep -- a seeded corpus of 400 lines from {@link
 *       HostileText}'s pools ({@code hostile.txt}) through every combination of mode, punctuation,
 *       compounds, user dictionary and n-best cost (0 or 2000), 48 configurations, rows with
 *       every attribute summarised per line as {@code config line rows fnv1a64}.
 *   <li>{@code graphviz_<config>.txt}: the {@link GraphvizFormatter} lattice of each corpus line.
 *   <li>{@code nbest_examples.tsv}: {@code calcNBestCost} of example strings.
 *   <li>{@code dictionary.tsv}: every 80th surface of the system dictionary's FST with each of its
 *       words' ids, connection ids, cost, part of speech, inflection, base form, reading and
 *       pronunciation.
 *   <li>{@code userdict_lookup.tsv}: {@code UserDictionary.lookup} over every corpus line.
 *   <li>{@code romanization.tsv}: {@code ToStringUtil.getRomanization} of every katakana pair
 *       followed by ウ and by ェ; {@code translations.tsv}: its three translation tables.
 *   <li>{@code chains/<name>.tsv}: {@code JapaneseAnalyzer} and {@code JapaneseCompletionAnalyzer}
 *       configurations, and every configuration of {@code corpus/analysis-kuromoji.conf} through
 *       {@code CustomAnalyzer} (the factories), over the corpus and {@code
 *       corpus/analysis-japanese-filters.txt}: {@link AnalysisRows}' rows, {@code S analyzer} (a
 *       configuration's {@code toString}), {@code N line normalized-bytes}, or {@code B Exception
 *       message} when it does not build.
 * </ul>
 *
 * Runs with lucene-analysis-kuromoji on its own classpath ({@code generator_classpath} in {@code
 * scripts/gen-fixtures.sh}). Deterministic. Read by {@code
 * crates/lucene-analysis-kuromoji/tests/kuromoji_fixtures.rs}.
 */
public class GenAnalysisKuromoji {

  static String esc(String s) {
    return s == null ? "null" : AnalysisRows.esc(s);
  }

  /**
   * The Kuromoji attributes' reflected values, tab-separated; their keys (always the same eleven,
   * in {@code reflectWith} order) are the file's first row, {@code K key...}.
   */
  static String reflect(TokenStream ts, StringBuilder keys) {
    StringBuilder b = new StringBuilder();
    StringBuilder k = new StringBuilder();
    ts.reflectWith(
        (attClass, key, value) -> {
          if (attClass.getName().startsWith("org.apache.lucene.analysis.ja.")) {
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

  static void dictionary(Path out, List<String> surfaces) throws Exception {
    TokenInfoDictionary dict = TokenInfoDictionary.getInstance();
    FST<Long> fst = systemFst();
    StringBuilder m = new StringBuilder();
    IntsRef ref = new IntsRef();
    for (int s = 0; s < surfaces.size(); s += 80) {
      String surface = surfaces.get(s);
      IntsRefBuilder key = new IntsRefBuilder();
      for (int i = 0; i < surface.length(); i++) key.append(surface.charAt(i));
      Long output = org.apache.lucene.util.fst.Util.get(fst, key.get());
      dict.lookupWordIds(output.intValue(), ref);
      char[] chars = surface.toCharArray();
      for (int i = 0; i < ref.length; i++) {
        int id = ref.ints[ref.offset + i];
        org.apache.lucene.analysis.ja.dict.JaMorphData md = dict.getMorphAttributes();
        m.append(esc(surface)).append('\t').append(output).append('\t').append(id)
            .append('\t').append(md.getLeftId(id)).append('\t').append(md.getRightId(id))
            .append('\t').append(md.getWordCost(id))
            .append('\t').append(esc(md.getPartOfSpeech(id)))
            .append('\t').append(esc(md.getInflectionType(id)))
            .append('\t').append(esc(md.getInflectionForm(id)))
            .append('\t').append(esc(md.getBaseForm(id, chars, 0, chars.length)))
            .append('\t').append(esc(md.getReading(id, chars, 0, chars.length)))
            .append('\t').append(esc(md.getPronunciation(id, chars, 0, chars.length)))
            .append('\n');
      }
    }
    Files.writeString(out.resolve("dictionary.tsv"), m.toString(), StandardCharsets.UTF_8);
  }

  /** Random runs of dictionary surfaces and characters, seeded. */
  static List<String> stress(List<String> surfaces) {
    Random r = new Random(20261007L);
    String extra = "、。！？・「」（）ー〜 　\t0123456789０１２３ABCabcｱｲｳｴｵ々ゝゞヽヾ😀𠮷";
    List<String> lines = new ArrayList<>();
    for (int n = 0; n < 300; n++) {
      StringBuilder b = new StringBuilder();
      int words = 1 + r.nextInt(n < 290 ? 20 : 300);
      for (int w = 0; w < words; w++) {
        int k = r.nextInt(10);
        if (k < 7) {
          b.append(surfaces.get(r.nextInt(surfaces.size())));
        } else if (k < 9) {
          b.appendCodePoint(extra.codePoints().toArray()[r.nextInt((int) extra.codePoints().count())]);
        } else {
          // kana and kanji runs
          int base = r.nextBoolean() ? 0x3041 : (r.nextBoolean() ? 0x30A1 : 0x4E00);
          int span = base == 0x4E00 ? 0x5000 : 0x56;
          for (int c = r.nextInt(6); c >= 0; c--) b.append((char) (base + r.nextInt(span)));
        }
      }
      lines.add(b.toString());
    }
    // A long run without a frontier forces the 1024-position backtrace.
    lines.add("あ".repeat(3000));
    lines.add("漢".repeat(1500) + "字".repeat(1500));
    lines.add("ア".repeat(2100) + "本日は晴天なり");
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
    List<String> lines = new ArrayList<>(AnalysisRows.corpus("analysis-japanese.txt"));
    lines.addAll(AnalysisRows.corpus("analysis-japanese-filters.txt"));
    Map<String, Supplier<Analyzer>> direct = new LinkedHashMap<>();
    direct.put("analyzer_default", JapaneseAnalyzer::new);
    direct.put("analyzer_user_normal", () -> new JapaneseAnalyzer(user, Mode.NORMAL,
        JapaneseAnalyzer.getDefaultStopSet(), JapaneseAnalyzer.getDefaultStopTags()));
    direct.put("analyzer_extended_nostop", () -> new JapaneseAnalyzer(null, Mode.EXTENDED, CharArraySet.EMPTY_SET, Set.of()));
    direct.put("completion_index", JapaneseCompletionAnalyzer::new);
    direct.put("completion_query", () -> new JapaneseCompletionAnalyzer(user, JapaneseCompletionFilter.Mode.QUERY));
    for (Map.Entry<String, Supplier<Analyzer>> e : direct.entrySet()) {
      try (Analyzer a = e.getValue().get()) {
        Files.writeString(out.resolve(e.getKey() + ".tsv"), AnalysisRows.rows(a, lines) + normalized(a, lines),
            StandardCharsets.UTF_8);
      }
    }
    for (String config : Files.readAllLines(Path.of(corpusDir, "analysis-kuromoji.conf"), StandardCharsets.UTF_8)) {
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

  @SuppressWarnings("unchecked")
  static void toStringUtil(Path out) throws Exception {
    StringBuilder m = new StringBuilder();
    for (char c1 = 0x30A0; c1 <= 0x30FF; c1++) {
      for (char c2 = 0x30A0; c2 <= 0x3100; c2++) {
        for (String c3 : new String[] {"ウ", "ェ"}) {
          String s = "" + c1 + c2 + c3;
          m.append(s).append('\t').append(ToStringUtil.getRomanization(s)).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("romanization.tsv"), m.toString(), StandardCharsets.UTF_8);
    StringBuilder t = new StringBuilder();
    for (String f : new String[] {"posTranslations", "inflTypeTranslations", "inflFormTranslations"}) {
      Field fl = ToStringUtil.class.getDeclaredField(f);
      fl.setAccessible(true);
      for (Map.Entry<String, String> e : new java.util.TreeMap<>((Map<String, String>) fl.get(null)).entrySet()) {
        t.append(f).append('\t').append(e.getKey()).append('\t').append(e.getValue()).append('\n');
      }
    }
    Files.writeString(out.resolve("translations.tsv"), t.toString(), StandardCharsets.UTF_8);
  }

  /** The configurations whose rows carry every attribute (the rest: term, offsets, positions). */
  static final List<String> FULL = List.of("normal", "search_compound", "user_search_compound", "nbest_normal_2000");

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_kuromoji");
    Files.createDirectories(out);
    List<String> corpus = AnalysisRows.corpus("analysis-japanese.txt");
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String userText = Files.readString(Path.of(corpusDir, "analysis-japanese-userdict.txt"), StandardCharsets.UTF_8);
    UserDictionary user = UserDictionary.open(new StringReader(userText));

    List<String> surfaces = surfaces();
    dictionary(out, surfaces);
    List<String> stress = stress(surfaces);
    Files.writeString(out.resolve("stress.txt"), String.join("\n", stress) + "\n", StandardCharsets.UTF_8);

    List<String> both = new ArrayList<>(corpus);
    both.addAll(stress);
    Map<String, Supplier<Tokenizer>> configs = new LinkedHashMap<>();
    configs.put("normal", () -> new JapaneseTokenizer(null, true, true, Mode.NORMAL));
    configs.put("normal_punct", () -> new JapaneseTokenizer(null, false, true, Mode.NORMAL));
    configs.put("search", () -> new JapaneseTokenizer(null, true, true, Mode.SEARCH));
    configs.put("search_compound", () -> new JapaneseTokenizer(null, true, false, Mode.SEARCH));
    configs.put("search_punct_compound", () -> new JapaneseTokenizer(null, false, false, Mode.SEARCH));
    configs.put("extended", () -> new JapaneseTokenizer(null, true, true, Mode.EXTENDED));
    configs.put("extended_punct_compound", () -> new JapaneseTokenizer(null, false, false, Mode.EXTENDED));
    configs.put("user_normal", () -> new JapaneseTokenizer(user, true, true, Mode.NORMAL));
    configs.put("user_search_compound", () -> new JapaneseTokenizer(user, false, false, Mode.SEARCH));
    configs.put("user_extended", () -> new JapaneseTokenizer(user, true, true, Mode.EXTENDED));
    int[][] nbest = {{0, 500}, {0, 2000}, {1, 2000}, {2, 1000}, {0, 10000}};
    for (int[] nb : nbest) {
      Mode mode = Mode.values()[nb[0]];
      configs.put("nbest_" + mode.name().toLowerCase() + "_" + nb[1], () -> {
        JapaneseTokenizer t = new JapaneseTokenizer(null, true, true, mode);
        t.setNBestCost(nb[1]);
        return t;
      });
    }
    configs.put("nbest_user_punct_2000", () -> {
      JapaneseTokenizer t = new JapaneseTokenizer(user, false, false, Mode.SEARCH);
      t.setNBestCost(2000);
      return t;
    });
    for (Map.Entry<String, Supplier<Tokenizer>> e : configs.entrySet()) {
      boolean attributes = FULL.contains(e.getKey());
      Files.writeString(out.resolve("tok_" + e.getKey() + ".tsv"), rows(e.getValue(), both, attributes), StandardCharsets.UTF_8);
    }

    // The hostile sweep: every mode combination over a seeded hostile corpus,
    // every attribute, one digest per line (HostileText).
    List<String> hostile = HostileText.lines(20261008L, 400);
    Files.writeString(out.resolve("hostile.txt"), String.join("\n", hostile) + "\n", StandardCharsets.UTF_8);
    StringBuilder sweep = new StringBuilder();
    for (Mode mode : Mode.values()) {
      for (boolean punct : new boolean[] {true, false}) {
        for (boolean compound : new boolean[] {true, false}) {
          for (UserDictionary ud : new UserDictionary[] {null, user}) {
            for (int nb : new int[] {0, 2000}) {
              String name = mode.name().toLowerCase() + "_dp" + (punct ? 1 : 0) + "_dc" + (compound ? 1 : 0)
                  + "_ud" + (ud == null ? 0 : 1) + "_nb" + nb;
              Supplier<Tokenizer> s = () -> {
                JapaneseTokenizer t = new JapaneseTokenizer(ud, punct, compound, mode);
                if (nb > 0) t.setNBestCost(nb);
                return t;
              };
              sweep.append(HostileText.digests(name, rows(s, hostile, true), hostile.size()));
            }
          }
        }
      }
    }
    Files.writeString(out.resolve("sweep.tsv"), sweep.toString(), StandardCharsets.UTF_8);

    // The lattice of each corpus line.
    Object[][] dots = {{"normal", Mode.NORMAL, null}, {"user_search", Mode.SEARCH, user}};
    for (Object[] d : dots) {
      StringBuilder m = new StringBuilder();
      for (int ln = 0; ln < corpus.size(); ln++) {
        JapaneseTokenizer t = new JapaneseTokenizer((UserDictionary) d[2], false, true, (Mode) d[1]);
        GraphvizFormatter<org.apache.lucene.analysis.ja.dict.JaMorphData> g =
            new GraphvizFormatter<>(ConnectionCosts.getInstance());
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

    // calcNBestCost
    String[] examples = {
      "", "/", "関西国際空港-国際", "東京特許許可局-許可", "すもももももももものうち-もも",
      "今日は良い天気-天気/国立国会図書館-国会", "あいうえお-かき", "日本経済新聞-経済新聞", "x", "a-b-c", "-関西"
    };
    StringBuilder nb = new StringBuilder();
    for (String ex : examples) {
      for (Mode mode : Mode.values()) {
        String v;
        try {
          v = Integer.toString(new JapaneseTokenizer(null, true, true, mode).calcNBestCost(ex));
        } catch (Exception e) {
          v = "!" + e.getClass().getSimpleName();
        }
        nb.append(esc(ex)).append('\t').append(mode).append('\t').append(v).append('\n');
      }
    }
    Files.writeString(out.resolve("nbest_examples.tsv"), nb.toString(), StandardCharsets.UTF_8);

    chains(out.resolve("chains"), user, corpusDir);
    toStringUtil(out);

    StringBuilder lu = new StringBuilder();
    for (String line : corpus) {
      char[] c = line.toCharArray();
      lu.append(esc(line));
      for (int[] r : user.lookup(c, 0, c.length)) {
        lu.append('\t').append(r[0]).append(',').append(r[1]).append(',').append(r[2]);
      }
      lu.append('\n');
    }
    Files.writeString(out.resolve("userdict_lookup.tsv"), lu.toString(), StandardCharsets.UTF_8);
  }
}
