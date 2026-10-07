import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.compound.DictionaryCompoundWordTokenFilter;
import org.apache.lucene.analysis.compound.HyphenationCompoundWordTokenFilter;
import org.apache.lucene.analysis.compound.hyphenation.Hyphenation;
import org.apache.lucene.analysis.compound.hyphenation.HyphenationTree;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.xml.sax.InputSource;

/**
 * M11 T11.6: the compound package. Loads {@code fixtures/corpus/hyphenation-test.xml} (a toy
 * grammar written for this project) into a {@link HyphenationTree}, records its hyphenation points
 * for words ({@code points.words}: "remain push word -> points|-") and runs the dictionary and
 * hyphenation decompounders, with their options, over {@code fixtures/corpus/analysis-compound.txt}
 * ({@code <chain>.tsv}, {@link AnalysisRows}' rows). {@code
 * crates/lucene-analysis/tests/analysis_compound_fixtures.rs} compares.
 */
public class GenAnalysisCompound {

  static final String[] DICT = {
    "rind", "fleisch", "donau", "dampf", "schiff", "fahrt", "kapitän", "fuss", "ball", "pumpe", "haus",
    "tür", "schloss", "garten", "zaun", "bahn", "hof", "kinder", "wagen", "see", "sonne", "blume",
    "brot", "kasten", "wurst", "ufer", "kaffee", "abend", "essen", "weiß", "kühl", "schrank", "bahnhof"
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_compound");
    Files.createDirectories(out);
    String dir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    HyphenationTree tree =
        HyphenationCompoundWordTokenFilter.getHyphenationTree(
            new InputSource(Path.of(dir, "hyphenation-test.xml").toUri().toString()));
    CharArraySet dict = new CharArraySet(List.of(DICT), true);

    List<String> lines = AnalysisRows.corpus("analysis-compound.txt");
    Set<String> words = new LinkedHashSet<>();
    for (String l : lines) for (String w : l.split(" ")) if (!w.isEmpty()) { words.add(w); words.add(w.toLowerCase(java.util.Locale.ROOT)); }
    for (String w : DICT) words.add(w);
    StringBuilder pts = new StringBuilder();
    for (int[] rp : new int[][] {{1, 1}, {2, 3}, {0, 0}}) {
      for (String w : words) {
        Hyphenation h = tree.hyphenate(w, rp[0], rp[1]);
        pts.append(rp[0]).append('\t').append(rp[1]).append('\t').append(AnalysisRows.esc(w)).append('\t');
        if (h == null) {
          pts.append('-');
        } else {
          int[] p = h.getHyphenationPoints();
          for (int i = 0; i < p.length; i++) pts.append(i == 0 ? "" : ",").append(p[i]);
        }
        pts.append('\n');
      }
    }
    Files.writeString(out.resolve("points.words"), pts.toString(), StandardCharsets.UTF_8);

    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("ws_lower_dict", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new DictionaryCompoundWordTokenFilter(new LowerCaseFilter(t), dict)));
    c.put("ws_dict_cased", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new DictionaryCompoundWordTokenFilter(t, dict)));
    c.put("ws_lower_dict_longest", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new DictionaryCompoundWordTokenFilter(new LowerCaseFilter(t), dict, 5, 2, 15, true, false)));
    c.put("ws_lower_dict_no_subwords", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new DictionaryCompoundWordTokenFilter(new LowerCaseFilter(t), dict, 5, 2, 15, true)));
    c.put("ws_lower_dict_sizes", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new DictionaryCompoundWordTokenFilter(new LowerCaseFilter(t), dict, 8, 3, 5, false)));
    c.put("std_lower_dict", () -> AnalysisRows.chain(StandardTokenizer::new, t -> new DictionaryCompoundWordTokenFilter(new LowerCaseFilter(t), dict)));
    c.put("ws_lower_hyph", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree)));
    c.put("ws_hyph_cased", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new HyphenationCompoundWordTokenFilter(t, tree)));
    c.put("ws_lower_hyph_dict", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree, dict)));
    c.put("ws_lower_hyph_dict_longest", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree, dict, 5, 2, 15, true)));
    c.put("ws_lower_hyph_dict_nosub", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree, dict, 5, 2, 15, false, true, false)));
    c.put("ws_lower_hyph_dict_nooverlap", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree, dict, 5, 2, 15, false, false, true)));
    c.put("ws_lower_hyph_sizes", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new HyphenationCompoundWordTokenFilter(new LowerCaseFilter(t), tree, 6, 3, 6)));
    AnalysisRows.writeChains(out, c, lines);
    random(out);
  }

  static final String RANDOM_ALPHA = "abcdefghä";

  /**
   * A random grammar at the scale of a real one ({@code random.xml}: 3000 patterns over {@link
   * #RANDOM_ALPHA}, its classes, 30 exceptions), which builds Java's {@code TernaryTree}s past their
   * first growth and balancing; its points for 4000 random words under random remain/push ({@code
   * random_points.words}); and 32 random decompounder configurations over 40 random lines ({@code
   * random_lines.txt}, {@code random_configs.txt}: a {@code D} row per dictionary word, a {@code C}
   * row per configuration; {@code random_chains.rows}: {@link AnalysisRows}' rows, each prefixed with
   * its configuration's number).
   */
  static void random(Path out) throws Exception {
    java.util.Random r = new java.util.Random(7);
    String alpha = RANDOM_ALPHA;
    StringBuilder x = new StringBuilder("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<hyphenation-info>\n<hyphen-char value=\"-\"/>\n<classes>\n");
    for (char ch : alpha.toCharArray()) x.append(ch).append(Character.toUpperCase(ch)).append(' ');
    x.append("\n</classes>\n<exceptions>\n");
    for (int i = 0; i < 30; i++) {
      StringBuilder w = new StringBuilder();
      int n = 3 + r.nextInt(6);
      for (int k = 0; k < n; k++) {
        w.append(alpha.charAt(r.nextInt(alpha.length())));
        if (k < n - 1 && r.nextInt(3) == 0) w.append('-');
      }
      x.append(w).append(i % 5 == 0 ? "\n" : " ");
    }
    x.append("\n</exceptions>\n<patterns>\n");
    for (int i = 0; i < 3000; i++) {
      StringBuilder p = new StringBuilder();
      int n = 1 + r.nextInt(7);
      if (r.nextInt(6) == 0) p.append('.');
      for (int k = 0; k < n; k++) {
        if (r.nextInt(3) == 0) p.append((char) ('0' + r.nextInt(10)));
        p.append(alpha.charAt(r.nextInt(alpha.length())));
      }
      if (r.nextInt(3) == 0) p.append((char) ('0' + r.nextInt(10)));
      if (r.nextInt(6) == 0) p.append('.');
      x.append(p).append(i % 12 == 11 ? "\n" : " ");
    }
    x.append("\n</patterns>\n</hyphenation-info>\n");
    Path xml = out.resolve("random.xml");
    Files.writeString(xml, x.toString(), StandardCharsets.UTF_8);
    HyphenationTree tree =
        HyphenationCompoundWordTokenFilter.getHyphenationTree(new InputSource(xml.toUri().toString()));

    StringBuilder pts = new StringBuilder();
    String walpha = alpha + "ABH-1 ß";
    for (int i = 0; i < 4000; i++) {
      StringBuilder w = new StringBuilder();
      int n = r.nextInt(i % 100 == 0 ? 60 : 14);
      for (int k = 0; k < n; k++) w.append(walpha.charAt(r.nextInt(walpha.length())));
      int rem = r.nextInt(4), push = r.nextInt(4);
      Hyphenation h = tree.hyphenate(w.toString(), rem, push);
      pts.append(rem).append('\t').append(push).append('\t').append(AnalysisRows.esc(w.toString())).append('\t');
      if (h == null) {
        pts.append('-');
      } else {
        int[] p = h.getHyphenationPoints();
        for (int k = 0; k < p.length; k++) pts.append(k == 0 ? "" : ",").append(p[k]);
      }
      pts.append('\n');
    }
    Files.writeString(out.resolve("random_points.words"), pts.toString(), StandardCharsets.UTF_8);

    String lalpha = alpha + "ABH";
    List<String> dict = new java.util.ArrayList<>();
    StringBuilder cfg = new StringBuilder();
    for (int i = 0; i < 300; i++) {
      StringBuilder w = new StringBuilder();
      int n = 1 + r.nextInt(6);
      for (int k = 0; k < n; k++) w.append(alpha.charAt(r.nextInt(alpha.length())));
      dict.add(w.toString());
      cfg.append("D\t").append(w).append('\n');
    }
    CharArraySet ds = new CharArraySet(dict, true);
    List<String> lines = new java.util.ArrayList<>();
    StringBuilder lf = new StringBuilder();
    for (int i = 0; i < 40; i++) {
      StringBuilder l = new StringBuilder();
      int n = r.nextInt(4);
      for (int w = 0; w < n; w++) {
        if (w > 0) l.append(' ');
        int k = r.nextInt(i % 20 == 0 ? 80 : 16);
        for (int j = 0; j < k; j++) l.append(lalpha.charAt(r.nextInt(lalpha.length())));
      }
      lines.add(l.toString());
      lf.append(l).append('\n');
    }
    StringBuilder rows = new StringBuilder();
    for (int c = 0; c < 32; c++) {
      int minW = r.nextInt(7), minS = r.nextInt(5), maxS = r.nextInt(12);
      boolean longest = r.nextBoolean(), noSub = r.nextBoolean(), noOverlap = r.nextBoolean();
      boolean useDict = r.nextBoolean(), hyph = r.nextBoolean();
      cfg.append("C\t").append(c).append('\t').append(minW).append('\t').append(minS).append('\t').append(maxS)
          .append('\t').append(longest).append('\t').append(noSub).append('\t').append(noOverlap)
          .append('\t').append(useDict).append('\t').append(hyph).append('\n');
      try (Analyzer an = hyph
          ? AnalysisRows.chain(WhitespaceTokenizer::new,
              t -> new HyphenationCompoundWordTokenFilter(t, tree, useDict ? ds : null, minW, minS, maxS, longest, noSub, noOverlap))
          : AnalysisRows.chain(WhitespaceTokenizer::new,
              t -> new DictionaryCompoundWordTokenFilter(t, ds, minW, minS, maxS, longest, noSub))) {
        for (String row : AnalysisRows.rows(an, lines).split("\n")) rows.append(c).append('\t').append(row).append('\n');
      }
    }
    Files.writeString(out.resolve("random_lines.txt"), lf.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("random_configs.txt"), cfg.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("random_chains.rows"), rows.toString(), StandardCharsets.UTF_8);
  }
}
