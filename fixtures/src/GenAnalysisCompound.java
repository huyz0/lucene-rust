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
  }
}
