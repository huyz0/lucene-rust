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
import org.apache.lucene.analysis.classic.ClassicAnalyzer;
import org.apache.lucene.analysis.classic.ClassicFilter;
import org.apache.lucene.analysis.classic.ClassicTokenizer;
import org.apache.lucene.analysis.wikipedia.WikipediaTokenizer;

/**
 * M11 T11.2: {@code ClassicTokenizer}, {@code ClassicFilter}, {@code ClassicAnalyzer} and {@code
 * WikipediaTokenizer} (all three output modes) over {@code fixtures/corpus/analysis-classic.txt}
 * ({@code <chain>.tsv}), and over 2,000 seeded joins of wiki-markup and classic-grammar fragments
 * ({@code fragments.txt}, then {@code frag_<chain>.tsv}). Rows are {@link AnalysisRows}'. {@code
 * crates/lucene-analysis/tests/analysis_classic_fixtures.rs} compares.
 */
public class GenAnalysisClassic {

  static final String[] FRAGMENTS = {
    "[[", "]]", "[", "]", "{{", "}}", "''", "'''", "'''''", "==", "===", "|", ":", "Category:",
    "Image:", "http://", "https://x.org", "www.", "example", ".com", "@", "&", "'", "'s", ".",
    "I.B.M.", "AT&T", "1.2", "42", " ", "  ", "\t", "\n", "word", "Word", "中文", "日本", "é", "-",
    "_", "/", "mailto:", "=", "!", "#", "ab", "x.y.", "O'Neil"
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_classic");
    Files.createDirectories(out);
    Set<String> some = Set.of("il", "c", "b", "h");
    Set<String> all = Set.of("il", "el", "elu", "ci", "c", "b", "i", "bi", "h", "sh");
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("classic_tokenizer", () -> AnalysisRows.tok(ClassicTokenizer::new));
    c.put("classic_tokenizer_max5", () -> AnalysisRows.tok(() -> {
      ClassicTokenizer t = new ClassicTokenizer();
      t.setMaxTokenLength(5);
      return t;
    }));
    c.put("classic_filter", () -> AnalysisRows.chain(ClassicTokenizer::new, ClassicFilter::new));
    c.put("classic_analyzer", ClassicAnalyzer::new);
    c.put("classic_analyzer_max4", () -> {
      ClassicAnalyzer a = new ClassicAnalyzer();
      a.setMaxTokenLength(4);
      return a;
    });
    c.put("wikipedia_tokens_only", () -> AnalysisRows.tok(WikipediaTokenizer::new));
    c.put("wikipedia_untokenized_some", () -> AnalysisRows.tok(() -> new WikipediaTokenizer(WikipediaTokenizer.UNTOKENIZED_ONLY, some)));
    c.put("wikipedia_both_some", () -> AnalysisRows.tok(() -> new WikipediaTokenizer(WikipediaTokenizer.BOTH, some)));
    c.put("wikipedia_both_all", () -> AnalysisRows.tok(() -> new WikipediaTokenizer(WikipediaTokenizer.BOTH, all)));
    c.put("wikipedia_untokenized_all", () -> AnalysisRows.tok(() -> new WikipediaTokenizer(WikipediaTokenizer.UNTOKENIZED_ONLY, all)));
    AnalysisRows.writeChains(out, c, AnalysisRows.corpus("analysis-classic.txt"));

    // Seeded joins of fragments: one per line (a "\n" fragment is written escaped).
    Random rnd = new Random(1133);
    List<String> lines = new ArrayList<>();
    StringBuilder file = new StringBuilder();
    for (int i = 0; i < 2000; i++) {
      StringBuilder text = new StringBuilder();
      int n = 1 + rnd.nextInt(12);
      for (int j = 0; j < n; j++) text.append(FRAGMENTS[rnd.nextInt(FRAGMENTS.length)]);
      lines.add(text.toString());
      file.append(AnalysisRows.esc(text.toString())).append('\n');
    }
    Files.writeString(out.resolve("fragments.txt"), file.toString(), StandardCharsets.UTF_8);
    Map<String, Supplier<Analyzer>> f = new LinkedHashMap<>();
    for (String name : List.of("classic_filter", "wikipedia_tokens_only", "wikipedia_both_all", "wikipedia_untokenized_some")) {
      f.put("frag_" + name, c.get(name));
    }
    AnalysisRows.writeChains(out, f, lines);
  }
}
