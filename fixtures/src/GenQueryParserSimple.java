import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.Query;

/**
 * M13 T13.1: {@code SimpleQueryParser} over {@code fixtures/corpus/queryparser-classic.txt} and
 * {@code fixtures/corpus/queryparser-simple.txt} ({@code <config>.tsv}), and over 3,000 seeded
 * joins of its operator fragments ({@code fragments.txt}, then {@code frag_<config>.tsv}). Rows
 * are {@link GenQueryParserClassic}'s: the escaped query, then {@code Q}, {@code toString("f")}
 * and the class tree, or {@code X} and the exception. {@code
 * crates/lucene-search/tests/queryparser_simple_fixtures.rs} compares.
 */
public class GenQueryParserSimple {

  static final String[] FRAGMENTS = {
    "a", "b", "foo", "Bar", "wi-fi", " ", " ", "  ", "\t", "\"", "(", ")", "+", "|", "-", "~", "~1",
    "~2", "~0", "~x", "*", "\\", "\\\"", "\\(", "é", "日本", "the", "1", "*", "~-1"
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_simple");
    Files.createDirectories(out);
    Map<String, Supplier<SimpleQueryParser>> c = new LinkedHashMap<>();
    c.put("std", () -> new SimpleQueryParser(new StandardAnalyzer(), "f"));
    c.put("std_must", () -> {
      SimpleQueryParser p = new SimpleQueryParser(new StandardAnalyzer(), "f");
      p.setDefaultOperator(BooleanClause.Occur.MUST);
      return p;
    });
    c.put("ws", () -> new SimpleQueryParser(new WhitespaceAnalyzer(), "f"));
    c.put("weights", () -> {
      Map<String, Float> w = new LinkedHashMap<>();
      w.put("f", 1.0f);
      w.put("g", 2.5f);
      return new SimpleQueryParser(new StandardAnalyzer(), w);
    });
    for (Map.Entry<String, Integer> flags : List.of(
            Map.entry("no_whitespace", ~SimpleQueryParser.WHITESPACE_OPERATOR),
            Map.entry("no_precedence", ~SimpleQueryParser.PRECEDENCE_OPERATORS),
            Map.entry("no_escape", ~SimpleQueryParser.ESCAPE_OPERATOR),
            Map.entry("no_phrase_near", ~(SimpleQueryParser.PHRASE_OPERATOR | SimpleQueryParser.NEAR_OPERATOR)),
            Map.entry("no_near", ~SimpleQueryParser.NEAR_OPERATOR),
            Map.entry("no_fuzzy_prefix", ~(SimpleQueryParser.FUZZY_OPERATOR | SimpleQueryParser.PREFIX_OPERATOR)),
            Map.entry("no_bool", ~(SimpleQueryParser.AND_OPERATOR | SimpleQueryParser.OR_OPERATOR | SimpleQueryParser.NOT_OPERATOR)),
            Map.entry("none", 0))) {
      c.put(flags.getKey(), () -> new SimpleQueryParser(new StandardAnalyzer(), Map.of("f", 1.0f), flags.getValue()));
    }
    List<String> corpus = new ArrayList<>(AnalysisRows.corpus("queryparser-classic.txt"));
    corpus.addAll(AnalysisRows.corpus("queryparser-simple.txt"));
    write(out, "", c, corpus);

    Random rnd = new Random(1311);
    List<String> lines = new ArrayList<>();
    StringBuilder file = new StringBuilder();
    for (int i = 0; i < 3000; i++) {
      StringBuilder text = new StringBuilder();
      int n = 1 + rnd.nextInt(8);
      for (int j = 0; j < n; j++) text.append(FRAGMENTS[rnd.nextInt(FRAGMENTS.length)]);
      lines.add(text.toString());
      file.append(AnalysisRows.esc(text.toString())).append('\n');
    }
    Files.writeString(out.resolve("fragments.txt"), file.toString(), StandardCharsets.UTF_8);
    Map<String, Supplier<SimpleQueryParser>> f = new LinkedHashMap<>();
    for (String name : List.of("std", "std_must", "no_whitespace", "no_escape")) f.put(name, c.get(name));
    write(out, "frag_", f, lines);
  }

  static void write(Path out, String prefix, Map<String, Supplier<SimpleQueryParser>> configs, List<String> queries)
      throws Exception {
    for (Map.Entry<String, Supplier<SimpleQueryParser>> e : configs.entrySet()) {
      StringBuilder b = new StringBuilder();
      for (String q : queries) {
        b.append(AnalysisRows.esc(q)).append('\t');
        try {
          Query parsed = e.getValue().get().parse(q);
          b.append("Q\t").append(AnalysisRows.esc(parsed.toString("f")))
              .append('\t').append(AnalysisRows.esc(GenQueryParserClassic.describe(parsed)));
        } catch (RuntimeException ex) {
          b.append("X\t").append(AnalysisRows.esc(ex.getClass().getSimpleName() + ": " + ex.getMessage()));
        }
        b.append('\n');
      }
      Files.writeString(out.resolve(prefix + e.getKey() + ".tsv"), b.toString(), StandardCharsets.UTF_8);
    }
  }
}
