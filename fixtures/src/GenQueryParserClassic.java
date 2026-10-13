import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.CharsRef;

/**
 * M13 T13.1: the classic {@code QueryParser} over {@code fixtures/corpus/queryparser-classic.txt}
 * ({@code <config>.tsv}) and over 3,000 seeded joins of syntax fragments ({@code fragments.txt},
 * then {@code frag_<config>.tsv}), under several parser configurations. A row is the escaped query
 * string, then {@code Q}, the query's {@code toString("f")} and its class tree, or {@code E} and
 * the {@code ParseException} message (position and expected tokens included), or {@code X} and
 * any other exception's class and message. {@code
 * crates/lucene-search/tests/queryparser_classic_fixtures.rs} compares.
 */
public class GenQueryParserClassic {

  static final String[] FRAGMENTS = {
    "a", "b", "foo", "Bar", "title", ":", " ", " ", "  ", "\t", "\"", "(", ")", "[", "]", "{", "}",
    " TO ", "TO", "*", "?", "~", "~2", "~0.5", "^", "^2", "^.5", "+", "-", "!", "&&", "||", " AND ",
    " OR ", " NOT ", "AND", "\\", "\\:", "\\\"", "/", "/x.*/", "1", "2.5", "é", "日本", "wi-fi", "the",
    "*:*", "-1", "a*", "?b", "\\u0041", "\\u00", "\\u", "x\\", ".", ","
  };

  /**
   * Queries the corpus file cannot hold as plain lines (control characters, U+3000), and the
   * lexer's rarer paths: non-ASCII text in every lexical state, unterminated tokens at the end of
   * the input, and the operator spellings' prefixes.
   */
  static final String[] EXTRA = {
    "[é TO 中]", "{日本 TO 語}", "[\"é\" TO \"\\\"中\"]", "[\"a\\\"b\" TO c]",
    "[a　TO　b]", "a　b", "+　a", "-　a", "!　a", "a^é", "a^　",
    "a~é", "a~2é", "\"a\"~é", "/é/", "/a\\/é/", "é*", "*é", "é?b",
    "a\\é*", "\"é\\中\"", "a\\中", "\\é", "a\\　b", "(\u0008", ")\u000c",
    "a\u0001(", "\u0007:", "'(", "a'", "\"a\u0008", "\"x\u000c", "\"'", "/\u0008", "&", "|", "&&&",
    "|||", "AN", "O", "NO", "N", "A", "TO", "AND)", "a OR)", "[T", "[TO", "[TOx TO y]", "[a TO T]",
    "[a TO TO]", "[a TO b c]", "\"a\n", "\"a\r", "a\\", "[a TO b]^é", "a^1.", "a^1.x", "a^.",
    "(a)^2.5^3", "a^2~", "\"a b\"~1.9", "\"a b\"~-2", "\"a b\"~x", "a~-1", "a~1.0", "a~NaN",
    "a~Infinity", "a~1e0", "a~2f", "/[/", "/a{2,1}/", "/<foo>/", "é", "😀",
    "a😀*", "*:é*", "*:a", "*:*^2", "title:*", "\\u00e9\\u4E2D", "\\u00zz", "a\\u12",
    "\u0000", "a\u0085b", "a b", "\\*\\*a", "a\\*\\?b*", "\\**", "[a\\ TO b]", "a~0", "fast", "Fast car", "\"fast car\"~2", "\"fast car\"~-1", "fast*", "car~1", "x fast", "/(a|b)*a(a|b){20}/", "/(a|b)*a(a|b){5}/", "a?b?c?d?e?f?g?h*i*j*k*", "a\\ *", "a\\😀*"
  };

  /** Whitespace, lower case, and the single-word synonyms fast = quick, car = auto. */
  static Analyzer synonyms() {
    SynonymMap.Builder b = new SynonymMap.Builder(true);
    b.add(new CharsRef("fast"), new CharsRef("quick"), true);
    b.add(new CharsRef("car"), new CharsRef("auto"), true);
    SynonymMap map;
    try {
      map = b.build();
    } catch (java.io.IOException e) {
      throw new java.io.UncheckedIOException(e);
    }
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer t = new org.apache.lucene.analysis.core.WhitespaceTokenizer();
        return new TokenStreamComponents(t, new SynonymGraphFilter(new LowerCaseFilter(t), map, true));
      }
    };
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_classic");
    Files.createDirectories(out);
    Map<String, Supplier<QueryParser>> c = new LinkedHashMap<>();
    c.put("std", () -> new QueryParser("f", new StandardAnalyzer()));
    c.put("std_and", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setDefaultOperator(QueryParser.Operator.AND);
      return p;
    });
    c.put("ws", () -> new QueryParser("f", new WhitespaceAnalyzer()));
    c.put("std_leading", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setAllowLeadingWildcard(true);
      return p;
    });
    c.put("std_autophrase", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setSplitOnWhitespace(true); // autoGeneratePhraseQueries needs it
      p.setAutoGeneratePhraseQueries(true);
      return p;
    });
    c.put("std_split", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setSplitOnWhitespace(true);
      return p;
    });
    c.put("std_tuned", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setFuzzyMinSim(1.0f);
      p.setFuzzyPrefixLength(2);
      p.setPhraseSlop(1);
      p.setEnablePositionIncrements(false);
      return p;
    });
    c.put("stop", () -> new QueryParser("f", new StopAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)));
    c.put("keyword", () -> new QueryParser("f", new KeywordAnalyzer()));
    String[] fields = {"title", "body"};
    c.put("multi", () -> new MultiFieldQueryParser(fields, new StandardAnalyzer()));
    c.put("multi_boost", () -> new MultiFieldQueryParser(fields, new StandardAnalyzer(), Map.of("title", 2.0f, "body", 0.5f)));
    c.put("syn", () -> new QueryParser("f", synonyms()));
    c.put("std_negative", () -> {
      QueryParser p = new QueryParser("f", new StandardAnalyzer());
      p.setFuzzyPrefixLength(-1);
      p.setPhraseSlop(-1);
      return p;
    });
    c.put("multi_negative", () -> {
      QueryParser p = new MultiFieldQueryParser(fields, new StandardAnalyzer());
      p.setPhraseSlop(-1);
      return p;
    });
    c.put("multi_perfield", () -> new MultiFieldQueryParser(fields,
        new PerFieldAnalyzerWrapper(new StandardAnalyzer(), Map.of("title", new KeywordAnalyzer()))));
    c.put("multi_and", () -> {
      QueryParser p = new MultiFieldQueryParser(fields, new StandardAnalyzer());
      p.setDefaultOperator(QueryParser.Operator.AND);
      return p;
    });
    List<String> corpus = new ArrayList<>(AnalysisRows.corpus("queryparser-classic.txt"));
    corpus.addAll(List.of(EXTRA));
    write(out, "", c, corpus);

    Random rnd = new Random(1301);
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
    Map<String, Supplier<QueryParser>> f = new LinkedHashMap<>();
    for (String name : List.of("std", "std_and", "ws", "std_leading", "multi", "multi_boost", "multi_and", "syn")) {
      f.put(name, c.get(name));
    }
    write(out, "frag_", f, lines);
    writeStatic(out);
  }

  /** {@code MultiFieldQueryParser}'s static {@code parse} helpers ({@code static.tsv}). */
  static void writeStatic(Path out) throws Exception {
    StringBuilder b = new StringBuilder();
    Analyzer std = new StandardAnalyzer();
    Analyzer stop = new StopAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
    String[] two = {"title", "body"};
    BooleanClause.Occur S = BooleanClause.Occur.SHOULD, M = BooleanClause.Occur.MUST, N = BooleanClause.Occur.MUST_NOT;
    row(b, "fields", () -> MultiFieldQueryParser.parse(new String[] {"foo", "bar baz"}, two, std));
    row(b, "fields_stop", () -> MultiFieldQueryParser.parse(new String[] {"the", "x"}, two, stop));
    row(b, "fields_mismatch", () -> MultiFieldQueryParser.parse(new String[] {"a"}, two, std));
    row(b, "flags", () -> MultiFieldQueryParser.parse("foo bar", two, new BooleanClause.Occur[] {M, N}, std));
    row(b, "flags_mismatch", () -> MultiFieldQueryParser.parse("foo", two, new BooleanClause.Occur[] {M}, std));
    row(b, "fields_flags", () -> MultiFieldQueryParser.parse(new String[] {"a", "the"}, two, new BooleanClause.Occur[] {S, M}, stop));
    row(b, "fields_flags_mismatch", () -> MultiFieldQueryParser.parse(new String[] {"a", "b"}, two, new BooleanClause.Occur[] {S}, std));
    row(b, "fields_error", () -> MultiFieldQueryParser.parse(new String[] {"a", "(b"}, two, std));
    Files.writeString(out.resolve("static.tsv"), b.toString(), StandardCharsets.UTF_8);
  }

  interface Parse {
    Query get() throws ParseException;
  }

  static void row(StringBuilder b, String name, Parse p) {
    b.append(name).append('\t');
    try {
      Query q = p.get();
      b.append("Q\t").append(AnalysisRows.esc(q.toString("f"))).append('\t').append(AnalysisRows.esc(describe(q)));
    } catch (ParseException ex) {
      b.append("E\t").append(AnalysisRows.esc(ex.getMessage()));
    } catch (RuntimeException ex) {
      b.append("X\t").append(AnalysisRows.esc(ex.getClass().getSimpleName() + ": " + ex.getMessage()));
    }
    b.append('\n');
  }

  static void write(Path out, String prefix, Map<String, Supplier<QueryParser>> configs, List<String> queries)
      throws Exception {
    for (Map.Entry<String, Supplier<QueryParser>> e : configs.entrySet()) {
      StringBuilder b = new StringBuilder();
      for (String q : queries) {
        b.append(AnalysisRows.esc(q)).append('\t');
        try {
          Query parsed = e.getValue().get().parse(q);
          b.append("Q\t").append(AnalysisRows.esc(parsed.toString("f")))
              .append('\t').append(AnalysisRows.esc(describe(parsed)));
        } catch (ParseException ex) {
          b.append("E\t").append(AnalysisRows.esc(ex.getMessage()));
        } catch (RuntimeException ex) {
          b.append("X\t").append(AnalysisRows.esc(ex.getClass().getSimpleName() + ": " + ex.getMessage()));
        }
        b.append('\n');
      }
      Files.writeString(out.resolve(prefix + e.getKey() + ".tsv"), b.toString(), StandardCharsets.UTF_8);
    }
  }

  /** The query's class tree: booleans and boosts opened up, every other query by class and text. */
  static String describe(Query q) {
    if (q instanceof BooleanQuery bq) {
      StringBuilder b = new StringBuilder("Boolean(");
      if (bq.getMinimumNumberShouldMatch() != 0) b.append("msm=").append(bq.getMinimumNumberShouldMatch()).append(' ');
      boolean first = true;
      for (BooleanClause c : bq.clauses()) {
        if (!first) b.append(' ');
        first = false;
        b.append(switch (c.occur()) {
          case MUST -> "+";
          case MUST_NOT -> "-";
          case FILTER -> "#";
          case SHOULD -> "";
        });
        b.append(describe(c.query()));
      }
      return b.append(')').toString();
    }
    if (q instanceof BoostQuery bq) {
      return "Boost(" + describe(bq.getQuery()) + "^" + bq.getBoost() + ")";
    }
    return q.getClass().getSimpleName() + "(" + q.toString("f") + ")";
  }
}
