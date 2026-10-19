import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.charfilter.MappingCharFilter;
import org.apache.lucene.analysis.charfilter.NormalizeCharMap;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.flexible.core.QueryNodeException;
import org.apache.lucene.queryparser.flexible.precedence.PrecedenceQueryParser;
import org.apache.lucene.queryparser.flexible.standard.QueryParserUtil;
import org.apache.lucene.queryparser.flexible.standard.StandardQueryParser;
import org.apache.lucene.queryparser.flexible.standard.config.StandardQueryConfigHandler;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.queryparser.charstream.FastCharStream;
import org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParserConstants;
import org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParserTokenManager;
import org.apache.lucene.queryparser.flexible.standard.parser.Token;
import org.apache.lucene.queryparser.flexible.standard.parser.TokenMgrError;

/**
 * M13 T13.1: the flexible query parser. {@code tokens.tsv} is {@code
 * StandardSyntaxParserTokenManager}'s token stream for each query of {@code
 * fixtures/corpus/queryparser-classic.txt} and {@code queryparser-flexible.txt}: the escaped query,
 * then each token as {@code kind:begin-column:image} (escaped, space-separated) up to the end of
 * input, or {@code E} and the {@code TokenMgrError} message. {@code
 * crates/lucene-search/src/queryparser/flexible/tests.rs} compares.
 *
 * <p>{@code <config>.tsv} is {@code StandardQueryParser.parse(query, field)} of each query under
 * one configuration ({@link #parser}): {@code Q}, {@code toString("f")} and the class tree, or
 * {@code X} and the exception's class and message. {@code
 * crates/lucene-search/tests/queryparser_flexible_fixtures.rs} compares.
 */
public class GenQueryParserFlexible {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_flexible");
    Files.createDirectories(out);
    List<String> corpus = new ArrayList<>(AnalysisRows.corpus("queryparser-classic.txt"));
    corpus.addAll(AnalysisRows.corpus("queryparser-flexible.txt"));
    StringBuilder b = new StringBuilder();
    for (String q : corpus) {
      b.append(AnalysisRows.esc(q)).append('\t').append(tokens(q)).append('\n');
    }
    Files.writeString(out.resolve("tokens.tsv"), b.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("util.tsv"), util(), StandardCharsets.UTF_8);
    // the classic parser's seeded joins of syntax fragments, unescaped: their
    // token streams and their parse under {@code std}
    List<String> fragments = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(args[0]).resolve("queryparser_classic/fragments.txt"), StandardCharsets.UTF_8)) {
      fragments.add(unesc(line));
    }
    // and seeded joins of the flexible syntax's own fragments
    java.util.Random rnd = new java.util.Random(1313);
    for (int i = 0; i < 4000; i++) {
      StringBuilder text = new StringBuilder();
      int n = 1 + rnd.nextInt(8);
      for (int j = 0; j < n; j++) text.append(FLEX_FRAGMENTS[rnd.nextInt(FLEX_FRAGMENTS.length)]);
      fragments.add(text.toString());
    }
    // every prefix of every function keyword, at the end of the input and
    // before a delimiter: the keyword DFA's every level, ending or not
    String[] keywords = {
      "atleast", "atLeast", "after", "before", "containedBy", "containedby", "containing",
      "extend", "or", "fuzzyterm", "fuzzyTerm", "maxgaps", "maxGaps", "maxwidth", "maxWidth",
      "nonOverlapping", "nonoverlapping", "notContainedBy", "notcontainedby", "notContaining",
      "notcontaining", "notWithin", "notwithin", "ordered", "overlapping", "phrase", "unordered",
      "unorderedNoOverlaps", "unorderednooverlaps", "wildcard", "within"
    };
    for (String kw : keywords) {
      for (int i = 1; i <= kw.length(); i++) {
        String prefix = kw.substring(0, i);
        fragments.add("fn:" + prefix);
        fragments.add("fn:" + prefix + "(a b)");
        fragments.add("fn:" + prefix + "x");
      }
    }
    StringBuilder fb = new StringBuilder();
    StandardQueryParser std = parser("std");
    for (String q : fragments) {
      fb.append(AnalysisRows.esc(q)).append('\t');
      fb.append(tokens(q)).append('\t');
      try {
        Query parsed = std.parse(q, "f");
        fb.append("Q").append('\t').append(AnalysisRows.esc(parsed.toString("f")))
            .append('\t').append(AnalysisRows.esc(GenQueryParserClassic.describe(parsed)));
      } catch (QueryNodeException | RuntimeException e) {
        fb.append("X").append('\t').append(AnalysisRows.esc(e.getClass().getName() + ": " + e.getMessage()));
      }
      fb.append('\n');
    }
    Files.writeString(out.resolve("fragments.tsv"), fb.toString(), StandardCharsets.UTF_8);
    for (String config : CONFIGS) {
      StandardQueryParser p = parser(config);
      String field = config.equals("multi") ? null : "f";
      StringBuilder rows = new StringBuilder();
      for (String q : corpus) {
        rows.append(AnalysisRows.esc(q)).append('\t');
        try {
          Query parsed = p.parse(q, field);
          rows.append("Q\t").append(AnalysisRows.esc(parsed.toString("f")))
              .append('\t').append(AnalysisRows.esc(GenQueryParserClassic.describe(parsed)));
        } catch (QueryNodeException | RuntimeException e) {
          rows.append("X\t").append(AnalysisRows.esc(e.getClass().getName() + ": " + e.getMessage()));
        }
        rows.append('\n');
      }
      Files.writeString(out.resolve(config + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
    }
    writeDates(out);
  }

  /**
   * {@code date_*.tsv}: {@code GenQueryParserClassic.DATES} through
   * {@code TermRangeQueryNodeProcessor}, the JVM's default zone GMT-05:00; {@code date_clock.txt}:
   * the clock the two-digit years were read against.
   */
  static void writeDates(Path out) throws Exception {
    java.util.TimeZone saved = java.util.TimeZone.getDefault();
    Files.writeString(out.resolve("date_clock.txt"), System.currentTimeMillis() + "\n", StandardCharsets.UTF_8);
    java.util.Locale savedLocale = java.util.Locale.getDefault();
    try {
      java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone("GMT-05:00"));
      java.util.Locale.setDefault(java.util.Locale.US);
      for (String config : List.of("us_day", "de_map", "root_ms", "none")) {
        StandardQueryParser p = new StandardQueryParser(new StandardAnalyzer());
        switch (config) {
          case "us_day" -> {
            p.setDateResolution(org.apache.lucene.document.DateTools.Resolution.DAY);
            p.setTimeZone(java.util.TimeZone.getTimeZone("GMT+02:00"));
          }
          case "de_map" -> {
            p.setLocale(java.util.Locale.GERMANY);
            Map<CharSequence, org.apache.lucene.document.DateTools.Resolution> m = new java.util.HashMap<>();
            m.put("d", org.apache.lucene.document.DateTools.Resolution.MINUTE);
            p.setDateResolutionMap(m);
          }
          case "root_ms" -> {
            p.setLocale(java.util.Locale.ROOT);
            p.setDateResolution(org.apache.lucene.document.DateTools.Resolution.MILLISECOND);
            p.setTimeZone(java.util.TimeZone.getTimeZone("GMT+05:30"));
          }
          default -> {}
        }
        StringBuilder rows = new StringBuilder();
        for (String q : GenQueryParserClassic.DATES) {
          rows.append(AnalysisRows.esc(q)).append('\t');
          try {
            Query parsed = p.parse(q, "f");
            rows.append("Q\t").append(AnalysisRows.esc(parsed.toString("f")))
                .append('\t').append(AnalysisRows.esc(GenQueryParserClassic.describe(parsed)));
          } catch (QueryNodeException | RuntimeException e) {
            rows.append("X\t").append(AnalysisRows.esc(e.getClass().getName() + ": " + e.getMessage()));
          }
          rows.append('\n');
        }
        Files.writeString(out.resolve("date_" + config + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
      }
    } finally {
      java.util.TimeZone.setDefault(saved);
      java.util.Locale.setDefault(savedLocale);
    }
  }

  /** The flexible syntax's own pieces: every function name and its prefixes, the comparisons. */
  static final String[] FLEX_FRAGMENTS = {
    "fn:", "fn", "f", "fn:a", "fn:o", "fn:or", "fn:ord", "fn:ordered", "fn:ordere", "fn:unordered",
    "fn:unorderedNoOverlaps", "fn:unorderednooverlaps", "fn:unorderedNo", "fn:atleast", "fn:atLeast",
    "fn:atl", "fn:maxgaps", "fn:maxGaps", "fn:maxg", "fn:maxwidth", "fn:maxWidth", "fn:max",
    "fn:wildcard", "fn:wild", "fn:fuzzyterm", "fn:fuzzyTerm", "fn:fuzzy", "fn:phrase", "fn:ph",
    "fn:after", "fn:af", "fn:before", "fn:be", "fn:containing", "fn:contain", "fn:containedBy",
    "fn:containedby", "fn:notContaining", "fn:notcontaining", "fn:notContainedBy",
    "fn:notcontainedby", "fn:not", "fn:within", "fn:with", "fn:notWithin", "fn:notwithin",
    "fn:overlapping", "fn:over", "fn:nonOverlapping", "fn:nonoverlapping", "fn:non", "fn:extend",
    "fn:ext", "(", ")", " ", " ", "a", "b", "x*", "\"q r\"", "1", "2", "3.5", "f:", "f=",
    "f<", "f<=", "f>", "f>=", "<", ">", "=", "@", "@2", "^", "^2", "~", "~1", "[a TO b]", "{",
    "]", "/r.*/", "AND", "OR", "NOT", "!", "+", "-", "&&", "||", "\\", "\\:", "日", "\t"
  };

  /** {@code QueryParserUtil}'s helpers over fixed arguments: {@code util.tsv}, a row a call. */
  static String util() {
    StringBuilder b = new StringBuilder();
    StandardAnalyzer a = new StandardAnalyzer();
    BooleanClause.Occur must = BooleanClause.Occur.MUST;
    BooleanClause.Occur not = BooleanClause.Occur.MUST_NOT;
    BooleanClause.Occur should = BooleanClause.Occur.SHOULD;
    BooleanClause.Occur filter = BooleanClause.Occur.FILTER;
    java.util.List<java.util.concurrent.Callable<Query>> calls = java.util.List.of(
        () -> QueryParserUtil.parse(new String[] {"a", "b c"}, new String[] {"f", "g"}, a),
        () -> QueryParserUtil.parse(new String[] {"a"}, new String[] {"f", "g"}, a),
        () -> QueryParserUtil.parse("a b", new String[] {"f", "g"}, new BooleanClause.Occur[] {must, not}, a),
        () -> QueryParserUtil.parse("a", new String[] {"f"}, new BooleanClause.Occur[] {must, not}, a),
        () -> QueryParserUtil.parse(new String[] {"a", "\"b c\""}, new String[] {"f", "g"},
            new BooleanClause.Occur[] {should, filter}, a),
        () -> QueryParserUtil.parse(new String[] {"a"}, new String[] {"f", "g"},
            new BooleanClause.Occur[] {should}, a),
        () -> QueryParserUtil.parse(new String[] {"a("}, new String[] {"f"}, a));
    for (java.util.concurrent.Callable<Query> call : calls) {
      try {
        Query q = call.call();
        b.append("Q\t").append(AnalysisRows.esc(q.toString("f"))).append('\t')
            .append(AnalysisRows.esc(GenQueryParserClassic.describe(q)));
      } catch (Exception e) {
        b.append("X\t").append(AnalysisRows.esc(e.getClass().getName() + ": " + e.getMessage()));
      }
      b.append('\n');
    }
    b.append("E\t").append(AnalysisRows.esc(QueryParserUtil.escape("a+b-c!(d):e^[f]\"g{h}~i*j?k|l&m/n\\o p"))).append('\n');
    return b.toString();
  }

  static final String[] CONFIGS = {"std", "and", "noanalyzer", "multi", "options", "syn", "stop", "negative", "prec", "prec_and", "mapping", "rewrite", "topterms"};

  /**
   * Whitespace, lower case, and a {@code MappingCharFilter} on both readers rewriting the two-char
   * runs {@code \*} to {@code S} and {@code \?} to {@code Q}: a normaliser that sees across a
   * chunk boundary, so the wildcard processor's chunking shows in its output.
   */
  static Analyzer mapping() {
    NormalizeCharMap.Builder b = new NormalizeCharMap.Builder();
    b.add("\\*", "S");
    b.add("\\?", "Q");
    NormalizeCharMap map = b.build();
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer t = new WhitespaceTokenizer();
        return new TokenStreamComponents(t, new LowerCaseFilter(t));
      }

      @Override
      protected TokenStream normalize(String field, TokenStream in) {
        return new LowerCaseFilter(in);
      }

      @Override
      protected Reader initReader(String field, Reader reader) {
        return new MappingCharFilter(map, reader);
      }

      @Override
      protected Reader initReaderForNormalization(String field, Reader reader) {
        return new MappingCharFilter(map, reader);
      }
    };
  }

  /** The parser of one configuration; the field is {@code f} but under {@code multi}. */
  static StandardQueryParser parser(String config) {
    Analyzer analyzer = switch (config) {
      case "syn" -> GenQueryParserClassic.synonyms();
      case "stop" -> new StopAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
      case "noanalyzer" -> null;
      case "mapping" -> mapping();
      default -> new StandardAnalyzer();
    };
    StandardQueryParser p =
        config.startsWith("prec") ? new PrecedenceQueryParser(analyzer) : new StandardQueryParser(analyzer);
    switch (config) {
      case "and", "prec_and" -> p.setDefaultOperator(StandardQueryConfigHandler.Operator.AND);
      case "multi" -> p.setMultiFields(new String[] {"title", "body"});
      case "options" -> {
        p.setAllowLeadingWildcard(true);
        p.setPhraseSlop(2);
        p.setFuzzyPrefixLength(1);
        p.setFuzzyMinSim(0.5f);
        p.setEnablePositionIncrements(false);
        Map<String, Float> boosts = new LinkedHashMap<>();
        boosts.put("f", 2f);
        boosts.put("title", 0.5f);
        p.setFieldsBoost(boosts);
      }
      case "negative" -> {
        p.setFuzzyPrefixLength(-1);
        p.setPhraseSlop(-1);
        Map<String, Float> boosts = new LinkedHashMap<>();
        boosts.put("f", -1f);
        p.setFieldsBoost(boosts);
      }
      case "rewrite" -> p.setMultiTermRewriteMethod(MultiTermQuery.SCORING_BOOLEAN_REWRITE);
      case "topterms" -> p.setMultiTermRewriteMethod(new MultiTermQuery.TopTermsBoostOnlyBooleanQueryRewrite(5));
      default -> {}
    }
    return p;
  }

  /** The token stream of {@code q}: {@code T} and the tokens, or {@code E} and the error. */
  static String tokens(String q) throws Exception {
    StringBuilder toks = new StringBuilder();
    try {
      StandardSyntaxParserTokenManager tm =
          new StandardSyntaxParserTokenManager(new FastCharStream(new StringReader(q)));
      while (true) {
        Token t = tm.getNextToken();
        if (toks.length() > 0) toks.append(' ');
        toks.append(t.kind).append(':').append(t.beginColumn).append(':')
            .append(AnalysisRows.esc(t.image).replace(" ", "\\u0020"));
        if (t.kind == StandardSyntaxParserConstants.EOF) break;
      }
      return "T\t" + toks;
    } catch (TokenMgrError e) {
      return "E\t" + AnalysisRows.esc(e.getMessage());
    }
  }

  /** {@code AnalysisRows.esc} undone. */
  static String unesc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\' || i + 1 == s.length()) {
        b.append(c);
        continue;
      }
      char n = s.charAt(++i);
      switch (n) {
        case 't' -> b.append('\t');
        case 'n' -> b.append('\n');
        case 'r' -> b.append('\r');
        case 'u' -> {
          b.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
          i += 4;
        }
        default -> b.append(n);
      }
    }
    return b.toString();
  }
}
