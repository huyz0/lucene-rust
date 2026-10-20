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
    writeNumbers(out);
    writePoints(out);
  }

  /**
   * The locales whose {@code NumberFormat}s {@code numbers.tsv} reads and writes: those whose
   * symbols JDK 21 (this generator) and JDK 25 (the Rust table, {@code GenDateLocales}) agree on --
   * not {@code ar-EG}, whose exponent and NaN CLDR changed.
   */
  static final String[] NUMBER_LOCALES = {"en-US", "de-DE", "fr-FR", "und", "sv-SE", "de-CH", "fa"};

  /**
   * {@code numbers.tsv}: each of {@link #NUMBER_LOCALES}' number ({@code N}) and integer ({@code
   * I}) instance parsing {@link #numberInputs} from index 0 ({@code P}: the input, then {@code
   * L<long>@<index>}, {@code D<raw bits>@<index>} or {@code E}) and formatting longs ({@code FL})
   * and doubles ({@code FD}, by raw bits).
   */
  static void writeNumbers(Path out) throws Exception {
    long[] longs = {0, -5, 7, 1000, 1234567, -1234567, Long.MAX_VALUE, Long.MIN_VALUE};
    double[] doubles = {
      0.0, -0.0, 1.5, -1.5, 0.0025, 0.0625, 0.5, 2.5, 1234.5678, 1e20, 1e-5, 123456789.123,
      Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1.1f, 3.4028234663852886E38,
      Double.MIN_VALUE, 999.9995, 0.1 + 0.2
    };
    StringBuilder rows = new StringBuilder();
    for (String tag : NUMBER_LOCALES) {
      java.util.Locale locale = java.util.Locale.forLanguageTag(tag);
      for (String kind : List.of("N", "I")) {
        java.text.NumberFormat f = kind.equals("I")
            ? java.text.NumberFormat.getIntegerInstance(locale)
            : java.text.NumberFormat.getNumberInstance(locale);
        java.text.DecimalFormatSymbols sym = ((java.text.DecimalFormat) f).getDecimalFormatSymbols();
        for (String text : numberInputs(sym)) {
          java.text.ParsePosition pos = new java.text.ParsePosition(0);
          Number n = f.parse(text, pos);
          String got = n == null ? "E"
              : (n instanceof Long ? "L" + n : "D" + Double.doubleToRawLongBits(n.doubleValue()))
                  + "@" + pos.getIndex();
          rows.append(tag).append('\t').append(kind).append("\tP\t").append(AnalysisRows.esc(text))
              .append('\t').append(got).append('\n');
        }
        for (long v : longs) {
          rows.append(tag).append('\t').append(kind).append("\tFL\t").append(v)
              .append('\t').append(AnalysisRows.esc(f.format(v))).append('\n');
        }
        for (double v : doubles) {
          rows.append(tag).append('\t').append(kind).append("\tFD\t").append(Double.doubleToRawLongBits(v))
              .append('\t').append(AnalysisRows.esc(f.format(v))).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("numbers.tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  /**
   * The texts a format parses: templates over its symbols ({@code {d}} decimal, {@code {g}}
   * grouping, {@code {-}} minus, {@code {e}} exponent, {@code {i}} infinity, {@code {n}} NaN),
   * each again in the locale's digits when they are not ASCII.
   */
  static List<String> numberInputs(java.text.DecimalFormatSymbols sym) {
    String[] templates = {
      "0", "12", "-12", "{-}12", "+5", " 5", "5 ", "1{g}234{g}567", "1{g}2{g}3", "1{g}", "{g}1",
      "1{d}5", "{d}5", "-{d}5", "{-}1{d}5", "1{d}5{d}6", "1{d}", "1{d}0", "0{d}000", "-0", "{-}0",
      "12abc", "abc", "", "1E3", "1{e}3", "1{e}-3", "1{e}{-}3", "2{d}5{e}2", "9223372036854775807",
      "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
      "123456789012345678901234567890", "0{d}1", "{i}", "{-}{i}", "{n}", "1 234", "1\u00a0234",
      "1\u202f234", "1'234", "1\u2019234", "1.234", "1,234", "1,5", "1.5", "1.234,5", "1,234.5",
      "\u2212" + "7", "1{g}234{d}5", "12{g}34"
    };
    List<String> out = new ArrayList<>();
    char zero = sym.getZeroDigit();
    for (String t : templates) {
      String text = t.replace("{d}", String.valueOf(sym.getDecimalSeparator()))
          .replace("{g}", String.valueOf(sym.getGroupingSeparator()))
          .replace("{-}", String.valueOf(sym.getMinusSign()))
          .replace("{e}", sym.getExponentSeparator())
          .replace("{i}", sym.getInfinity())
          .replace("{n}", sym.getNaN());
      out.add(text);
      if (zero != '0') {
        StringBuilder local = new StringBuilder();
        for (char c : text.toCharArray()) {
          local.append(c >= '0' && c <= '9' ? (char) (zero + (c - '0')) : c);
        }
        out.add(local.toString());
      }
    }
    return out;
  }

  /** The point fields of {@code points*.tsv}. */
  static Map<String, org.apache.lucene.queryparser.flexible.standard.config.PointsConfig> pointsConfigs() {
    Map<String, org.apache.lucene.queryparser.flexible.standard.config.PointsConfig> m = new LinkedHashMap<>();
    m.put("i", new org.apache.lucene.queryparser.flexible.standard.config.PointsConfig(
        java.text.NumberFormat.getNumberInstance(java.util.Locale.US), Integer.class));
    m.put("l", new org.apache.lucene.queryparser.flexible.standard.config.PointsConfig(
        java.text.NumberFormat.getIntegerInstance(java.util.Locale.GERMANY), Long.class));
    m.put("fl", new org.apache.lucene.queryparser.flexible.standard.config.PointsConfig(
        java.text.NumberFormat.getNumberInstance(java.util.Locale.FRANCE), Float.class));
    m.put("d", new org.apache.lucene.queryparser.flexible.standard.config.PointsConfig(
        java.text.NumberFormat.getNumberInstance(java.util.Locale.forLanguageTag("sv-SE")), Double.class));
    m.put("dt", new org.apache.lucene.queryparser.flexible.standard.config.PointsConfig(
        new org.apache.lucene.queryparser.flexible.standard.config.NumberDateFormat(
            java.text.DateFormat.getDateInstance(java.text.DateFormat.SHORT, java.util.Locale.US)),
        Long.class));
    return m;
  }

  /** The queries of {@code points.tsv}. */
  static final String[] POINT_QUERIES = {
    "i:5", "i:-5", "i:1,234", "i:1,234.9", "i:2147483648", "i:-2147483649", "i:abc", "i:5abc",
    "i:\"5\"", "i:5*", "i:5~", "i:5^2", "+i:5 -i:6", "i:(5 6)", "i:[1 TO 10]", "i:{1 TO 10}",
    "i:[1 TO 10}", "i:{1 TO 10]", "i:[* TO 10]", "i:{* TO 10}", "i:[1 TO *]", "i:{1 TO *}",
    "i:[* TO *]", "i:{2147483647 TO *]", "i:[* TO -2147483648}", "i:[x TO 5]", "i:[5 TO x]",
    "i:>5", "i:>=5", "i:<5", "i:<=5", "i:\\-5", "i:\"\"", "i:[\"\" TO 5]", "i:{\"1\" TO \"2\"}",
    "l:1.234.567", "l:1,5", "l:-9223372036854775808", "l:{-9223372036854775808 TO 9223372036854775807}",
    "l:[1.000 TO 2.000]", "l:9223372036854775808", "l:x",
    "fl:1,5", "fl:-1,5", "fl:{1,5 TO 2,5}", "fl:[* TO 0]", "fl:{* TO *}", "fl:1\u202f000", "fl:1 000",
    "fl:1.5", "fl:3,4028236E38", "fl:{0 TO 1}",
    "d:1,5", "d:\u22121,5", "d:-1,5", "d:{1,5 TO *}", "d:{* TO 0}", "d:1\u00a0000,25",
    "d:1e3", "d:x",
    "dt:\"1/2/2020\"", "dt:[\"1/1/2020\" TO \"12/31/2020\"]", "dt:{\"1/1/2020\" TO *}", "dt:{* TO \"1/1/2020\"}", "dt:1/2/2020",
    "dt:foo", "dt:\"13/45/2020\"", "dt:\"1/2/2020x\"",
    "x:5", "i:5 x:6", "f:[1 TO 2]"
  };

  /**
   * The rows of the escaping: each piece after 1024 terms {@code z}, so that the boolean has one
   * clause too many and its message prints the whole query through {@code toQueryString}; the
   * message's 1024 {@code f:z } compressed to {@code <FILL>}.
   */
  static final String[] ESCAPE_PIECES = {
    "a\\:b", "\"a:b\\\"c\"", "a\\ b", "\\+a", "\\-a", "\\@a", "and", "\\AND", "Within",
    "within", "notx", "İNORDER", "wıthİn", "İn","a\\(b\\)", "a\\*b*", "a?\\?", "\\*a", "a\\\\b", "a\\u3000b", "a\u3000b",
    "a\\~b", "a\\^b", "a\\/b", "a\\<b", "a\\=b", "a\\!b", "a\\[b\\]", "a\\{b\\}", "a\\>b",
    "\u0130\\:", "\u0130\\:a", "\u0130a\\:", "a\\:\u0130", "I\u0307\\:", "\u00cc\\:", "\u00cc",
    "I\u0300\\:", "J\u0301\\:", "\u0128\\:", "\u03a3\\:", "K\\:",
    "(a b)", "+a -b", "a AND b", "a OR b", "NOT a", "a && b", "a^2", "a^2.5", "a^0.1", "(a b)^3",
    "\"a b\"~3", "\"a\"", "\"a:b c\"", "a~", "a~0.5", "a~1", "a~2", "[a TO b]", "{a TO *]",
    "[* TO *]", "[\\: TO b\\:]", "/a.b/", "/a\\/b/", "*:*", "(a b)@2", "g:a", "_plain:a",
    "a*", "a*b?c", "f:(a b)", "((a))", "+(a b)", "-(a b)", "!a", "+(+a)", "-(-a)", "a\\ AND",
    "fn:ordered(a b)", "\"fast car\"", "fast", "car", "the", "a b c",
    "i:5", "i:1234", "i:[1 TO 10]", "i:[1 TO *]", "i:{* TO 5}", "l:1234567", "fl:1,5", "d:-1,25",
    "d:{* TO 1}", "dt:\"1/2/2020\"", "dt:[\"1/1/2020\" TO \"12/31/2020\"]", "dt:[* TO \"1/1/2020\"]"
  };

  /** Escaping's configurations: the parser and {@code Locale.getDefault()}. */
  static final String[] ESCAPE_CONFIGS = {"noanalyzer", "noanalyzer_tr", "noanalyzer_lt", "std", "syn", "prec", "points"};

  /**
   * {@code points.tsv} ({@link #POINT_QUERIES} under {@link #pointsConfigs}) and {@code
   * escape_<config>.tsv} ({@link #ESCAPE_PIECES}), in the default zone GMT-05:00 the date format
   * was made in.
   */
  static void writePoints(Path out) throws Exception {
    java.util.TimeZone saved = java.util.TimeZone.getDefault();
    java.util.Locale savedLocale = java.util.Locale.getDefault();
    try {
      java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone("GMT-05:00"));
      java.util.Locale.setDefault(java.util.Locale.US);
      StandardQueryParser points = new StandardQueryParser(new StandardAnalyzer());
      points.setPointsConfigMap(pointsConfigs());
      StringBuilder rows = new StringBuilder();
      for (String q : POINT_QUERIES) {
        rows.append(AnalysisRows.esc(q)).append('\t').append(parseRow(points, q, null)).append('\n');
      }
      Files.writeString(out.resolve("points.tsv"), rows.toString(), StandardCharsets.UTF_8);
      String fill = "z ".repeat(1024);
      for (String config : ESCAPE_CONFIGS) {
        StandardQueryParser p = switch (config) {
          case "points" -> points;
          case "noanalyzer_tr", "noanalyzer_lt" -> parser("noanalyzer");
          default -> parser(config);
        };
        java.util.Locale.setDefault(switch (config) {
          case "noanalyzer_tr" -> java.util.Locale.forLanguageTag("tr-TR");
          case "noanalyzer_lt" -> java.util.Locale.forLanguageTag("lt");
          case "noanalyzer" -> java.util.Locale.ROOT;
          default -> java.util.Locale.US;
        });
        StringBuilder esc = new StringBuilder();
        for (String piece : ESCAPE_PIECES) {
          esc.append(AnalysisRows.esc(piece)).append('\t').append(parseRow(p, fill + piece, "f:z ".repeat(1024)))
              .append('\n');
        }
        java.util.Locale.setDefault(java.util.Locale.US);
        Files.writeString(out.resolve("escape_" + config + ".tsv"), esc.toString(), StandardCharsets.UTF_8);
      }
    } finally {
      java.util.TimeZone.setDefault(saved);
      java.util.Locale.setDefault(savedLocale);
    }
  }

  /** {@code Q}, the query and its class tree, or {@code X} and the exception; {@code fill} compressed. */
  static String parseRow(StandardQueryParser p, String q, String fill) {
    try {
      Query parsed = p.parse(q, "f");
      return "Q\t" + AnalysisRows.esc(parsed.toString("f")) + "\t" + AnalysisRows.esc(GenQueryParserClassic.describe(parsed));
    } catch (QueryNodeException | RuntimeException e) {
      String m = e.getClass().getName() + ": " + e.getMessage();
      if (fill != null) m = m.replace(fill, "<FILL>");
      return "X\t" + AnalysisRows.esc(m);
    }
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
