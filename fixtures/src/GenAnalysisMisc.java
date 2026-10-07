import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.miscellaneous.DateRecognizerFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceFilter;
import org.apache.lucene.analysis.reverse.ReverseStringFilter;
import org.apache.lucene.analysis.util.CSVUtil;

/**
 * M11 T11.6: small packages. {@code WordDelimiterFilter} (the deprecated one) over {@link
 * #WDF_LINES} ({@code wdf_lines.txt}, {@code wdf_<chain>.tsv}); {@code ReverseStringFilter} (with and without each marker) over the
 * lines of {@link #LINES} (written here; surrogate pairs at every position), stored with the
 * fixture as {@code lines.txt} and run as {@code <chain>.tsv} ({@link AnalysisRows}' rows); {@code
 * CSVUtil.parse}/{@code quoteEscape} over {@link #CSV} ({@code csv.words}); {@code
 * DateRecognizerFilter}'s {@code DateFormat.parse}: every pattern of {@link #DATE_PATTERNS} over
 * seeded texts, valid ones and mutations of them ({@code date_formats.txt}: a {@code #pattern} row per
 * pattern, then {@code input<TAB>1|0}), and the filter over {@link #DATE_LINES} ({@code
 * date_<chain>.tsv}). {@code
 * crates/lucene-analysis/tests/analysis_misc_fixtures.rs} compares.
 */
public class GenAnalysisMisc {

  static final String[] LINES = {
    "A BA CBA DCBA abcdef",
    "易经 中文字符 日本語",
    "😀 😀a a😀 a😀b 😀😁 😀a😁 😀😁a ab😀😁 😀ab😁cd",
    "😀😁😂 x😀😁y",
    "Ünïcödé naïve façade résumé",
    "",
    "single"
  };

  /** Lines for the deprecated {@code WordDelimiterFilter} (stored as {@code wdf_lines.txt}). */
  static final String[] WDF_LINES = {
    "Wi-Fi PowerShot500 O'Neil's 3.14 AT&T j2se wi-fi-4000 SD500 foo's",
    "  --x--  -- 123-456 abc_def a1b2c3 ABCdef abcDEF 42nd 1st-2nd",
    "ŁódźPolska ÀÉÎ-òü 中文-字符 😀-a b😀c",
    "x-xx-xxx xxx-xx-x -a- a- -a a--b--c",
    "",
    "protected-word keyword-term Hello-World HELLO-world"
  };

  static final int WDF_DEFAULT =
      WordDelimiterFilter.GENERATE_WORD_PARTS | WordDelimiterFilter.GENERATE_NUMBER_PARTS
          | WordDelimiterFilter.SPLIT_ON_CASE_CHANGE | WordDelimiterFilter.SPLIT_ON_NUMERICS
          | WordDelimiterFilter.STEM_ENGLISH_POSSESSIVE;

  static final String[] CSV = {
    "a,b,c", "\"a,b\",c", "\"x\"\"y\",z", "a,\"b,c\"", "\"a,b", "", "\"\",x", "a,,b", ",", "\"q\",\"r\"",
    "\"\"\"\",x", "x,\"y\"\"z\",w", "abc", "\"a\"b\",c", "日本,\"語,文\""
  };

  /** {@code SimpleDateFormat} patterns ({@code DEFAULT}: the English date instance). */
  static final String[] DATE_PATTERNS = {
    "DEFAULT", "yyyy-MM-dd", "dd/MM/yyyy", "MM/dd/yy", "yyyyMMdd", "ddMMyy", "d MMMM yyyy",
    "EEE, d MMM yyyy HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX", "h:mm a", "hh 'o''clock' a",
    "G yyyy", "yyyy.MM.dd G 'at' HH:mm:ss", "EEEE", "MMMMM", "LLL", "L", "yy", "y", "Y-ww-u", "D",
    "F", "W", "k:K", "H", "S", "mm:ss", "X", "XX", "XXX", "EEEdMMM", "dMMMyyyy", "MMMd",
    "ddMMMyyyy", "yyyyMMddHHmmssSSS", "u", "''", "'lit'", "", "MMMMd", "aK", "Ga", "dd.MM.yyyy",
    "yyyyDDD", "HHmmX", "yyyy-MM-dd HH:mm:ss.SSS", "E MMM dd HH:mm:ss yyyy", "M/d"
  };

  /** Fragments the date texts are built from and mutated with. */
  static final String[] DATE_BITS = {
    "0", "1", "2", "5", "9", "12", "31", "59", "2020", "-", "+", "E", "e", " ", "  ", "\t", ":", ".",
    ",", "/", "'", "T", "Z", "z", "NaN", "\u221e", "\u0665", "\u0661\u0662", "\uff15", "\u0e55",
    "\ud835\udfd3", "\u2212", "Jan", "January", "jan", "SEP", "Sept", "May", "Mon",
    "Monday", "Thu", "AM", "pm", "AD", "Anno Domini", "bc", "Before Christ", "\u017fep", "\u0131",
    "\u212a", "A", "x", "Ma", "E3", "E-2", "+01", "-0800", "+01:00", "+24", "00", "\n"
  };

  /** Hand-picked edges, pattern then inputs (appended after the seeded texts). */
  static final String[][] DATE_EDGES = {
    {"d", "5", " 5", "\t5", "  5", "-5", "+5", "\u0665", "\u0661\u0662", "\uff15", "\u0e55", "\ud835\udfd3",
        "\u00a05", "\u20035", "\n5", "\u000b5", "\u001c5", "\u2212\u0035", "5x", "x", "", "--5", "- 5", "-",
        "5E3", "5E\u0663", "5E-\u0663", "5E--3", "5E 3", "5E", "5E-", "E3", "-E3", "1E400", "5.0", "5,0", "\u221e",
        "-\u221e", "+\u221e", "NaN", "-NaN", "nan", "0x5", "0\u0660", "99999999999999999999"},
    {"dd", " 5", "  5", "\t\t5", "NaN", "NaN5", "5"},
    {"ddMM", "0512", "5 12", "05 12", "051", "-0512", "-5-12", " 0512", "  0512", "NaN12", "5NaN", "\u221e12",
        "-512", "- 512", "1E12", "1E-3", "\t512"},
    {"HHmm", "1234", "12 34", "1 234", " 1234", "1\t234"},
    {"MMM", "Jan", " Jan", "jAN", "January", "Janu", "1", "Ma", "\u017fep", "SEP", "Sept", "\u212aan", "J\u0131n"},
    {"MMM d", "Jan 5", "Jan\t5", "Jan\u00855", "Jan\u200b5", "Jan\u20285", "Jan  5", "Jan5"},
    {"d-M", "5--1", "5-1", "5- 1", "5 -1", "-5--1"},
    {"X", "Z", "z", " Z", "+01", "+1", "+0100", "+01:00", "-00", "+23", "+24", "+99", "+01:60", " +01",
        "\u0661\u0662", "+\u0661\u0662", "\uff0b01", "\u221201"},
    {"XX", "+01", "+0100", "+0159", "+0160", "+01:00", "Z", "-2359", "+2400"},
    {"XXX", "+01", "+0100", "+01:00", "+01:59", "+01:60", "Z", "+01:\u0663\u0663", "+01-00"},
    {"G", "AD", "ad", "BC", "B", "A", "Anno Domini", "Before Christ", "before christ", "BCE", " AD"},
    {"EEEEE", "T", "Thu", "Thursday", "thurs", "Mo"},
    {"a", "AM", "pm", "P", "p.m.", "A.M.", "\u0101m"},
    {"'T'd", "T5", "t5"},
    {"''d''", "'5'", "5"},
    {"", "", "x"}
  };

  static String dateBit(java.util.Random r) {
    return DATE_BITS[r.nextInt(DATE_BITS.length)];
  }

  static java.text.DateFormat dateFormat(String pattern) {
    return pattern.equals("DEFAULT")
        ? java.text.DateFormat.getDateInstance(java.text.DateFormat.DEFAULT, java.util.Locale.ENGLISH)
        : new java.text.SimpleDateFormat(pattern, java.util.Locale.ENGLISH);
  }

  /** Whether {@code text} parses: {@code DateRecognizerFilter.accept()}. */
  static boolean parses(java.text.DateFormat f, String text) {
    try {
      f.parse(text);
      return true;
    } catch (java.text.ParseException e) {
      return false;
    }
  }

  static String dateFormats() {
    StringBuilder o = new StringBuilder();
    for (int pi = 0; pi < DATE_PATTERNS.length; pi++) {
      String pattern = DATE_PATTERNS[pi];
      java.text.DateFormat f = dateFormat(pattern);
      f.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
      java.util.Random r = new java.util.Random(1000 + pi);
      o.append("#pattern\t").append(AnalysisRows.esc(pattern)).append('\n');
      java.util.Set<String> seen = new java.util.LinkedHashSet<>();
      while (seen.size() < 300) {
        // Year -9999..9999 either side of the epoch.
        long millis = (r.nextLong() % 315537897600000L);
        String text = f.format(new java.util.Date(millis));
        int edits = r.nextInt(4);
        for (int e = 0; e < edits; e++) {
          int at = text.isEmpty() ? 0 : r.nextInt(text.length() + 1);
          switch (r.nextInt(5)) {
            case 0 -> text = text.substring(0, at) + dateBit(r) + text.substring(at);
            case 1 -> text = at < text.length() ? text.substring(0, at) + text.substring(at + 1) : text;
            case 2 -> text = text.substring(0, at);
            case 3 -> text = at < text.length() ? text.substring(0, at) + dateBit(r) + text.substring(at + 1) : text;
            default -> text = r.nextBoolean() ? text.toUpperCase(java.util.Locale.ROOT) : text.toLowerCase(java.util.Locale.ROOT);
          }
        }
        if (r.nextInt(8) == 0) {
          StringBuilder b = new StringBuilder();
          for (int k = r.nextInt(6); k >= 0; k--) b.append(dateBit(r));
          text = b.toString();
        }
        // A split surrogate pair cannot reach a Rust term.
        boolean lone = false;
        for (int i = 0; i < text.length(); i++) {
          char c = text.charAt(i);
          if (Character.isHighSurrogate(c) && (i + 1 == text.length() || !Character.isLowSurrogate(text.charAt(i + 1)))) lone = true;
          if (Character.isLowSurrogate(c) && (i == 0 || !Character.isHighSurrogate(text.charAt(i - 1)))) lone = true;
        }
        // A space separator other than U+0020 matches a pattern's space in JDK 23+'s lenient
        // parse, not JDK 21's: left to the Rust unit tests, so the file is the same under both.
        boolean otherSpace = false;
        for (int i = 0; i < text.length(); i++) {
          char c = text.charAt(i);
          if (c != ' ' && Character.getType(c) == Character.SPACE_SEPARATOR) otherSpace = true;
        }
        if (lone || otherSpace || !seen.add(text)) continue;
        o.append(AnalysisRows.esc(text)).append('\t').append(parses(f, text) ? '1' : '0').append('\n');
      }
    }
    for (String[] edge : DATE_EDGES) {
      java.text.DateFormat f = dateFormat(edge[0]);
      o.append("#pattern\t").append(AnalysisRows.esc(edge[0])).append('\n');
      for (int i = 1; i < edge.length; i++) {
        o.append(AnalysisRows.esc(edge[i])).append('\t').append(parses(f, edge[i]) ? '1' : '0').append('\n');
      }
    }
    return o.toString();
  }

  /** Lines for the filter itself (stored as {@code date_lines.txt}). */
  static final String[] DATE_LINES = {
    "Jan 5, 2020",
    "2020-01-05 not 2020-13-45 x 05/01/2020 1999-12-31T23:59 -1-1-1 NaN-1-1",
    "May 1, 1999 and June 31, 2000",
    ""
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_misc");
    Files.createDirectories(out);
    StringBuilder lines = new StringBuilder();
    for (String l : LINES) lines.append(AnalysisRows.esc(l)).append('\n');
    Files.writeString(out.resolve("lines.txt"), lines.toString(), StandardCharsets.UTF_8);
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("ws_reverse", () -> AnalysisRows.chain(WhitespaceTokenizer::new, ReverseStringFilter::new));
    c.put("ws_reverse_soh", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new ReverseStringFilter(t, ReverseStringFilter.START_OF_HEADING_MARKER)));
    c.put("ws_reverse_pua", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new ReverseStringFilter(t, ReverseStringFilter.PUA_EC00_MARKER)));
    c.put("ws_reverse_rtl", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new ReverseStringFilter(t, ReverseStringFilter.RTL_DIRECTION_MARKER)));
    c.put("ws_reverse_is", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new ReverseStringFilter(t, ReverseStringFilter.INFORMATION_SEPARATOR_MARKER)));
    AnalysisRows.writeChains(out, c, List.of(LINES));
    StringBuilder wdfLines = new StringBuilder();
    for (String l : WDF_LINES) wdfLines.append(AnalysisRows.esc(l)).append('\n');
    Files.writeString(out.resolve("wdf_lines.txt"), wdfLines.toString(), StandardCharsets.UTF_8);
    org.apache.lucene.analysis.CharArraySet prot = AnalysisRows.set(false, "protected-word", "AT&T");
    org.apache.lucene.analysis.CharArraySet kw = AnalysisRows.set(false, "keyword-term");
    int all = WDF_DEFAULT | WordDelimiterFilter.CATENATE_WORDS | WordDelimiterFilter.CATENATE_NUMBERS
        | WordDelimiterFilter.CATENATE_ALL | WordDelimiterFilter.PRESERVE_ORIGINAL;
    Map<String, Supplier<Analyzer>> w = new LinkedHashMap<>();
    w.put("wdf_default", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new WordDelimiterFilter(t, WDF_DEFAULT, null)));
    w.put("wdf_all", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new WordDelimiterFilter(t, all, prot)));
    w.put("wdf_catenate_only", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new WordDelimiterFilter(t, WordDelimiterFilter.CATENATE_WORDS | WordDelimiterFilter.CATENATE_NUMBERS | WordDelimiterFilter.CATENATE_ALL, null)));
    w.put("wdf_ignore_keywords", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new WordDelimiterFilter(new SetKeywordMarkerFilter(t, kw), WDF_DEFAULT | WordDelimiterFilter.IGNORE_KEYWORDS | WordDelimiterFilter.CATENATE_ALL, null)));
    w.put("wdf_illegal_offsets", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new WordDelimiterFilter(new PatternReplaceFilter(t, java.util.regex.Pattern.compile("x"), "yy", true), all, null)));
    w.put("wdf_no_case_numerics", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new WordDelimiterFilter(t, WordDelimiterFilter.GENERATE_WORD_PARTS | WordDelimiterFilter.GENERATE_NUMBER_PARTS | WordDelimiterFilter.CATENATE_ALL, null)));
    AnalysisRows.writeChains(out, w, List.of(WDF_LINES));
    StringBuilder csv = new StringBuilder();
    for (String s : CSV) {
      csv.append(AnalysisRows.esc(s)).append('\t').append(AnalysisRows.esc(CSVUtil.quoteEscape(s)));
      for (String f : CSVUtil.parse(s)) csv.append('\t').append(AnalysisRows.esc(f));
      csv.append('\n');
    }
    Files.writeString(out.resolve("csv.words"), csv.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("date_formats.txt"), dateFormats(), StandardCharsets.UTF_8);
    StringBuilder dateLines = new StringBuilder();
    for (String l : DATE_LINES) dateLines.append(AnalysisRows.esc(l)).append('\n');
    Files.writeString(out.resolve("date_lines.txt"), dateLines.toString(), StandardCharsets.UTF_8);
    Map<String, Supplier<Analyzer>> d = new LinkedHashMap<>();
    d.put("date_default_kw", () -> AnalysisRows.chain(KeywordTokenizer::new, DateRecognizerFilter::new));
    d.put("date_iso_ws", () -> AnalysisRows.chain(WhitespaceTokenizer::new,
        t -> new DateRecognizerFilter(t, new java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.ENGLISH))));
    AnalysisRows.writeChains(out, d, List.of(DATE_LINES));
  }
}
