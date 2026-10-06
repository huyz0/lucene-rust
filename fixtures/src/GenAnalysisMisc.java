import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
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
 * CSVUtil.parse}/{@code quoteEscape} over {@link #CSV} ({@code csv.words}). {@code
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
  }
}
