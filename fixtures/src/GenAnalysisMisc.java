import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.reverse.ReverseStringFilter;
import org.apache.lucene.analysis.util.CSVUtil;

/**
 * M11 T11.6: small packages. {@code ReverseStringFilter} (with and without each marker) over the
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
    StringBuilder csv = new StringBuilder();
    for (String s : CSV) {
      csv.append(AnalysisRows.esc(s)).append('\t').append(AnalysisRows.esc(CSVUtil.quoteEscape(s)));
      for (String f : CSVUtil.parse(s)) csv.append('\t').append(AnalysisRows.esc(f));
      csv.append('\n');
    }
    Files.writeString(out.resolve("csv.words"), csv.toString(), StandardCharsets.UTF_8);
  }
}
