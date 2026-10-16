import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.monitor.Monitor;
import org.apache.lucene.monitor.MonitorConfiguration;
import org.apache.lucene.monitor.MonitorQuerySerializer;
import org.apache.lucene.monitor.QueryMatch;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.5d: a query index the Rust port wrote ({@code write_monitor_index_fixture}), opened by
 * Lucene's own {@code Monitor} with a writable {@code WritableQueryIndex}. It must pass {@code
 * CheckIndex}, hold the same 40 queries in 59 pieces Lucene itself writes for the corpus, and match
 * each document of {@code fixtures/data/monitor/match.tsv} to the same query ids.
 */
public class VerifyMonitorIndex {
  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0]);
    try (FSDirectory fs = FSDirectory.open(dir);
        CheckIndex check = new CheckIndex(fs)) {
      CheckIndex.Status status = check.checkIndex();
      if (!status.clean) {
        throw new AssertionError("CheckIndex found problems in " + dir);
      }
    }
    StandardAnalyzer analyzer = new StandardAnalyzer();
    QueryParser parser = new QueryParser("body", analyzer);
    MonitorQuerySerializer ser =
        MonitorQuerySerializer.fromParser(
            t -> {
              try {
                return parser.parse(t);
              } catch (Exception e) {
                throw new RuntimeException(e);
              }
            });
    try (Monitor monitor =
        new Monitor(analyzer, new MonitorConfiguration().setIndexPath(dir, ser))) {
      String[] want = Files.readString(Path.of("fixtures/data/monitor/persisted.txt")).trim().split(" ");
      if (monitor.getQueryCount() != Integer.parseInt(want[0])
          || monitor.getDisjunctCount() != Integer.parseInt(want[1])) {
        throw new AssertionError(
            "counts " + monitor.getQueryCount() + " " + monitor.getDisjunctCount()
                + " != " + String.join(" ", want));
      }
      List<String> rows =
          Files.readAllLines(Path.of("fixtures/data/monitor/match.tsv"), StandardCharsets.UTF_8);
      for (String row : rows) {
        String[] cols = row.split("\t", -1);
        Document d = new Document();
        for (String f : unesc(cols[0]).split("\\|")) {
          int eq = f.indexOf('=');
          d.add(new TextField(f.substring(0, eq), f.substring(eq + 1), Field.Store.NO));
        }
        List<String> got = new ArrayList<>();
        for (var m : monitor.match(d, QueryMatch.SIMPLE_MATCHER).getMatches()) {
          got.add(m.getQueryId());
        }
        Collections.sort(got);
        List<String> expected = new ArrayList<>();
        for (String id : unesc(cols[1]).split(" ")) {
          if (!id.isEmpty()) expected.add(id);
        }
        Collections.sort(expected);
        if (!got.equals(expected)) {
          throw new AssertionError("document " + cols[0] + ": got " + got + ", want " + expected);
        }
      }
    }
    System.out.println("ok");
  }

  /** {@code AnalysisRows.esc} undone: only the escapes the corpus uses. */
  static String unesc(String s) {
    return s.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\");
  }
}
