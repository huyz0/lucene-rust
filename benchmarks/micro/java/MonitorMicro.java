import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.monitor.Monitor;
import org.apache.lucene.monitor.MonitorQuery;
import org.apache.lucene.monitor.QueryMatch;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;

/**
 * M13 T13.5c's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_monitor.rs}): a
 * {@code Monitor} over the 41 queries of {@code fixtures/corpus/monitor-queries.txt}. {@code
 * mon_register}: a fresh monitor, all queries registered (units: queries); {@code mon_match}: four
 * documents matched against the registered set (units: documents). Output is {@code
 * name\tns_per_unit\tunits}.
 */
public class MonitorMicro {
  static final String[][] DOCS = {
    {"body=the quick brown fox"},
    {"body=a b", "title=quick"},
    {"body=fox fox fox", "title=brown"},
    {"body=zzzz extraordinarily"},
  };

  static Document doc(String[] fields) {
    Document d = new Document();
    for (String f : fields) {
      int eq = f.indexOf('=');
      d.add(new TextField(f.substring(0, eq), f.substring(eq + 1), Field.Store.NO));
    }
    return d;
  }

  public static void main(String[] args) throws Exception {
    StandardAnalyzer analyzer = new StandardAnalyzer();
    QueryParser parser = new QueryParser("body", analyzer);
    List<MonitorQuery> queries = new ArrayList<>();
    int i = 0;
    for (String q : Files.readAllLines(Path.of("fixtures/corpus/monitor-queries.txt"), StandardCharsets.UTF_8)) {
      Query parsed = parser.parse(q);
      queries.add(new MonitorQuery("q" + i++, parsed, q, Map.of()));
    }
    SweepMicro.measure("mon_register", () -> {
      try (Monitor m = new Monitor(analyzer)) {
        m.register(queries);
        SweepMicro.sink += m.getQueryCount();
      }
      return queries.size();
    });
    Monitor monitor = new Monitor(analyzer);
    monitor.register(queries);
    List<Document> docs = new ArrayList<>();
    for (String[] d : DOCS) {
      docs.add(doc(d));
    }
    SweepMicro.measure("mon_match", () -> {
      for (Document d : docs) {
        SweepMicro.sink += monitor.match(d, QueryMatch.SIMPLE_MATCHER).getMatchCount();
      }
      return docs.size();
    });
    // the monitor's purge thread keeps the JVM alive until it is closed
    monitor.close();
  }
}
