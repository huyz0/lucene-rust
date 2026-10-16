import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.monitor.QueryDecomposer;
import org.apache.lucene.monitor.MonitorFixtureAccess;
import org.apache.lucene.monitor.QueryTree;
import org.apache.lucene.monitor.TermWeightor;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.5a: {@code QueryDecomposer} and {@code QueryAnalyzer} (the term tree the
 * presearcher indexes) over {@code fixtures/corpus/monitor-queries.txt}, one classic-syntax query a
 * line, default field {@code body}, parsed with {@code StandardAnalyzer}. {@code decompose.tsv}:
 * per query, the escaped query, then each decomposed sub-query's {@code toString()} (escaped,
 * sorted, tab-separated) -- or {@code X} and the exception. {@code tree.tsv}: per query, the
 * escaped query, the tree's {@code toString()} and its collected terms under the default weightor,
 * or {@code X} and the exception. {@code weights.tsv}: the named weightors' weights over a few
 * terms. {@code crates/lucene-search/tests/monitor_fixtures.rs} compares.
 */
public class GenMonitor {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static List<String> queries() throws Exception {
    return AnalysisRows.corpus("monitor-queries.txt");
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("monitor");
    Files.createDirectories(out);
    List<String> qs = queries();
    QueryParser parser = new QueryParser("body", new StandardAnalyzer());
    QueryDecomposer decomposer = new QueryDecomposer();
    TermWeightor weightor = TermWeightor.DEFAULT;

    StringBuilder decompose = new StringBuilder();
    StringBuilder tree = new StringBuilder();
    for (String text : qs) {
      decompose.append(esc(text)).append('\t');
      tree.append(esc(text)).append('\t');
      try {
        Query q = parser.parse(text);
        List<String> subs = new ArrayList<>();
        for (Query sub : decomposer.decompose(q)) {
          subs.add(sub.toString());
        }
        java.util.Collections.sort(subs);
        decompose.append("Q");
        for (String s : subs) {
          decompose.append('\t').append(esc(s));
        }
      } catch (Exception e) {
        decompose.append("X\t").append(esc(e.getClass().getName() + ": " + e.getMessage()));
      }
      decompose.append('\n');

      try {
        Query q = parser.parse(text);
        QueryTree t = MonitorFixtureAccess.buildTree(q, weightor);
        List<String> terms = new ArrayList<>();
        t.collectTerms((field, bytes) -> terms.add(field + ":" + bytes.utf8ToString()));
        tree.append("Q\t").append(esc(t.toString())).append('\t')
            .append(esc(String.join(" ", terms)));
      } catch (Exception e) {
        tree.append("X\t").append(esc(e.getClass().getName() + ": " + e.getMessage()));
      }
      tree.append('\n');
    }
    Files.writeString(out.resolve("decompose.tsv"), decompose.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("tree.tsv"), tree.toString(), StandardCharsets.UTF_8);

    // T13.5b: the presearcher's indexed document per query, and a batch's query
    StringBuilder presearch = new StringBuilder();
    org.apache.lucene.monitor.Presearcher plain = new org.apache.lucene.monitor.TermFilteredPresearcher();
    org.apache.lucene.monitor.Presearcher multi =
        new org.apache.lucene.monitor.MultipassTermFilteredPresearcher(2);
    for (String text : qs) {
      presearch.append(esc(text));
      try {
        Query q = parser.parse(text);
        presearch.append("	P");
        for (String f : MonitorFixtureAccess.indexedFields(plain, q, java.util.Map.of())) {
          presearch.append('	').append(esc(f));
        }
        presearch.append("	M");
        for (String f : MonitorFixtureAccess.indexedFields(multi, q, java.util.Map.of())) {
          presearch.append('	').append(esc(f));
        }
      } catch (Exception e) {
        presearch.append("	X	").append(esc(e.getClass().getName() + ": " + e.getMessage()));
      }
      presearch.append('\n');
    }
    Files.writeString(out.resolve("presearch.tsv"), presearch.toString(), StandardCharsets.UTF_8);

    // the batch query: one document, its fields analysed, under both presearchers
    StringBuilder batches = new StringBuilder();
    String[][] docs = {
      {"body=the quick brown fox"},
      {"body=a b", "title=quick"},
      {"body=fox fox fox", "title=brown"},
      {"body=zzzz extraordinarily"},
    };
    StandardAnalyzer batchAnalyzer = new StandardAnalyzer();
    for (String[] doc : docs) {
      batches.append(esc(String.join("|", doc))).append('	');
      batches.append("P	").append(esc(MonitorFixtureAccess.batchQuery(plain, batchAnalyzer, doc)));
      batches.append("	M	").append(esc(MonitorFixtureAccess.batchQuery(multi, batchAnalyzer, doc)));
      batches.append('\n');
    }
    Files.writeString(out.resolve("batch.tsv"), batches.toString(), StandardCharsets.UTF_8);

    // T13.5c: a real Monitor -- register the corpus, match a batch of documents,
    // and dump each document's matched query ids, in Java's order
    org.apache.lucene.monitor.Monitor monitor =
        new org.apache.lucene.monitor.Monitor(new StandardAnalyzer());
    List<org.apache.lucene.monitor.MonitorQuery> registered = new ArrayList<>();
    for (int i = 0; i < qs.size(); i++) {
      registered.add(new org.apache.lucene.monitor.MonitorQuery("q" + i, parser.parse(qs.get(i)), qs.get(i), java.util.Map.of()));
    }
    monitor.register(registered);
    StringBuilder matchRows = new StringBuilder();
    StringBuilder scoreRows = new StringBuilder();
    for (String[] doc : docs) {
      org.apache.lucene.document.Document d = new org.apache.lucene.document.Document();
      for (String f : doc) {
        int eq = f.indexOf('=');
        d.add(new org.apache.lucene.document.TextField(f.substring(0, eq), f.substring(eq + 1), org.apache.lucene.document.Field.Store.NO));
      }
      var result = monitor.match(d, org.apache.lucene.monitor.QueryMatch.SIMPLE_MATCHER);
      java.util.List<String> ids = new ArrayList<>();
      for (var m : result.getMatches()) {
        ids.add(m.getQueryId());
      }
      java.util.Collections.sort(ids);
      var scored = monitor.match(d, org.apache.lucene.monitor.ScoringMatch.DEFAULT_MATCHER);
      java.util.List<String> scoreCells = new ArrayList<>();
      for (var m : scored.getMatches()) {
        scoreCells.add(m.getQueryId() + "=" + Integer.toHexString(Float.floatToIntBits(m.getScore())));
      }
      java.util.Collections.sort(scoreCells);
      scoreRows.append(esc(String.join("|", doc))).append((char) 9).append(esc(String.join(" ", scoreCells))).append((char) 10);
      matchRows.append(esc(String.join("|", doc))).append('\t').append(esc(String.join(" ", ids))).append('\n');
    }
    Files.writeString(out.resolve("match.tsv"), matchRows.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("score.tsv"), scoreRows.toString(), StandardCharsets.UTF_8);
    // T13.5d: whole batches under each matcher; per batch and matcher, per document, the
    // matched query ids with their detail (score bits, explanation, hits)
    String[][][] batchSets = {
      {{"body=the quick brown fox"}, {"body=a b", "title=quick"}},
      {{"body=the quick brown fox"}, {"body=a b", "title=quick"}, {"body=fox fox fox", "title=brown"}, {"body=zzzz extraordinarily"}},
      {{"body=quick quick brown"}, {"body=fox jumps over the lazy dog"}},
    };
    StringBuilder batchMatch = new StringBuilder();
    for (String[][] batchDocs : batchSets) {
      org.apache.lucene.document.Document[] arr = new org.apache.lucene.document.Document[batchDocs.length];
      for (int di = 0; di < batchDocs.length; di++) {
        org.apache.lucene.document.Document d = new org.apache.lucene.document.Document();
        for (String f : batchDocs[di]) {
          int eq = f.indexOf('=');
          d.add(new org.apache.lucene.document.TextField(f.substring(0, eq), f.substring(eq + 1), org.apache.lucene.document.Field.Store.NO));
        }
        arr[di] = d;
      }
      StringBuilder label = new StringBuilder();
      for (String[] bd : batchDocs) label.append(String.join("+", bd)).append("|");
      var simple = monitor.match(arr, org.apache.lucene.monitor.QueryMatch.SIMPLE_MATCHER);
      var scoring = monitor.match(arr, org.apache.lucene.monitor.ScoringMatch.DEFAULT_MATCHER);
      var explaining = monitor.match(arr, org.apache.lucene.monitor.ExplainingMatch.MATCHER);
      var highlights = monitor.match(arr, org.apache.lucene.monitor.HighlightsMatch.MATCHER);
      for (int di = 0; di < arr.length; di++) {
        List<String> cells = new ArrayList<>();
        for (var m : simple.getMatches(di)) cells.add("S:" + m.getQueryId());
        for (var m : scoring.getMatches(di)) cells.add("C:" + m.getQueryId() + "=" + Integer.toHexString(Float.floatToIntBits(m.getScore())));
        for (var m : explaining.getMatches(di)) cells.add("E:" + m.getQueryId() + "=" + m.getExplanation().toString());
        for (var m : highlights.getMatches(di)) cells.add("H:" + m.getQueryId() + "=" + m.getHits());
        java.util.Collections.sort(cells);
        batchMatch.append(esc(label.toString())).append((char) 9).append(di).append((char) 9)
            .append(esc(String.join(String.valueOf((char) 9), cells))).append((char) 10);
      }
    }
    Files.writeString(out.resolve("batch_match.tsv"), batchMatch.toString(), StandardCharsets.UTF_8);
    // T13.5d: MonitorQuerySerializer's bytes for queries with and without metadata
    org.apache.lucene.monitor.MonitorQuerySerializer ser =
        org.apache.lucene.monitor.MonitorQuerySerializer.fromParser(t -> {
          try {
            return parser.parse(t);
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        });
    StringBuilder serial = new StringBuilder();
    String[][] serialCases = {
      {"q1", "fox", ""},
      {"q2", "title:quick AND body:fox", "a=1"},
      {"q3", "\"quick brown\"~2", "owner=ann,team=search"},
      {"id with spaces", "café OR naïve", "k=v"},
      {"", "x", ""},
    };
    for (String[] c : serialCases) {
      java.util.Map<String, String> md = new java.util.TreeMap<>();
      if (!c[2].isEmpty()) {
        for (String kv : c[2].split(",")) {
          md.put(kv.split("=")[0], kv.split("=")[1]);
        }
      }
      org.apache.lucene.monitor.MonitorQuery mq =
          new org.apache.lucene.monitor.MonitorQuery(c[0], parser.parse(c[1]), c[1], md);
      org.apache.lucene.util.BytesRef bytes = ser.serialize(mq);
      StringBuilder hex = new StringBuilder();
      for (int i = 0; i < bytes.length; i++) {
        hex.append(String.format("%02x", bytes.bytes[bytes.offset + i] & 0xff));
      }
      serial.append(esc(c[0])).append((char) 9).append(esc(c[1])).append((char) 9)
          .append(esc(c[2])).append((char) 9).append(hex).append((char) 10);
    }
    Files.writeString(out.resolve("serialized.tsv"), serial.toString(), StandardCharsets.UTF_8);
    // a persisted monitor: Lucene writes the query index, Rust opens it
    Path persisted = out.resolve("index");
    if (Files.exists(persisted)) {
      try (var walk = Files.walk(persisted)) {
        for (Path f : (Iterable<Path>) walk.sorted(java.util.Comparator.reverseOrder())::iterator) {
          Files.delete(f);
        }
      }
    }
    try (org.apache.lucene.monitor.Monitor pm =
        new org.apache.lucene.monitor.Monitor(
            new StandardAnalyzer(),
            new org.apache.lucene.monitor.MonitorConfiguration().setIndexPath(persisted, ser))) {
      pm.register(registered);
      Files.writeString(
          out.resolve("persisted.txt"),
          pm.getQueryCount() + " " + pm.getDisjunctCount() + "\n",
          StandardCharsets.UTF_8);
    }
    // the monitor's purge thread keeps the JVM alive until it is closed
    monitor.close();

    StringBuilder weights = new StringBuilder();
    Term[] probes = {
      new Term("body", "fox"), new Term("body", "a"), new Term("title", "quick"),
      new Term("body", "extraordinarily"), new Term("body", "")
    };
    java.util.Map<String, TermWeightor> named = new java.util.LinkedHashMap<>();
    named.put("default", TermWeightor.DEFAULT);
    named.put("length", TermWeightor.lengthWeightor(3, 0.3f));
    named.put("field", TermWeightor.fieldWeightor(2.0, "title"));
    named.put("term", TermWeightor.termWeightor(3.0, new BytesRef("fox")));
    named.put("freq", TermWeightor.termFreqWeightor(java.util.Map.of("fox", 4), 10, 1));
    named.put("combined", TermWeightor.combine(TermWeightor.DEFAULT, TermWeightor.fieldWeightor(2.0, "title")));
    for (var e : named.entrySet()) {
      weights.append(e.getKey());
      for (Term p : probes) {
        weights.append('\t').append(Double.doubleToLongBits(e.getValue().applyAsDouble(p)));
      }
      weights.append('\n');
    }
    Files.writeString(out.resolve("weights.tsv"), weights.toString(), StandardCharsets.UTF_8);
  }
}
