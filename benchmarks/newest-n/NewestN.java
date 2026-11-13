import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;

/**
 * Newest-N (top N by a time field) on stock Lucene, over real Loghub lines.
 *
 *   build <data dir> <out>   25 segments of 250k lines, no merges
 *   merge <idx> <n>          force-merge to n segments
 *   query <idx> <out dir> [cold]
 *                            every shape in {@link #shapes}: median latency, and the hits
 *                            (global doc, sort value) written to out dir for comparison
 *
 * Fields: ts (LongPoint + NumericDocValues, unique ms), ts_s (doc values with a skip index, no
 * points), tsec (ts truncated to the second: ~10 lines a value, ties), tmiss (ts on 9 lines of
 * 10, and on none of the newest 0.5%), src (keyword term: bgl, hdfs, openssh, thunderbird).
 */
public class NewestN {
  static final String[] SETS = {"bgl", "hdfs", "openssh", "thunderbird"};
  static final long T0 = 1730000000000L, SPAN = 7L * 24 * 3600 * 1000;
  static final int TOTAL = 6_454_808;

  public static void main(String[] a) throws Exception {
    switch (a[0]) {
      case "build" -> build(a[1], a[2]);
      case "query" -> query(a[1], a[2], a.length > 3 && a[3].equals("cold"));
      case "merge" -> {
        IndexWriterConfig c = new IndexWriterConfig();
        c.setUseCompoundFile(false);
        try (IndexWriter w = new IndexWriter(FSDirectory.open(Paths.get(a[1])), c)) { w.forceMerge(Integer.parseInt(a[2])); }
      }
      default -> throw new IllegalArgumentException(a[0]);
    }
  }

  static void build(String dir, String out) throws IOException {
    IndexWriterConfig c = new IndexWriterConfig();
    c.setUseCompoundFile(false);
    // -Dmerges=tiered: the default merge policy (it merges segments that are
    // not adjacent, as an OpenSearch shard does); otherwise no merges.
    if (!"tiered".equals(System.getProperty("merges"))) c.setMergePolicy(NoMergePolicy.INSTANCE);
    c.setMaxBufferedDocs(250_000);
    c.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    BufferedReader[] r = new BufferedReader[SETS.length];
    for (int i = 0; i < SETS.length; i++)
      r[i] = Files.newBufferedReader(Paths.get(dir, SETS[i] + ".txt"), StandardCharsets.ISO_8859_1);
    boolean[] done = new boolean[SETS.length];
    int n = 0, live = SETS.length;
    Random rnd = new Random(7);
    long t0 = System.nanoTime();
    try (IndexWriter w = new IndexWriter(FSDirectory.open(Paths.get(out)), c)) {
      while (live > 0) {
        int s = rnd.nextInt(SETS.length);
        if (done[s]) continue;
        String line = r[s].readLine();
        if (line == null) { done[s] = true; live--; continue; }
        if (line.isEmpty()) continue;
        long ts = T0 + (long) n * SPAN / TOTAL;
        Document d = new Document();
        d.add(new LongPoint("ts", ts));
        d.add(new NumericDocValuesField("ts", ts));
        d.add(NumericDocValuesField.indexedField("ts_s", ts));
        long sec = ts / 1000 * 1000;
        d.add(new LongPoint("tsec", sec));
        d.add(new NumericDocValuesField("tsec", sec));
        if (n % 10 != 0 && n < TOTAL - TOTAL / 200) {
          d.add(new LongPoint("tmiss", ts));
          d.add(new NumericDocValuesField("tmiss", ts));
        }
        d.add(new StringField("src", SETS[s], Field.Store.NO));
        w.addDocument(d);
        n++;
      }
      w.commit();
    }
    System.out.printf("built %d docs in %.1f s%n", n, (System.nanoTime() - t0) / 1e9);
  }

  record Shape(String name, Query q, Sort sort, int n, boolean page2) {}

  static List<Shape> shapes() {
    long to = T0 + SPAN, from = to - 24L * 3600 * 1000;
    Query all = new MatchAllDocsQuery();
    Query ssh = new TermQuery(new Term("src", "openssh"));
    Query hdfs = new TermQuery(new Term("src", "hdfs"));
    Query range = LongPoint.newRangeQuery("ts", from, to);
    Query rangeSsh = new BooleanQuery.Builder().add(range, BooleanClause.Occur.FILTER).add(ssh, BooleanClause.Occur.FILTER).build();
    Query rangeHdfs = new BooleanQuery.Builder().add(range, BooleanClause.Occur.FILTER).add(hdfs, BooleanClause.Occur.FILTER).build();
    Sort desc = new Sort(new SortField("ts", SortField.Type.LONG, true));
    Sort asc = new Sort(new SortField("ts", SortField.Type.LONG, false));
    Sort skipDesc = new Sort(new SortField("ts_s", SortField.Type.LONG, true));
    Sort skipAsc = new Sort(new SortField("ts_s", SortField.Type.LONG, false));
    Sort ties = new Sort(new SortField("tsec", SortField.Type.LONG, true));
    Sort tiesAsc = new Sort(new SortField("tsec", SortField.Type.LONG, false));
    SortField ml = new SortField("tmiss", SortField.Type.LONG, true);
    ml.setMissingValue(Long.MIN_VALUE);
    Sort missLast = new Sort(ml);
    SortField mf = new SortField("tmiss", SortField.Type.LONG, true);
    mf.setMissingValue(Long.MAX_VALUE);
    Sort missFirst = new Sort(mf);
    SortField mla = new SortField("tmiss", SortField.Type.LONG, false);
    mla.setMissingValue(Long.MAX_VALUE);
    Sort missLastAsc = new Sort(mla);
    List<Shape> s = new ArrayList<>();
    for (int n : new int[] {10, 100, 1000}) {
      s.add(new Shape("match_all ts desc", all, desc, n, false));
      s.add(new Shape("match_all ts asc", all, asc, n, false));
      s.add(new Shape("match_all ts_s(skipper) desc", all, skipDesc, n, false));
      s.add(new Shape("match_all ts_s(skipper) asc", all, skipAsc, n, false));
      s.add(new Shape("src=openssh ts desc", ssh, desc, n, false));
      s.add(new Shape("src=hdfs ts desc", hdfs, desc, n, false));
      s.add(new Shape("src=openssh ts_s(skipper) desc", ssh, skipDesc, n, false));
      s.add(new Shape("range24h ts desc", range, desc, n, false));
      s.add(new Shape("range24h&hdfs ts desc", rangeHdfs, desc, n, false));
      s.add(new Shape("range24h&openssh(0 hits) ts desc", rangeSsh, desc, n, false));
      s.add(new Shape("match_all tsec(ties) desc", all, ties, n, false));
      s.add(new Shape("match_all tsec(ties) asc", all, tiesAsc, n, false));
      s.add(new Shape("match_all tmiss desc missing_last", all, missLast, n, false));
      s.add(new Shape("match_all tmiss asc missing_last", all, missLastAsc, n, false));
      s.add(new Shape("match_all tmiss desc missing_first", all, missFirst, n, false));
      s.add(new Shape("match_all ts desc search_after", all, desc, n, true));
      s.add(new Shape("src=hdfs tsec(ties) desc search_after", hdfs, ties, n, true));
    }
    // Filters whose best matches lie far from the key's best, which the
    // bounded search hands back to a whole pass (after the shapes above, so
    // their indices stay put).
    Query oldHalf = LongPoint.newRangeQuery("tsec", T0, T0 + SPAN / 2);
    Query middle = LongPoint.newRangeQuery("tsec", T0 + SPAN / 3, T0 + 2 * SPAN / 3);
    for (int n : new int[] {10, 100, 1000}) {
      s.add(new Shape("src=openssh ts asc", ssh, asc, n, false));
      s.add(new Shape("tsec(old half) ts desc", oldHalf, desc, n, false));
      s.add(new Shape("tsec(middle third) ts desc", middle, desc, n, false));
      s.add(new Shape("tsec(middle third) ts asc", middle, asc, n, false));
    }
    return s;
  }

  static TopFieldDocs run(IndexSearcher s, Shape sh) throws IOException {
    if (!sh.page2) return s.search(sh.q, sh.n, sh.sort);
    TopFieldDocs first = s.search(sh.q, sh.n, sh.sort);
    if (first.scoreDocs.length == 0) return first;
    return (TopFieldDocs) s.searchAfter(first.scoreDocs[first.scoreDocs.length - 1], sh.q, sh.n, sh.sort);
  }

  static void query(String idx, String outDir, boolean cold) throws IOException {
    Files.createDirectories(Paths.get(outDir));
    List<Shape> shapes = shapes();
    try (PrintWriter csv = new PrintWriter(Files.newBufferedWriter(Paths.get(outDir, cold ? "java-cold.csv" : "java.csv")))) {
      csv.println("shape,n,median_ms,p90_ms,reps,hits,total,relation");
      if (cold) {
        // One fresh reader a shape, first query only (page cache dropped by the caller between runs).
        int i = Integer.getInteger("shape", -1);
        Shape sh = shapes.get(i);
        long st = System.nanoTime();
        DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(idx)));
        IndexSearcher s = new IndexSearcher(r);
        s.setQueryCache(null);
        TopFieldDocs td = run(s, sh);
        double ms = (System.nanoTime() - st) / 1e6;
        System.out.printf("%s,%d,%.3f%n", sh.name, sh.n, ms);
        return;
      }
      DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(idx)));
      IndexSearcher s = new IndexSearcher(r);
      s.setQueryCache(null);
      System.out.println("segments=" + r.leaves().size() + " docs=" + r.maxDoc());
      for (Shape sh : shapes) {
        TopFieldDocs td = null;
        long end = System.nanoTime() + 600_000_000L;
        int warm = 0;
        while (System.nanoTime() < end && warm < 30) { td = run(s, sh); warm++; }
        long[] ts = new long[400];
        int reps = 0;
        end = System.nanoTime() + 1_200_000_000L;
        while ((System.nanoTime() < end || reps < 3) && reps < ts.length) {
          long st = System.nanoTime();
          td = run(s, sh);
          ts[reps++] = System.nanoTime() - st;
        }
        Arrays.sort(ts, 0, reps);
        System.out.printf("%-45s n=%-5d median=%9.3f ms reps=%d hits=%d total=%d%s%n", sh.name, sh.n, ts[reps / 2] / 1e6, reps, td.scoreDocs.length, td.totalHits.value(), td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "" : "+");
        csv.printf("%s,%d,%.4f,%.4f,%d,%d,%d,%s%n", sh.name, sh.n, ts[reps / 2] / 1e6, ts[Math.min(reps - 1, reps * 9 / 10)] / 1e6, reps, td.scoreDocs.length, td.totalHits.value(), td.totalHits.relation());
        try (PrintWriter hw = new PrintWriter(Files.newBufferedWriter(Paths.get(outDir, "java-" + sh.name.replaceAll("[^A-Za-z0-9]+", "_") + "-" + sh.n + ".hits")))) {
          for (ScoreDoc d : td.scoreDocs) hw.println(d.doc + " " + ((FieldDoc) d).fields[0]);
        }
      }
    }
  }
}
