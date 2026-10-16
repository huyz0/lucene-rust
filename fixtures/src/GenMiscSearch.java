import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.search.DiversifiedTopDocsCollector;
import org.apache.lucene.misc.search.DiversifiedTopDocsCollector.ScoreDocKey;
import org.apache.lucene.misc.search.DiversifiedTopDocsCollectorManager;
import org.apache.lucene.misc.search.DocValuesStats;
import org.apache.lucene.misc.search.DocValuesStatsCollector;
import org.apache.lucene.misc.search.MemoryAccountingBitsetCollectorManager;
import org.apache.lucene.misc.CollectorMemoryTracker;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.IndexSearcher.LeafReaderContextPartition;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;

/**
 * M14 T14.3 `misc`: the {@code misc.search} collectors. Writes {@code misc/search_src} (two
 * segments of 40 and 35 documents) and {@code misc/search.tsv}:
 *
 * <pre>
 * S  query  field  count missing min max sum mean variance stdev valuesCount   a doc-values statistic
 * D  config rank doc scoreBits key      a diversified hit (total in rank -1)
 * B  config doc                         a document of the memory-accounting bitset (length in doc -1)
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_search_fixtures.rs}.
 */
public class GenMiscSearch {

  /** A searcher that can run one leaf, as a {@code CollectorManager} slice does. */
  static class LeafSearcher extends IndexSearcher {
    LeafSearcher(IndexReader r) {
      super(r);
    }

    void leaf(LeafReaderContext ctx, Query q, Collector c) throws Exception {
      Weight w = createWeight(rewrite(q), c.scoreMode(), 1f);
      search(new LeafReaderContextPartition[] {LeafReaderContextPartition.createForEntireSegment(ctx)}, w, c);
    }
  }

  static String hex(BytesRef b) {
    if (b == null) return "null";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return sb.toString();
  }

  static String dbits(double d) {
    return Long.toHexString(Double.doubleToRawLongBits(d));
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path src = out.resolve("search_src");
    Random r = new Random(11);
    try (Directory dir = FSDirectory.open(src);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int seg = 0; seg < 2; seg++) {
        int n = seg == 0 ? 40 : 35;
        for (int i = 0; i < n; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", "d" + (seg * 40 + i), Field.Store.YES));
          StringBuilder t = new StringBuilder("a");
          for (int k = r.nextInt(6); k > 0; k--) t.append(r.nextBoolean() ? " a" : " c");
          if (r.nextInt(3) == 0) t.append(" b");
          doc.add(new TextField("txt", t.toString(), Field.Store.NO));
          if (r.nextInt(5) != 0) doc.add(new NumericDocValuesField("lv", r.nextInt(2000) - 1000L));
          if (r.nextInt(5) != 0) doc.add(new DoubleDocValuesField("dv", (r.nextInt(2000) - 1000) / 8.0));
          if (r.nextInt(4) != 0) {
            for (int k = 1 + r.nextInt(3); k > 0; k--) {
              doc.add(new SortedNumericDocValuesField("snl", r.nextInt(500) - 250L));
            }
          }
          if (r.nextInt(4) != 0) {
            for (int k = 1 + r.nextInt(3); k > 0; k--) {
              doc.add(
                  new SortedNumericDocValuesField(
                      "snd", Double.doubleToLongBits((r.nextInt(500) - 250) / 4.0)));
            }
          }
          if (r.nextInt(5) != 0) doc.add(new SortedDocValuesField("sd", new BytesRef("s" + r.nextInt(30))));
          if (r.nextInt(5) != 0) {
            for (int k = 1 + r.nextInt(3); k > 0; k--) {
              doc.add(new SortedSetDocValuesField("ssd", new BytesRef("v" + r.nextInt(40))));
            }
          }
          if (seg == 1 && r.nextInt(2) == 0) doc.add(new NumericDocValuesField("late", r.nextInt(100)));
          if (r.nextInt(6) != 0) doc.add(new NumericDocValuesField("key", r.nextInt(5)));
          w.addDocument(doc);
        }
        w.commit();
      }
    }

    try (PrintWriter tsv =
            new PrintWriter(Files.newBufferedWriter(out.resolve("search.tsv"), StandardCharsets.UTF_8));
        Directory dir = FSDirectory.open(src);
        IndexReader reader = DirectoryReader.open(dir)) {
      LeafSearcher searcher = new LeafSearcher(reader);
      for (String q : new String[] {"a", "b"}) {
        Query query = new TermQuery(new Term("txt", q));
        stats(tsv, searcher, reader, q, query);
      }
      Query all = new TermQuery(new Term("txt", "a"));
      diversified(tsv, searcher, reader, all);
      bitset(tsv, searcher, reader, new TermQuery(new Term("txt", "b")));
    }
    System.exit(0);
  }

  static <S extends DocValuesStats<?>> void one(
      PrintWriter tsv, LeafSearcher searcher, IndexReader reader, String q, Query query,
      Supplier<S> supplier, String kind) throws Exception {
    // one collector per leaf, merged by hand as DocValuesStatsCollectorManager.reduce does
    S merged = supplier.get();
    java.lang.reflect.Method merge = DocValuesStats.class.getDeclaredMethod("merge", DocValuesStats.class);
    merge.setAccessible(true);
    java.lang.reflect.Method getStats = DocValuesStatsCollector.class.getDeclaredMethod("getStats");
    getStats.setAccessible(true);
    for (LeafReaderContext ctx : reader.leaves()) {
      DocValuesStatsCollector c = new DocValuesStatsCollector(supplier.get());
      searcher.leaf(ctx, query, c);
      merge.invoke(merged, getStats.invoke(c));
    }
    String min;
    String max;
    String extra = "-\t-\t-\t-\t-";
    if (merged instanceof DocValuesStats.LongDocValuesStats s) {
      min = "" + s.min();
      max = "" + s.max();
      extra = s.sum() + "\t" + dbits(s.mean()) + "\t" + dbits(s.variance()) + "\t" + dbits(s.stdev()) + "\t-";
    } else if (merged instanceof DocValuesStats.DoubleDocValuesStats s) {
      min = dbits(s.min());
      max = dbits(s.max());
      extra = dbits(s.sum()) + "\t" + dbits(s.mean()) + "\t" + dbits(s.variance()) + "\t" + dbits(s.stdev()) + "\t-";
    } else if (merged instanceof DocValuesStats.SortedLongDocValuesStats s) {
      min = "" + s.min();
      max = "" + s.max();
      extra = s.sum() + "\t" + dbits(s.mean()) + "\t" + dbits(s.variance()) + "\t" + dbits(s.stdev()) + "\t" + s.valuesCount();
    } else if (merged instanceof DocValuesStats.SortedDoubleDocValuesStats s) {
      min = dbits(s.min());
      max = dbits(s.max());
      extra = dbits(s.sum()) + "\t" + dbits(s.mean()) + "\t" + dbits(s.variance()) + "\t" + dbits(s.stdev()) + "\t" + s.valuesCount();
    } else {
      min = hex((BytesRef) merged.min());
      max = hex((BytesRef) merged.max());
    }
    tsv.printf("S\t%s\t%s\t%d\t%d\t%s\t%s\t%s%n", q, kind + ":" + merged.field(), merged.count(), merged.missing(), min, max, extra);
  }

  static void stats(PrintWriter tsv, LeafSearcher s, IndexReader r, String q, Query query) throws Exception {
    one(tsv, s, r, q, query, () -> new DocValuesStats.LongDocValuesStats("lv"), "long");
    one(tsv, s, r, q, query, () -> new DocValuesStats.DoubleDocValuesStats("dv"), "double");
    one(tsv, s, r, q, query, () -> new DocValuesStats.SortedLongDocValuesStats("snl"), "sortedlong");
    one(tsv, s, r, q, query, () -> new DocValuesStats.SortedDoubleDocValuesStats("snd"), "sorteddouble");
    one(tsv, s, r, q, query, () -> new Sorted("sd"), "sorted");
    one(tsv, s, r, q, query, () -> new SortedSet("ssd"), "sortedset");
    one(tsv, s, r, q, query, () -> new DocValuesStats.LongDocValuesStats("late"), "long");
    one(tsv, s, r, q, query, () -> new DocValuesStats.LongDocValuesStats("nothing"), "long");
  }

  static class Sorted extends DocValuesStats.SortedDocValuesStats {
    Sorted(String f) {
      super(f);
    }
  }

  static class SortedSet extends DocValuesStats.SortedSetDocValuesStats {
    SortedSet(String f) {
      super(f);
    }
  }

  static class Div extends DiversifiedTopDocsCollector {
    Div(int n, int m) {
      super(n, m);
    }

    @Override
    protected NumericDocValues getKeys(LeafReaderContext context) {
      try {
        return context.reader().getNumericDocValues("key");
      } catch (java.io.IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  static class DivManager extends DiversifiedTopDocsCollectorManager<Div> {
    final int n, m;

    DivManager(int n, int m) {
      super(n, m);
      this.n = n;
      this.m = m;
    }

    @Override
    public Div newCollector() {
      return new Div(n, m);
    }
  }

  static void printTop(PrintWriter tsv, String config, TopDocs top) {
    int rank = 0;
    for (ScoreDoc sd : top.scoreDocs) {
      ScoreDocKey k = (ScoreDocKey) sd;
      tsv.printf("D\t%s\t%d\t%d\t%d\t%d%n", config, rank++, sd.doc, Float.floatToRawIntBits(sd.score), k.getKey());
    }
    tsv.printf("D\t%s\t-1\t%d\t0\t0%n", config, top.totalHits.value());
  }

  static void diversified(PrintWriter tsv, LeafSearcher searcher, IndexReader reader, Query query)
      throws Exception {
    for (int[] c : new int[][] {{10, 2}, {5, 1}, {50, 3}}) {
      String name = c[0] + "_" + c[1];
      Div single = new Div(c[0], c[1]);
      searcher.search(query, single);
      printTop(tsv, "c" + name, single.topDocs());
      DivManager mgr = new DivManager(c[0], c[1]);
      List<Div> cs = new ArrayList<>();
      for (LeafReaderContext ctx : reader.leaves()) {
        Div d = mgr.newCollector();
        searcher.leaf(ctx, query, d);
        cs.add(d);
      }
      printTop(tsv, "m" + name, mgr.reduce(cs));
    }
  }

  static void bitset(PrintWriter tsv, LeafSearcher searcher, IndexReader reader, Query query)
      throws Exception {
    MemoryAccountingBitsetCollectorManager mgr =
        new MemoryAccountingBitsetCollectorManager(new CollectorMemoryTracker("t", Long.MAX_VALUE));
    var cs = new ArrayList<org.apache.lucene.misc.search.MemoryAccountingBitsetCollector>();
    for (LeafReaderContext ctx : reader.leaves()) {
      var c = mgr.newCollector();
      searcher.leaf(ctx, query, c);
      cs.add(c);
    }
    FixedBitSet bits = mgr.reduce(cs).bitSet();
    tsv.printf("B\tb\t-1\t%d%n", bits.length());
    for (int d = bits.nextSetBit(0); d != -1; d = d + 1 < bits.length() ? bits.nextSetBit(d + 1) : -1) {
      tsv.printf("B\tb\t%d\t%d%n", d, d);
    }
  }
}
