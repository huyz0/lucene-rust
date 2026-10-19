import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.sandbox.document.FloatPointNearestNeighbor;
import org.apache.lucene.sandbox.document.IntPointMultiRangeBuilder;
import org.apache.lucene.sandbox.search.DocValuesMultiRangeQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the lucene-sandbox benchmark pair (M14.f); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_sandbox.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 200 000 documents in four segments, each with an {@code IntPoint p}, up
 * to three {@code SortedNumericDocValuesField sn} values, one or two skip-indexed {@code snx} values
 * that grow with the document, up to two {@code SortedSetDocValuesField ss} terms and a 3-D {@code
 * FloatPoint v} -- and the 16 range sets both engines search ({@code sandbox-ranges.tsv}, ten
 * {@code lo:hi} ranges a line). Each case runs, per range set, one top-10 search: {@code
 * IntPointMultiRangeBuilder} over {@code p}, the sorted-numeric multi-range query over {@code sn} and
 * over {@code snx}, the sorted-set one over {@code ss}; and {@code FloatPointNearestNeighbor}'s 10
 * nearest to an origin read off the set. Each prints a {@code #check} digest of its hits (documents
 * and scores, or distances; not the total, which Java's threshold leaves a lower bound).
 */
public final class SandboxMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;

  interface Op {
    long run() throws IOException;
  }

  static void measure(String name, Op op) throws IOException {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) System.err.print("");
    return units;
  }

  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }
  }

  static String term(int t) {
    return String.format("t%05d", t);
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1019L);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new IntPoint("p", r.nextInt(1_000_000)));
        int nsn = r.nextInt(4);
        for (int k = 0; k < nsn; k++) {
          doc.add(new SortedNumericDocValuesField("sn", r.nextInt(1_000_000)));
        }
        int nsx = 1 + r.nextInt(2);
        for (int k = 0; k < nsx; k++) {
          doc.add(SortedNumericDocValuesField.indexedField("snx", i * 5L + r.nextInt(1000)));
        }
        int nss = r.nextInt(3);
        for (int k = 0; k < nss; k++) {
          doc.add(new SortedSetDocValuesField("ss", new BytesRef(term(r.nextInt(20_000)))));
        }
        doc.add(new FloatPoint("v", r.nextFloat() * 100, r.nextFloat() * 100, r.nextFloat() * 100));
        w.addDocument(doc);
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
      w.commit();
    }
    StringBuilder q = new StringBuilder();
    for (int k = 0; k < 16; k++) {
      for (int j = 0; j < 10; j++) {
        int lo = r.nextInt(1_000_000);
        int hi = lo + r.nextInt(5_000);
        q.append(j == 0 ? "" : ";").append(lo).append(':').append(hi);
      }
      q.append('\n');
    }
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("sandbox-ranges.tsv"), q.toString());
  }

  interface Case {
    Query query(long[][] ranges);
  }

  static void cases(String name, List<long[][]> sets, IndexSearcher s, Case c) throws IOException {
    Fnv f = new Fnv();
    for (long[][] set : sets) digest(f, s.search(c.query(set), 10));
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, sets.size());
    measure(
        name,
        () -> {
          Fnv g = new Fnv();
          for (long[][] set : sets) digest(g, s.search(c.query(set), 10));
          sink += g.h;
          return sets.size();
        });
  }

  static void digest(Fnv f, TopDocs td) {
    for (ScoreDoc sd : td.scoreDocs) {
      f.add(sd.doc);
      if (sd instanceof FieldDoc fd) {
        f.add(Float.floatToIntBits((Float) fd.fields[0]));
      } else {
        f.add(Float.floatToIntBits(sd.score));
      }
    }
  }

  static float[] origin(long[][] set) {
    return new float[] {set[0][0] % 100, set[1][0] % 100, set[2][0] % 100};
  }

  static void run(Path dir) throws IOException {
    List<long[][]> sets = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("sandbox-ranges.tsv"))) {
      String[] parts = line.split(";");
      long[][] set = new long[parts.length][];
      for (int i = 0; i < parts.length; i++) {
        String[] lh = parts[i].split(":");
        set[i] = new long[] {Long.parseLong(lh[0]), Long.parseLong(lh[1])};
      }
      sets.add(set);
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      cases(
          "sb_multi_range_points",
          sets,
          s,
          set -> {
            IntPointMultiRangeBuilder b = new IntPointMultiRangeBuilder("p", 1);
            for (long[] r : set) b.add(new int[] {(int) r[0]}, new int[] {(int) r[1]});
            return b.build();
          });
      cases(
          "sb_dv_multi_range",
          sets,
          s,
          set -> {
            var b = new DocValuesMultiRangeQuery.SortedNumericStabbingBuilder("sn");
            for (long[] r : set) b.add(r[0], r[1]);
            return b.build();
          });
      cases(
          "sb_dv_multi_range_skip",
          sets,
          s,
          set -> {
            var b = new DocValuesMultiRangeQuery.SortedNumericStabbingBuilder("snx");
            for (long[] r : set) b.add(r[0], r[1]);
            return b.build();
          });
      cases(
          "sb_ss_multi_range",
          sets,
          s,
          set -> {
            var b = new DocValuesMultiRangeQuery.SortedSetStabbingBuilder("ss");
            for (long[] r : set) {
              b.add(new BytesRef(term((int) (r[0] / 50))), new BytesRef(term((int) (r[1] / 50))));
            }
            return b.build();
          });
      Fnv f = new Fnv();
      for (long[][] set : sets) digest(f, FloatPointNearestNeighbor.nearest(s, "v", 10, origin(set)));
      System.out.printf("#check\t%s\t%016x\t%d%n", "sb_nearest", f.h, sets.size());
      measure(
          "sb_nearest",
          () -> {
            Fnv g = new Fnv();
            for (long[][] set : sets) {
              digest(g, FloatPointNearestNeighbor.nearest(s, "v", 10, origin(set)));
            }
            sink += g.h;
            return sets.size();
          });
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("sandbox-ranges.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
