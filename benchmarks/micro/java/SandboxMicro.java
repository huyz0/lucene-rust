import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.sandbox.document.FloatPointNearestNeighbor;
import org.apache.lucene.sandbox.document.IntPointMultiRangeBuilder;
import org.apache.lucene.sandbox.search.CombinedFieldQuery;
import org.apache.lucene.sandbox.search.CoveringQuery;
import org.apache.lucene.sandbox.search.DocValuesMultiRangeQuery;
import org.apache.lucene.sandbox.search.LargeNumHitsTopDocsCollectorManager;
import org.apache.lucene.sandbox.search.PhraseWildcardQuery;
import org.apache.lucene.sandbox.search.TermAutomatonQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
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
 *
 * <p>The scoring queries run over a second index, {@code text/} (built once too): 200 000
 * documents in four segments, a {@code body} of 5 to 30 and a {@code title} of 1 to 4 words
 * {@code w0000..w1999} skewed toward the low ids, and an {@code mm} numeric doc value 0 to 3. Per
 * range set, four words {@code a b c d} (see {@link #words}) and one top-10 search each:
 * the sandbox {@code CombinedFieldQuery} of {@code a b c} over {@code body} and {@code title^2},
 * {@code CoveringQuery} of the four as body terms with {@code mm} as the minimum, the {@code
 * TermAutomatonQuery} {@code a (b | c d)}, and {@code PhraseWildcardQuery} {@code a b*} (the prefix
 * {@code b}'s first four characters, 128 expansions).
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

  static String word(int w) {
    return String.format("w%04d", w);
  }

  static String words(Random r, int n) {
    StringBuilder b = new StringBuilder();
    for (int k = 0; k < n; k++) {
      double u = r.nextDouble();
      b.append(k == 0 ? "" : " ").append(word((int) (2000 * u * u * u)));
    }
    return b.toString();
  }

  static void buildText(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1020L);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new TextField("body", words(r, 5 + r.nextInt(26)), Field.Store.NO));
        doc.add(new TextField("title", words(r, 1 + r.nextInt(4)), Field.Store.NO));
        doc.add(new NumericDocValuesField("mm", r.nextInt(4)));
        w.addDocument(doc);
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
      w.commit();
    }
    Files.writeString(dir.resolve("built"), "");
  }

  /** The words a range set's scoring queries name. */
  static String[] words(long[][] set) {
    return new String[] {
      word((int) (set[0][0] % 50)),
      word((int) (set[1][0] % 200)),
      word((int) (set[2][0] % 200)),
      word((int) (set[3][0] % 50))
    };
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
      runText(dir.resolve("text"), sets);
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

  static void runText(Path dir, List<long[][]> sets) throws IOException {
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      cases(
          "sb_combined_terms",
          sets,
          s,
          set -> {
            String[] w = words(set);
            CombinedFieldQuery.Builder b = new CombinedFieldQuery.Builder();
            for (int k = 0; k < 3; k++) b.addTerm(new BytesRef(w[k]));
            return b.addField("body", 1).addField("title", 2).build();
          });
      cases(
          "sb_covering",
          sets,
          s,
          set -> {
            List<Query> qs = new ArrayList<>();
            for (String w : words(set)) qs.add(new TermQuery(new Term("body", w)));
            return new CoveringQuery(qs, LongValuesSource.fromLongField("mm"));
          });
      cases(
          "sb_term_automaton",
          sets,
          s,
          set -> {
            String[] w = words(set);
            TermAutomatonQuery q = new TermAutomatonQuery("body");
            for (int k = 0; k < 4; k++) q.createState();
            q.setAccept(3, true);
            q.addTransition(0, 1, w[0]);
            q.addTransition(1, 3, w[1]);
            q.addTransition(1, 2, w[2]);
            q.addTransition(2, 3, w[3]);
            q.finish();
            return q;
          });
      cases(
          "sb_phrase_wildcard",
          sets,
          s,
          set -> {
            String[] w = words(set);
            PhraseWildcardQuery.Builder b = new PhraseWildcardQuery.Builder("body", 128);
            b.addTerm(new BytesRef(w[0]));
            b.addMultiTerm(new PrefixQuery(new Term("body", w[1].substring(0, 4))));
            return b.build();
          });
      // LargeNumHitsTopDocsCollector: the top 1000 of two mid-frequency words.
      {
        List<Query> qs = new ArrayList<>();
        for (long[][] set : sets) {
          String[] w = words(set);
          qs.add(
              new BooleanQuery.Builder()
                  .add(new TermQuery(new Term("body", w[1])), BooleanClause.Occur.SHOULD)
                  .add(new TermQuery(new Term("body", w[2])), BooleanClause.Occur.SHOULD)
                  .build());
        }
        Fnv f = new Fnv();
        for (Query q : qs) digest(f, s.search(q, new LargeNumHitsTopDocsCollectorManager(1000)));
        System.out.printf("#check	%s	%016x	%d%n", "sb_large_num_hits", f.h, sets.size());
        measure(
            "sb_large_num_hits",
            () -> {
              Fnv g = new Fnv();
              for (Query q : qs) {
                digest(g, s.search(q, new LargeNumHitsTopDocsCollectorManager(1000)));
              }
              sink += g.h;
              return sets.size();
            });
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (!Files.exists(dir.resolve("text").resolve("built"))) {
        Files.createDirectories(dir.resolve("text"));
        buildText(dir.resolve("text"));
      }
      if (Files.exists(dir.resolve("sandbox-ranges.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
