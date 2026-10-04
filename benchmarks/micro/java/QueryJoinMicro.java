import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.JoinUtil;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.packed.PackedInts;

/**
 * Java side of the query-time join benchmark pair (M10 T10.3); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_query_join.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 200 000 documents in four segments, half "from" documents referencing
 * one of 20 000 keys (a {@code SORTED} {@code fk}, a {@code SORTED_SET} {@code fkm} of one to three
 * keys, a {@code NUMERIC} {@code nfk}, the {@code SORTED} join field {@code gj}) and half "to"
 * documents holding one (an indexed {@code pk}, a {@code LongPoint} {@code npk}, {@code gj}) -- and
 * the word list both engines draw from-queries from ({@code qjoin-words.tsv}). Each case runs, per
 * word, {@code JoinUtil.createJoinQuery} over {@code +type:from +body:word} (collecting the from
 * side) and the top 10 of the join it returns: the terms join (single-valued in each score mode,
 * multi-valued), the numeric join, and the global-ordinals join (with min/max). Each prints a
 * {@code #check} digest of its hits the report compares first.
 */
public final class QueryJoinMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;
  static final int KEYS = 20_000;
  static final String[] WORDS = {
    "red", "blue", "green", "fast", "slow", "big", "small", "shiny", "old", "new", "cheap", "rare",
    "warm", "cold", "soft", "hard"
  };

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

  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.6) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static String body(Random r) {
    int n = 2 + r.nextInt(8);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      b.append(word(r));
    }
    return b.toString();
  }

  static String key(int k) {
    return "k" + k;
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_3_2026_1004L);
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(DOCS / 4);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        int k = r.nextInt(KEYS);
        doc.add(new TextField("body", body(r), Field.Store.NO));
        if (i % 2 == 0) {
          doc.add(new StringField("type", "from", Field.Store.NO));
          doc.add(new SortedDocValuesField("fk", new BytesRef(key(k))));
          doc.add(new NumericDocValuesField("nfk", k));
          doc.add(new SortedDocValuesField("gj", new BytesRef(key(k))));
          int n = 1 + r.nextInt(3);
          for (int j = 0; j < n; j++) {
            doc.add(new SortedSetDocValuesField("fkm", new BytesRef(key(r.nextInt(KEYS)))));
          }
        } else {
          doc.add(new StringField("type", "to", Field.Store.NO));
          doc.add(new StringField("pk", key(k), Field.Store.NO));
          doc.add(new LongPoint("npk", k));
          doc.add(new SortedDocValuesField("gj", new BytesRef(key(k))));
        }
        w.addDocument(doc);
      }
      w.commit();
    }
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 16; i++) q.append(word(r)).append('\n');
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("qjoin-words.tsv"), q.toString());
  }

  static Query from(String word) {
    return new BooleanQuery.Builder()
        .add(new TermQuery(new Term("type", "from")), BooleanClause.Occur.FILTER)
        .add(new TermQuery(new Term("body", word)), BooleanClause.Occur.MUST)
        .build();
  }

  interface Join {
    Query build(String word) throws IOException;
  }

  static void cases(String name, IndexSearcher s, List<String> words, Join join)
      throws IOException {
    Fnv f = new Fnv();
    for (String w : words) {
      TopDocs td = s.search(join.build(w), 10);
      for (ScoreDoc sd : td.scoreDocs) {
        f.add(sd.doc);
        f.add(Float.floatToIntBits(sd.score));
      }
    }
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, words.size());
    measure(
        name,
        () -> {
          for (String w : words) sink += s.search(join.build(w), 10).totalHits.value();
          return words.size();
        });
  }

  static void run(Path dir) throws IOException {
    List<String> words = Files.readAllLines(dir.resolve("qjoin-words.tsv"));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] names = {"none", "avg", "max", "total", "min"};
      ScoreMode[] modes = {ScoreMode.None, ScoreMode.Avg, ScoreMode.Max, ScoreMode.Total, ScoreMode.Min};
      for (int m = 0; m < modes.length; m++) {
        ScoreMode mode = modes[m];
        cases("qjoin_terms_" + names[m], s, words, w -> JoinUtil.createJoinQuery("fk", false, "pk", from(w), s, mode));
      }
      cases("qjoin_terms_mv_max", s, words, w -> JoinUtil.createJoinQuery("fkm", true, "pk", from(w), s, ScoreMode.Max));
      cases("qjoin_numeric_none", s, words, w -> JoinUtil.createJoinQuery("nfk", false, "npk", Long.class, from(w), s, ScoreMode.None));
      cases("qjoin_numeric_max", s, words, w -> JoinUtil.createJoinQuery("nfk", false, "npk", Long.class, from(w), s, ScoreMode.Max));
      SortedDocValues[] values = new SortedDocValues[reader.leaves().size()];
      for (LeafReaderContext leaf : reader.leaves()) {
        values[leaf.ord] = DocValues.getSorted(leaf.reader(), "gj");
      }
      OrdinalMap map = OrdinalMap.build(null, values, PackedInts.DEFAULT);
      Query to = new TermQuery(new Term("type", "to"));
      cases("qjoin_gord_none", s, words, w -> JoinUtil.createJoinQuery("gj", from(w), to, s, ScoreMode.None, map));
      cases("qjoin_gord_max", s, words, w -> JoinUtil.createJoinQuery("gj", from(w), to, s, ScoreMode.Max, map));
      cases("qjoin_gord_avg_minmax", s, words, w -> JoinUtil.createJoinQuery("gj", from(w), to, s, ScoreMode.Avg, map, 2, 10));
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("qjoin-words.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
