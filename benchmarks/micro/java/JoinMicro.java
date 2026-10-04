import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.join.ParentChildrenBlockJoinQuery;
import org.apache.lucene.search.join.ParentsChildrenBlockJoinQuery;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinSortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Java side of the block-join benchmark pair (M10 T10.2); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_join.rs}, with the same case names. {@code build <dir>} writes
 * the corpus once -- 60 000 blocks of zero to ten children (a body of words, a {@code price}) and
 * a parent (a body and a {@code rank}), force-merged to one segment -- and the word list ({@code
 * join-words.tsv}) both engines draw their queries from. Cases: {@code ToParentBlockJoinQuery} in
 * each score mode, alone and inside a boolean, {@code ToChildBlockJoinQuery}, {@code
 * ParentChildrenBlockJoinQuery} over 200 parents, {@code ParentsChildrenBlockJoinQuery}, a {@code
 * ToParentBlockJoinSortField} sort and {@code DiversifyingChildrenFloatKnnVectorQuery} over the
 * children's eight-dimensional vectors, unfiltered and with a child filter (query vectors in {@code
 * join-vectors.tsv}); each top 10 through {@code IndexSearcher.search}. Each
 * prints a {@code #check} digest of its hits (doc and score bits, or sort value) the report
 * compares first.
 */
public final class JoinMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int BLOCKS = 60_000;
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

  static void check(String c, Fnv d, long n) {
    System.out.printf("#check\t%s\t%016x\t%d%n", c, d.h, n);
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

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_2026_1003L);
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    cfg.setParentField("_parent");
    // Plain files: the Rust side opens the vector files by name.
    cfg.setUseCompoundFile(false);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int b = 0; b < BLOCKS; b++) {
        List<Document> docs = new ArrayList<>();
        int children = r.nextInt(11);
        for (int c = 0; c < children; c++) {
          Document doc = new Document();
          doc.add(new StringField("type", "child", Field.Store.NO));
          doc.add(new TextField("body", body(r), Field.Store.NO));
          doc.add(new SortedNumericDocValuesField("price", r.nextInt(100_000)));
          doc.add(new KnnFloatVectorField("vec", vector(r), VectorSimilarityFunction.EUCLIDEAN));
          docs.add(doc);
        }
        Document p = new Document();
        p.add(new StringField("type", "parent", Field.Store.NO));
        p.add(new TextField("body", body(r), Field.Store.NO));
        p.add(new NumericDocValuesField("rank", r.nextInt(1000)));
        docs.add(p);
        w.addDocuments(docs);
      }
      w.forceMerge(1);
    }
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 24; i++) {
      q.append(word(r)).append('\t').append(word(r)).append('\n');
    }
    StringBuilder v = new StringBuilder();
    for (int i = 0; i < 24; i++) {
      float[] x = vector(r);
      for (int k = 0; k < x.length; k++) {
        if (k > 0) v.append('\t');
        v.append(Integer.toHexString(Float.floatToIntBits(x[k])));
      }
      v.append('\n');
    }
    Files.writeString(dir.resolve("join-vectors.tsv"), v.toString());
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("join-words.tsv"), q.toString());
  }

  static float[] vector(Random r) {
    float[] v = new float[8];
    for (int k = 0; k < v.length; k++) v[k] = r.nextInt(2001) / 1000f - 1;
    return v;
  }

  static Query level(String type, String word) {
    return new BooleanQuery.Builder()
        .add(new TermQuery(new Term("type", type)), BooleanClause.Occur.FILTER)
        .add(new TermQuery(new Term("body", word)), BooleanClause.Occur.MUST)
        .build();
  }

  static void hits(Fnv f, TopDocs td) {
    for (ScoreDoc sd : td.scoreDocs) {
      f.add(sd.doc);
      f.add(Float.floatToIntBits(sd.score));
    }
  }

  static void cases(String name, IndexSearcher s, List<Query> qs) throws IOException {
    Fnv f = new Fnv();
    for (Query q : qs) hits(f, s.search(q, 10));
    check(name, f, qs.size());
    measure(name, () -> {
      for (Query q : qs) sink += s.search(q, 10).totalHits.value();
      return qs.size();
    });
  }

  static void run(Path dir) throws IOException {
    List<String[]> words = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("join-words.tsv"))) {
      words.add(line.split("\t"));
    }
    BitSetProducer parents = new QueryBitSetProducer(new TermQuery(new Term("type", "parent")));
    BitSetProducer children = new QueryBitSetProducer(new TermQuery(new Term("type", "child")));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] modes = {"none", "avg", "max", "total", "min"};
      ScoreMode[] sm = {ScoreMode.None, ScoreMode.Avg, ScoreMode.Max, ScoreMode.Total, ScoreMode.Min};
      for (int m = 0; m < modes.length; m++) {
        List<Query> qs = new ArrayList<>();
        for (String[] w : words) qs.add(new ToParentBlockJoinQuery(level("child", w[0]), parents, sm[m]));
        cases("join_to_parent_" + modes[m], s, qs);
      }
      List<Query> nested = new ArrayList<>();
      for (String[] w : words) {
        nested.add(
            new BooleanQuery.Builder()
                .add(new ToParentBlockJoinQuery(level("child", w[0]), parents, ScoreMode.Avg), BooleanClause.Occur.MUST)
                .add(new TermQuery(new Term("body", w[1])), BooleanClause.Occur.SHOULD)
                .build());
      }
      cases("join_to_parent_in_bool", s, nested);
      List<Query> toChild = new ArrayList<>();
      for (String[] w : words) toChild.add(new ToChildBlockJoinQuery(level("parent", w[0]), parents));
      cases("join_to_child", s, toChild);
      List<Query> parentsChildren = new ArrayList<>();
      for (String[] w : words) {
        parentsChildren.add(
            new ParentsChildrenBlockJoinQuery(parents, level("parent", w[0]), level("child", w[1]), 3));
      }
      cases("join_parents_children", s, parentsChildren);
      // The children of 200 parents, every one of them.
      List<Query> parentChildren = new ArrayList<>();
      LeafReaderContext leaf = reader.leaves().get(0);
      var bits = parents.getBitSet(leaf);
      int parent = bits.nextSetBit(0);
      for (int i = 0; i < 200; i++) {
        parentChildren.add(
            new ParentChildrenBlockJoinQuery(parents, level("child", words.get(i % words.size())[0]), parent));
        for (int k = 0; k < 250 && parent + 1 < reader.maxDoc(); k++) {
          parent = bits.nextSetBit(parent + 1);
        }
      }
      cases("join_parent_children", s, parentChildren);
      // Parents by their children's lowest price.
      List<Query> sortQs = new ArrayList<>();
      for (String[] w : words.subList(0, 8)) sortQs.add(level("parent", w[0]));
      Sort sort =
          new Sort(new ToParentBlockJoinSortField("price", SortField.Type.LONG, false, parents, children));
      Fnv f = new Fnv();
      for (Query q : sortQs) {
        for (ScoreDoc sd : s.search(q, 10, sort).scoreDocs) {
          f.add(sd.doc);
          f.add((Long) ((FieldDoc) sd).fields[0]);
        }
      }
      check("join_sort", f, sortQs.size());
      measure("join_sort", () -> {
        for (Query q : sortQs) sink += s.search(q, 10, sort).scoreDocs.length;
        return sortQs.size();
      });
      // The nearest child of each of the ten nearest parents, unfiltered and filtered.
      List<float[]> vectors = new ArrayList<>();
      for (String line : Files.readAllLines(dir.resolve("join-vectors.tsv"))) {
        String[] parts = line.split("\t");
        float[] v = new float[parts.length];
        for (int k = 0; k < v.length; k++) v[k] = Float.intBitsToFloat(Integer.parseUnsignedInt(parts[k], 16));
        vectors.add(v);
      }
      List<Query> knn = new ArrayList<>();
      List<Query> knnFiltered = new ArrayList<>();
      for (int i = 0; i < vectors.size(); i++) {
        knn.add(new DiversifyingChildrenFloatKnnVectorQuery("vec", vectors.get(i), null, 10, parents));
        knnFiltered.add(
            new DiversifyingChildrenFloatKnnVectorQuery(
                "vec", vectors.get(i), level("child", words.get(i)[0]), 10, parents));
      }
      cases("join_knn_diversify", s, knn);
      cases("join_knn_diversify_filtered", s, knnFiltered);
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[args.length - 1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("join-words.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
