import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatDocValuesField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.function.FunctionMatchQuery;
import org.apache.lucene.queries.function.FunctionQuery;
import org.apache.lucene.queries.function.FunctionRangeQuery;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.queries.function.ValueSource;
import org.apache.lucene.queries.function.valuesource.ConstValueSource;
import org.apache.lucene.queries.function.valuesource.FloatFieldSource;
import org.apache.lucene.queries.function.valuesource.IDFValueSource;
import org.apache.lucene.queries.function.valuesource.IntFieldSource;
import org.apache.lucene.queries.function.valuesource.JoinDocFreqValueSource;
import org.apache.lucene.queries.function.valuesource.LinearFloatFunction;
import org.apache.lucene.queries.function.valuesource.LongFieldSource;
import org.apache.lucene.queries.function.valuesource.ProductFloatFunction;
import org.apache.lucene.queries.function.valuesource.ReciprocalFloatFunction;
import org.apache.lucene.queries.function.valuesource.SumFloatFunction;
import org.apache.lucene.queries.function.valuesource.TFValueSource;
import org.apache.lucene.queries.function.valuesource.TermFreqValueSource;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the function-query benchmark pair (M10 T10.5); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_function.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 200 000 documents in four segments, each with a text {@code body}, a
 * {@code NUMERIC} int {@code i} (one in ten missing), a {@code NUMERIC} long {@code n} and a float
 * {@code f} -- and the word list both engines search ({@code function-words.tsv}). Each case runs,
 * per word, one top-10 search: {@code FunctionScoreQuery} over {@code body:word} with a field
 * source and {@code boostByValue} with a composite source, {@code FunctionQuery} over a field and
 * over a composite arithmetic source (every document), {@code FunctionRangeQuery} alone and as a
 * filter, {@code FunctionQuery(termfreq)}, {@code tf * idf} under {@code ClassicSimilarity}, and
 * {@code FunctionMatchQuery}. Each prints a {@code #check} digest of its hits the report compares
 * first.
 */
public final class FunctionMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;
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

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1005L);
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    // `k`, for `joindf(k,body)`: a quarter of the documents name a body word,
    // the rest one of 20 000 keys no body holds. Its own generator, so the
    // other fields are what they were before `k` was added.
    Random rk = new Random(0x10_5_2026_1006L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        int n = 2 + r.nextInt(8);
        StringBuilder b = new StringBuilder();
        for (int k = 0; k < n; k++) {
          if (k > 0) b.append(' ');
          b.append(word(r));
        }
        doc.add(new TextField("body", b.toString(), Field.Store.NO));
        if (r.nextInt(10) != 0) doc.add(new NumericDocValuesField("i", r.nextInt(1000)));
        doc.add(new NumericDocValuesField("n", r.nextInt(100_000)));
        doc.add(new FloatDocValuesField("f", r.nextFloat() * 100f));
        String key = rk.nextInt(4) == 0 ? WORDS[rk.nextInt(WORDS.length)] : "k" + rk.nextInt(20_000);
        doc.add(new SortedDocValuesField("k", new BytesRef(key)));
        w.addDocument(doc);
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
      w.commit();
    }
    StringBuilder q = new StringBuilder();
    for (int k = 0; k < 16; k++) q.append(word(r)).append('\n');
    Files.writeString(dir.resolve("function-words.tsv"), q.toString());
    // Last: its presence is what marks the corpus built (with `k`).
    Files.writeString(dir.resolve("function-k"), "k\n");
  }

  interface Case {
    Query query(String word);
  }

  static void cases(String name, List<String> words, IndexSearcher s, Case c) throws IOException {
    Fnv f = new Fnv();
    for (String w : words) digest(f, s.search(c.query(w), 10));
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, words.size());
    measure(
        name,
        () -> {
          Fnv g = new Fnv();
          for (String w : words) digest(g, s.search(c.query(w), 10));
          sink += g.h;
          return words.size();
        });
  }

  static void digest(Fnv f, TopDocs td) {
    f.add(td.totalHits.value());
    for (ScoreDoc sd : td.scoreDocs) {
      f.add(sd.doc);
      f.add(Float.floatToIntBits(sd.score));
    }
  }

  static Query q(String word) {
    return new TermQuery(new Term("body", word));
  }

  /** {@code sum(product(int(i),0.5), linear(float(f),2,1), recip(long(n),0.001,10,1))}. */
  static ValueSource composite() {
    return new SumFloatFunction(
        new ValueSource[] {
          new ProductFloatFunction(new ValueSource[] {new IntFieldSource("i"), new ConstValueSource(0.5f)}),
          new LinearFloatFunction(new FloatFieldSource("f"), 2f, 1f),
          new ReciprocalFloatFunction(new LongFieldSource("n"), 0.001f, 10f, 1f)
        });
  }

  static void run(Path dir) throws IOException {
    List<String> words = Files.readAllLines(dir.resolve("function-words.tsv"));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      IndexSearcher classic = new IndexSearcher(reader);
      classic.setQueryCache(null);
      classic.setSimilarity(new ClassicSimilarity());
      cases("fn_score_field", words, s, w -> new FunctionScoreQuery(q(w), DoubleValuesSource.fromFloatField("f")));
      cases("fn_boost_composite", words, s, w -> FunctionScoreQuery.boostByValue(q(w), composite().asDoubleValuesSource()));
      cases("fn_query_field", words, s, w -> new FunctionQuery(new FloatFieldSource("f")));
      cases("fn_query_composite", words, s, w -> new FunctionQuery(composite()));
      cases(
          "fn_range",
          words,
          s,
          w -> new FunctionRangeQuery(new IntFieldSource("i"), "100", "180", true, false));
      cases(
          "fn_range_filter",
          words,
          s,
          w ->
              new BooleanQuery.Builder()
                  .add(q(w), BooleanClause.Occur.MUST)
                  .add(
                      new FunctionRangeQuery(new IntFieldSource("i"), "100", "500", true, true),
                      BooleanClause.Occur.FILTER)
                  .build());
      cases(
          "fn_termfreq",
          words,
          s,
          w -> new FunctionQuery(new TermFreqValueSource("body", w, "body", new BytesRef(w))));
      cases(
          "fn_tf_idf",
          words,
          classic,
          w ->
              new FunctionScoreQuery(
                  q(w),
                  new ProductFloatFunction(
                          new ValueSource[] {
                            new TFValueSource("body", w, "body", new BytesRef(w)),
                            new IDFValueSource("body", w, "body", new BytesRef(w))
                          })
                      .asDoubleValuesSource()));
      // `joindf`: Java seeks the top-level terms per visited document.
      cases(
          "fn_joindf",
          words,
          s,
          w -> new FunctionScoreQuery(q(w), new JoinDocFreqValueSource("k", "body").asDoubleValuesSource()));
      cases(
          "fn_match",
          words,
          s,
          w ->
              new BooleanQuery.Builder()
                  .add(q(w), BooleanClause.Occur.MUST)
                  .add(
                      new FunctionMatchQuery(DoubleValuesSource.fromIntField("i"), v -> v > 500),
                      BooleanClause.Occur.FILTER)
                  .build());
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("function-k"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
