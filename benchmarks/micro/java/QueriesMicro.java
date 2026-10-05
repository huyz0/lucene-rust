import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.CommonTermsQuery;
import org.apache.lucene.queries.intervals.IntervalQuery;
import org.apache.lucene.queries.intervals.Intervals;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M10 T10.6's benchmark pair, against {@code benchmarks/rust-runner/src/micro_queries.rs}: per
 * word, one top-10 search of each case (interval queries, {@code CommonTermsQuery}) over a 200 000-document, four-segment index of text with
 * positions ({@code body}) and positions with one-byte payloads ({@code pay}).
 *
 * <p>Usage: {@code QueriesMicro build <dir>}, then {@code QueriesMicro run <dir>}.
 */
public final class QueriesMicro {
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

  /** One-byte payloads, {@code position % 4}, none on every fifth position. */
  static final class PayloadAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer t = new StandardTokenizer();
      TokenStream ts =
          new TokenFilter(new LowerCaseFilter(t)) {
            final PayloadAttribute pay = addAttribute(PayloadAttribute.class);
            int pos;

            @Override
            public boolean incrementToken() throws IOException {
              if (!input.incrementToken()) return false;
              int p = pos++;
              pay.setPayload(p % 5 == 4 ? null : new BytesRef(new byte[] {(byte) (p % 4)}));
              return true;
            }

            @Override
            public void reset() throws IOException {
              super.reset();
              pos = 0;
            }
          };
      return new TokenStreamComponents(t, ts);
    }
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1066L);
    IndexWriterConfig cfg =
        new IndexWriterConfig(
            new PerFieldAnalyzerWrapper(
                new StandardAnalyzer(), java.util.Map.of("pay", new PayloadAnalyzer())));
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        int n = 4 + r.nextInt(17);
        StringBuilder b = new StringBuilder();
        for (int k = 0; k < n; k++) {
          if (k > 0) b.append(' ');
          b.append(word(r));
        }
        doc.add(new TextField("body", b.toString(), Field.Store.NO));
        doc.add(new TextField("pay", b.toString(), Field.Store.NO));
        w.addDocument(doc);
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
      w.commit();
    }
    StringBuilder q = new StringBuilder();
    for (int k = 0; k < 16; k++) q.append(word(r)).append('\n');
    Files.writeString(dir.resolve("queries-words.tsv"), q.toString());
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("queries-built"), "1\n");
  }

  interface Case {
    Query query(String word, String next, String third);
  }

  static void cases(String name, List<String> words, IndexSearcher s, Case c) throws IOException {
    int n = words.size();
    Query[] qs = new Query[n];
    for (int i = 0; i < n; i++) {
      qs[i] = c.query(words.get(i), words.get((i + 1) % n), words.get((i + 2) % n));
    }
    Fnv f = new Fnv();
    for (Query q : qs) digest(f, s.search(q, 10));
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, n);
    measure(
        name,
        () -> {
          Fnv g = new Fnv();
          for (Query q : qs) digest(g, s.search(q, 10));
          sink += g.h;
          return n;
        });
  }

  static void digest(Fnv f, TopDocs td) {
    f.add(td.totalHits.value());
    for (ScoreDoc sd : td.scoreDocs) {
      f.add(sd.doc);
      f.add(Float.floatToIntBits(sd.score));
    }
  }

  static void run(Path dir) throws IOException {
    List<String> words = Files.readAllLines(dir.resolve("queries-words.tsv"));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      cases(
          "iv_ordered",
          words,
          s,
          (a, b, c) -> new IntervalQuery("body", Intervals.ordered(Intervals.term(a), Intervals.term(b))));
      cases(
          "iv_phrase",
          words,
          s,
          (a, b, c) -> new IntervalQuery("body", Intervals.phrase(a, b)));
      cases(
          "iv_unordered_maxgaps",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "body",
                  Intervals.maxgaps(2, Intervals.unordered(Intervals.term(a), Intervals.term(b)))));
      cases(
          "iv_or_phrase",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "body",
                  Intervals.phrase(Intervals.or(Intervals.term(a), Intervals.term(b)), Intervals.term(c))));
      cases(
          "iv_containing",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "body",
                  Intervals.containing(
                      Intervals.maxwidth(6, Intervals.ordered(Intervals.term(a), Intervals.term(c))),
                      Intervals.term(b))));
      cases(
          "iv_atleast",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "body",
                  Intervals.maxwidth(
                      5, Intervals.atLeast(2, Intervals.term(a), Intervals.term(b), Intervals.term(c)))));
      cases(
          "iv_prefix",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "body",
                  Intervals.ordered(
                      Intervals.prefix(new BytesRef(a.substring(0, 2))), Intervals.term(b))));
      cases(
          "iv_payload",
          words,
          s,
          (a, b, c) ->
              new IntervalQuery(
                  "pay",
                  Intervals.ordered(
                      Intervals.term(a, p -> p != null && p.bytes[p.offset] >= 2),
                      Intervals.term(b))));
      cases(
          "ct_split",
          words,
          s,
          (a, b, c) -> {
            CommonTermsQuery q =
                new CommonTermsQuery(BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD, 0.5f);
            q.add(new Term("body", a));
            q.add(new Term("body", b));
            q.add(new Term("body", c));
            return q;
          });
      cases(
          "ct_msm",
          words,
          s,
          (a, b, c) -> {
            CommonTermsQuery q =
                new CommonTermsQuery(BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD, 0.45f);
            q.add(new Term("body", a));
            q.add(new Term("body", b));
            q.add(new Term("body", c));
            q.add(new Term("body", "missing"));
            q.setLowFreqMinimumNumberShouldMatch(0.5f);
            return q;
          });
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("queries-built"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
