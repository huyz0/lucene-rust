import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.zip.CRC32;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.QueryBuilder;
import org.apache.lucene.util.Version;

/**
 * Java side of M7's benchmark pairs; the Rust side is {@code
 * benchmarks/rust-runner/src/micro_m7.rs}, with the same bench and case names.
 *
 * <p>The queries are parsed by the fixture generators' own {@code parse} ({@link GenM7Queries},
 * {@link GenSimilaritySearch}, {@code BenchRunner.Sexpr}/{@code BenchRunner.sort}), compiled
 * alongside this class by {@code scripts/bench-micro.sh}, so both engines time the shapes the
 * differential tests verify. Every case prints a {@code #check} digest of its results (FNV-1a over
 * each top hit's document and score bits) that the report compares across the engines before it
 * shows a ratio.
 *
 * <p>Usage: {@code M7Micro <bench> [index-dir]}
 */
public final class M7Micro {

  interface Op {
    long run() throws IOException;
  }

  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;

  /** {@code -Dcase=<name>} runs one case alone, as {@code MICRO_CASE} does on the Rust side. */
  static final String ONLY = System.getProperty("case");

  /** {@code -Dquery=<n>} keeps only the n-th query of every case, as {@code MICRO_QUERY} does. */
  static boolean querySelected(int k) {
    String only = System.getProperty("query");
    return only == null || Integer.parseInt(only) == k;
  }

  static void measure(String name, Op op) throws IOException {
    if (ONLY != null && !ONLY.equals(name)) return;
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
    if (sink == 0xDEADBEEFL) {
      System.err.print("");
    }
    return units;
  }

  /** FNV-1a over 64-bit words, identical to the Rust side's {@code Fnv}. */
  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }

    void hit(int doc, float score) {
      add(doc & 0xffffffffL);
      add(Float.floatToRawIntBits(score) & 0xffffffffL);
    }

    void bytes(byte[] b) {
      for (byte x : b) add(x & 0xffL);
    }
  }

  static void check(String c, Fnv d, int n) {
    if (ONLY != null && !ONLY.equals(c)) return;
    System.out.printf("#check\t%s\t%016x\t%d%n", c, d.h, n);
  }

  public static void main(String[] args) throws Exception {
    String bench = args[0];
    String index = args.length > 1 ? args[1] : null;
    switch (bench) {
      case "m7_fixture" -> m7Fixture();
      case "m7_corpus" -> m7Corpus(index);
      case "similarity" -> similarity(index);
      case "sort_pruning" -> sortPruning(index);
      case "query_builder" -> queryBuilder();
      case "stored_fields_write" -> storedFieldsWrite();
      default -> {
        System.err.println("M7Micro: unknown bench " + bench);
        System.exit(2);
      }
    }
  }

  /** Rust {@code family}: a query's first M7 op, the multi-term ones by rewrite method. */
  static String family(String q) {
    String[] toks = q.split(" ");
    for (String t : toks) {
      String f =
          switch (t) {
            case "S" -> "synonym";
            case "CF" -> "combined_field";
            case "PP" -> "phrase_positions";
            case "NG" -> "ngram_phrase";
            case "MP" -> "multi_phrase";
            case "F" -> "fuzzy";
            case "PF", "W", "RE", "R", "A" -> "mtq_" + toks[toks.length - 1].split(":")[0];
            case "BT" -> "blended";
            case "IA" -> "indri";
            case "LO" -> "log_odds";
            case "BY" -> "bayesian";
            case "NR" -> "dv_range";
            case "ISR" -> "index_sort_range";
            case "P2R", "I1R", "I1S", "LS", "LR", "IPR", "IPS" -> "points";
            case "IODV" -> "index_or_dv";
            case "VSF", "VSFF", "VSB" -> "vector_similarity";
            case "KF", "KFF", "KB", "KFS" -> "knn";
            case "PKF" -> "knn_patience";
            case "SKF" -> "knn_seeded";
            default -> null;
          };
      if (f != null) return f;
    }
    return "other";
  }

  static Query parseM7(String q) {
    GenM7Queries.pos = 0;
    String[] tok = q.split(" ");
    Query query = GenM7Queries.parse(tok);
    if (GenM7Queries.pos != tok.length) throw new AssertionError("trailing tokens in " + q);
    return query;
  }

  record Search(IndexSearcher searcher, Query query, int top) {}

  static void runM7Cases(String prefix, Path dir, List<String[]> lines) throws IOException {
    try (FSDirectory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      Map<String, IndexSearcher> searchers = new TreeMap<>();
      Map<String, List<Search>> groups = new TreeMap<>();
      for (String[] l : lines) {
        IndexSearcher s =
            searchers.computeIfAbsent(
                l[0],
                k -> {
                  IndexSearcher is = new IndexSearcher(reader);
                  is.setSimilarity(GenM7Queries.sims().get(k));
                  return is;
                });
        groups
            .computeIfAbsent(prefix + "_" + family(l[1]), k -> new ArrayList<>())
            .add(new Search(s, parseM7(l[1]), Integer.parseInt(l[2])));
      }
      for (Map.Entry<String, List<Search>> e : groups.entrySet()) {
        List<Search> searches = new ArrayList<>();
        for (int k = 0; k < e.getValue().size(); k++) {
          if (querySelected(k)) searches.add(e.getValue().get(k));
        }
        Fnv digest = new Fnv();
        for (Search s : searches) {
          for (ScoreDoc sd : s.searcher.search(s.query, s.top).scoreDocs) digest.hit(sd.doc, sd.score);
        }
        check(e.getKey(), digest, searches.size());
        measure(
            e.getKey(),
            () -> {
              Fnv f = new Fnv();
              for (Search s : searches) {
                for (ScoreDoc sd : s.searcher.search(s.query, s.top).scoreDocs) {
                  f.hit(sd.doc, sd.score);
                }
              }
              sink += f.h;
              return searches.size();
            });
      }
    }
  }

  static List<String[]> fixtureLines(String name, int top) throws IOException {
    List<String[]> out = new ArrayList<>();
    for (String line : Files.readAllLines(Paths.get("fixtures/data", name, "searches.tsv"))) {
      String[] f = line.split("\t");
      out.add(new String[] {f[0], f[1], Integer.toString(top)});
    }
    return out;
  }

  static void m7Fixture() throws IOException {
    float[] qvec = GenM7Queries.QVEC;
    runM7Cases("fx", Paths.get("fixtures/data/m7_queries_index"), fixtureLines("m7_queries_index", 20));
    GenM7Queries.QVEC = new float[] {0.1f, -0.3f, 0.25f, 0.6f, -0.05f, 0.4f, -0.2f, 0.15f};
    runM7Cases("fxknn", Paths.get("fixtures/data/m7_knn_index"), fixtureLines("m7_knn_index", 50));
    GenM7Queries.QVEC = qvec;
    runM7Cases("fxdv", Paths.get("fixtures/data/m7_dv_index"), fixtureLines("m7_dv_index", 50));
  }

  /** Rust {@code CORPUS_QUERIES}, character for character. */
  static final String[] CORPUS_QUERIES = {
    "S body 2 t1 1 t2 1",
    "S body 3 tz 1 t2s 0.5 t1z4 1",
    "B 1 1 0 T body t0 S body 2 t3 1 t11 0.5",
    "CF t1 2 body 1 title 1",
    "CF tz 2 body 1 title 3",
    "B 0 2 0 CF t3 2 body 1 title 2 T body t4",
    "PP body 0 2 t0 0 t1 2",
    "PP body 1 3 t0 0 t1 1 t2 3",
    "NG 2 body 0 3 t0 t1 t2",
    "MP body 0 2 2 t0 t1 0 1 t2 1",
    "MP body 1 2 1 t1 0 2 t0 t3 1",
    "F body t123 1 0",
    "F title t1z4 2 1",
    "PF body t4a sb",
    "PF body t4a csbool",
    "PF body t1 tts:50",
    "PF body t1 ttb:50",
    "PF body t1 ttbf:50",
    "PF body t1 cs",
    "PF body t1 csb",
    "W body t?3 csb",
    "RE body t[0-2]. tts:20",
    "R body t10 t20 1 0 csb",
    "R body t4a t4b 1 0 sb",
    "A body t3.* csb",
    "A body (t1|t2)5? sb",
    "PF keyword t1z dv",
    "R keyword t10 t11 1 1 dv",
    "BT bool 3 body t1 1 body t2 1 title t1 1",
    "BT dismax:0.1 2 body tz 1 title tz 1",
    "IA 2 T body t1 T body t2",
    "IA 2 T body tz T body t2s",
    "LO 0.5 - - 2 T body t1 T title t1",
    "LO 1 0.7,0.3 - 2 T body tz T title t2",
    "BY 1.5 1 0 T body t2",
    "BY 0.5 2 0.1 B 0 2 0 T body t1 T title t1",
    "NR num 1000 2000",
    "B 1 0 0 T body t1 F2 NR num 0 100000",
    "B 1 0 0 T body tz F2 NR num 0 500000",
    "LR num 1000 2000",
    "LS num 5 7 99 1000 123456 777777",
    "B 1 0 0 T body t2 F2 LR num 0 300000",
    "IODV LR num 0 100000 NR num 0 100000",
    "B 1 0 0 T body t2s F2 IODV LR num 0 500000 NR num 0 500000",
    "B 1 0 0 T body t1z4 F2 IODV LR num 0 900000 NR num 0 900000",
    "B 0 2 0 IA 2 T body t1 T body t2 T title t3",
    "B 0 2 0 BY 1.5 1 0 T body t2 T body t5",
  };

  static void m7Corpus(String index) throws IOException {
    List<String[]> lines = new ArrayList<>();
    for (String q : CORPUS_QUERIES) lines.add(new String[] {"bm25", q, "10"});
    runM7Cases("c", Paths.get(index), lines);
  }

  static final String[] SIM_QUERIES = {
    "T body t1",
    "T body tz",
    "B 0 2 0 T body t1 T body t2",
    "B 2 0 0 T body t0 T body tz",
    "B 0 3 0 T body tz T body t2s T title t1",
    "P body 0 2 t0 t1",
  };

  static final String[] SPAN_QUERIES = {
    "N 0 1 2 S body t0 S body t1",
    "N 3 0 2 S body t2 S body t0",
    "O 2 S body tz S body t2s",
    "B 0 2 0 N 1 0 2 S body t4 S body t0 T body t2",
  };

  static Query parseSim(String q) {
    GenSimilaritySearch.pos = 0;
    String[] tok = q.split(" ");
    Query query = GenSimilaritySearch.parse(tok);
    if (GenSimilaritySearch.pos != tok.length) throw new AssertionError("trailing tokens in " + q);
    return query;
  }

  static void similarity(String index) throws IOException {
    try (FSDirectory d = FSDirectory.open(Paths.get(index));
        DirectoryReader reader = DirectoryReader.open(d)) {
      Map<String, Similarity> sims = GenSimilaritySearch.sims();
      List<Object[]> cases = new ArrayList<>();
      for (Map.Entry<String, Similarity> e : sims.entrySet()) {
        cases.add(new Object[] {"sim_" + e.getKey(), e.getValue(), SIM_QUERIES});
      }
      String[] spanCases = {"span_near_ordered", "span_near_unordered", "span_or", "span_in_boolean"};
      for (int k = 0; k < spanCases.length; k++) {
        cases.add(new Object[] {spanCases[k], new BM25Similarity(), new String[] {SPAN_QUERIES[k]}});
      }
      cases.add(new Object[] {"span_classic", sims.get("classic"), SPAN_QUERIES});
      for (Object[] c : cases) {
        String name = (String) c[0];
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity((Similarity) c[1]);
        List<Query> queries = new ArrayList<>();
        String[] texts = (String[]) c[2];
        for (int k = 0; k < texts.length; k++) {
          if (querySelected(k)) queries.add(parseSim(texts[k]));
        }
        Fnv digest = new Fnv();
        for (Query q : queries) {
          for (ScoreDoc sd : searcher.search(q, 10).scoreDocs) digest.hit(sd.doc, sd.score);
        }
        check(name, digest, queries.size());
        measure(
            name,
            () -> {
              Fnv f = new Fnv();
              for (Query q : queries) {
                for (ScoreDoc sd : searcher.search(q, 10).scoreDocs) f.hit(sd.doc, sd.score);
              }
              sink += f.h;
              return queries.size();
            });
      }
    }
  }

  static void sortPruning(String index) throws IOException {
    try (FSDirectory d = FSDirectory.open(Paths.get(index));
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setSimilarity(new BM25Similarity());
      for (String line : Files.readAllLines(Paths.get("benchmarks/queries.tsv"))) {
        if (line.startsWith("#")) continue;
        String[] f = line.split("\t");
        if (f.length < 5 || !f[1].equals("sorted")) continue;
        String name = "sort_" + f[0];
        Query query = BenchRunner.Sexpr.parse(f[2], new BenchRunner.Sexpr(f[3]));
        Sort sort = BenchRunner.sort(f[4]);
        Op op =
            () -> {
              TopDocs td = searcher.search(query, new TopFieldCollectorManager(sort, 10, null, 1000));
              Fnv fn = new Fnv();
              for (ScoreDoc sd : td.scoreDocs) fn.add(sd.doc & 0xffffffffL);
              sink += fn.h;
              return 1;
            };
        TopDocs td = searcher.search(query, new TopFieldCollectorManager(sort, 10, null, 1000));
        Fnv digest = new Fnv();
        for (ScoreDoc sd : td.scoreDocs) digest.add(sd.doc & 0xffffffffL);
        check(name, digest, 1);
        measure(name, op);
      }
    }
  }

  /** Rust {@code query_texts}, character for character. */
  static List<String> queryTexts() {
    SweepMicroRng r = new SweepMicroRng(0x5EED0F00D15EA5E5L);
    List<String> out = new ArrayList<>();
    for (int q = 0; q < 2000; q++) {
      int words = 1 + (int) Long.remainderUnsigned(r.next(), 8);
      StringBuilder s = new StringBuilder();
      for (int wi = 0; wi < words; wi++) {
        if (wi > 0) s.append(' ');
        long x = r.next();
        String word = "t" + Long.toString(Long.remainderUnsigned(x >>> 8, 5000), 36);
        if ((x & 7) == 0) word = word.toUpperCase();
        s.append(word);
        if ((x & 15) == 1) s.append(',');
      }
      out.add(s.toString());
    }
    return out;
  }

  /** xorshift64, identical to {@code SweepMicro.Rng} and the Rust {@code Rng}. */
  static final class SweepMicroRng {
    long s;

    SweepMicroRng(long seed) {
      s = seed;
    }

    long next() {
      s ^= s << 13;
      s ^= s >>> 7;
      s ^= s << 17;
      return s;
    }
  }

  static void queryBuilder() throws IOException {
    QueryBuilder qb = new QueryBuilder(new StandardAnalyzer());
    List<String> texts = queryTexts();
    Map<String, Function<String, Query>> cases = new java.util.LinkedHashMap<>();
    cases.put("qb_should", t -> qb.createBooleanQuery("body", t));
    cases.put("qb_must", t -> qb.createBooleanQuery("body", t, BooleanClause.Occur.MUST));
    cases.put("qb_phrase", t -> qb.createPhraseQuery("body", t));
    cases.put("qb_phrase_slop", t -> qb.createPhraseQuery("body", t, 2));
    cases.put("qb_min_should_match", t -> qb.createMinShouldMatchQuery("body", t, 0.5f));
    for (Map.Entry<String, Function<String, Query>> e : cases.entrySet()) {
      Function<String, Query> build = e.getValue();
      Fnv digest = new Fnv();
      for (String t : texts) {
        Query q = build.apply(t);
        digest.bytes(String.valueOf(q == null ? null : q.toString("body")).getBytes(StandardCharsets.UTF_8));
      }
      check(e.getKey(), digest, texts.size());
      measure(
          e.getKey(),
          () -> {
            for (String t : texts) {
              Query q = build.apply(t);
              if (q != null) sink += q.hashCode();
            }
            return texts.size();
          });
    }
  }

  static FieldInfo storedField(String name, int number) {
    return new FieldInfo(
        name,
        number,
        false,
        false,
        false,
        IndexOptions.NONE,
        DocValuesType.NONE,
        DocValuesSkipIndexType.NONE,
        -1,
        Collections.emptyMap(),
        0,
        0,
        0,
        0,
        VectorEncoding.FLOAT32,
        VectorSimilarityFunction.EUCLIDEAN,
        false,
        false);
  }

  /** {@link GenStoredFieldsDeflate}'s 500 documents, fields numbered text 0 .. big 4. */
  static List<Document> storedDocs() {
    GenStoredFieldsDeflate.seed = 20260930L;
    List<Document> docs = new ArrayList<>();
    for (int i = 0; i < 500; i++) docs.add(GenStoredFieldsDeflate.doc(i));
    return docs;
  }

  /** Writes {@code docs} into a fresh in-memory directory; returns it. */
  static ByteBuffersDirectory writeStored(
      Lucene90StoredFieldsFormat format, List<Document> docs, Map<String, FieldInfo> infos, byte[] id)
      throws IOException {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    SegmentInfo si =
        new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            docs.size(),
            false,
            false,
            new Lucene104Codec(),
            Collections.emptyMap(),
            id,
            Collections.emptyMap(),
            null);
    try (StoredFieldsWriter w = format.fieldsWriter(dir, si, IOContext.DEFAULT)) {
      for (Document doc : docs) {
        w.startDocument();
        for (IndexableField f : doc.getFields()) {
          FieldInfo fi = infos.get(f.name());
          if (f.numericValue() != null) {
            w.writeField(fi, f.numericValue().intValue());
          } else if (f.binaryValue() != null) {
            w.writeField(fi, f.binaryValue());
          } else {
            w.writeField(fi, f.stringValue());
          }
        }
        w.finishDocument();
      }
      w.finish(docs.size());
    }
    return dir;
  }

  static long crc(ByteBuffersDirectory dir, String ext) throws IOException {
    for (String f : dir.listAll()) {
      if (f.endsWith(ext)) {
        try (IndexInput in = dir.openInput(f, IOContext.READONCE)) {
          byte[] b = new byte[(int) in.length()];
          in.readBytes(b, 0, b.length);
          CRC32 c = new CRC32();
          c.update(b);
          return c.getValue();
        }
      }
    }
    throw new AssertionError("no " + ext);
  }

  static void storedFieldsWrite() throws IOException {
    List<Document> docs = storedDocs();
    Map<String, FieldInfo> infos = new TreeMap<>();
    String[] names = {"text", "blob", "run", "num", "big"};
    for (int k = 0; k < names.length; k++) infos.put(names[k], storedField(names[k], k));
    byte[] id = new byte[16];
    java.util.Arrays.fill(id, (byte) 7);
    Object[][] cases = {
      {"sf_write_best_speed", new Lucene90StoredFieldsFormat(Lucene90StoredFieldsFormat.Mode.BEST_SPEED)},
      {
        "sf_write_best_compression",
        new Lucene90StoredFieldsFormat(Lucene90StoredFieldsFormat.Mode.BEST_COMPRESSION)
      },
    };
    for (Object[] c : cases) {
      String name = (String) c[0];
      Lucene90StoredFieldsFormat format = (Lucene90StoredFieldsFormat) c[1];
      ByteBuffersDirectory dir = writeStored(format, docs, infos, id);
      Fnv digest = new Fnv();
      digest.add(crc(dir, ".fdt"));
      digest.add(crc(dir, ".fdx"));
      check(name, digest, docs.size());
      measure(
          name,
          () -> {
            sink += writeStored(format, docs, infos, id).listAll().length;
            return docs.size();
          });
    }
  }
}
