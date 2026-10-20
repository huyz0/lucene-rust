import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HexFormat;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.bitvectors.HnswBitVectorsFormat;
import org.apache.lucene.codecs.blocktreeords.BlockTreeOrdsPostingsFormat;
import org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.codecs.memory.DirectPostingsFormat;
import org.apache.lucene.codecs.memory.FSTPostingsFormat;
import org.apache.lucene.codecs.uniformsplit.UniformSplitPostingsFormat;
import org.apache.lucene.codecs.uniformsplit.sharedterms.STUniformSplitPostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the lucene-codecs benchmark pair (M14.g); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_codecs.rs}, with the same case names. {@code build <dir>}
 * writes each codec's index once, in its own subdirectory; {@code run <dir>} times every case.
 *
 * <p>{@code bloom/}: 200 000 documents, one segment, whose unique {@code id} is routed to {@code
 * BloomFilteringPostingsFormat} over the default {@code Lucene104PostingsFormat}, and {@code
 * targets.tsv}: 200 000 lookups, a quarter of them present -- the primary-key lookup the format
 * exists for, where most misses are answered by the filter without touching the terms index.
 *
 * <p>{@code memory/}: the same 200 000 ids twice, {@code fid} routed to {@code FSTPostingsFormat}
 * and {@code did} to {@code DirectPostingsFormat}, and the same kind of {@code targets.tsv}:
 * {@code cx_fst_seek} and {@code cx_direct_seek} are {@code seekExact} on each.
 *
 * <p>{@code bitvectors/}: 100 000 documents, each with a 32-byte (256-bit) vector {@code bv} routed
 * to {@code HnswBitVectorsFormat}, and {@code queries.tsv}: 2 000 random queries; {@code
 * cx_bitvec_search} is a top-10 {@code searchNearestVectors} per query.
 *
 * <p>{@code ords/}: the same 200 000 ids, {@code oid} routed to {@code BlockTreeOrdsPostingsFormat},
 * the same kind of {@code targets.tsv} (half present) and {@code ords.tsv}, one random ord per
 * target: {@code cx_ords_seek} is, per target, {@code seekExact(term)} (and its {@code ord()})
 * then {@code seekExact(ord)} (and its term).
 *
 * <p>{@code usplit/}: the same 200 000 ids, {@code uid} routed to {@code
 * UniformSplitPostingsFormat} (default 32/3 block lines, no encoder), and the same kind of {@code
 * targets.tsv} (half present): {@code cx_usplit_seek} is {@code seekExact} on each.
 *
 * <p>{@code stsplit/}: the same 200 000 ids in {@code sa}, every third one also in {@code sb},
 * both routed to one {@code STUniformSplitPostingsFormat} (default 32/3 block lines, one shared
 * dictionary), and the same kind of {@code targets.tsv} (half present in {@code sa}): {@code
 * cx_stsplit_seek} is {@code seekExact} on {@code sb} for each, most hits landing on a line only
 * {@code sa} has.
 */
public final class CodecsMicro {
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

  static boolean selected(String name) {
    String only = System.getenv("MICRO_CASE");
    return only == null || only.equals(name);
  }

  /** Document {@code i}'s id: a scrambled base-36 number. */
  static String idOf(long i) {
    return Long.toString((i * 0x9E3779B97F4A7C15L) >>> 24, 36);
  }

  static IndexWriterConfig config(String field, PostingsFormat format) {
    return config(Map.of(field, format));
  }

  static IndexWriterConfig config(Map<String, PostingsFormat> formats) {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setUseCompoundFile(false);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public PostingsFormat getPostingsFormatForField(String f) {
            PostingsFormat format = formats.get(f);
            return format != null ? format : super.getPostingsFormatForField(f);
          }
        });
    return cfg;
  }

  static void buildBloom(Path dir) throws IOException {
    Random r = new Random(0x10_5_2026_1020L);
    PostingsFormat bloom = new BloomFilteringPostingsFormat(new Lucene104PostingsFormat());
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("id", bloom))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // A quarter present (ids below DOCS), the rest absent.
      targets.add(idOf(r.nextInt(4) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildMemory(Path dir) throws IOException {
    Random r = new Random(0x14_7_2026_1020L);
    Map<String, PostingsFormat> formats =
        Map.of("fid", new FSTPostingsFormat(), "did", new DirectPostingsFormat());
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config(formats))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("fid", idOf(i), Field.Store.NO));
        doc.add(new StringField("did", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildOrds(Path dir) throws IOException {
    Random r = new Random(0x0d_5_2026_1020L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("oid", new BlockTreeOrdsPostingsFormat()))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("oid", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    List<String> ords = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent; every ord below the term count.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
      ords.add(Integer.toString(r.nextInt(DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.write(dir.resolve("ords.tsv"), ords);
    Files.writeString(dir.resolve("built"), "");
  }

  static void buildBitVectors(Path dir) throws IOException {
    Random r = new Random(0x14_8_2026_1020L);
    KnnVectorsFormat format = new HnswBitVectorsFormat();
    IndexWriterConfig cfg = config(Map.of());
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String f) {
            return format;
          }
        });
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < 100_000; i++) {
        byte[] v = new byte[32];
        r.nextBytes(v);
        Document doc = new Document();
        doc.add(new KnnByteVectorField("bv", v, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> queries = new ArrayList<>();
    for (int k = 0; k < 2_000; k++) {
      byte[] q = new byte[32];
      r.nextBytes(q);
      queries.add(HexFormat.of().formatHex(q));
    }
    Files.write(dir.resolve("queries.tsv"), queries);
    Files.writeString(dir.resolve("built"), "");
  }

  /** {@code cx_bitvec_search}: a top-10 graph search per query. */
  static void runBitVectors(Path dir) throws IOException {
    if (!selected("cx_bitvec_search")) return;
    List<byte[]> queries = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("queries.tsv"))) {
      queries.add(HexFormat.of().parseHex(line));
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      LeafReader leaf = reader.leaves().get(0).reader();
      AcceptDocs all = AcceptDocs.fromLiveDocs(null, leaf.maxDoc());
      // The per-field vectors reader, as the Rust side calls its reader.
      KnnVectorsReader vectors = ((CodecReader) leaf).getVectorReader();
      Op op =
          () -> {
            Fnv g = new Fnv();
            for (byte[] q : queries) {
              TopKnnCollector c = new TopKnnCollector(10, Integer.MAX_VALUE);
        vectors.search("bv", q, c, all);
        TopDocs top = c.topDocs();
              for (ScoreDoc sd : top.scoreDocs) {
                g.add(sd.doc);
                g.add(Float.floatToIntBits(sd.score));
              }
            }
            sink += g.h;
            return queries.size();
          };
      Fnv f = new Fnv();
      for (byte[] q : queries) {
        TopKnnCollector c = new TopKnnCollector(10, Integer.MAX_VALUE);
        vectors.search("bv", q, c, all);
        TopDocs top = c.topDocs();
        for (ScoreDoc sd : top.scoreDocs) {
          f.add(sd.doc);
          f.add(Float.floatToIntBits(sd.score));
        }
      }
      System.out.printf("#check	cx_bitvec_search	%016x	%d%n", f.h, queries.size());
      measure("cx_bitvec_search", op);
    }
  }

  /** Builds {@code usplit/}: {@code cx_usplit_seek}'s ids under UniformSplit. */
  static void buildUniformSplit(Path dir) throws IOException {
    Random r = new Random(0x15_5_2026_1020L);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config("uid", new UniformSplitPostingsFormat()))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("uid", idOf(i), Field.Store.NO));
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present, half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  /** Builds {@code stsplit/}: {@code cx_stsplit_seek}'s ids under SharedTermsUniformSplit. */
  static void buildSharedTerms(Path dir) throws IOException {
    Random r = new Random(0x15_5_2026_1021L);
    PostingsFormat shared = new STUniformSplitPostingsFormat();
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, config(Map.of("sa", shared, "sb", shared)))) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("sa", idOf(i), Field.Store.NO));
        if (i % 3 == 0) {
          doc.add(new StringField("sb", idOf(i), Field.Store.NO));
        }
        w.addDocument(doc);
      }
      w.commit();
    }
    List<String> targets = new ArrayList<>();
    for (int k = 0; k < 200_000; k++) {
      // Half present (in sa), half absent.
      targets.add(idOf(r.nextInt(2) == 0 ? r.nextInt(DOCS) : DOCS + r.nextInt(10 * DOCS)));
    }
    Files.write(dir.resolve("targets.tsv"), targets);
    Files.writeString(dir.resolve("built"), "");
  }

  /** {@code cx_bloom_seek}: {@code seekExact} on one filtered enum, per target. */
  static void runBloom(Path dir) throws IOException {
    runSeek(dir, "cx_bloom_seek", "id");
  }

  /** {@code cx_fst_seek}/{@code cx_direct_seek}: {@code seekExact} on FST50 and Direct. */
  static void runMemory(Path dir) throws IOException {
    runSeek(dir, "cx_fst_seek", "fid");
    runSeek(dir, "cx_direct_seek", "did");
  }

  /**
   * {@code cx_ords_seek}: per target, {@code seekExact(term)} hashing its {@code ord()} (-1 when
   * absent), then {@code seekExact(ord)} hashing the term's length and last byte, on one enum.
   */
  static void runOrds(Path dir) throws IOException {
    if (!selected("cx_ords_seek")) return;
    List<BytesRef> ids = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("targets.tsv"))) {
      ids.add(new BytesRef(line));
    }
    List<String> ordLines = Files.readAllLines(dir.resolve("ords.tsv"));
    long[] ords = new long[ordLines.size()];
    for (int i = 0; i < ords.length; i++) ords[i] = Long.parseLong(ordLines.get(i));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      Terms terms = reader.leaves().get(0).reader().terms("oid");
      Op op =
          () -> {
            Fnv g = new Fnv();
            ordsPass(terms, ids, ords, g);
            sink += g.h;
            return ids.size();
          };
      Fnv f = new Fnv();
      ordsPass(terms, ids, ords, f);
      System.out.printf("#check	%s	%016x	%d%n", "cx_ords_seek", f.h, ids.size());
      measure("cx_ords_seek", op);
    }
  }

  static void ordsPass(Terms terms, List<BytesRef> ids, long[] ords, Fnv f) throws IOException {
    TermsEnum te = terms.iterator();
    for (int i = 0; i < ords.length; i++) {
      f.add(te.seekExact(ids.get(i)) ? te.ord() : -1);
      te.seekExact(ords[i]);
      BytesRef t = te.term();
      f.add(((long) t.length << 8) | (t.bytes[t.offset + t.length - 1] & 0xff));
    }
  }

  static void runSeek(Path dir, String name, String field) throws IOException {
    if (!selected(name)) return;
    List<BytesRef> ids = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("targets.tsv"))) {
      ids.add(new BytesRef(line));
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      Terms terms = reader.leaves().get(0).reader().terms(field);
      Op op =
          () -> {
            TermsEnum te = terms.iterator();
            Fnv g = new Fnv();
            for (BytesRef id : ids) {
              g.add(te.seekExact(id) ? te.docFreq() : -1);
            }
            sink += g.h;
            return ids.size();
          };
      Fnv f = new Fnv();
      TermsEnum te = terms.iterator();
      for (BytesRef id : ids) {
        f.add(te.seekExact(id) ? te.docFreq() : -1);
      }
      System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, ids.size());
      measure(name, op);
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (!Files.exists(dir.resolve("bloom").resolve("built"))) {
        Files.createDirectories(dir.resolve("bloom"));
        buildBloom(dir.resolve("bloom"));
      }
      if (!Files.exists(dir.resolve("memory").resolve("built"))) {
        Files.createDirectories(dir.resolve("memory"));
        buildMemory(dir.resolve("memory"));
      }
      if (!Files.exists(dir.resolve("bitvectors").resolve("built"))) {
        Files.createDirectories(dir.resolve("bitvectors"));
        buildBitVectors(dir.resolve("bitvectors"));
      }
      if (!Files.exists(dir.resolve("ords").resolve("built"))) {
        Files.createDirectories(dir.resolve("ords"));
        buildOrds(dir.resolve("ords"));
      }
      if (!Files.exists(dir.resolve("usplit").resolve("built"))) {
        Files.createDirectories(dir.resolve("usplit"));
        buildUniformSplit(dir.resolve("usplit"));
      }
      if (!Files.exists(dir.resolve("stsplit").resolve("built"))) {
        Files.createDirectories(dir.resolve("stsplit"));
        buildSharedTerms(dir.resolve("stsplit"));
      }
    } else {
      runBloom(dir.resolve("bloom"));
      runMemory(dir.resolve("memory"));
      runBitVectors(dir.resolve("bitvectors"));
      runOrds(dir.resolve("ords"));
      runSeek(dir.resolve("usplit"), "cx_usplit_seek", "uid");
      runSeek(dir.resolve("stsplit"), "cx_stsplit_seek", "sb");
    }
  }
}
