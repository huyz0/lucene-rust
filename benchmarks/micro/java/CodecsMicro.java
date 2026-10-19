import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
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

  /** {@code cx_bloom_seek}: {@code seekExact} on one filtered enum, per target. */
  static void runBloom(Path dir) throws IOException {
    runSeek(dir, "cx_bloom_seek", "id");
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
    } else {
      runBloom(dir.resolve("bloom"));
    }
  }
}
