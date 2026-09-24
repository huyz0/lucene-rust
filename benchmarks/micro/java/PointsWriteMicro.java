import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.FSDirectory;

/**
 * Points write throughput: documents carrying the four point fields of {@code
 * write_points_segment_fixture} (a multi-valued {@code LongPoint}, a sparse {@code IntPoint}, a
 * {@code DoublePoint}, a two-dimensional {@code IntPoint}) and nothing else.
 *
 * <ul>
 *   <li>{@code flush}: index 200 000 documents into one segment and commit.
 *   <li>{@code merge}: {@code forceMerge(1)} of four 50 000-document segments, each run from a fresh
 *       copy (not timed).
 * </ul>
 *
 * Neither side writes compound files. Both sides also store every document's values in
 * {@code .fdt}: this port's {@code Document} is its stored fields, so the Java documents carry
 * a {@code StoredField} beside each point to match.
 *
 * <p>The Rust side is {@code bench_points_write} in {@code benchmarks/rust-runner/src/micro.rs}.
 * Emits {@code case<TAB>ns_per_doc<TAB>docs}.
 */
public final class PointsWriteMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static final int DOCS = 200_000;
  static final int SEGMENTS = 4;

  static Document document(int i) {
    Document d = new Document();
    d.add(new LongPoint("lp", 7L * i - 1000));
    d.add(new StoredField("lp", 7L * i - 1000));
    if (i % 4 == 0) {
      d.add(new LongPoint("lp", -i));
      d.add(new StoredField("lp", (long) -i));
    }
    if (i % 3 != 0) {
      d.add(new IntPoint("ip", i % 1000));
      d.add(new StoredField("ip", i % 1000));
    }
    d.add(new DoublePoint("dp", i / 8.0 - 100.0));
    d.add(new StoredField("dp", i / 8.0 - 100.0));
    d.add(new IntPoint("xy", i % 97, i % 89 - 44));
    d.add(new StoredField("xy", IntPoint.pack(i % 97, i % 89 - 44)));
    return d;
  }

  static void delete(Path p) throws IOException {
    if (!Files.exists(p)) return;
    try (Stream<Path> walk = Files.walk(p)) {
      for (Path f : (Iterable<Path>) walk.sorted(Comparator.reverseOrder())::iterator) {
        Files.delete(f);
      }
    }
  }

  static void index(Path dirPath, int docsPerSegment) throws IOException {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(docsPerSegment);
    cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    // This port writes no compound files; neither does this side.
    cfg.setUseCompoundFile(false);
    try (FSDirectory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        w.addDocument(document(i));
      }
      w.commit();
    }
  }

  interface Run {
    long nanos() throws IOException;
  }

  static void measure(String name, Run run) throws IOException {
    long warmEnd = System.nanoTime() + warmupMs * 1_000_000L;
    while (System.nanoTime() < warmEnd) {
      run.nanos();
    }
    long total = 0;
    long docs = 0;
    long end = System.nanoTime() + measureMs * 1_000_000L;
    do {
      total += run.nanos();
      docs += DOCS;
    } while (System.nanoTime() < end);
    System.out.printf("%s\t%.3f\t%d%n", name, (double) total / docs, docs);
  }

  public static void main(String[] args) throws IOException {
    Path root = Files.createTempDirectory("points-write-micro");
    Path flushDir = root.resolve("flush");
    measure(
        "flush",
        () -> {
          delete(flushDir);
          long start = System.nanoTime();
          index(flushDir, DOCS);
          return System.nanoTime() - start;
        });

    Path source = root.resolve("source");
    index(source, DOCS / SEGMENTS);
    Path work = root.resolve("work");
    measure(
        "merge",
        () -> {
          delete(work);
          Files.createDirectories(work);
          try (Stream<Path> files = Files.list(source)) {
            for (Path f : (Iterable<Path>) files::iterator) {
              Files.copy(f, work.resolve(f.getFileName()));
            }
          }
          IndexWriterConfig cfg = new IndexWriterConfig();
          TieredMergePolicy mp = new TieredMergePolicy();
          mp.setNoCFSRatio(0.0);
          cfg.setMergePolicy(mp);
          cfg.setUseCompoundFile(false);
          long start = System.nanoTime();
          try (FSDirectory dir = FSDirectory.open(work);
              IndexWriter w = new IndexWriter(dir, cfg)) {
            w.forceMerge(1);
            w.commit();
          }
          return System.nanoTime() - start;
        });
    delete(root);
  }
}
