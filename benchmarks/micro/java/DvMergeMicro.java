import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Doc-values merge throughput: four segments of documents carrying the five sparse doc-values
 * columns of {@code write_sparse_doc_values_fixture} (every type, each missing on a different
 * stride), merged into one with {@code forceMerge(1)}. The Rust side is {@code bench_dv_merge} in
 * {@code benchmarks/rust-runner/src/micro.rs}, over the same documents.
 *
 * <p>Each timed merge starts from a fresh copy of the unmerged segments; the copy is not timed.
 * Emits {@code case<TAB>ns_per_doc<TAB>docs}.
 */
public final class DvMergeMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static final int DOCS = 200_000;
  static final int SEGMENTS = 4;

  static Document document(int i) {
    Document d = new Document();
    if (i % 3 != 0) d.add(new NumericDocValuesField("num", 7L * i - 1000));
    if (i % 5 != 0) d.add(new BinaryDocValuesField("bin", new BytesRef("b" + i)));
    if (i % 7 != 0) d.add(new SortedDocValuesField("sorted", new BytesRef("s" + (i % 50))));
    if (i % 11 != 0) {
      for (long v : new long[] {i % 13, i, -i}) d.add(new SortedNumericDocValuesField("snum", v));
    }
    if (i % 13 != 0) {
      d.add(new SortedSetDocValuesField("sset", new BytesRef("t" + (i % 17))));
      d.add(new SortedSetDocValuesField("sset", new BytesRef("t" + (i % 19))));
    }
    return d;
  }

  static void copy(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    try (Stream<Path> files = Files.list(from)) {
      for (Path f : (Iterable<Path>) files::iterator) {
        Files.copy(f, to.resolve(f.getFileName()));
      }
    }
  }

  static void delete(Path p) throws IOException {
    if (!Files.exists(p)) return;
    try (Stream<Path> walk = Files.walk(p)) {
      for (Path f : (Iterable<Path>) walk.sorted(Comparator.reverseOrder())::iterator) {
        Files.delete(f);
      }
    }
  }

  static long mergeOnce(Path source, Path work) throws IOException {
    delete(work);
    copy(source, work);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMergePolicy(new TieredMergePolicy());
    long start = System.nanoTime();
    try (FSDirectory dir = FSDirectory.open(work);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      w.forceMerge(1);
      w.commit();
    }
    return System.nanoTime() - start;
  }

  public static void main(String[] args) throws IOException {
    Path root = Files.createTempDirectory("dv-merge-micro");
    Path source = root.resolve("source");
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(DOCS / SEGMENTS);
    cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    try (FSDirectory dir = FSDirectory.open(source);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        w.addDocument(document(i));
      }
      w.commit();
    }
    Path work = root.resolve("work");
    long warmEnd = System.nanoTime() + warmupMs * 1_000_000L;
    while (System.nanoTime() < warmEnd) {
      mergeOnce(source, work);
    }
    long total = 0;
    long docs = 0;
    long measureEnd = System.nanoTime() + measureMs * 1_000_000L;
    do {
      total += mergeOnce(source, work);
      docs += DOCS;
    } while (System.nanoTime() < measureEnd);
    System.out.printf("sparse_5_types\t%.3f\t%d%n", (double) total / docs, docs);
    delete(root);
  }
}
