import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.FSDirectory;

/**
 * Concurrent indexing throughput (M4's T4.3): {@code IndexWriter} shared by several threads, each
 * adding its share of the same 100 000 documents, then one commit.
 *
 * <ul>
 *   <li>{@code add_t1}, {@code add_t4}: {@code addDocument} from one and from four threads.
 *   <li>{@code update_t4}: {@code updateDocument(id, doc)} from four threads -- what OpenSearch
 *       issues for every indexed document, so each add also buffers a term delete.
 * </ul>
 *
 * Each document is a stored {@code StringField} id and a stored {@code TextField} body of twelve
 * words. {@code maxBufferedDocs} 10 000 (per indexing thread, as in Lucene), no merges, no
 * compound files. The Rust side is {@code bench_concurrent_index} in {@code
 * benchmarks/rust-runner/src/micro.rs}, over {@code ConcurrentIndexWriter}. Emits {@code
 * case<TAB>ns_per_doc<TAB>docs}; ns per document is wall time, so four threads that scale show a
 * quarter of one thread's figure.
 */
public final class ConcurrentIndexMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static final int DOCS = 100_000;

  /** Must match the Rust side's {@code body}. */
  static String body(int i) {
    StringBuilder b = new StringBuilder();
    for (int k = 0; k < 12; k++) {
      int w = (int) (((long) i * (2 * k + 7) + (long) k * k * 31) % (50 + 40 * k));
      if (k > 0) b.append(' ');
      b.append('w').append(k).append('x').append(w);
    }
    return b.toString();
  }

  static Document document(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.YES));
    d.add(new TextField("body", body(i), Field.Store.YES));
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

  static long index(Path dirPath, int threads, boolean update) throws Exception {
    delete(dirPath);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(10_000);
    cfg.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setUseCompoundFile(false);
    long start = System.nanoTime();
    try (FSDirectory dir = FSDirectory.open(dirPath);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      List<Thread> workers = new ArrayList<>();
      Exception[] failure = new Exception[1];
      for (int t = 0; t < threads; t++) {
        final int first = t;
        Thread worker =
            new Thread(
                () -> {
                  try {
                    for (int i = first; i < DOCS; i += threads) {
                      if (update) {
                        w.updateDocument(new Term("id", "d" + i), document(i));
                      } else {
                        w.addDocument(document(i));
                      }
                    }
                  } catch (Exception e) {
                    failure[0] = e;
                  }
                });
        worker.start();
        workers.add(worker);
      }
      for (Thread worker : workers) {
        worker.join();
      }
      if (failure[0] != null) throw failure[0];
      w.commit();
    }
    return System.nanoTime() - start;
  }

  interface Run {
    long nanos() throws Exception;
  }

  static void measure(String name, Run run) throws Exception {
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

  public static void main(String[] args) throws Exception {
    Path root = Files.createTempDirectory("concurrent-index-micro");
    Path dir = root.resolve("index");
    measure("add_t1", () -> index(dir, 1, false));
    measure("add_t4", () -> index(dir, 4, false));
    measure("update_t4", () -> index(dir, 4, true));
    delete(root);
  }
}
