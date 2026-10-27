import java.io.IOException;
import java.lang.ref.Reference;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/** Native term-vector routing. Flush times prepared-document insertion + commit;
 * merge times forceMerge + commit. Setup/readback/accounting are outside timers.
 * Compile with fixtures/src/testframework/TermVectorsRoutingSupport.java.
 */
public final class TermVectorsWriterRoutingBench {
  static final int DOCS = 2048, SOURCES = 4;
  private static volatile long sink;
  private static final TermVectorsRoutingSupport.Profile PROFILE =
      TermVectorsRoutingSupport.Profile.rich();

  private static IndexWriter prepare(Directory directory, boolean asserting, int mode)
      throws IOException {
    var writer = new IndexWriter(directory,
        TermVectorsRoutingSupport.config(asserting, mode == 2, DOCS, null));
    if (mode != 0) {
      for (int doc = 0; doc < DOCS; doc++) {
        writer.addDocument(TermVectorsRoutingSupport.document(doc, DOCS, SOURCES, 0, PROFILE));
        if ((doc + 1) % (DOCS / SOURCES) == 0) writer.commit();
      }
      for (int doc = 0; doc < DOCS; doc++) if (doc % 7 == 0)
        writer.deleteDocuments(new Term("id", Integer.toString(doc)));
      writer.commit();
      TermVectorsRoutingSupport.require(
          SegmentInfos.readLatestCommit(directory).size() == SOURCES, "source segment count");
    }
    return writer;
  }
  private static List<Document> preparedDocuments(int mode) {
    if (mode != 0) return List.of();
    var documents = new ArrayList<Document>(DOCS);
    for (int doc = 0; doc < DOCS; doc++)
      documents.add(TermVectorsRoutingSupport.document(doc, DOCS, SOURCES, 0, PROFILE));
    return documents;
  }
  private static TermVectorsRoutingSupport.Check verify(Directory directory, int mode)
      throws IOException {
    var check = TermVectorsRoutingSupport.verify(directory,
        TermVectorsRoutingSupport.ids(DOCS, SOURCES, mode != 0, mode == 2),
        DOCS, SOURCES, PROFILE, true, null);
    sink = check.digest();
    return check;
  }
  private static String name(boolean asserting, int mode) {
    return "term_vectors_routing_" + (asserting ? "asserting" : "default") + "_"
        + new String[] {"flush", "merge", "sorted_merge"}[mode] + "_commit";
  }
  private static void output(String name, double ns, long iterations,
      TermVectorsRoutingSupport.Check check, long accounted) {
    System.out.printf(Locale.ROOT, "#check\t%s\t%s\t%d\t%d\t%d\t%d\t%d%n",
        name, Long.toUnsignedString(check.digest()), check.docs(), check.fields(),
        check.terms(), check.occurrences(), check.payloadBytes());
    System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, ns, iterations);
    System.out.printf(Locale.ROOT,
        "%s_encoded_bytes\t%d%n%s_writer_accounted_bytes\t%d%n%s_logical_payload_bytes\t%d%n",
        name, check.encodedBytes(), name, accounted, name, check.payloadBytes());
  }
  private static void run(boolean asserting, int mode, long millis, long fixed, boolean print)
      throws IOException {
    long nanos = 0, iterations = 0;
    for (;;) {
      try (var directory = new ByteBuffersDirectory();
           var writer = prepare(directory, asserting, mode)) {
        var documents = preparedDocuments(mode);
        long iterationNanos = 0;
        if (mode == 0) {
          // Java encodes term-vector chunks during addDocument. Include the
          // matching insertion work on both sides rather than timing commit
          // after Java's writer has already compressed most of the vectors.
          long insertionStart = System.nanoTime();
          for (var document : documents) writer.addDocument(document);
          iterationNanos = System.nanoTime() - insertionStart;
        }
        long accounted = writer.ramBytesUsed();
        long start = System.nanoTime();
        if (mode != 0) writer.forceMerge(1);
        writer.commit();
        nanos += iterationNanos + System.nanoTime() - start;
        iterations++;
        var check = verify(directory, mode);
        // Rust's prepared input Vec remains owned through readback too.
        Reference.reachabilityFence(documents);
        if ((fixed > 0 && iterations == fixed)
            || (fixed == 0 && nanos >= millis * 1_000_000L)) {
          if (print) output(name(asserting, mode), (double) nanos / iterations, iterations,
                            check, accounted);
          return;
        }
      }
    }
  }
  private static int mode(String text) {
    int value = Integer.parseInt(text);
    TermVectorsRoutingSupport.require(value >= 0 && value <= 2, "mode must be 0, 1 or 2");
    return value;
  }
  public static void main(String[] args) throws Exception {
    TermVectorsRoutingSupport.require(TermVectorsWriterRoutingBench.class.desiredAssertionStatus(),
                                      "run with -ea");
    if (args.length > 0 && args[0].equals("--verify-dir")) {
      try (var directory = FSDirectory.open(Path.of(args[1]))) {
        int mode = mode(args[2]);
        output(name(true, mode), 1, 1, verify(directory, mode), 0);
      }
      return;
    }
    if (args.length > 0 && args[0].equals("--memory-case")) {
      long fixed = Long.parseLong(args[2]);
      TermVectorsRoutingSupport.require(fixed > 0, "memory iterations");
      for (int mode = 0; mode < 3; mode++) for (boolean asserting : new boolean[] {false, true})
        if (name(asserting, mode).equals(args[1])) {
          run(asserting, mode, 0, fixed, true);
          return;
        }
      throw new IllegalArgumentException("unknown case: " + args[1]);
    }
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    TermVectorsRoutingSupport.require(warmup > 0 && measure > 0, "positive durations");
    for (int mode = 0; mode < 3; mode++) for (boolean asserting : new boolean[] {false, true}) {
      run(asserting, mode, warmup, 0, false);
      run(asserting, mode, measure, 0, true);
    }
  }
}
