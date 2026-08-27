import java.io.IOException;
import java.nio.file.Paths;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.store.MMapDirectory;

/**
 * Java side of the reader-open microbenchmark. The Rust side is
 * {@code benchmarks/rust-runner/src/micro.rs}'s {@code reader_open} case; see its comment for why
 * this measurement exists -- this port materializes every term of every field when a segment is
 * opened, where {@code SegmentTermsEnum} navigates the term dictionary lazily and holds nothing.
 *
 * <p>Emits TSV {@code open<TAB>ns_per_open<TAB>opens} on stdout.
 */
public final class ReaderOpenMicro {

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("usage: ReaderOpenMicro <index-dir>");
      System.exit(2);
    }
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 2000);

    try (MMapDirectory dir = new MMapDirectory(Paths.get(args[0]))) {
      open(dir, warmupMs);
      long opens = open(dir, measureMs);
      System.out.printf("open\t%.3f\t%d%n", (double) lastNanos / opens, opens);
    }
  }

  private static long lastNanos;
  private static int sink;

  private static long open(MMapDirectory dir, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long opens = 0;
    long start = System.nanoTime();
    long elapsed;
    do {
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        sink += reader.leaves().size();
      }
      opens++;
      elapsed = System.nanoTime() - start;
    } while (elapsed < budgetNs);
    lastNanos = elapsed;
    if (sink == 0xDEADBEEF) {
      System.err.print("");
    }
    return opens;
  }
}
