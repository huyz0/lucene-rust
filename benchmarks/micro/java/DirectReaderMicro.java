import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;

/**
 * Java side of the DirectReader microbenchmark -- the per-value read behind doc values and
 * monotonic sequences. The Rust side is {@code benchmarks/rust-runner/src/micro.rs}'s
 * {@code direct_reader} case.
 *
 * <p>Both sides read a fixed odd stride through a packed array of 2^17 values rather than
 * sweeping it sequentially: this primitive exists to serve random per-document lookups, and a
 * sequential sweep would measure the hardware prefetcher instead.
 *
 * <p>Values are written with {@link DirectWriter} rather than generated identically to the Rust
 * side, because unlike ForUtil this benchmark does not need the two to decode the same *bytes* --
 * it needs them to do the same *work*, and the work is a function of `bitsPerValue` and the access
 * pattern, not of the values. Each side round-trip-checks its own array before timing.
 *
 * <p>Emits TSV {@code bitsNN<TAB>ns_per_op<TAB>ops} on stdout.
 */
public final class DirectReaderMicro {

  private static final int COUNT = 1 << 17;
  private static final int STRIDE = 4099;

  public static void main(String[] args) throws IOException {
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 2000);

    Path dir = Files.createTempDirectory("directreader-micro");
    try (Directory directory = new MMapDirectory(dir)) {
      for (int bits : new int[] {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64}) {
        String name = "bits" + bits;
        long mask = bits >= 64 ? -1L : (1L << bits) - 1;
        long[] values = new long[COUNT];
        long state = 0x243F6A8885A308D3L ^ bits;
        for (int i = 0; i < COUNT; ++i) {
          state ^= state << 13;
          state ^= state >>> 7;
          state ^= state << 17;
          values[i] = state & mask;
        }

        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
          DirectWriter w = DirectWriter.getInstance(out, COUNT, bits);
          for (long v : values) {
            w.add(v);
          }
          w.finish();
        }

        try (IndexInput in = directory.openInput(name, IOContext.DEFAULT)) {
          RandomAccessInput slice = in.randomAccessSlice(0, in.length());
          LongValues reader = DirectReader.getInstance(slice, bits);

          for (int i : new int[] {0, 1, COUNT / 2, COUNT - 1}) {
            if (reader.get(i) != values[i]) {
              throw new AssertionError("round-trip failed at bits=" + bits + " i=" + i);
            }
          }

          walk(reader, warmupMs);
          long ops = walk(reader, measureMs);
          System.out.printf("bits%02d\t%.3f\t%d%n", bits, (double) lastNanos / ops, ops);
        }
      }
    } finally {
      for (Path p : Files.list(dir).toList()) {
        Files.deleteIfExists(p);
      }
      Files.deleteIfExists(dir);
    }
  }

  private static long lastNanos;
  private static long sink;

  private static long walk(LongValues reader, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long ops = 0;
    int i = 0;
    long start = System.nanoTime();
    long elapsed;
    do {
      for (int n = 0; n < 4096; ++n) {
        i = (i + STRIDE) & (COUNT - 1);
        sink += reader.get(i);
      }
      ops += 4096;
      elapsed = System.nanoTime() - start;
    } while (elapsed < budgetNs);
    lastNanos = elapsed;
    if (sink == 0xDEADBEEFL) {
      System.err.print("");
    }
    return ops;
  }
}
