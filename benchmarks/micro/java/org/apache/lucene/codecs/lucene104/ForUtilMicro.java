package org.apache.lucene.codecs.lucene104;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;

/**
 * Java side of the ForUtil decode microbenchmark. The Rust side is
 * {@code crates/lucene-codecs/benches/for_util_decode.rs}; {@code scripts/bench-micro.sh}
 * runs both and joins them on the case name.
 *
 * <p>This class lives in {@code org.apache.lucene.codecs.lucene104} deliberately: {@code ForUtil}
 * is package-private, and {@link PostingIndexInput} -- whose own javadoc says it "mostly exists to
 * enable benchmarking the decoding logic of postings since it internally calls code that may only
 * be called from the lucene-core JAR" -- is the supported way in. Run with
 * {@code --add-modules jdk.incubator.vector}, otherwise {@code VectorizationProvider} falls back to
 * the scalar default and the comparison flatters the Rust side.
 *
 * <p>Reads from an {@link MMapDirectory} because that is what selects
 * {@code MemorySegmentPostingDecodingUtil}: the Panama path loads {@code IntVector}s straight out
 * of the mapped segment, and a heap-backed directory would silently measure something else.
 *
 * <p>Emits TSV {@code case<TAB>ns_per_block<TAB>blocks} on stdout.
 */
public final class ForUtilMicro {

  private static final int BLOCK_SIZE = ForUtil.BLOCK_SIZE;

  /**
   * Same generator as the Rust side, bit for bit: a xorshift32 seeded {@code 0x9E3779B9 ^ bits},
   * masked to {@code bitsPerValue}. Both harnesses must decode identical bytes or the comparison is
   * between two different workloads.
   */
  private static int[] blockFor(int bits) {
    int[] out = new int[BLOCK_SIZE];
    int state = 0x9E3779B9 ^ bits;
    int mask = bits >= 32 ? -1 : (1 << bits) - 1;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      out[i] = state & mask;
    }
    return out;
  }

  public static void main(String[] args) throws IOException {
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 2000);

    Path dir = Files.createTempDirectory("forutil-micro");
    try (Directory directory = new MMapDirectory(dir)) {
      // 1..=31, not 1..=32: decodeSlow indexes MASKS32, which is declared
      // new int[32], so bitsPerValue == 32 throws AIOOBE here. The Rust side
      // decodes it (its mask saturates rather than indexing a table) but has
      // nothing to be compared against, so the shared range stops at 31.
      for (int bits = 1; bits <= 31; ++bits) {
        String name = "bits" + bits;
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
          // encode() collapses in place, so hand it a fresh copy.
          new ForUtil().encode(blockFor(bits), bits, out);
        }
        try (IndexInput in = directory.openInput(name, IOContext.DEFAULT)) {
          PostingIndexInput pii = new PostingIndexInput(in, new ForUtil());
          int[] decoded = new int[BLOCK_SIZE];

          // Guard the fixture: a decode benchmark over bytes that do not round-trip
          // measures the wrong work and still looks fast.
          in.seek(0);
          pii.decode(bits, decoded);
          int[] expected = blockFor(bits);
          for (int i = 0; i < BLOCK_SIZE; ++i) {
            if (decoded[i] != expected[i]) {
              throw new AssertionError(
                  "round-trip failed at bits=" + bits + " i=" + i
                      + " got=" + decoded[i] + " want=" + expected[i]);
            }
          }

          long blocks = timedLoop(in, pii, bits, decoded, warmupMs);
          blocks = timedLoop(in, pii, bits, decoded, measureMs);
          long nanos = lastNanos;
          System.out.printf(
              "%s\t%.3f\t%d%n", String.format("bits%02d", bits), (double) nanos / blocks, blocks);
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

  /**
   * Runs decode for at least {@code budgetMs}, in batches, and returns the block count. Called
   * twice per case: the first call is warmup whose result is discarded, so C2 has compiled and
   * settled before the measured call starts.
   */
  private static long timedLoop(
      IndexInput in, PostingIndexInput pii, int bits, int[] decoded, long budgetMs)
      throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long blocks = 0;
    long start = System.nanoTime();
    long elapsed;
    do {
      for (int i = 0; i < 1024; ++i) {
        in.seek(0);
        pii.decode(bits, decoded);
      }
      blocks += 1024;
      elapsed = System.nanoTime() - start;
    } while (elapsed < budgetNs);
    lastNanos = elapsed;
    // Keep the decoded array observably live so nothing above can be eliminated.
    if (decoded[0] == 0xDEADBEEF) {
      System.err.print("");
    }
    return blocks;
  }
}
