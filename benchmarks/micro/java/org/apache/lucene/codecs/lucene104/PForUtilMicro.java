package org.apache.lucene.codecs.lucene104;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.internal.vectorization.PostingDecodingUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;

/**
 * Java side of the {@code PForUtil} (patched frame-of-reference) decode microbenchmark -- the
 * frequency blocks every scoring postings walk decodes. The Rust side is {@code micro.rs}'s
 * {@code pfor_decode} case.
 *
 * <p>Lives in this package for the same reason {@link ForUtilMicro} does: {@code PForUtil} is
 * package-private. {@code VectorizationProvider.getInstance()} refuses callers outside Lucene, so
 * the {@link PostingDecodingUtil} is taken from a {@link PostingIndexInput} -- which is allowed to
 * build one -- by reflection.
 *
 * <p>Each case is a block of 256 values below {@code 2^bits} with three exceptions, values that
 * need eight more bits and so take PForUtil's patch path. Emits {@code case<TAB>ns_per_block}.
 */
public final class PForUtilMicro {

  static int[] block(int bits) {
    int[] out = new int[ForUtil.BLOCK_SIZE];
    int state = 0x51ED270B ^ bits;
    int mask = (1 << bits) - 1;
    for (int i = 0; i < out.length; ++i) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      out[i] = state & mask;
    }
    // Three exceptions at fixed slots, each needing 8 extra bits.
    for (int slot : new int[] {17, 101, 230}) {
      out[slot] = (out[slot] & mask) | (0xA5 << bits);
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 2000);
    Field pduField = PostingIndexInput.class.getDeclaredField("postingDecodingUtil");
    pduField.setAccessible(true);

    Path dir = Files.createTempDirectory("pforutil-micro");
    try (Directory directory = new MMapDirectory(dir)) {
      for (int bits = 1; bits <= 23; ++bits) {
        String name = "bits" + bits;
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
          new PForUtil(new ForUtil()).encode(block(bits), out);
        }
        try (IndexInput in = directory.openInput(name, IOContext.DEFAULT)) {
          PostingIndexInput pii = new PostingIndexInput(in, new ForUtil());
          PostingDecodingUtil pdu = (PostingDecodingUtil) pduField.get(pii);
          PForUtil pfor = new PForUtil(new ForUtil());
          int[] decoded = new int[ForUtil.BLOCK_SIZE];
          in.seek(0);
          pfor.decode(pdu, decoded);
          int[] expected = block(bits);
          for (int i = 0; i < decoded.length; ++i) {
            if (decoded[i] != expected[i]) {
              throw new AssertionError("round-trip failed at bits=" + bits + " i=" + i);
            }
          }
          long blocks = loop(in, pdu, pfor, decoded, warmupMs);
          blocks = loop(in, pdu, pfor, decoded, measureMs);
          System.out.printf(
              "%s\t%.3f\t%d%n", String.format("bits%02d", bits), (double) lastNanos / blocks, blocks);
        }
      }
    } finally {
      try (var s = Files.list(dir)) {
        for (Path p : s.toList()) Files.deleteIfExists(p);
      }
      Files.deleteIfExists(dir);
    }
  }

  private static long lastNanos;

  private static long loop(
      IndexInput in, PostingDecodingUtil pdu, PForUtil pfor, int[] decoded, long budgetMs)
      throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long blocks = 0;
    long start = System.nanoTime();
    long elapsed;
    do {
      for (int i = 0; i < 1024; ++i) {
        in.seek(0);
        pfor.decode(pdu, decoded);
      }
      blocks += 1024;
      elapsed = System.nanoTime() - start;
    } while (elapsed < budgetNs);
    lastNanos = elapsed;
    if (decoded[0] == 0xDEADBEEF) {
      System.err.print("");
    }
    return blocks;
  }
}
