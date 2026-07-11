import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.util.BitUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Generates byte-level fixtures for lucene-rust primitive decoders, pinned to the
 * Lucene version on the classpath (must match OpenSearch's pin).
 *
 * Each fixture is a pair: <name>.bin (Lucene-encoded bytes) and <name>.expected
 * (one decoded value per line, decimal) so the Rust side needs no Java to verify.
 *
 * Deterministic: fixed seed. Usage: java GenPrimitives <outdir>
 */
public class GenPrimitives {
  static final long SEED = 0xC0FFEE_2026_0711L;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);

    // Edge cases + random values across all byte-length buckets.
    long[] vlongs = interesting(new Random(SEED), 500, false);
    int[] vints = new int[500];
    long[] zlongs = interesting(new Random(SEED + 1), 500, true);
    Random r = new Random(SEED + 2);
    int[] edges = {0, 1, 127, 128, 16383, 16384, Integer.MAX_VALUE, -1, Integer.MIN_VALUE};
    for (int i = 0; i < vints.length; i++) {
      vints[i] = i < edges.length ? edges[i] : r.nextInt();
    }

    // vint
    ByteBuffersDataOutput o = new ByteBuffersDataOutput();
    StringBuilder exp = new StringBuilder();
    for (int v : vints) { o.writeVInt(v); exp.append(v).append('\n'); }
    write(out, "vint", o, exp);

    // vlong (non-negative only; Lucene forbids negatives for writeVLong)
    o = new ByteBuffersDataOutput(); exp = new StringBuilder();
    for (long v : vlongs) { o.writeVLong(v); exp.append(v).append('\n'); }
    write(out, "vlong", o, exp);

    // zigzag long
    o = new ByteBuffersDataOutput(); exp = new StringBuilder();
    for (long v : zlongs) { o.writeZLong(v); exp.append(v).append('\n'); }
    write(out, "zlong", o, exp);

    // group-varint: 1024 ints (multiple full groups of 4 + tail), unsigned int range
    o = new ByteBuffersDataOutput(); exp = new StringBuilder();
    Random gr = new Random(SEED + 3);
    long[] gvals = new long[1024];
    for (int i = 0; i < gvals.length; i++) {
      // spread across 1..4 byte encodings
      int bits = 1 + gr.nextInt(32);
      gvals[i] = gr.nextLong() >>> (64 - bits);
      exp.append(gvals[i]).append('\n');
    }
    o.writeGroupVInts(gvals, gvals.length);
    write(out, "group_vint", o, exp);

    // sanity: re-read group vints with Lucene itself
    byte[] bytes = o.toArrayCopy();
    ByteArrayDataInput in = new ByteArrayDataInput(bytes);
    long[] back = new long[1024];
    org.apache.lucene.util.GroupVIntUtil.readGroupVInts(in, back, back.length);
    for (int i = 0; i < back.length; i++) {
      if (back[i] != gvals[i]) throw new AssertionError("self-check failed at " + i);
    }

    // zigzag primitive reference values for unit tests
    StringBuilder zz = new StringBuilder();
    for (long v : new long[] {0, -1, 1, -2, 2, Long.MAX_VALUE, Long.MIN_VALUE}) {
      zz.append(v).append(' ').append(BitUtil.zigZagEncode(v)).append('\n');
    }
    Files.writeString(out.resolve("zigzag_pairs.expected"), zz.toString());

    System.out.println("fixtures written to " + out + " (lucene " +
        org.apache.lucene.util.Version.LATEST + ")");
  }

  static long[] interesting(Random r, int n, boolean allowNegative) {
    long[] a = new long[n];
    long[] edges = allowNegative
        ? new long[] {0, 1, -1, 63, 64, -64, -65, Long.MAX_VALUE, Long.MIN_VALUE + 1}
        : new long[] {0, 1, 127, 128, 16383, 16384, (1L << 62) - 1, Long.MAX_VALUE};
    for (int i = 0; i < n; i++) {
      if (i < edges.length) { a[i] = edges[i]; continue; }
      int bits = 1 + r.nextInt(allowNegative ? 63 : 63);
      long v = r.nextLong() >>> (64 - bits);
      a[i] = allowNegative && r.nextBoolean() ? -v : v;
    }
    return a;
  }

  static void write(Path dir, String name, ByteBuffersDataOutput o, StringBuilder exp)
      throws IOException {
    Files.write(dir.resolve(name + ".bin"), o.toArrayCopy());
    Files.writeString(dir.resolve(name + ".expected"), exp.toString());
  }
}
