import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.SegmentInfoFormat;
import org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

/** Actual cranky wrapper, real .si files and the same batches as the Rust twin. */
public class CrankySegmentInfoMicro {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static volatile long sink;
  static long hash(long digest, long value) { return (digest ^ value) * 0x100000001b3L; }
  static SegmentInfo info(Directory dir, String name) {
    var info = new SegmentInfo(dir, Version.fromBits(10, 5, 0), null, name, 7,
        false, false, null, Map.of(), ID.clone(), Map.of(), null);
    info.setFiles(Set.of());
    return info;
  }
  static long batch(SegmentInfoFormat format, Random random, Directory base, boolean write) throws Exception {
    random.setSeed(42);
    long digest = 0xcbf29ce484222325L;
    try (var directory = new ByteBuffersDirectory()) {
      for (int op = 0; op < 256; op++) {
        if (write) {
          String name = "_s" + op;
          var info = info(directory, name);
          try {
            format.write(directory, info, IOContext.DEFAULT);
            digest = hash(digest, info.files().size());
            try (var input = directory.openInput(name + ".si", IOContext.DEFAULT)) {
              while (input.getFilePointer() < input.length()) digest = hash(digest, input.readByte() & 255);
            }
          } catch (IOException e) {
            if (!e.getMessage().equals("Fake IOException from SegmentInfoFormat.write()")) throw e;
            if (!info.files().isEmpty()) throw new AssertionError("mutated files on injection");
            digest = hash(digest, -1L);
          }
        } else {
          var read = format.read(base, "_base", ID, IOContext.DEFAULT);
          digest = hash(hash(digest, read.maxDoc()), read.files().size());
        }
      }
    }
    sink = digest;
    return digest;
  }
  public static void main(String[] args) throws Exception {
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    var constructor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankySegmentInfoFormat")
        .getDeclaredConstructor(SegmentInfoFormat.class, Random.class);
    constructor.setAccessible(true);
    Random random = new Random(42);
    var delegate = new Lucene99SegmentInfoFormat();
    var format = (SegmentInfoFormat) constructor.newInstance(delegate, random);
    long ownedBytes = org.apache.lucene.util.RamUsageEstimator.shallowSizeOf(format)
        + org.apache.lucene.util.RamUsageEstimator.shallowSizeOf(delegate)
        + org.apache.lucene.util.RamUsageEstimator.shallowSizeOf(random)
        + org.apache.lucene.util.RamUsageEstimator.shallowSizeOf(new java.util.concurrent.atomic.AtomicLong());
    System.out.printf("cranky_segment_info_owned_bytes\t%d%n", ownedBytes);
    try (var base = new ByteBuffersDirectory()) {
      new Lucene99SegmentInfoFormat().write(base, info(base, "_base"), IOContext.DEFAULT);
      for (boolean write : new boolean[] {false, true}) {
        String name = write ? "cranky_segment_info_write" : "cranky_segment_info_read";
        long expected = batch(format, random, base, write);
        for (int phase = 0; phase < 2; phase++) {
          long budget = phase == 0 ? warmup : measure;
          long start = System.nanoTime(), operations = 0;
          do {
            if (batch(format, random, base, write) != expected) throw new AssertionError("unstable digest");
            operations += 256;
          } while (System.nanoTime() - start < budget * 1000000);
          if (phase == 1) System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name, (double)(System.nanoTime() - start) / operations, operations);
        }
        System.out.printf("#check\t%s\t%s\t256%n", name, Long.toUnsignedString(expected));
      }
    }
  }
}
