import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Map;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Actual assertion-enabled Java wrapper: 4096-bit check and 65-bit real write. */
public class AssertingLiveDocsMicro {
  static volatile long sink;
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static AssertingLiveDocsFormat format(LiveDocsFormat delegate) throws Exception {
    AssertingLiveDocsFormat format = new AssertingLiveDocsFormat();
    var field = AssertingLiveDocsFormat.class.getDeclaredField("in");
    field.setAccessible(true);
    field.set(format, delegate);
    return format;
  }
  static class Counter extends LiveDocsFormat {
    long count;
    @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) { throw new AssertionError(); }
    @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info, int count, IOContext context) { this.count++; }
    @Override public void files(SegmentCommitInfo info, Collection<String> files) { throw new AssertionError(); }
  }
  static long batch(boolean real, Counter counter, AssertingLiveDocsFormat guard, AssertingLiveDocsFormat writer) throws Exception {
    counter.count = 0;
    int length = real ? 65 : 4096;
    FixedBitSet bits = new FixedBitSet(length);
    int deleted = 0;
    for (int i = 0; i < length; i++) { if (i % 3 == 0) deleted++; else bits.set(i); }
    try (Directory empty = new ByteBuffersDirectory()) {
      SegmentInfo si = new SegmentInfo(empty, Version.LATEST, Version.LATEST, "_ald", length,
          false, false, null, Map.of(), ID, Map.of(), null);
      SegmentCommitInfo info = new SegmentCommitInfo(si, 0, 0, -1, -1, -1, null);
      long digest = 0;
      for (int i = 0; i < 256; i++) {
        if (real) {
          try (Directory dir = new ByteBuffersDirectory()) {
            writer.writeLiveDocs(bits, dir, info, deleted, IOContext.DEFAULT);
            try (var input = dir.openInput("_ald_1.liv", IOContext.DEFAULT)) {
              while (input.getFilePointer() < input.length()) digest += input.readByte() & 255;
            }
          }
        } else { guard.writeLiveDocs(bits, empty, info, deleted, IOContext.DEFAULT); }
      }
      if (!real) digest += counter.count * 31;
      return digest;
    }
  }
  static long[] run(boolean real, long ms, Counter counter, AssertingLiveDocsFormat guard, AssertingLiveDocsFormat writer) throws Exception {
    long start = System.nanoTime(), batches = 0;
    do { sink = batch(real, counter, guard, writer); batches++; } while (System.nanoTime() - start < ms * 1_000_000L);
    return new long[] {batches, System.nanoTime() - start};
  }
  public static void main(String[] args) throws Exception {
    if (!AssertingLiveDocsMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Counter counter = new Counter();
    AssertingLiveDocsFormat guard = format(counter);
    AssertingLiveDocsFormat writer = format(new Lucene90LiveDocsFormat());
    // Requested format wrapper only; excludes delegate, input bits, directory
    // and transient output allocations, matching Rust's generic wrapper size.
    System.out.printf("#memory\tasserting_live_docs_wrapper\t%d%n", RamUsageEstimator.shallowSizeOf(writer));
    for (boolean real : new boolean[] {false, true}) {
      String name = real ? "asserting_live_docs_write" : "asserting_live_docs_guard";
      System.out.printf("#check\t%s\t%d\t256%n", name, batch(real, counter, guard, writer));
      run(real, 1000, counter, guard, writer);
      long[] result = run(real, 1500, counter, guard, writer);
      long operations = result[0] * 256;
      System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result[1] / operations, operations);
    }
  }
}
