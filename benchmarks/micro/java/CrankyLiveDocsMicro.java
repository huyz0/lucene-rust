import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.cranky.CrankyCodec;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Actual CrankyCodec live-docs wrapper: guard-only and real .liv writes. */
public class CrankyLiveDocsMicro {
  static volatile long sink;
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static LiveDocsFormat cranky(LiveDocsFormat liveDocsDelegate, Random random) {
    return new CrankyCodec(new FilterCodec("LiveDocsMicro", Codec.getDefault()) {
      @Override public LiveDocsFormat liveDocsFormat() { return liveDocsDelegate; }
    }, random).liveDocsFormat();
  }
  static class Counter extends LiveDocsFormat {
    long count;
    @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) { throw new AssertionError(); }
    @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info, int count, IOContext context) { this.count++; }
    @Override public void files(SegmentCommitInfo info, Collection<String> files) { throw new AssertionError(); }
  }
  static long batch(boolean real) throws Exception {
    Random random = new Random(42);
    Counter counter = new Counter();
    LiveDocsFormat guard = cranky(counter, random);
    LiveDocsFormat writer = cranky(new Lucene90LiveDocsFormat(), random);
    FixedBitSet bits = new FixedBitSet(65);
    bits.set(0, 64);
    try (Directory emptyDir = new ByteBuffersDirectory()) {
      SegmentInfo si = new SegmentInfo(emptyDir, Version.LATEST, Version.LATEST, "_0", 65,
          false, false, Codec.getDefault(), Map.of(), ID, Map.of(), null);
      SegmentCommitInfo info = new SegmentCommitInfo(si, 0, 0, -1, -1, -1, null);
      long digest = 0;
      for (int i = 0; i < 256; i++) {
        if (real) {
          try (Directory dir = new ByteBuffersDirectory()) {
            try {
              writer.writeLiveDocs(bits, dir, info, 1, IOContext.DEFAULT);
              try (var input = dir.openInput("_0_1.liv", IOContext.DEFAULT)) {
                while (input.getFilePointer() < input.length()) digest += input.readByte() & 255;
              }
            } catch (IOException error) {
              if (!error.getMessage().equals("Fake IOException from LiveDocsFormat.writeLiveDocs()")) throw error;
              digest += 7;
            }
          }
        } else {
          try { guard.writeLiveDocs(bits, emptyDir, info, 1, IOContext.DEFAULT); }
          catch (IOException error) {
            if (!error.getMessage().equals("Fake IOException from LiveDocsFormat.writeLiveDocs()")) throw error;
            digest++;
          }
        }
      }
      if (!real) digest += counter.count * 31;
      return digest;
    }
  }
  static long[] run(boolean real, long ms) throws Exception {
    long start = System.nanoTime(), batches = 0;
    do { sink = batch(real); batches++; } while (System.nanoTime() - start < ms * 1_000_000L);
    return new long[] {batches, System.nanoTime() - start};
  }
  public static void main(String[] args) throws Exception {
    Random random = new Random(42);
    LiveDocsFormat format = cranky(new Lucene90LiveDocsFormat(), random);
    // Wrapper + shared Random + its AtomicLong once. Excludes delegate, bits,
    // transient directories/outputs, and allocator metadata on both sides.
    long owned = RamUsageEstimator.shallowSizeOf(format) + RamUsageEstimator.shallowSizeOf(random)
        + RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
    System.out.printf("#memory\tcranky_live_docs_wrapper\t%d%n", owned);
    for (boolean real : new boolean[] {false, true}) {
      String name = real ? "cranky_live_docs_write" : "cranky_live_docs_guard";
      System.out.printf("#check\t%s\t%d\t256%n", name, batch(real));
      run(real, 1000);
      long[] result = run(real, 1500);
      long operations = result[0] * 256;
      System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result[1] / operations, operations);
    }
  }
}
