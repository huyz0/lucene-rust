package org.apache.lucene.tests.codecs.cranky;

import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.RamUsageEstimator;

/** Actual CrankyFieldInfosFormat, with the same bounded counting delegate as Rust. */
public class CrankyFieldInfosMicro {
  static volatile long sink;
  static final FieldInfos EMPTY = new FieldInfos(new FieldInfo[0]);
  static final String[] NAMES = {"", "a", "abc", "abcdefg"};
  static final String ERROR = "Fake IOException from FieldInfosFormat.getFieldInfosWriter()";
  static class Counting extends FieldInfosFormat {
    long reads, writes;
    @Override public FieldInfos read(Directory d, SegmentInfo s, String suffix, IOContext c) {
      reads += suffix.length() + 1;
      return new FieldInfos(new FieldInfo[0]);
    }
    @Override public void write(Directory d, SegmentInfo s, String suffix, FieldInfos f, IOContext c) {
      writes += suffix.length() + 1;
    }
  }

  static FieldInfosFormat cranky(Counting delegate, Random random) {
    return new CrankyFieldInfosFormat(delegate, random);
  }

  static long batch(boolean write) throws Exception {
    Random random = new Random(0);
    Counting delegate = new Counting();
    FieldInfosFormat format = cranky(delegate, random);
    long failures = 0;
    for (int i = 0; i < 256; i++) {
      String suffix = NAMES[i % NAMES.length];
      if (write) {
        try { format.write(null, null, suffix, EMPTY, IOContext.DEFAULT); }
        catch (IOException error) {
          if (!ERROR.equals(error.getMessage())) throw new AssertionError(error);
          failures++;
        }
      } else if (format.read(null, null, suffix, IOContext.DEFAULT).size() != 0)
        throw new AssertionError("delegate result");
    }
    return delegate.reads * 17 + delegate.writes * 31 + failures * 101
        + Integer.toUnsignedLong(random.nextInt());
  }

  static long[] run(boolean write, long ms) throws Exception {
    long start = System.nanoTime(), count = 0;
    do { sink = batch(write); count += 256; }
    while (System.nanoTime() - start < ms * 1_000_000L);
    return new long[] {System.nanoTime() - start, count};
  }

  public static void main(String[] args) throws Exception {
    Counting delegate = new Counting();
    Random random = new Random(0);
    FieldInfosFormat format = cranky(delegate, random);
    long owned = RamUsageEstimator.shallowSizeOf(format) + RamUsageEstimator.shallowSizeOf(delegate)
        + RamUsageEstimator.shallowSizeOf(random)
        + RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
    System.out.printf("#memory\tcranky_field_infos\t%d%n", owned);
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    for (boolean write : new boolean[] {false, true}) {
      String name = write ? "cranky_field_infos_write" : "cranky_field_infos_read";
      System.out.printf("#check\t%s\t%d\t256%n", name, batch(write));
      run(write, warmup);
      long[] result = run(write, measure);
      System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name,
          (double) result[0] / result[1], result[1]);
    }
  }
}
