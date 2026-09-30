import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.packed.BlockPackedReaderIterator;
import org.apache.lucene.util.packed.BlockPackedWriter;
import org.apache.lucene.util.packed.GrowableWriter;
import org.apache.lucene.util.packed.MonotonicBlockPackedReader;
import org.apache.lucene.util.packed.MonotonicBlockPackedWriter;
import org.apache.lucene.util.packed.PackedDataInput;
import org.apache.lucene.util.packed.PackedDataOutput;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;
import org.apache.lucene.util.packed.PagedGrowableWriter;
import org.apache.lucene.util.packed.PagedMutable;

/**
 * Differential fixtures for the packed-ints family (`org.apache.lucene.util.packed`), both the
 * in-memory arrays and the serialized forms.
 *
 * <p>Writes `packed_ints/ops.txt`: a script of operations and what Lucene 10.5.0 made of them,
 * which the Rust test replays line by line. Each record starts with a header line naming the
 * structure and its parameters; operation lines follow; `expect`-style lines carry Lucene's
 * results (backing `long[]` words read by reflection, per-page widths, serialized bytes in hex).
 * Every serialized form is read back through Lucene's own reader before it is written out.
 *
 * <p>Deterministic: fixed seeds. Usage: java GenPackedInts <outdir>
 */
public class GenPackedInts {
  static final StringBuilder OUT = new StringBuilder();

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String hex(byte[] b, int len) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) sb.append(String.format("%02x", b[i] & 0xFF));
    return len == 0 ? "-" : sb.toString();
  }

  static String longs(long[] v, int from, int to) {
    StringBuilder sb = new StringBuilder();
    for (int i = from; i < to; i++) {
      if (i > from) sb.append(' ');
      sb.append(v[i]);
    }
    return sb.toString();
  }

  static long rnd(Random r, int bpv) {
    long v = r.nextLong();
    return bpv == 64 ? v : v & PackedInts.maxValue(bpv);
  }

  static Object field(Object o, String name) throws Exception {
    Class<?> c = o.getClass();
    while (c != null) {
      try {
        Field f = c.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(o);
      } catch (NoSuchFieldException e) {
        c = c.getSuperclass();
      }
    }
    throw new NoSuchFieldException(name);
  }

  static void mutables(Random r) throws Exception {
    for (PackedInts.Format format : PackedInts.Format.values()) {
      for (int bpv = 1; bpv <= 64; bpv++) {
        if (!format.isSupported(bpv)) continue;
        int valueCount = r.nextInt(400) + 1;
        PackedInts.Mutable m = PackedInts.getMutable(valueCount, bpv, format);
        line("mutable", format.getId(), bpv, valueCount);
        for (int op = 0; op < 60; op++) {
          int kind = r.nextInt(10);
          if (kind < 5) {
            int i = r.nextInt(valueCount);
            long v = rnd(r, bpv);
            m.set(i, v);
            line("set", i, v);
          } else if (kind < 7) {
            int i = r.nextInt(valueCount);
            int len = 1 + r.nextInt(Math.min(200, valueCount));
            long[] arr = new long[len];
            for (int k = 0; k < len; k++) arr[k] = rnd(r, bpv);
            int got = m.set(i, arr, 0, len);
            line("bset", i, got, longs(arr, 0, len));
          } else if (kind < 9) {
            int from = r.nextInt(valueCount);
            int to = from + r.nextInt(valueCount - from + 1);
            long v = rnd(r, bpv);
            m.fill(from, to, v);
            line("fill", from, to, v);
          } else {
            int i = r.nextInt(valueCount);
            int len = 1 + r.nextInt(valueCount);
            long[] arr = new long[len];
            int got = m.get(i, arr, 0, len);
            line("bget", i, len, got, longs(arr, 0, got));
          }
        }
        long[] blocks = (long[]) field(m, "blocks");
        line("blocks", longs(blocks, 0, blocks.length));
        long[] all = new long[valueCount];
        for (int i = 0; i < valueCount; i++) all[i] = m.get(i);
        line("values", longs(all, 0, valueCount));
        line("end");
      }
    }
  }

  static void growable(Random r) {
    float[] ratios = {PackedInts.COMPACT, PackedInts.DEFAULT, PackedInts.FAST, PackedInts.FASTEST};
    for (int c = 0; c < 12; c++) {
      int start = 1 + r.nextInt(10);
      int valueCount = 1 + r.nextInt(300);
      float ratio = ratios[c % ratios.length];
      GrowableWriter w = new GrowableWriter(start, valueCount, ratio);
      line("growable", start, valueCount, Float.floatToIntBits(ratio), w.getBitsPerValue());
      for (int op = 0; op < 40; op++) {
        int bits = 1 + r.nextInt(op < 30 ? 20 : 64);
        long v = bits == 64 ? r.nextLong() : r.nextLong() & PackedInts.maxValue(bits);
        int kind = r.nextInt(8);
        if (kind < 6) {
          int i = r.nextInt(valueCount);
          w.set(i, v);
          line("set", i, v, w.getBitsPerValue());
        } else if (kind == 6) {
          int from = r.nextInt(valueCount);
          int to = from + r.nextInt(valueCount - from + 1);
          w.fill(from, to, v);
          line("fill", from, to, v, w.getBitsPerValue());
        } else {
          int newSize = 1 + r.nextInt(400);
          w = w.resize(newSize);
          valueCount = newSize;
          line("resize", newSize, w.getBitsPerValue());
        }
      }
      long[] all = new long[valueCount];
      for (int i = 0; i < valueCount; i++) all[i] = w.get(i);
      line("values", longs(all, 0, valueCount));
      line("end");
    }
  }

  static void paged(Random r) throws Exception {
    float[] ratios = {PackedInts.COMPACT, PackedInts.DEFAULT, PackedInts.FASTEST};
    for (int c = 0; c < 12; c++) {
      boolean growable = c % 2 == 1;
      long size = 1 + r.nextInt(3000);
      int pageSize = 64 << r.nextInt(4);
      int bpv = 1 + r.nextInt(growable ? 8 : 40);
      float ratio = ratios[c % ratios.length];
      line(growable ? "pagedgrowable" : "paged", size, pageSize, bpv, Float.floatToIntBits(ratio));
      Object p =
          growable
              ? new PagedGrowableWriter(size, pageSize, bpv, ratio)
              : new PagedMutable(size, pageSize, bpv, ratio);
      for (int op = 0; op < 80; op++) {
        int kind = r.nextInt(20);
        if (kind < 18) {
          long i = (long) (r.nextDouble() * size);
          int bits = growable ? 1 + r.nextInt(30) : bpv;
          long v = r.nextLong() & PackedInts.maxValue(Math.min(bits, 63));
          if (growable) ((PagedGrowableWriter) p).set(i, v);
          else ((PagedMutable) p).set(i, v);
          line("set", i, v);
        } else if (kind == 18) {
          long newSize = 1 + r.nextInt(3000);
          p = growable ? ((PagedGrowableWriter) p).resize(newSize) : ((PagedMutable) p).resize(newSize);
          size = newSize;
          line("resize", newSize);
        } else {
          long min = size + r.nextInt(500);
          p = growable ? ((PagedGrowableWriter) p).grow(min) : ((PagedMutable) p).grow(min);
          size = growable ? ((PagedGrowableWriter) p).size() : ((PagedMutable) p).size();
          line("grow", min, size);
        }
      }
      PackedInts.Mutable[] subs = (PackedInts.Mutable[]) field(p, "subMutables");
      StringBuilder widths = new StringBuilder();
      for (int i = 0; i < subs.length; i++) {
        if (i > 0) widths.append(' ');
        widths.append(subs[i].getBitsPerValue());
      }
      line("pagebits", widths);
      long[] all = new long[(int) size];
      for (int i = 0; i < size; i++)
        all[i] = growable ? ((PagedGrowableWriter) p).get(i) : ((PagedMutable) p).get(i);
      line("values", longs(all, 0, (int) size));
      line("end");
    }
  }

  static void longValues(Random r) throws Exception {
    float[] ratios = {PackedInts.COMPACT, PackedInts.DEFAULT, PackedInts.FASTEST};
    String[] kinds = {"packed", "delta", "monotonic"};
    for (int c = 0; c < 18; c++) {
      String kind = kinds[c % 3];
      int pageSize = 64 << r.nextInt(3);
      float ratio = ratios[(c / 3) % ratios.length];
      PackedLongValues.Builder b =
          switch (kind) {
            case "packed" -> PackedLongValues.packedBuilder(pageSize, ratio);
            case "delta" -> PackedLongValues.deltaPackedBuilder(pageSize, ratio);
            default -> PackedLongValues.monotonicBuilder(pageSize, ratio);
          };
      int n = r.nextInt(2000);
      long[] vals = new long[n];
      int shape = c % 6;
      long acc = r.nextInt(1000);
      for (int i = 0; i < n; i++) {
        switch (shape) {
          case 0 -> vals[i] = r.nextLong();
          case 1 -> vals[i] = r.nextInt(1 << (1 + r.nextInt(20)));
          case 2 -> vals[i] = (acc += r.nextInt(50));
          case 3 -> vals[i] = i / 100 == 3 ? 0 : 1000 + r.nextInt(64);
          case 4 -> vals[i] = (acc += 1 + r.nextInt(3)) * 1_000_000_007L;
          default -> vals[i] = r.nextBoolean() ? Long.MIN_VALUE + r.nextInt(10) : Long.MAX_VALUE - r.nextInt(10);
        }
        b.add(vals[i]);
      }
      PackedLongValues built = b.build();
      line("longvalues", kind, pageSize, Float.floatToIntBits(ratio));
      line("add", n == 0 ? "-" : longs(vals, 0, n));
      PackedInts.Reader[] pages = (PackedInts.Reader[]) field(built, "values");
      StringBuilder widths = new StringBuilder();
      for (int i = 0; i < pages.length; i++) {
        if (i > 0) widths.append(' ');
        widths.append(pages[i] instanceof PackedInts.Mutable mm ? mm.getBitsPerValue() : 0);
      }
      line("pagebits", pages.length == 0 ? "-" : widths);
      if (!kind.equals("packed")) {
        long[] mins = (long[]) field(built, "mins");
        line("mins", mins.length == 0 ? "-" : longs(mins, 0, mins.length));
      }
      if (kind.equals("monotonic")) {
        float[] avgs = (float[]) field(built, "averages");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < avgs.length; i++) {
          if (i > 0) sb.append(' ');
          sb.append(Float.floatToIntBits(avgs[i]));
        }
        line("avgbits", avgs.length == 0 ? "-" : sb);
      }
      PackedLongValues.Iterator it = built.iterator();
      for (int i = 0; i < n; i++) {
        long got = it.next();
        if (got != vals[i] || built.get(i) != vals[i]) throw new AssertionError("longvalues " + i);
      }
      line("end");
    }
  }

  static void writers(Random r) throws IOException {
    for (PackedInts.Format format : PackedInts.Format.values()) {
      for (int bpv = 1; bpv <= 64; bpv++) {
        if (!format.isSupported(bpv)) continue;
        int valueCount = r.nextInt(300);
        int written = valueCount == 0 ? 0 : r.nextInt(valueCount + 1);
        int mem = r.nextInt(3) == 0 ? 0 : r.nextInt(4096);
        ByteBuffersDataOutput out = new ByteBuffersDataOutput();
        PackedInts.Writer w = PackedInts.getWriterNoHeader(out, format, valueCount, bpv, mem);
        long[] vals = new long[valueCount];
        for (int i = 0; i < written; i++) {
          vals[i] = rnd(r, bpv);
          w.add(vals[i]);
        }
        w.finish();
        byte[] bytes = out.toArrayCopy();
        // round trip through Lucene's own iterator
        PackedInts.ReaderIterator it =
            PackedInts.getReaderIteratorNoHeader(
                new ByteArrayDataInput(bytes), format, PackedInts.VERSION_CURRENT, valueCount, bpv, mem);
        for (int i = 0; i < valueCount; i++) {
          if (it.next() != vals[i]) throw new AssertionError("writer round trip " + bpv);
        }
        line("writer", format.getId(), bpv, valueCount, mem, written);
        line("add", written == 0 ? "-" : longs(vals, 0, written));
        line("bytes", hex(bytes, bytes.length));
        line("end");
      }
    }
  }

  static void directSingleBlock(Random r) throws Exception {
    Class<?> cls = Class.forName("org.apache.lucene.util.packed.DirectPacked64SingleBlockReader");
    Constructor<?> ctor = cls.getDeclaredConstructor(int.class, int.class, IndexInput.class);
    ctor.setAccessible(true);
    for (int bpv : new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 21, 32}) {
      int valueCount = 1 + r.nextInt(200);
      PackedInts.Mutable m = PackedInts.getMutable(valueCount, bpv, PackedInts.Format.PACKED_SINGLE_BLOCK);
      for (int i = 0; i < valueCount; i++) m.set(i, rnd(r, bpv));
      long[] blocks = (long[]) field(m, "blocks");
      try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
        try (IndexOutput o = dir.createOutput("f", IOContext.DEFAULT)) {
          o.writeByte((byte) 0x5A); // a leading byte, so the reader starts mid-file
          for (long b : blocks) o.writeLong(b);
        }
        byte[] bytes;
        long[] got = new long[valueCount];
        try (IndexInput in = dir.openInput("f", IOContext.DEFAULT)) {
          bytes = new byte[(int) in.length()];
          in.readBytes(bytes, 0, bytes.length);
          in.seek(1);
          PackedInts.Reader reader = (PackedInts.Reader) ctor.newInstance(bpv, valueCount, in);
          for (int i = 0; i < valueCount; i++) {
            got[i] = reader.get(i);
            if (got[i] != m.get(i)) throw new AssertionError("direct single block " + bpv);
          }
        }
        line("directsingleblock", bpv, valueCount);
        line("bytes", hex(bytes, bytes.length));
        line("values", longs(got, 0, valueCount));
        line("end");
      }
    }
  }

  static void packedData(Random r) throws IOException {
    for (int c = 0; c < 6; c++) {
      ByteBuffersDataOutput out = new ByteBuffersDataOutput();
      PackedDataOutput pdo = new PackedDataOutput(out);
      int n = 1 + r.nextInt(200);
      List<long[]> items = new ArrayList<>();
      StringBuilder ops = new StringBuilder();
      for (int i = 0; i < n; i++) {
        int bpv = 1 + r.nextInt(64);
        long v = rnd(r, bpv);
        boolean flush = r.nextInt(10) == 0;
        pdo.writeLong(v, bpv);
        if (flush) pdo.flush();
        items.add(new long[] {v, bpv, flush ? 1 : 0});
        if (i > 0) ops.append(' ');
        ops.append(v).append(':').append(bpv).append(':').append(flush ? 1 : 0);
      }
      pdo.flush();
      byte[] bytes = out.toArrayCopy();
      PackedDataInput pdi = new PackedDataInput(new ByteArrayDataInput(bytes));
      for (long[] it : items) {
        if (pdi.readLong((int) it[1]) != it[0]) throw new AssertionError("packed data");
        if (it[2] == 1) pdi.skipToNextByte();
      }
      line("packeddata");
      line("items", ops);
      line("bytes", hex(bytes, bytes.length));
      line("end");
    }
  }

  static long[] shaped(Random r, int n, int shape) {
    long[] v = new long[n];
    long acc = r.nextInt(100);
    for (int i = 0; i < n; i++) {
      switch (shape) {
        case 0 -> v[i] = (acc += r.nextInt(1000));
        case 1 -> v[i] = 5L * i + 17;
        case 2 -> v[i] = (acc += r.nextInt(3)) + (r.nextInt(20) == 0 ? 100_000 : 0);
        case 3 -> v[i] = r.nextInt(1 << 20);
        default -> v[i] = 123456789L;
      }
    }
    return v;
  }

  static void monotonic(Random r) throws IOException {
    for (int c = 0; c < 15; c++) {
      int blockSize = 64 << r.nextInt(4);
      int n = r.nextInt(1500);
      long[] vals = shaped(r, n, c % 5);
      ByteBuffersDirectory dir = new ByteBuffersDirectory();
      try (IndexOutput o = dir.createOutput("m", IOContext.DEFAULT)) {
        MonotonicBlockPackedWriter w = new MonotonicBlockPackedWriter(o, blockSize);
        for (long v : vals) w.add(v);
        w.finish();
      }
      byte[] bytes;
      try (IndexInput in = dir.openInput("m", IOContext.DEFAULT)) {
        bytes = new byte[(int) in.length()];
        in.readBytes(bytes, 0, bytes.length);
        in.seek(0);
        MonotonicBlockPackedReader rd =
            MonotonicBlockPackedReader.of(in, PackedInts.VERSION_CURRENT, blockSize, n);
        for (int i = 0; i < n; i++) {
          if (rd.get(i) != vals[i]) throw new AssertionError("monotonic " + i);
        }
      }
      line("monotonic", blockSize);
      line("add", n == 0 ? "-" : longs(vals, 0, n));
      line("bytes", hex(bytes, bytes.length));
      line("end");
    }
  }

  static void blockPacked(Random r) throws IOException {
    for (int c = 0; c < 12; c++) {
      int blockSize = 64 << r.nextInt(3);
      int n = r.nextInt(1500);
      long[] vals = new long[n];
      for (int i = 0; i < n; i++) {
        vals[i] =
            switch (c % 4) {
              case 0 -> r.nextLong();
              case 1 -> r.nextInt(1000) - 500;
              case 2 -> 1_000_000 + r.nextInt(16);
              default -> i % 300 < 150 ? 0 : 7;
            };
      }
      ByteBuffersDataOutput out = new ByteBuffersDataOutput();
      BlockPackedWriter w = new BlockPackedWriter(out, blockSize);
      for (long v : vals) w.add(v);
      w.finish();
      byte[] bytes = out.toArrayCopy();
      BlockPackedReaderIterator it =
          new BlockPackedReaderIterator(
              new ByteArrayDataInput(bytes), PackedInts.VERSION_CURRENT, blockSize, n);
      for (int i = 0; i < n; i++) {
        if (it.next() != vals[i]) throw new AssertionError("block packed " + i);
      }
      line("blockpacked", blockSize);
      line("add", n == 0 ? "-" : longs(vals, 0, n));
      line("bytes", hex(bytes, bytes.length));
      line("end");
    }
  }

  static void fastest() {
    float[] ratios = {-1f, 0f, 0.1f, 0.25f, 0.5f, 1f, 3f, 7f, 100f};
    for (float ratio : ratios) {
      StringBuilder sb = new StringBuilder();
      for (int bpv = 1; bpv <= 64; bpv++) {
        if (bpv > 1) sb.append(' ');
        sb.append(PackedInts.fastestFormatAndBits(100, bpv, ratio).bitsPerValue());
      }
      line("fastest", Float.floatToIntBits(ratio), sb);
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("packed_ints");
    Files.createDirectories(out);
    Random r = new Random(0x9AC4ED_2026_0930L);
    fastest();
    mutables(r);
    growable(r);
    paged(r);
    longValues(r);
    writers(r);
    directSingleBlock(r);
    packedData(r);
    monotonic(r);
    blockPacked(r);
    Files.writeString(out.resolve("ops.txt"), OUT.toString());
  }
}
