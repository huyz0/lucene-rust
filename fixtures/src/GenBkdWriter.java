import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;
import org.apache.lucene.codecs.MutablePointTree;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.bkd.BKDConfig;
import org.apache.lucene.util.bkd.BKDReader;
import org.apache.lucene.util.bkd.BKDWriter;

/**
 * Differential fixtures for {@link BKDWriter}: all three entry points (the flush path over a
 * {@link MutablePointTree}, {@code add} + {@code finish} with and without spilling to temp files,
 * and the one-dimensional {@code merge}) over point sets generated from a formula both sides share
 * (a 64-bit LCG; see {@code Points}), so nothing large is checked in.
 *
 * <p>Writes `bkd_writer/cases.txt`: per case its parameters, then the length and CRC32 of the
 * meta, index and data bytes BKDWriter wrote (the data bytes in hex too, for small cases).
 *
 * <p>Usage: java GenBkdWriter <outdir>
 */
public class GenBkdWriter {
  /** The shared point generator: an LCG, one point per call. */
  static final class Points {
    long seed;
    final int numDims, bpd, card, multi;
    final boolean shuffledDocs;
    final long docStride;
    int doc = 0;

    Points(long seed, int numDims, int bpd, int card, int multi, boolean shuffledDocs, long docStride) {
      this.seed = seed;
      this.numDims = numDims;
      this.bpd = bpd;
      this.card = card;
      this.multi = multi;
      this.shuffledDocs = shuffledDocs;
      this.docStride = docStride;
    }

    long next() {
      seed = seed * 6364136223846793005L + 1442695040888963407L;
      return seed;
    }

    int nextInt(int bound) {
      return (int) Long.remainderUnsigned(next() >>> 16, bound);
    }

    /** Fills {@code value} and returns the doc of the next point. */
    int point(byte[] value) {
      for (int d = 0; d < numDims; d++) {
        int v = nextInt(card);
        for (int b = 0; b < bpd; b++) {
          int fromEnd = bpd - 1 - b;
          value[d * bpd + b] = fromEnd < 4 ? (byte) (v >>> (fromEnd * 8)) : (byte) (d * 17 + 1);
        }
      }
      int thisDoc = doc;
      if (multi == 0 || nextInt(multi) != 0) {
        doc++;
      }
      if (shuffledDocs) {
        return (int) ((thisDoc * 7919L) % 100003L * docStride);
      }
      return (int) (thisDoc * docStride);
    }
  }

  /**
   * A {@link MutablePointTree} shaped like {@code PointValuesWriter}'s: the points stay put and an
   * ordinal array is permuted, so a {@code BytesRef} a sorter holds as its pivot stays valid.
   */
  static final class ArrayTree extends MutablePointTree {
    final int[] docs;
    final byte[] values;
    final int stride;
    final int[] ords;
    final int[] temp;

    ArrayTree(int[] docs, byte[] values, int stride) {
      this.docs = docs;
      this.values = values;
      this.stride = stride;
      this.ords = new int[docs.length];
      for (int i = 0; i < ords.length; i++) ords[i] = i;
      this.temp = new int[docs.length];
    }

    @Override
    public void getValue(int i, BytesRef packedValue) {
      packedValue.bytes = values;
      packedValue.offset = ords[i] * stride;
      packedValue.length = stride;
    }

    @Override
    public byte getByteAt(int i, int k) {
      return values[ords[i] * stride + k];
    }

    @Override
    public int getDocID(int i) {
      return docs[ords[i]];
    }

    @Override
    public void swap(int i, int j) {
      int t = ords[i];
      ords[i] = ords[j];
      ords[j] = t;
    }

    @Override
    public void save(int i, int j) {
      temp[j] = ords[i];
    }

    @Override
    public void restore(int i, int j) {
      System.arraycopy(temp, i, ords, i, j - i);
    }

    @Override
    public long size() {
      return docs.length;
    }

    @Override
    public void visitDocValues(PointValues.IntersectVisitor visitor) throws java.io.IOException {
      byte[] v = new byte[stride];
      for (int i = 0; i < docs.length; i++) {
        System.arraycopy(values, ords[i] * stride, v, 0, stride);
        visitor.visit(docs[ords[i]], v);
      }
    }
  }

  static String crc(ByteBuffersDataOutput o) {
    byte[] b = o.toArrayCopy();
    CRC32 c = new CRC32();
    c.update(b);
    return b.length + ":" + Long.toHexString(c.getValue());
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xff));
    return sb.length() == 0 ? "-" : sb.toString();
  }

  static final class Outputs {
    final ByteBuffersDataOutput meta = new ByteBuffersDataOutput();
    final ByteBuffersDataOutput index = new ByteBuffersDataOutput();
    final ByteBuffersDataOutput data = new ByteBuffersDataOutput();
    final IndexOutput metaOut = new ByteBuffersIndexOutput(meta, "meta", "meta");
    final IndexOutput indexOut = new ByteBuffersIndexOutput(index, "index", "index");
    final IndexOutput dataOut = new ByteBuffersIndexOutput(data, "data", "data");

    String describe(boolean withData) {
      return crc(meta) + " " + crc(index) + " " + crc(data) + " " + (withData ? hex(data.toArrayCopy()) : "-");
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0], "bkd_writer");
    Files.createDirectories(out);
    StringBuilder sb = new StringBuilder();
    // name mode numDims numIndexDims bpd maxLeaf n seed card multi shuffled docStride maxMB
    Object[][] cases = {
      {"flush_1d_int", "flush", 1, 1, 4, 512, 5000, 1L, 1000, 3, false, 1L, 16.0},
      {"flush_1d_long_lowcard", "flush", 1, 1, 8, 100, 3000, 2L, 50, 0, true, 1L, 16.0},
      {"flush_1d_16", "flush", 1, 1, 16, 64, 1200, 3L, 1000000, 2, false, 1L, 16.0},
      {"flush_1d_sparse_docs", "flush", 1, 1, 4, 128, 2000, 4L, 100000, 0, false, 9000L, 16.0},
      {"flush_1d_mid_docs", "flush", 1, 1, 4, 128, 2000, 5L, 100000, 0, true, 40L, 16.0},
      {"flush_2d_ties", "flush", 2, 2, 4, 100, 6000, 6L, 300, 2, false, 1L, 16.0},
      {"flush_2d_small", "flush", 2, 2, 4, 20, 150, 7L, 10, 2, false, 1L, 16.0},
      {"flush_3d", "flush", 3, 3, 4, 50, 4000, 8L, 500, 3, false, 1L, 16.0},
      {"flush_4d_2index", "flush", 4, 2, 4, 80, 3000, 9L, 200, 2, true, 1L, 16.0},
      {"flush_2d_1index", "flush", 2, 1, 8, 100, 2000, 10L, 400, 0, false, 1L, 16.0},
      {"flush_2d_constant", "flush", 2, 2, 4, 100, 1000, 11L, 1, 0, false, 1L, 16.0},
      {"add_2d_heap", "add", 2, 2, 4, 100, 6000, 12L, 300, 2, false, 1L, 16.0},
      {"add_2d_offline", "add", 2, 2, 4, 64, 6000, 13L, 300, 2, false, 1L, 0.002},
      {"add_3d_offline", "add", 3, 3, 4, 64, 5000, 14L, 40, 2, true, 1L, 0.002},
      {"add_4d_2index_offline", "add", 4, 2, 4, 64, 5000, 15L, 30, 3, false, 1L, 0.004},
      {"add_1d_offline", "add", 1, 1, 4, 64, 4000, 16L, 100, 2, false, 1L, 0.001},
      {"merge_1d", "merge", 1, 1, 4, 128, 3000, 17L, 2000, 2, false, 1L, 16.0},
    };
    for (Object[] c : cases) {
      String name = (String) c[0];
      String mode = (String) c[1];
      int numDims = (Integer) c[2], numIndexDims = (Integer) c[3], bpd = (Integer) c[4];
      int maxLeaf = (Integer) c[5], n = (Integer) c[6];
      long seed = (Long) c[7];
      int card = (Integer) c[8], multi = (Integer) c[9];
      boolean shuffled = (Boolean) c[10];
      long docStride = (Long) c[11];
      double maxMB = (Double) c[12];
      BKDConfig config = new BKDConfig(numDims, numIndexDims, bpd, maxLeaf);
      Points gen = new Points(seed, numDims, bpd, card, multi, shuffled, docStride);
      int stride = numDims * bpd;
      int[] docs = new int[n];
      byte[] values = new byte[n * stride];
      byte[] v = new byte[stride];
      int maxDoc = 0;
      for (int i = 0; i < n; i++) {
        docs[i] = gen.point(v);
        System.arraycopy(v, 0, values, i * stride, stride);
        maxDoc = Math.max(maxDoc, docs[i] + 1);
      }
      String result;
      try (Directory tmp = new ByteBuffersDirectory()) {
        Outputs o = new Outputs();
        if (mode.equals("flush")) {
          try (BKDWriter w = new BKDWriter(maxDoc, tmp, "_0", config, maxMB, n)) {
            IORunnable fin =
                w.writeField(o.metaOut, o.indexOut, o.dataOut, "f", new ArrayTree(docs, values, stride));
            fin.run();
          }
        } else if (mode.equals("add")) {
          try (BKDWriter w = new BKDWriter(maxDoc, tmp, "_0", config, maxMB, n)) {
            for (int i = 0; i < n; i++) {
              System.arraycopy(values, i * stride, v, 0, stride);
              w.add(v, docs[i]);
            }
            IORunnable fin = w.finish(o.metaOut, o.indexOut, o.dataOut);
            fin.run();
          }
          if (tmp.listAll().length != 0) throw new AssertionError("temp files left");
        } else {
          // Two segments: the first and second half of the points, each flushed, then merged
          // with doc maps that shift the second segment and delete every 7th doc.
          int half = n / 2;
          List<PointValues> readers = new ArrayList<>();
          List<MergeState.DocMap> maps = new ArrayList<>();
          int[] segMaxDoc = new int[2];
          for (int s = 0; s < 2; s++) {
            int from = s == 0 ? 0 : half, to = s == 0 ? half : n;
            int[] sd = new int[to - from];
            byte[] sv = new byte[(to - from) * stride];
            int segMax = 0;
            for (int i = from; i < to; i++) {
              sd[i - from] = docs[i] - docs[from];
              segMax = Math.max(segMax, sd[i - from] + 1);
            }
            System.arraycopy(values, from * stride, sv, 0, (to - from) * stride);
            segMaxDoc[s] = segMax;
            Outputs so = new Outputs();
            try (BKDWriter w = new BKDWriter(segMax, tmp, "_s" + s, config, maxMB, to - from)) {
              IORunnable fin = w.writeField(so.metaOut, so.indexOut, so.dataOut, "f", new ArrayTree(sd, sv, stride));
              fin.run();
            }
            for (String f : new String[] {"meta", "index", "data"}) {
              ByteBuffersDataOutput b = f.equals("meta") ? so.meta : f.equals("index") ? so.index : so.data;
              try (IndexOutput io = tmp.createOutput("s" + s + f, IOContext.DEFAULT)) {
                io.writeBytes(b.toArrayCopy(), 0, (int) b.size());
              }
            }
            IndexInput mi = tmp.openInput("s" + s + "meta", IOContext.DEFAULT);
            IndexInput ii = tmp.openInput("s" + s + "index", IOContext.DEFAULT);
            IndexInput di = tmp.openInput("s" + s + "data", IOContext.DEFAULT);
            readers.add(new BKDReader(mi, ii, di));
          }
          final int base = segMaxDoc[0];
          maps.add(doc -> doc % 7 == 0 ? -1 : doc);
          maps.add(doc -> doc % 7 == 3 ? -1 : doc + base);
          try (BKDWriter w = new BKDWriter(base + segMaxDoc[1], tmp, "_m", config, maxMB, n)) {
            IORunnable fin = w.merge(o.metaOut, o.indexOut, o.dataOut, maps, readers);
            fin.run();
          }
        }
        result = o.describe(n <= 1200);
      }
      sb.append(name);
      for (int i = 1; i < c.length; i++) sb.append(' ').append(c[i]);
      sb.append(' ').append(maxDoc).append(' ').append(result).append('\n');
    }
    Files.writeString(out.resolve("cases.txt"), sb.toString());
  }
}
