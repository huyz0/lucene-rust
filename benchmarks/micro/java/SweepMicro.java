import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.GroupVIntUtil;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * Java side of the per-area component sweep. Each {@code <bench>} argument selects a group of
 * cases; the Rust side is {@code benchmarks/rust-runner/src/micro.rs}, which implements the same
 * group names and case names over the same generated inputs (or the same corpus directory), so
 * {@code scripts/bench-micro.sh --bench <bench>} can join the two on case name.
 *
 * <p>Every timed operation performs a <em>batch</em> of work (one whole decode pass, a thousand
 * lookups, ...) and reports nanoseconds per unit of work. That keeps the harness's own dispatch --
 * one megamorphic {@link Op#run} call per batch -- out of the per-unit figure on both sides.
 *
 * <p>Emits TSV {@code case<TAB>ns_per_unit<TAB>units} on stdout. Memory cases emit bytes in the
 * same column, so the report's {@code java/rust} ratio reads the same way: above 1.0, Rust uses
 * less.
 *
 * <p>Usage: {@code SweepMicro <bench> [index-dir]}
 */
public final class SweepMicro {

  /** One batch of work; returns the number of units it performed. */
  interface Op {
    long run() throws IOException;
  }

  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;

  static void measure(String name, Op op) throws IOException {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) {
      System.err.print("");
    }
    return units;
  }

  /** xorshift64, identical on the Rust side. */
  static final class Rng {
    long s;

    Rng(long seed) {
      s = seed;
    }

    long next() {
      s ^= s << 13;
      s ^= s >>> 7;
      s ^= s << 17;
      return s;
    }
  }

  public static void main(String[] args) throws Exception {
    String bench = args[0];
    String index = args.length > 1 ? args[1] : null;
    switch (bench) {
      case "vint" -> vint();
      case "bitset" -> bitset();
      case "lz4" -> lz4();
      case "direct_monotonic" -> directMonotonic();
      case "checksum" -> checksum();
      case "analysis" -> analysis();
      case "vectors" -> vectors();
      case "quantized" -> quantized();
      case "fst_build" -> fstBuild();
      case "bytes_ref_hash" -> bytesRefHash();
      case "bkd_build" -> bkdBuild();
      case "postings_adv" -> withLeaf(index, SweepMicro::postingsAdvance);
      case "postings_freq" -> withLeaf(index, SweepMicro::postingsFreq);
      case "positions" -> withLeaf(index, SweepMicro::positions);
      case "term_seek" -> withLeaf(index, SweepMicro::termSeek);
      case "doc_values" -> withLeaf(index, SweepMicro::docValues);
      case "norms" -> withLeaf(index, SweepMicro::norms);
      case "points" -> withLeaf(index, SweepMicro::points);
      case "memory" -> memory(index);
      default -> {
        System.err.println("SweepMicro: unknown bench " + bench);
        System.exit(2);
      }
    }
  }

  interface LeafBench {
    void run(LeafReader leaf) throws IOException;
  }

  static void withLeaf(String index, LeafBench b) throws IOException {
    try (MMapDirectory dir = new MMapDirectory(Paths.get(index));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      // The merged corpus is one segment; the sweep's per-area cases measure a
      // codec component, so one leaf is the unit on both sides.
      b.run(reader.leaves().get(0).reader());
    }
  }

  // ---------------------------------------------------------------- primitives

  /** Mixed-length values: a random value shifted right by a random amount. */
  static int[] vintValues(int n) {
    Rng r = new Rng(0x1234_5678_9ABC_DEF1L);
    int[] v = new int[n];
    for (int i = 0; i < n; i++) {
      long x = r.next();
      v[i] = (int) ((x >>> 33) >>> (int) ((x & 0xFFFF) % 31));
    }
    return v;
  }

  static long[] vlongValues(int n) {
    Rng r = new Rng(0x0FED_CBA9_8765_4321L);
    long[] v = new long[n];
    for (int i = 0; i < n; i++) {
      long x = r.next();
      v[i] = (x >>> 1) >>> (int) ((x & 0xFFFF) % 63);
    }
    return v;
  }

  /**
   * Varint decode through an mmap'd {@link IndexInput}: the input type every codec actually reads
   * through (a {@code MemorySegmentIndexInput}), not a byte-array wrapper Lucene never decodes
   * postings from.
   */
  static void vint() throws IOException {
    final int n = 1 << 20;
    Path tmp = Files.createTempDirectory("sweep-vint");
    try (Directory d = new MMapDirectory(tmp)) {
      int[] ints = vintValues(n);
      long[] longs = vlongValues(n);
      try (IndexOutput o = d.createOutput("vint", IOContext.DEFAULT)) {
        for (int x : ints) o.writeVInt(x);
      }
      try (IndexOutput o = d.createOutput("vlong", IOContext.DEFAULT)) {
        for (long x : longs) o.writeVLong(x);
      }
      try (IndexOutput o = d.createOutput("group", IOContext.DEFAULT)) {
        // 128-value groups, the shape the postings tail block uses.
        for (int off = 0; off < n; off += 128) {
          int[] g = new int[128];
          System.arraycopy(ints, off, g, 0, 128);
          o.writeGroupVInts(g, 128);
        }
      }
      try (IndexInput in = d.openInput("vint", IOContext.DEFAULT)) {
        in.seek(0);
        for (int i = 0; i < n; i++) {
          if (in.readVInt() != ints[i]) throw new AssertionError("vint " + i);
        }
        measure(
            "vint",
            () -> {
              in.seek(0);
              int acc = 0;
              for (int i = 0; i < n; i++) acc += in.readVInt();
              sink += acc;
              return n;
            });
      }
      try (IndexInput in = d.openInput("vlong", IOContext.DEFAULT)) {
        measure(
            "vlong",
            () -> {
              in.seek(0);
              long acc = 0;
              for (int i = 0; i < n; i++) acc += in.readVLong();
              sink += acc;
              return n;
            });
      }
      try (IndexInput in = d.openInput("group", IOContext.DEFAULT)) {
        int[] dst = new int[128];
        measure(
            "group_vint",
            () -> {
              in.seek(0);
              for (int off = 0; off < n; off += 128) {
                GroupVIntUtil.readGroupVInts(in, dst, 128);
                sink += dst[127];
              }
              return n;
            });
      }
    } finally {
      deleteAll(tmp);
    }
  }

  static FixedBitSet randomBits(int numBits, long seed, int densityPct) {
    FixedBitSet b = new FixedBitSet(numBits);
    Rng r = new Rng(seed);
    for (int i = 0; i < numBits; i++) {
      if (Long.remainderUnsigned(r.next(), 100) < densityPct) b.set(i);
    }
    return b;
  }

  static void bitset() throws IOException {
    final int numBits = 1 << 22;
    FixedBitSet a = randomBits(numBits, 0x1111_2222_3333_4444L, 10);
    FixedBitSet b = randomBits(numBits, 0x5555_6666_7777_8888L, 10);
    final int setBits = a.cardinality();
    measure(
        "cardinality",
        () -> {
          sink += a.cardinality();
          return numBits / 64;
        });
    measure(
        "next_set_bit",
        () -> {
          int n = 0;
          for (int i = a.nextSetBit(0);
              i != DocIdSetIterator.NO_MORE_DOCS;
              i = i + 1 >= numBits ? DocIdSetIterator.NO_MORE_DOCS : a.nextSetBit(i + 1)) {
            n++;
          }
          sink += n;
          return setBits;
        });
    measure(
        "intersection_count",
        () -> {
          sink += FixedBitSet.intersectionCount(a, b);
          return numBits / 64;
        });
    FixedBitSet c = a.clone();
    measure(
        "or",
        () -> {
          c.or(b);
          return numBits / 64;
        });
    final int[] probes = new int[1 << 16];
    Rng r = new Rng(0x9999_AAAA_BBBB_CCCCL);
    for (int i = 0; i < probes.length; i++) probes[i] = (int) Long.remainderUnsigned(r.next(), numBits);
    measure(
        "get_random",
        () -> {
          int n = 0;
          for (int p : probes) if (a.get(p)) n++;
          sink += n;
          return probes.length;
        });
  }

  /** Text-like bytes: space-separated base-36 words from a 2000-word vocabulary. */
  static byte[] textBytes(int len, long seed) {
    Rng r = new Rng(seed);
    StringBuilder sb = new StringBuilder(len + 16);
    while (sb.length() < len) {
      if (sb.length() > 0) sb.append(' ');
      sb.append('t').append(Long.toString(Long.remainderUnsigned(r.next(), 2000), 36));
    }
    sb.setLength(len);
    return sb.toString().getBytes(java.nio.charset.StandardCharsets.US_ASCII);
  }

  static void lz4() throws IOException {
    for (int len : new int[] {16 * 1024, 60 * 1024}) {
      byte[] src = textBytes(len, 0xABCD_EF01_2345_6789L ^ len);
      String sz = (len / 1024) + "k";
      ByteBuffersDataOutput out = new ByteBuffersDataOutput();
      LZ4.compress(src, 0, len, out, new LZ4.FastCompressionHashTable());
      byte[] compressed = out.toArrayCopy();
      byte[] dst = new byte[len];
      ByteArrayDataInput in = new ByteArrayDataInput(compressed);
      LZ4.decompress(in, len, dst, 0);
      if (!java.util.Arrays.equals(src, dst)) throw new AssertionError("lz4 round trip");
      measure(
          "decompress_" + sz,
          () -> {
            in.reset(compressed);
            LZ4.decompress(in, len, dst, 0);
            sink += dst[len - 1];
            return len;
          });
      LZ4.FastCompressionHashTable fast = new LZ4.FastCompressionHashTable();
      ByteBuffersDataOutput o2 = new ByteBuffersDataOutput(len);
      measure(
          "compress_fast_" + sz,
          () -> {
            o2.reset();
            LZ4.compress(src, 0, len, o2, fast);
            sink += o2.size();
            return len;
          });
      LZ4.HighCompressionHashTable high = new LZ4.HighCompressionHashTable();
      measure(
          "compress_high_" + sz,
          () -> {
            o2.reset();
            LZ4.compress(src, 0, len, o2, high);
            sink += o2.size();
            return len;
          });
    }
  }

  static void directMonotonic() throws IOException {
    final int n = 1 << 20;
    final int blockShift = 16;
    Rng r = new Rng(0x7777_1234_ABCD_0001L);
    long[] values = new long[n];
    long acc = 0;
    for (int i = 0; i < n; i++) {
      acc += Long.remainderUnsigned(r.next(), 1000);
      values[i] = acc;
    }
    Path tmp = Files.createTempDirectory("sweep-dm");
    try (Directory d = new MMapDirectory(tmp)) {
      try (IndexOutput meta = d.createOutput("meta", IOContext.DEFAULT);
          IndexOutput data = d.createOutput("data", IOContext.DEFAULT)) {
        DirectMonotonicWriter w = DirectMonotonicWriter.getInstance(meta, data, n, blockShift);
        for (long v : values) w.add(v);
        w.finish();
      }
      try (IndexInput metaIn = d.openInput("meta", IOContext.DEFAULT);
          IndexInput dataIn = d.openInput("data", IOContext.DEFAULT)) {
        DirectMonotonicReader.Meta m = DirectMonotonicReader.loadMeta(metaIn, n, blockShift);
        RandomAccessInput ra = dataIn.randomAccessSlice(0, dataIn.length());
        DirectMonotonicReader reader = DirectMonotonicReader.getInstance(m, ra);
        for (int i : new int[] {0, 1, n / 2, n - 1}) {
          if (reader.get(i) != values[i]) throw new AssertionError("dm " + i);
        }
        final int stride = 4099;
        measure(
            "get_random",
            () -> {
              int i = 0;
              long s = 0;
              for (int k = 0; k < 4096; k++) {
                i = (i + stride) & (n - 1);
                s += reader.get(i);
              }
              sink += s;
              return 4096;
            });
        measure(
            "get_seq",
            () -> {
              long s = 0;
              for (int i = 0; i < n; i++) s += reader.get(i);
              sink += s;
              return n;
            });
      }
    } finally {
      deleteAll(tmp);
    }
  }

  static void checksum() throws IOException {
    final int len = 16 << 20;
    byte[] data = textBytes(len, 0x5151_5151_5151_5151L);
    measure(
        "crc32_16m",
        () -> {
          CRC32 c = new CRC32();
          c.update(data, 0, len);
          sink += c.getValue();
          return len;
        });
  }

  /** The corpus's own sentence shape: Zipf-ish base-36 words, 40..160 per document. */
  static List<String> analysisDocs() {
    Rng r = new Rng(0x2468_ACE0_1357_9BDFL);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 2000; d++) {
      int words = 40 + (int) Long.remainderUnsigned(r.next(), 120);
      StringBuilder sb = new StringBuilder();
      for (int w = 0; w < words; w++) {
        if (w > 0) sb.append(' ');
        long x = r.next();
        // Skewed rank: min of two uniforms, over a 50k vocabulary, some capitalized.
        long a = Long.remainderUnsigned(x, 50000), b = Long.remainderUnsigned(x >>> 20, 50000);
        String word = "t" + Long.toString(Math.min(a, b), 36);
        if ((x & 7) == 0) word = word.toUpperCase(java.util.Locale.ROOT);
        sb.append(word);
        if ((x & 31) == 1) sb.append(',');
        if ((x & 63) == 2) sb.append('.');
      }
      docs.add(sb.toString());
    }
    return docs;
  }

  static void analysis() throws IOException {
    List<String> docs = analysisDocs();
    try (StandardAnalyzer a = new StandardAnalyzer()) {
      measure(
          "standard",
          () -> {
            long tokens = 0;
            for (String text : docs) {
              try (TokenStream ts = a.tokenStream("body", text)) {
                CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
                ts.reset();
                while (ts.incrementToken()) {
                  tokens++;
                  sink += term.length();
                }
                ts.end();
              }
            }
            return tokens;
          });
    }
  }

  /** Deterministic vectors in [-0.5, 0.5), the same on both sides. */
  static float[][] floatVectors(int n, int dim, long seed) {
    Rng r = new Rng(seed);
    float[][] v = new float[n][dim];
    for (float[] row : v) {
      for (int i = 0; i < dim; i++) {
        row[i] = (float) ((r.next() >>> 40) / (double) (1L << 24)) - 0.5f;
      }
    }
    return v;
  }

  static byte[][] byteVectors(int n, int dim, long seed) {
    Rng r = new Rng(seed);
    byte[][] v = new byte[n][dim];
    for (byte[] row : v) {
      for (int i = 0; i < dim; i++) row[i] = (byte) (r.next() >>> 56);
    }
    return v;
  }

  /**
   * The scalar-quantized distance kernels (one per {@code ScalarEncoding}) over 1024 stored
   * vectors against one query, and {@code OptimizedScalarQuantizer.scalarQuantize} -- the
   * port's {@code bench_quantized}.
   */
  static void quantized() throws IOException {
    for (int dim : new int[] {128, 768}) {
      byte[][] bdocs = byteVectors(1024, dim, 0xB17E + dim);
      byte[] bq = byteVectors(1, dim, 0xB0B + dim)[0];
      measure(
          "uint8_dot_" + dim,
          () -> {
            long s = 0;
            for (byte[] d : bdocs) s += org.apache.lucene.util.VectorUtil.uint8DotProduct(bq, d);
            sink += s;
            return bdocs.length;
          });
      byte[] q4 = new byte[dim];
      for (int i = 0; i < dim; i++) q4[i] = (byte) (bq[i] & 0x0f);
      byte[][] packed = new byte[1024][];
      byte[][] bits = new byte[1024][];
      byte[][] dibits = new byte[1024][];
      for (int i = 0; i < 1024; i++) {
        packed[i] = java.util.Arrays.copyOf(bdocs[i], dim / 2);
        bits[i] = java.util.Arrays.copyOf(bdocs[i], dim / 8);
        dibits[i] = java.util.Arrays.copyOf(bdocs[i], dim / 4);
      }
      byte[] tq = java.util.Arrays.copyOf(bq, dim / 2);
      measure(
          "int4_packed_dot_" + dim,
          () -> {
            long s = 0;
            for (byte[] d : packed) {
              s += org.apache.lucene.util.VectorUtil.int4DotProductSinglePacked(q4, d);
            }
            sink += s;
            return packed.length;
          });
      measure(
          "int4_bit_dot_" + dim,
          () -> {
            long s = 0;
            for (byte[] d : bits) s += org.apache.lucene.util.VectorUtil.int4BitDotProduct(tq, d);
            sink += s;
            return bits.length;
          });
      measure(
          "int4_dibit_dot_" + dim,
          () -> {
            long s = 0;
            for (byte[] d : dibits) s += org.apache.lucene.util.VectorUtil.int4DibitDotProduct(tq, d);
            sink += s;
            return dibits.length;
          });
      float[][] fdocs = floatVectors(64, dim, 0xF00D + dim);
      float[] centroid = floatVectors(1, dim, 0xCE17 + dim)[0];
      org.apache.lucene.util.quantization.OptimizedScalarQuantizer q =
          new org.apache.lucene.util.quantization.OptimizedScalarQuantizer(
              org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN);
      byte[] dest = new byte[dim];
      float[] scratch = new float[dim];
      measure(
          "osq_quantize4_" + dim,
          () -> {
            long s = 0;
            for (float[] d : fdocs) {
              System.arraycopy(d, 0, scratch, 0, d.length);
              s += q.scalarQuantize(scratch, dest, (byte) 4, centroid).quantizedComponentSum();
            }
            sink += s;
            return fdocs.length;
          });
    }
  }

  /**
   * {@code BytesRefHash}: 200k adds of decimal terms drawn from 50k distinct ones ({@code (i *
   * 7919) % 50000}), then {@code sort()}. One op is one add. Mirrors the port's {@code
   * bench_bytes_ref_hash}.
   */
  static void bytesRefHash() throws IOException {
    org.apache.lucene.util.BytesRef[] terms = new org.apache.lucene.util.BytesRef[200_000];
    for (int i = 0; i < terms.length; i++) {
      terms[i] = new org.apache.lucene.util.BytesRef(Long.toString((i * 7919L) % 50_000));
    }
    measure(
        "bytes_ref_hash_add_sort_200k",
        () -> {
          org.apache.lucene.util.BytesRefHash h = new org.apache.lucene.util.BytesRefHash();
          long s = 0;
          for (org.apache.lucene.util.BytesRef t : terms) s += h.add(t);
          s += h.sort()[0];
          sink += s;
          return terms.length;
        });
  }

  /**
   * {@code BKDWriter.writeField} (the flush path) plus {@code writeIndex} over 200k points of one
   * and of two 4-byte dimensions (dim {@code d} of point {@code i} is {@code (i * 7919 + d *
   * 104729) % 1000003}, big-endian; doc {@code i}), 512 points per leaf. One op is one point.
   * Mirrors the port's {@code bench_bkd_build}.
   */
  static void bkdBuild() throws IOException {
    final int n = 200_000;
    for (int dims = 1; dims <= 2; dims++) {
      final org.apache.lucene.util.bkd.BKDConfig config =
          new org.apache.lucene.util.bkd.BKDConfig(dims, dims, 4, 512);
      final int stride = dims * 4;
      final byte[] values = new byte[n * stride];
      final int[] docs = new int[n];
      for (int i = 0; i < n; i++) {
        docs[i] = i;
        for (int d = 0; d < dims; d++) {
          int x = (int) ((i * 7919L + d * 104729L) % 1_000_003);
          int o = i * stride + d * 4;
          values[o] = (byte) (x >>> 24);
          values[o + 1] = (byte) (x >>> 16);
          values[o + 2] = (byte) (x >>> 8);
          values[o + 3] = (byte) x;
        }
      }
      measure(
          "bkd_flush_" + dims + "d_200k",
          () -> {
            ByteBuffersDataOutput meta = new ByteBuffersDataOutput();
            ByteBuffersDataOutput index = new ByteBuffersDataOutput();
            ByteBuffersDataOutput data = new ByteBuffersDataOutput();
            try (org.apache.lucene.util.bkd.BKDWriter w =
                new org.apache.lucene.util.bkd.BKDWriter(n, null, "_0", config, 16.0, n)) {
              org.apache.lucene.util.IORunnable fin =
                  w.writeField(
                      new org.apache.lucene.store.ByteBuffersIndexOutput(meta, "m", "m"),
                      new org.apache.lucene.store.ByteBuffersIndexOutput(index, "i", "i"),
                      new org.apache.lucene.store.ByteBuffersIndexOutput(data, "d", "d"),
                      "f",
                      new BkdArrayTree(docs, values.clone(), stride));
              fin.run();
            }
            sink += meta.size() + index.size() + data.size();
            return n;
          });
    }
  }

  /**
   * {@code PointValuesWriter}'s shape of {@code MutablePointTree}: an ordinal array over flat
   * points.
   */
  static final class BkdArrayTree extends org.apache.lucene.codecs.MutablePointTree {
    final int[] docs;
    final byte[] values;
    final int stride;
    final int[] ords;
    final int[] temp;

    BkdArrayTree(int[] docs, byte[] values, int stride) {
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
    public void visitDocValues(PointValues.IntersectVisitor visitor) throws IOException {
      byte[] v = new byte[stride];
      for (int i = 0; i < docs.length; i++) {
        System.arraycopy(values, ords[i] * stride, v, 0, stride);
        visitor.visit(docs[ords[i]], v);
      }
    }
  }

  /**
   * {@code FSTCompiler} over 50k sorted decimal keys ({@code i * 7919 % 1000003}) with {@code
   * PositiveIntOutputs}: one op is one key added, the compile included. Mirrors the port's {@code
   * bench_fst_build}.
   */
  static void fstBuild() throws IOException {
    java.util.TreeMap<org.apache.lucene.util.BytesRef, Long> sorted = new java.util.TreeMap<>();
    for (long i = 0; i < 50_000; i++) {
      long v = i * 7919 % 1_000_003;
      sorted.put(new org.apache.lucene.util.BytesRef(Long.toString(v)), v);
    }
    org.apache.lucene.util.IntsRefBuilder scratch = new org.apache.lucene.util.IntsRefBuilder();
    measure(
        "fst_build_50k",
        () -> {
          org.apache.lucene.util.fst.FSTCompiler<Long> c =
              new org.apache.lucene.util.fst.FSTCompiler.Builder<>(
                      org.apache.lucene.util.fst.FST.INPUT_TYPE.BYTE1,
                      org.apache.lucene.util.fst.PositiveIntOutputs.getSingleton())
                  .build();
          for (java.util.Map.Entry<org.apache.lucene.util.BytesRef, Long> e : sorted.entrySet()) {
            c.add(org.apache.lucene.util.fst.Util.toIntsRef(e.getKey(), scratch), e.getValue());
          }
          c.compile();
          sink += c.fstSizeInBytes();
          return sorted.size();
        });
  }

  /**
   * The similarity kernels every vector search spends its time in, over 1024 stored vectors
   * against one query: {@code org.apache.lucene.util.VectorUtil}, which runs Lucene's Panama
   * implementation here.
   */
  static void vectors() throws IOException {
    for (int dim : new int[] {128, 768}) {
      float[][] docs = floatVectors(1024, dim, 0xF00D + dim);
      float[] q = floatVectors(1, dim, 0xBEEF + dim)[0];
      measure(
          "dot_f32_" + dim,
          () -> {
            float s = 0;
            for (float[] d : docs) s += org.apache.lucene.util.VectorUtil.dotProduct(q, d);
            sink += (long) s;
            return docs.length;
          });
      measure(
          "l2_f32_" + dim,
          () -> {
            float s = 0;
            for (float[] d : docs) s += org.apache.lucene.util.VectorUtil.squareDistance(q, d);
            sink += (long) s;
            return docs.length;
          });
      measure(
          "cos_f32_" + dim,
          () -> {
            float s = 0;
            for (float[] d : docs) s += org.apache.lucene.util.VectorUtil.cosine(q, d);
            sink += (long) s;
            return docs.length;
          });
      byte[][] bdocs = byteVectors(1024, dim, 0xB17E + dim);
      byte[] bq = byteVectors(1, dim, 0xB0B + dim)[0];
      measure(
          "dot_u8_" + dim,
          () -> {
            long s = 0;
            for (byte[] d : bdocs) s += org.apache.lucene.util.VectorUtil.dotProduct(bq, d);
            sink += s;
            return bdocs.length;
          });
    }
  }

  // -------------------------------------------------------------- codec reads

  static void postingsAdvance(LeafReader leaf) throws IOException {
    Terms terms = leaf.terms("body");
    for (String term : new String[] {"t0", "t1", "tz"}) {
      for (int gap : new int[] {8, 64, 1024}) {
        TermsEnum te = terms.iterator();
        if (!te.seekExact(new BytesRef(term))) continue;
        measure(
            term + "_gap" + gap,
            () -> {
              PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
              long n = 0;
              int doc = pe.advance(0);
              while (doc != DocIdSetIterator.NO_MORE_DOCS) {
                n++;
                doc = pe.advance(doc + gap);
              }
              sink += n;
              return n;
            });
      }
    }
  }

  static void postingsFreq(LeafReader leaf) throws IOException {
    Terms terms = leaf.terms("body");
    for (String term : new String[] {"t0", "t1", "tz", "t2s"}) {
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new BytesRef(term))) continue;
      measure(
          term,
          () -> {
            PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
            long n = 0, f = 0;
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
              n++;
              f += pe.freq();
            }
            sink += f;
            return n;
          });
    }
  }

  static void positions(LeafReader leaf) throws IOException {
    Terms terms = leaf.terms("body");
    for (String term : new String[] {"t1", "tz", "t2s"}) {
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new BytesRef(term))) continue;
      measure(
          term,
          () -> {
            PostingsEnum pe = te.postings(null, PostingsEnum.POSITIONS);
            long n = 0, s = 0;
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
              int f = pe.freq();
              for (int i = 0; i < f; i++) s += pe.nextPosition();
              n += f;
            }
            sink += s;
            return n;
          });
    }
  }

  static void termSeek(LeafReader leaf) throws IOException {
    String field = "body";
    List<BytesRef> terms = new ArrayList<>();
    TermsEnum all = leaf.terms(field).iterator();
    long n = 0;
    for (BytesRef t = all.next(); t != null; t = all.next()) {
      if (n % 97 == 0) terms.add(BytesRef.deepCopyOf(t));
      n++;
    }
    // Same xorshift64 shuffle as the Rust side.
    long state = 0x9E3779B97F4A7C15L;
    for (int i = terms.size() - 1; i >= 1; i--) {
      state ^= state << 13;
      state ^= state >>> 7;
      state ^= state << 17;
      int j = (int) Long.remainderUnsigned(state, i + 1);
      BytesRef tmp = terms.get(i);
      terms.set(i, terms.get(j));
      terms.set(j, tmp);
    }
    final List<BytesRef> hits = new ArrayList<>(terms.subList(0, Math.min(2000, terms.size())));
    final List<BytesRef> misses = new ArrayList<>();
    for (BytesRef t : hits) {
      byte[] b = java.util.Arrays.copyOf(t.bytes, t.offset + t.length + 1);
      b[t.offset + t.length] = '~';
      misses.add(new BytesRef(b, t.offset, t.length + 1));
    }
    for (var c : List.of(List.of("seek_hit", hits), List.of("seek_miss", misses))) {
      @SuppressWarnings("unchecked")
      List<BytesRef> targets = (List<BytesRef>) c.get(1);
      measure(
          (String) c.get(0),
          () -> {
            TermsEnum te = leaf.terms(field).iterator();
            for (BytesRef t : targets) if (te.seekExact(t)) sink += te.docFreq();
            return targets.size();
          });
    }
    measure(
        "next_all",
        () -> {
          TermsEnum te = leaf.terms(field).iterator();
          long ops = 0;
          for (BytesRef t = te.next(); t != null; t = te.next()) {
            sink += t.length + te.docFreq();
            ops++;
          }
          return ops;
        });
  }

  static void docValues(LeafReader leaf) throws IOException {
    final int maxDoc = leaf.maxDoc();
    measure(
        "numeric_seq",
        () -> {
          NumericDocValues v = leaf.getNumericDocValues("num");
          long s = 0;
          for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
            s += v.longValue();
          }
          sink += s;
          return maxDoc;
        });
    final int stride = 37;
    measure(
        "numeric_stride37",
        () -> {
          NumericDocValues v = leaf.getNumericDocValues("num");
          long s = 0, n = 0;
          for (int d = 0; d < maxDoc; d += stride) {
            if (v.advanceExact(d)) s += v.longValue();
            n++;
          }
          sink += s;
          return n;
        });
    measure(
        "sorted_ord_seq",
        () -> {
          SortedDocValues v = leaf.getSortedDocValues("keyword");
          long s = 0;
          for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
            s += v.ordValue();
          }
          sink += s;
          return maxDoc;
        });
    measure(
        "sorted_set_seq",
        () -> {
          SortedSetDocValues v = leaf.getSortedSetDocValues("cat");
          long s = 0;
          for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
            int c = v.docValueCount();
            for (int i = 0; i < c; i++) s += v.nextOrd();
          }
          sink += s;
          return maxDoc;
        });
    SortedDocValues sorted = leaf.getSortedDocValues("keyword");
    final int valueCount = sorted.getValueCount();
    measure(
        "lookup_ord",
        () -> {
          long s = 0;
          int ord = 0;
          for (int k = 0; k < 4096; k++) {
            ord = (ord + 4099) % valueCount;
            s += sorted.lookupOrd(ord).length;
          }
          sink += s;
          return 4096;
        });
  }

  static void norms(LeafReader leaf) throws IOException {
    final int maxDoc = leaf.maxDoc();
    measure(
        "body_seq",
        () -> {
          NumericDocValues v = leaf.getNormValues("body");
          long s = 0;
          for (int d = v.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = v.nextDoc()) {
            s += v.longValue();
          }
          sink += s;
          return maxDoc;
        });
    measure(
        "body_stride37",
        () -> {
          NumericDocValues v = leaf.getNormValues("body");
          long s = 0, n = 0;
          for (int d = 0; d < maxDoc; d += 37) {
            if (v.advanceExact(d)) s += v.longValue();
            n++;
          }
          sink += s;
          return n;
        });
  }

  /** Counts points in [lo, hi] with a plain visitor, the shape PointRangeQuery's count uses. */
  static final class RangeCounter implements PointValues.IntersectVisitor {
    final byte[] lo = new byte[8], hi = new byte[8];
    long count;

    RangeCounter(long l, long h) {
      org.apache.lucene.document.LongPoint.encodeDimension(l, lo, 0);
      org.apache.lucene.document.LongPoint.encodeDimension(h, hi, 0);
    }

    @Override
    public void visit(int docID) {
      count++;
    }

    @Override
    public void visit(int docID, byte[] packed) {
      if (java.util.Arrays.compareUnsigned(packed, 0, 8, lo, 0, 8) >= 0
          && java.util.Arrays.compareUnsigned(packed, 0, 8, hi, 0, 8) <= 0) {
        count++;
      }
    }

    @Override
    public PointValues.Relation compare(byte[] min, byte[] max) {
      if (java.util.Arrays.compareUnsigned(min, 0, 8, hi, 0, 8) > 0
          || java.util.Arrays.compareUnsigned(max, 0, 8, lo, 0, 8) < 0) {
        return PointValues.Relation.CELL_OUTSIDE_QUERY;
      }
      if (java.util.Arrays.compareUnsigned(min, 0, 8, lo, 0, 8) >= 0
          && java.util.Arrays.compareUnsigned(max, 0, 8, hi, 0, 8) <= 0) {
        return PointValues.Relation.CELL_INSIDE_QUERY;
      }
      return PointValues.Relation.CELL_CROSSES_QUERY;
    }
  }

  static void points(LeafReader leaf) throws IOException {
    PointValues pv = leaf.getPointValues("num");
    for (long[] range : new long[][] {{0, 1000}, {0, 100_000}, {250_000, 750_000}}) {
      String name = "range_" + range[0] + "_" + range[1];
      measure(
          name,
          () -> {
            RangeCounter c = new RangeCounter(range[0], range[1]);
            pv.intersect(c);
            sink += c.count;
            return 1;
          });
    }
  }

  // -------------------------------------------------------------------- memory

  static long usedHeap() {
    for (int i = 0; i < 4; i++) {
      System.gc();
      try {
        Thread.sleep(50);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    Runtime rt = Runtime.getRuntime();
    return rt.totalMemory() - rt.freeMemory();
  }

  /**
   * Heap retained by an open reader. Both sides exclude the mmap'd files themselves (off-heap in
   * Java, outside the allocator in Rust) and count what opening the index made the process keep.
   */
  static void memory(String index) throws IOException {
    try (MMapDirectory dir = new MMapDirectory(Paths.get(index))) {
      long before = usedHeap();
      DirectoryReader reader = DirectoryReader.open(dir);
      long after = usedHeap();
      System.out.printf("open_heap_bytes\t%d\t1%n", after - before);
      // Touch every field's terms and postings once, as a first query would, and
      // count what that leaves resident too.
      for (var leaf : reader.leaves()) {
        for (String f : new String[] {"body", "title", "keyword"}) {
          TermsEnum te = leaf.reader().terms(f).iterator();
          if (te.seekExact(new BytesRef("t0"))) {
            PostingsEnum pe = te.postings(null, PostingsEnum.FREQS);
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) sink += pe.freq();
          }
        }
      }
      long touched = usedHeap();
      System.out.printf("after_query_heap_bytes\t%d\t1%n", touched - before);
      sink += reader.maxDoc();
      reader.close();
    }
  }

  static void deleteAll(Path dir) throws IOException {
    try (var s = Files.list(dir)) {
      for (Path p : s.toList()) Files.deleteIfExists(p);
    }
    Files.deleteIfExists(dir);
  }
}
