import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.Random;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.MathUtil;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.SmallFloat;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;

/**
 * Differential fixtures for the `org.apache.lucene.util` primitives: BitUtil, MathUtil,
 * NumericUtils, SmallFloat, StringHelper, Version and FixedBitSet's full API.
 *
 * <p>Writes `util_primitives/cases.txt`, one case per line: an op name, its inputs, and what
 * Lucene 10.5.0 returned. Floats and doubles travel as raw bits; byte arrays as hex (`-` when
 * empty); a thrown exception as `ERR` (Version: `ERR <message>`). `fbs` blocks replay a random
 * operation script on one FixedBitSet (plus a second operand set for the binary ops).
 *
 * <p>`randomId` is pinned by running with `tests.seed` set (StringHelper reads it in its static
 * initializer), so the id stream is deterministic. Usage: java GenUtilPrimitives <outdir>
 */
public class GenUtilPrimitives {
  static final String TESTS_SEED = "5EED0F1D2C3B4A59";
  static final StringBuilder OUT = new StringBuilder();

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String hex(byte[] b) {
    if (b.length == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xFF));
    return sb.toString();
  }

  static byte[] bytes(Random r, int n) {
    byte[] b = new byte[n];
    r.nextBytes(b);
    return b;
  }

  static long interesting(Random r) {
    switch (r.nextInt(6)) {
      case 0:
        return r.nextLong();
      case 1:
        return r.nextInt(100) - 50;
      case 2:
        return 1L << r.nextInt(64);
      case 3:
        return (1L << r.nextInt(64)) - 1;
      case 4:
        return new long[] {Long.MIN_VALUE, Long.MAX_VALUE, 0, -1}[r.nextInt(4)];
      default:
        return r.nextInt();
    }
  }

  static double interestingDouble(Random r) {
    switch (r.nextInt(5)) {
      case 0:
        return Double.longBitsToDouble(r.nextLong());
      case 1:
        return (r.nextDouble() - 0.5) * Math.pow(10, r.nextInt(20) - 10);
      case 2:
        return new double[] {
          0.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
          Double.MIN_VALUE, Double.MAX_VALUE, 1.0, -1.0
        }[r.nextInt(9)];
      default:
        return r.nextGaussian() * 3;
    }
  }

  public static void main(String[] args) throws Exception {
    System.setProperty("tests.seed", TESTS_SEED);
    Path out = Path.of(args[0], "util_primitives");
    Files.createDirectories(out);
    Random r = new Random(0x5EEDL);

    // --- BitUtil ---
    for (int i = 0; i < 200; i++) {
      int e = (int) interesting(r), o = (int) interesting(r);
      long m = BitUtil.interleave(e, o);
      line("interleave", e, o, m);
      long b = interesting(r);
      line("deinterleave", b, BitUtil.deinterleave(b));
      line("flipflop", b, BitUtil.flipFlop(b));
      int v = (int) interesting(r);
      line("nhp32", v, BitUtil.nextHighestPowerOfTwo(v));
      line("nhp64", b, BitUtil.nextHighestPowerOfTwo(b));
      line("izp", v, BitUtil.isZeroOrPowerOfTwo(v));
      line("zz", b, BitUtil.zigZagEncode(b), BitUtil.zigZagEncode(v));
    }

    // --- MathUtil ---
    for (int i = 0; i < 200; i++) {
      long x = interesting(r);
      int base = 2 + r.nextInt(20);
      line("log", x, base, MathUtil.log(x, base));
      line("log", x, 2, MathUtil.log(x, 2));
      long a = interesting(r), c = interesting(r);
      if (r.nextInt(3) == 0) {
        long g = 1 + r.nextInt(1000);
        a = (a % 100000) * g;
        c = (c % 100000) * g;
      }
      line("gcd", a, c, MathUtil.gcd(a, c));
      double d = interestingDouble(r);
      double bd = 1.0 + r.nextDouble() * 20;
      line("logd", Double.doubleToRawLongBits(bd), Double.doubleToRawLongBits(d),
          Double.doubleToRawLongBits(MathUtil.log(bd, d)));
      line("asinh", Double.doubleToRawLongBits(d), Double.doubleToRawLongBits(MathUtil.asinh(d)));
      double ac = 1.0 + Math.abs(d);
      line("acosh", Double.doubleToRawLongBits(ac), Double.doubleToRawLongBits(MathUtil.acosh(ac)));
      double at = r.nextDouble() * 2 - 1;
      line("atanh", Double.doubleToRawLongBits(at), Double.doubleToRawLongBits(MathUtil.atanh(at)));
      int n = r.nextInt(1000) - 5;
      line("sumrel", n, Double.doubleToRawLongBits(MathUtil.sumRelativeErrorBound(n)));
      line("sumup", Double.doubleToRawLongBits(d), n,
          Double.doubleToRawLongBits(MathUtil.sumUpperBound(d, n)));
      int ua = (int) interesting(r), ub = (int) interesting(r);
      line("umin", ua, ub, MathUtil.unsignedMin(ua, ub));
    }
    for (int base : new int[] {1, 0, -3}) {
      try {
        line("log", 5, base, MathUtil.log(5, base));
      } catch (IllegalArgumentException e) {
        line("log", 5, base, "ERR");
      }
    }

    // --- NumericUtils ---
    for (int i = 0; i < 200; i++) {
      double d = interestingDouble(r);
      long raw = Double.doubleToRawLongBits(d);
      line("d2sl", raw, NumericUtils.doubleToSortableLong(d));
      long enc = interesting(r);
      line("sl2d", enc, Double.doubleToRawLongBits(NumericUtils.sortableLongToDouble(enc)));
      float f = (float) d;
      if (r.nextInt(10) == 0) f = Float.intBitsToFloat(0x7f800001 + r.nextInt(1000)); // odd NaN
      line("f2si", Float.floatToRawIntBits(f), NumericUtils.floatToSortableInt(f));
      int iv = (int) interesting(r);
      byte[] ib = new byte[4];
      NumericUtils.intToSortableBytes(iv, ib, 0);
      line("i2sb", iv, hex(ib));
      byte[] lb = new byte[8];
      NumericUtils.longToSortableBytes(enc, lb, 0);
      line("l2sb", enc, hex(lb));

      BigInteger big = new BigInteger(1 + r.nextInt(100), r);
      if (r.nextBoolean()) big = big.negate();
      int size = 1 + r.nextInt(14);
      byte[] bb = new byte[size];
      String res;
      try {
        NumericUtils.bigIntToSortableBytes(big, size, bb, 0);
        res = hex(bb);
        line("sb2bi", res, hex(NumericUtils.sortableBytesToBigInt(bb, 0, size).toByteArray()));
      } catch (IllegalArgumentException e) {
        res = "ERR";
      }
      line("bi2sb", hex(big.toByteArray()), size, res);

      int bpd = 1 + r.nextInt(12), dims = 1 + r.nextInt(3), dim = r.nextInt(dims);
      byte[] a = bytes(r, bpd * dims), b = bytes(r, bpd * dims);
      if (r.nextInt(4) == 0) System.arraycopy(a, 0, b, 0, a.length / 2);
      byte[] sum = new byte[bpd], diff = new byte[bpd];
      String sres, dres;
      try {
        NumericUtils.add(bpd, dim, a, b, sum);
        sres = hex(sum);
      } catch (IllegalArgumentException e) {
        sres = "ERR";
      }
      try {
        NumericUtils.subtract(bpd, dim, a, b, diff);
        dres = hex(diff);
      } catch (IllegalArgumentException e) {
        dres = "ERR";
      }
      line("add", bpd, dim, hex(a), hex(b), sres);
      line("sub", bpd, dim, hex(a), hex(b), dres);
    }

    // --- SmallFloat ---
    for (int b = 0; b < 256; b++) {
      line("b315", b, Float.floatToRawIntBits(SmallFloat.byte315ToFloat((byte) b)));
      line("b2f", b, 5, 2, Float.floatToRawIntBits(SmallFloat.byteToFloat((byte) b, 5, 2)));
      line("b42i", b, SmallFloat.byte4ToInt((byte) b));
    }
    for (int i = 0; i < 300; i++) {
      float f = (float) interestingDouble(r);
      if (Float.isNaN(f)) f = 1.5f;
      line("f315", Float.floatToRawIntBits(f), SmallFloat.floatToByte315(f) & 0xFF);
      int mb = 1 + r.nextInt(7), ze = r.nextInt(30);
      line("f2b", Float.floatToRawIntBits(f), mb, ze, SmallFloat.floatToByte(f, mb, ze) & 0xFF);
      long lv = interesting(r) & Long.MAX_VALUE;
      line("l2i4", lv, SmallFloat.longToInt4(lv));
      int i4 = r.nextInt(256);
      line("i42l", i4, SmallFloat.int4ToLong(i4));
      int len = (int) (interesting(r) & Integer.MAX_VALUE);
      line("i2b4", len, SmallFloat.intToByte4(len) & 0xFF);
    }

    // --- StringHelper ---
    for (int i = 0; i < 150; i++) {
      byte[] data = bytes(r, r.nextInt(70));
      int seed = (int) interesting(r);
      line("mm32", hex(data), seed, StringHelper.murmurhash3_x86_32(data, 0, data.length, seed));
      long[] h = StringHelper.murmurhash3_x64_128(data, 0, data.length, seed);
      line("mm128", hex(data), seed, h[0], h[1]);
      long[] hd = StringHelper.murmurhash3_x64_128(new BytesRef(data));
      line("mm128d", hex(data), hd[0], hd[1]);
      byte[] other = data.clone();
      if (other.length > 0 && r.nextBoolean()) {
        int p = r.nextInt(other.length);
        other[p]++;
        other = java.util.Arrays.copyOf(other, p + 1 + r.nextInt(other.length - p));
      } else if (r.nextBoolean()) {
        other = java.util.Arrays.copyOf(other, other.length + 1 + r.nextInt(3));
      }
      String bd;
      try {
        bd = Integer.toString(StringHelper.bytesDifference(new BytesRef(data), new BytesRef(other)));
      } catch (IllegalArgumentException e) {
        bd = "ERR";
      }
      line("bdiff", hex(data), hex(other), bd);
      byte[] pre = java.util.Arrays.copyOf(data, Math.min(data.length, r.nextInt(8)));
      if (r.nextInt(4) == 0 && pre.length > 0) pre[0]++;
      line("sw", hex(data), hex(pre), StringHelper.startsWith(new BytesRef(data), new BytesRef(pre)));
      byte[] suf = java.util.Arrays.copyOfRange(data, Math.max(0, data.length - r.nextInt(8)), data.length);
      if (r.nextInt(4) == 0 && suf.length > 0) suf[0]++;
      line("ew", hex(data), hex(suf), StringHelper.endsWith(new BytesRef(data), new BytesRef(suf)));
    }
    for (int i = 0; i < 20; i++) {
      line("id", TESTS_SEED, i, hex(StringHelper.randomId()));
    }
    line("idstr", hex(new byte[] {1, 2, (byte) 0xab}), StringHelper.idToString(new byte[] {1, 2, (byte) 0xab}));

    // --- Version ---
    String[] versions = {
      "10.5.0", "10.5", "9.12.4", "1.2.3", "8.0.0.1", "8.0.0.2", "8.0.0.3", "8.1.0.1", "8.0.0.0",
      "1.2.3.1.1", "10", "", ".", "1.", "x.1", "1.y", "1.2.z", "1.0.0.w", "256.0", "1.256", "1.1.256",
      "-1.0", "+3.4", "99999999999.0", "0.0.0", "255.255.255", "1..2", "LUCENE_10_5_0", "latest",
      "LATEST", "lucene_current", "LUCENE_9_12", "LUCENE_95", "LUCENE_123", "lucene_1_2_3_4",
      "Lucene_10_4_0", "LUCENE_X_1", "foo", "10.5.0-SNAPSHOT"
    };
    for (String v : versions) {
      String s = v.isEmpty() ? "<empty>" : v;
      try {
        Version p = Version.parse(v);
        line("vparse", s, p.major, p.minor, p.bugfix, p.prerelease, p.toString());
      } catch (ParseException e) {
        line("vparse", s, "ERR", e.getMessage());
      }
      try {
        Version p = Version.parseLeniently(v);
        line("vlenient", s, p.major, p.minor, p.bugfix, p.prerelease, p.toString());
      } catch (ParseException e) {
        line("vlenient", s, "ERR", e.getMessage());
      }
    }
    Version prev = null;
    for (java.lang.reflect.Field fld : Version.class.getFields()) {
      if (fld.getType() == Version.class) {
        Version v = (Version) fld.get(null);
        line("vconst", fld.getName(), v.toString(), v.hashCode(), prev == null ? "-" : v.onOrAfter(prev));
        prev = v;
      }
    }
    line("vmin", Version.MIN_SUPPORTED_MAJOR);

    // --- FixedBitSet ---
    for (int t = 0; t < 40; t++) {
      int numBits = 1 + r.nextInt(t < 5 ? 70 : 3000);
      FixedBitSet a = new FixedBitSet(numBits), b = new FixedBitSet(numBits);
      line("fbs", numBits);
      for (int op = 0; op < 60; op++) {
        int i = r.nextInt(numBits), j = r.nextInt(numBits + 1);
        int lo = Math.min(i, j), hi = Math.max(i, j);
        switch (r.nextInt(22)) {
          case 0 -> line("set", i);
          case 1 -> line("clear", i);
          case 2 -> line("getAndSet", i, a.getAndSet(i));
          case 3 -> line("getAndClear", i, a.getAndClear(i));
          case 4 -> line("flip", i);
          case 5 -> line("flipRange", lo, hi);
          case 6 -> line("setRange", lo, hi);
          case 7 -> line("clearRange", lo, hi);
          case 8 -> line("prevSetBit", i, a.prevSetBit(i));
          case 9 -> line("nextSetBit", i, a.nextSetBit(i));
          case 10 -> line("nextSetBitRange", i, hi, i < hi ? a.nextSetBit(i, hi) : "-");
          case 11 -> line("nextClearBit", i, a.nextClearBit(i));
          case 12 -> line("nextClearBitRange", i, hi, i < hi ? a.nextClearBit(i, hi) : "-");
          case 13 -> line("cardRange", lo, hi, a.cardinality(lo, hi));
          case 14 -> {
            for (int k = 0; k < 1 + r.nextInt(numBits); k++) b.set(r.nextInt(numBits));
            line("bset", hex(toBytes(b)));
          }
          case 15 -> line("counts", FixedBitSet.unionCount(a, b), FixedBitSet.andNotCount(a, b),
              FixedBitSet.intersectionCount(a, b), a.intersects(b));
          case 16 -> line("xor");
          case 17 -> {
            int len = Math.min(numBits - lo, 1 + r.nextInt(64));
            int from = r.nextInt(numBits - len + 1);
            line(r.nextBoolean() ? "orRange" : "andRange", from, lo, len);
          }
          case 18 -> {
            int ml = 1 + r.nextInt(64);
            int sb = r.nextInt(numBits - Math.min(ml, numBits) + 1);
            if (sb + ml <= numBits) line("orMask", sb, r.nextLong() >>> (64 - ml), ml);
          }
          case 19 -> line("state", a.cardinality(), a.approximateCardinality(), a.hashCode(), a.scanIsEmpty());
          case 20 -> {
            int[] arr = new int[numBits];
            int n = a.intoArray(lo, hi, 7, arr);
            line("intoArray", lo, hi, n, n == 0 ? "-" : join(arr, n));
          }
          default -> {
            int desired = r.nextInt(numBits * 2 + 1);
            FixedBitSet g = FixedBitSet.ensureCapacity(a.clone(), desired);
            line("ensureCapacity", desired, g.length(), g.cardinality());
          }
        }
        applyLast(a, b);
        line("sum", a.hashCode(), a.cardinality());
      }
      line("words", hex(toBytes(a)));
    }

    Files.writeString(out.resolve("cases.txt"), OUT.toString());
  }

  static String join(int[] a, int n) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) sb.append(',');
      sb.append(a[i]);
    }
    return sb.toString();
  }

  static byte[] toBytes(FixedBitSet s) {
    long[] w = s.getBits();
    java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(w.length * 8);
    for (long x : w) bb.putLong(x);
    return bb.array();
  }

  /** Applies the mutating op of the line just written (reads it back from OUT). */
  static void applyLast(FixedBitSet a, FixedBitSet b) {
    int end = OUT.length() - 1;
    int start = OUT.lastIndexOf("\n", end - 1) + 1;
    String[] p = OUT.substring(start, end).split(" ");
    switch (p[0]) {
      case "set" -> a.set(Integer.parseInt(p[1]));
      case "clear" -> a.clear(Integer.parseInt(p[1]));
      case "flip" -> a.flip(Integer.parseInt(p[1]));
      case "flipRange" -> a.flip(Integer.parseInt(p[1]), Integer.parseInt(p[2]));
      case "setRange" -> a.set(Integer.parseInt(p[1]), Integer.parseInt(p[2]));
      case "clearRange" -> a.clear(Integer.parseInt(p[1]), Integer.parseInt(p[2]));
      case "xor" -> a.xor(b);
      case "orRange" -> FixedBitSet.orRange(b, Integer.parseInt(p[1]), a, Integer.parseInt(p[2]),
          Integer.parseInt(p[3]));
      case "andRange" -> FixedBitSet.andRange(b, Integer.parseInt(p[1]), a, Integer.parseInt(p[2]),
          Integer.parseInt(p[3]));
      case "orMask" -> a.orMask(Integer.parseInt(p[1]), Long.parseLong(p[2]), Integer.parseInt(p[3]));
      default -> {}
    }
  }
}
