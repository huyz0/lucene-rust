import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Function;
import org.apache.lucene.misc.util.fst.ListOfOutputs;
import org.apache.lucene.misc.util.fst.UpToTwoPositiveIntOutputs;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Outputs;
import org.apache.lucene.util.fst.PositiveIntOutputs;
import org.apache.lucene.util.fst.Util;

/**
 * M14 T14.3 `misc`: {@link UpToTwoPositiveIntOutputs} and {@link ListOfOutputs}. Writes
 * {@code misc/fst/<case>.txt}, one per outputs type, whose lines are
 *
 * <pre>
 * k    hexKey  value        one add() call, in order (a key is added again for a second value; "-" is the empty key)
 * stats nodes arcs
 * fst  hex                  FST.save(out, out)
 * get  hexKey  result       Util.get: L:v, T:first,second, or M:v;v;...
 * </pre>
 *
 * Read by {@code crates/lucene-search/tests/misc_fst_fixtures.rs}.
 */
public class GenMiscFst {
  static Path out;

  static String hex(byte[] b, int off, int len) {
    if (len == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = off; i < off + len; i++) sb.append(String.format("%02x", b[i] & 0xFF));
    return sb.toString();
  }

  static String key(int[] k) {
    byte[] b = new byte[k.length];
    for (int i = 0; i < k.length; i++) b[i] = (byte) k[i];
    return hex(b, 0, b.length);
  }

  /** The value of an add, or of a lookup, in the fixture's text form. */
  static String show(Object o) {
    if (o instanceof Long l) return "L:" + l;
    if (o instanceof UpToTwoPositiveIntOutputs.TwoLongs t) return "T:" + t.first() + "," + t.second();
    if (o instanceof BytesRef b) return "L:" + hex(b.bytes, b.offset, b.length);
    if (o instanceof List<?> list) {
      StringBuilder sb = new StringBuilder("M:");
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) sb.append(';');
        sb.append(show(list.get(i)).substring(2));
      }
      return sb.toString();
    }
    throw new IllegalStateException(String.valueOf(o));
  }

  /**
   * Adds every key {@code repeats[i]} times (consecutively, as the outputs require), each time with
   * a fresh value, then compiles and reads every key back.
   */
  static <T> void run(
      String name,
      Outputs<T> outputs,
      List<int[]> keys,
      int[] repeats,
      Function<Random, T> value,
      long seed)
      throws Exception {
    Random r = new Random(seed);
    FSTCompiler<T> c = new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs).build();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < keys.size(); i++) {
      int[] k = keys.get(i);
      for (int rep = 0; rep < repeats[i]; rep++) {
        T v = value.apply(r);
        c.add(new IntsRef(k, 0, k.length), v);
        sb.append("k ").append(key(k)).append(' ').append(show(v)).append('\n');
      }
    }
    FST.FSTMetadata<T> meta = c.compile();
    sb.append("stats ").append(c.getNodeCount()).append(' ').append(c.getArcCount()).append('\n');
    FST<T> fst = FST.fromFSTReader(meta, c.getFSTReader());
    ByteBuffersDataOutput o = new ByteBuffersDataOutput();
    fst.save(o, o);
    byte[] b = o.toArrayCopy();
    sb.append("fst ").append(hex(b, 0, b.length)).append('\n');
    for (int[] k : keys) {
      byte[] kb = new byte[k.length];
      for (int i = 0; i < k.length; i++) kb[i] = (byte) k[i];
      sb.append("get ").append(key(k)).append(' ').append(show(Util.get(fst, new BytesRef(kb)))).append('\n');
    }
    Files.writeString(out.resolve(name + ".txt"), sb.toString());
  }

  static List<int[]> sortedKeys(Random r, int n, int maxLen, int alphabet, boolean allowEmpty) {
    TreeSet<int[]> set = new TreeSet<>(Arrays::compare);
    if (allowEmpty) set.add(new int[0]);
    while (set.size() < n) {
      int len = 1 + r.nextInt(maxLen);
      int[] k = new int[len];
      for (int i = 0; i < len; i++) k[i] = 'a' + r.nextInt(alphabet);
      set.add(k);
    }
    return new ArrayList<>(set);
  }

  static int[] repeats(Random r, int n, int max) {
    int[] a = new int[n];
    for (int i = 0; i < n; i++) a[i] = 1 + r.nextInt(max);
    return a;
  }

  /** Positive values from a small pool (so neighbours share prefixes) and the odd large one. */
  static long positive(Random r) {
    return switch (r.nextInt(4)) {
      case 0 -> 1 + r.nextInt(8);
      case 1 -> 100 + r.nextInt(4);
      case 2 -> 1 + r.nextInt(100_000);
      default -> (1L << 40) + r.nextInt(1000);
    };
  }

  static BytesRef bytes(Random r) {
    byte[] b = new byte[1 + r.nextInt(3)];
    for (int i = 0; i < b.length; i++) b[i] = (byte) ('p' + r.nextInt(3));
    return new BytesRef(b);
  }

  public static void main(String[] args) throws Exception {
    out = Path.of(args[0]).resolve("misc").resolve("fst");
    Files.createDirectories(out);
    Random r = new Random(1464L);
    for (boolean share : new boolean[] {true, false}) {
      List<int[]> keys = sortedKeys(r, 120, 6, 4, share);
      run(
          share ? "up_to_two_share" : "up_to_two_noshare",
          UpToTwoPositiveIntOutputs.getSingleton(share),
          keys,
          repeats(r, keys.size(), 2),
          rnd -> UpToTwoPositiveIntOutputs.getSingleton(share).get(positive(rnd)),
          share ? 11 : 12);
    }
    List<int[]> keys = sortedKeys(r, 120, 6, 4, true);
    run(
        "list_of_longs",
        new ListOfOutputs<Long>(PositiveIntOutputs.getSingleton()),
        keys,
        repeats(r, keys.size(), 4),
        rnd -> positive(rnd),
        13);
    keys = sortedKeys(r, 120, 6, 4, false);
    run(
        "list_of_bytes",
        new ListOfOutputs<BytesRef>(ByteSequenceOutputs.getSingleton()),
        keys,
        repeats(r, keys.size(), 3),
        rnd -> bytes(rnd),
        14);
  }
}
