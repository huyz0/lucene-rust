import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Function;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.CharSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.IntSequenceOutputs;
import org.apache.lucene.util.fst.NoOutputs;
import org.apache.lucene.util.fst.Outputs;
import org.apache.lucene.util.fst.PairOutputs;
import org.apache.lucene.util.fst.PositiveIntOutputs;

/**
 * Differential fixtures for {@link FSTCompiler}: byte-for-byte FST output.
 *
 * <p>Writes `fst_compiler/<case>.txt`, one per compiler configuration: a `config` line (outputs
 * type, input type, suffixRAMLimitMB, allowFixedLengthArcs, directAddressingMaxOversizingFactor,
 * version), one `k <input> <output>` line per `add` call in order, a `stats` line (node and arc
 * counts) and the `fst` line: `FST.save(out, out)` in hex. The Rust test replays the adds through
 * its port and compares the saved bytes exactly.
 *
 * <p>Inputs: hex for BYTE1, comma-separated ints for BYTE2/BYTE4, `-` when empty. Outputs: hex
 * (bytes), decimal (long), comma list (ints/chars), `-` (NoOutputs), `long|hex` (pair).
 *
 * <p>Deterministic: fixed seeds. Usage: java GenFstCompiler <outdir>
 */
public class GenFstCompiler {
  static Path out;

  static String hex(byte[] b, int off, int len) {
    if (len == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = off; i < off + len; i++) sb.append(String.format("%02x", b[i] & 0xFF));
    return sb.toString();
  }

  static String ints(int[] a, int off, int len) {
    if (len == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) {
      if (i > 0) sb.append(',');
      sb.append(a[off + i]);
    }
    return sb.toString();
  }

  static String input(FST.INPUT_TYPE t, int[] key) {
    if (t == FST.INPUT_TYPE.BYTE1) {
      byte[] b = new byte[key.length];
      for (int i = 0; i < key.length; i++) b[i] = (byte) key[i];
      return hex(b, 0, b.length);
    }
    return ints(key, 0, key.length);
  }

  static <T> void run(
      String name,
      Outputs<T> outputs,
      String outputsName,
      FST.INPUT_TYPE inputType,
      double ramMB,
      boolean allowFixed,
      float daFactor,
      int version,
      List<int[]> keys,
      List<T> values,
      Function<T, String> fmt)
      throws Exception {
    FSTCompiler<T> c =
        new FSTCompiler.Builder<>(inputType, outputs)
            .suffixRAMLimitMB(ramMB)
            .allowFixedLengthArcs(allowFixed)
            .directAddressingMaxOversizingFactor(daFactor)
            .setVersion(version)
            .build();
    StringBuilder sb = new StringBuilder();
    sb.append("config ")
        .append(outputsName)
        .append(' ')
        .append(inputType)
        .append(' ')
        .append(ramMB)
        .append(' ')
        .append(allowFixed)
        .append(' ')
        .append(daFactor)
        .append(' ')
        .append(version)
        .append('\n');
    for (int i = 0; i < keys.size(); i++) {
      int[] k = keys.get(i);
      c.add(new IntsRef(k, 0, k.length), values.get(i));
      sb.append("k ").append(input(inputType, k)).append(' ').append(fmt.apply(values.get(i))).append('\n');
    }
    FST.FSTMetadata<T> meta = c.compile();
    sb.append("stats ").append(c.getNodeCount()).append(' ').append(c.getArcCount()).append('\n');
    if (meta == null) {
      sb.append("fst -\n");
    } else {
      FST<T> fst = FST.fromFSTReader(meta, c.getFSTReader());
      ByteBuffersDataOutput o = new ByteBuffersDataOutput();
      fst.save(o, o);
      byte[] b = o.toArrayCopy();
      sb.append("fst ").append(hex(b, 0, b.length)).append('\n');
    }
    Files.writeString(out.resolve(name + ".txt"), sb.toString());
  }

  static List<int[]> sortedKeys(Random r, int n, int maxLen, int alphabet, int base, boolean allowEmpty) {
    TreeSet<int[]> set = new TreeSet<>(Arrays::compare);
    while (set.size() < n) {
      int len = (allowEmpty ? 0 : 1) + r.nextInt(maxLen);
      int[] k = new int[len];
      for (int i = 0; i < len; i++) k[i] = base + r.nextInt(alphabet);
      set.add(k);
    }
    return new ArrayList<>(set);
  }

  static byte[] randBytes(Random r, int max) {
    byte[] b = new byte[r.nextInt(max + 1)];
    r.nextBytes(b);
    return b;
  }

  static List<BytesRef> byteValues(Random r, List<int[]> keys, int max) {
    // Outputs that share prefixes with their neighbours, so output pushing has work to do.
    List<BytesRef> v = new ArrayList<>();
    byte[] prev = new byte[0];
    for (int i = 0; i < keys.size(); i++) {
      byte[] b;
      if (r.nextInt(3) == 0 && prev.length > 0) {
        byte[] tail = randBytes(r, 3);
        b = Arrays.copyOf(prev, r.nextInt(prev.length + 1) + tail.length);
        System.arraycopy(tail, 0, b, b.length - tail.length, tail.length);
      } else {
        b = randBytes(r, max);
      }
      v.add(new BytesRef(b));
      prev = b;
    }
    return v;
  }

  static List<Long> longValues(Random r, int n, int bound) {
    List<Long> v = new ArrayList<>();
    for (int i = 0; i < n; i++) v.add(r.nextInt(5) == 0 ? 0L : (long) r.nextInt(bound));
    return v;
  }

  static String bytesFmt(BytesRef b) {
    return hex(b.bytes, b.offset, b.length);
  }

  public static void main(String[] args) throws Exception {
    out = Path.of(args[0], "fst_compiler");
    Files.createDirectories(out);
    ByteSequenceOutputs bso = ByteSequenceOutputs.getSingleton();
    PositiveIntOutputs pio = PositiveIntOutputs.getSingleton();
    final FST.INPUT_TYPE B1 = FST.INPUT_TYPE.BYTE1;
    final int V = FST.VERSION_CURRENT;

    // Hand-picked: shared suffixes and pushed outputs.
    {
      String[] words = {"", "cat", "cats", "catsup", "dog", "doge", "dogs", "top", "tops", "zebra"};
      List<int[]> keys = new ArrayList<>();
      List<BytesRef> vals = new ArrayList<>();
      for (int i = 0; i < words.length; i++) {
        byte[] w = words[i].getBytes("UTF-8");
        int[] k = new int[w.length];
        for (int j = 0; j < w.length; j++) k[j] = w[j] & 0xff;
        keys.add(k);
        vals.add(new BytesRef(("out" + (i % 4)).getBytes("UTF-8")));
      }
      run("bytes_words", bso, "bytes", B1, 32, true, 1f, V, keys, vals, GenFstCompiler::bytesFmt);
    }

    Random r = new Random(0xF57);
    {
      List<int[]> keys = sortedKeys(r, 2000, 12, 6, 'a', false);
      run("bytes_random", bso, "bytes", B1, 32, true, 1f, V, keys, byteValues(r, keys, 6),
          GenFstCompiler::bytesFmt);
    }
    {
      // Every single byte and many two-byte keys: continuous and direct-addressing nodes.
      TreeSet<int[]> set = new TreeSet<>(Arrays::compare);
      for (int a = 0; a < 256; a++) {
        set.add(new int[] {a});
        if (a % 4 == 0) {
          for (int b = 0; b < 256; b += 1 + (a % 7) * (a % 5)) set.add(new int[] {a, b});
        }
      }
      List<int[]> keys = new ArrayList<>(set);
      run("bytes_dense", bso, "bytes", B1, 32, true, 1f, V, keys, byteValues(r, keys, 3),
          GenFstCompiler::bytesFmt);
    }
    List<int[]> wide = sortedKeys(r, 1500, 8, 256, 0, false);
    List<Long> wideVals = longValues(r, wide.size(), 1_000_000);
    run("long_random", pio, "long", B1, 32, true, 1f, V, wide, wideVals, String::valueOf);
    run("long_nofixed", pio, "long", B1, 32, false, 1f, V, wide, wideVals, String::valueOf);
    run("long_da_2", pio, "long", B1, 32, true, 2f, V, wide, wideVals, String::valueOf);
    run("long_da_0", pio, "long", B1, 32, true, 0f, V, wide, wideVals, String::valueOf);
    run("long_nodedup", pio, "long", B1, 0, true, 1f, V, wide, wideVals, String::valueOf);
    run("long_v8", pio, "long", B1, 32, true, 1f, FST.VERSION_90, wide, wideVals, String::valueOf);
    {
      List<int[]> keys = sortedKeys(r, 3000, 10, 4, 'w', false);
      List<Long> vals = longValues(r, keys.size(), 50);
      run("long_tinyram", pio, "long", B1, 0.002, true, 1f, V, keys, vals, String::valueOf);
      run("long_smallram", pio, "long", B1, 0.02, true, 1f, V, keys, vals, String::valueOf);
      run("long_fullram", pio, "long", B1, 32, true, 1f, V, keys, vals, String::valueOf);
    }
    {
      List<int[]> keys = sortedKeys(r, 1500, 5, 100000, 0, false);
      for (int i = 0; i < 20; i++) // Not within 8 of Integer.MAX_VALUE: Java's getNumPresenceBytes overflows there
        // and FSTCompiler throws (a Java bug the Rust port does not share).
        keys.add(new int[] {Integer.MAX_VALUE - 100 + i, r.nextInt(5)});
      List<IntsRef> vals = new ArrayList<>();
      for (int i = 0; i < keys.size(); i++) {
        int[] v = new int[r.nextInt(4)];
        for (int j = 0; j < v.length; j++) v[j] = r.nextInt(3) == 0 ? r.nextInt(3) : r.nextInt();
        if (v.length > 0 && v[0] < 0) v[0] = 1;
        vals.add(new IntsRef(v, 0, v.length));
      }
      run("ints_byte4", IntSequenceOutputs.getSingleton(), "ints", FST.INPUT_TYPE.BYTE4, 32, true, 1f,
          V, keys, vals, x -> ints(x.ints, x.offset, x.length));
    }
    {
      List<int[]> keys = sortedKeys(r, 1500, 6, 400, 0x4e00, true);
      for (int i = 0; i < 12; i++) keys.add(new int[] {65520 + i});
      List<CharsRef> vals = new ArrayList<>();
      for (int i = 0; i < keys.size(); i++) {
        char[] v = new char[r.nextInt(4)];
        for (int j = 0; j < v.length; j++) v[j] = (char) (r.nextInt(3) == 0 ? 'a' : r.nextInt(65536));
        vals.add(new CharsRef(v, 0, v.length));
      }
      run("chars_byte2", CharSequenceOutputs.getSingleton(), "chars", FST.INPUT_TYPE.BYTE2, 32, true,
          1f, V, keys, vals, x -> {
            int[] a = new int[x.length];
            for (int j = 0; j < x.length; j++) a[j] = x.chars[x.offset + j];
            return ints(a, 0, a.length);
          });
    }
    {
      List<int[]> base = sortedKeys(r, 800, 7, 26, 'a', true);
      List<int[]> keys = new ArrayList<>();
      List<Object> vals = new ArrayList<>();
      for (int[] k : base) {
        keys.add(k);
        vals.add(NoOutputs.getSingleton().getNoOutput());
        if (r.nextInt(10) == 0) {
          keys.add(k);
          vals.add(NoOutputs.getSingleton().getNoOutput());
        }
      }
      run("none_dups", NoOutputs.getSingleton(), "none", B1, 32, true, 1f, V, keys, vals, x -> "-");
    }
    {
      PairOutputs<Long, BytesRef> po = new PairOutputs<>(pio, bso);
      List<int[]> keys = sortedKeys(r, 1200, 9, 30, 'A', false);
      List<BytesRef> bv = byteValues(r, keys, 4);
      List<PairOutputs.Pair<Long, BytesRef>> vals = new ArrayList<>();
      for (int i = 0; i < keys.size(); i++) {
        vals.add(po.newPair(r.nextInt(4) == 0 ? 0L : (long) r.nextInt(1000), bv.get(i)));
      }
      run("pair", po, "pair", B1, 32, true, 1f, V, keys, vals,
          x -> x.output1 + "|" + bytesFmt(x.output2));
    }
    run("empty_only", bso, "bytes", B1, 32, true, 1f, V, List.of(new int[0]),
        List.of(new BytesRef(new byte[] {7, 8})), GenFstCompiler::bytesFmt);
    run("empty_noout", pio, "long", B1, 32, true, 1f, V, List.of(new int[0]), List.of(0L),
        String::valueOf);
    run("single", pio, "long", B1, 32, true, 1f, V, List.of(new int[] {'x', 'y'}), List.of(42L),
        String::valueOf);
    run("nothing", pio, "long", B1, 32, true, 1f, V, List.of(), List.of(), String::valueOf);
  }
}
