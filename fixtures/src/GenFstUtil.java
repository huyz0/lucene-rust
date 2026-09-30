import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.TreeMap;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Outputs;
import org.apache.lucene.util.fst.PairOutputs;
import org.apache.lucene.util.fst.PositiveIntOutputs;
import org.apache.lucene.util.fst.Util;

/**
 * Differential fixtures for {@code org.apache.lucene.util.fst.Util} and FST arc reading through
 * typed outputs: {@code Util.get}, {@code shortestPaths}/{@code TopNSearcher} (from the root and
 * from prefix nodes, with and without an {@code acceptResult} filter), {@code readCeilArc}, {@code
 * FST.readLastTargetArc} and {@code toDot}.
 *
 * <p>Writes `fst_util/<case>.txt`: an `fst` line (the saved FST, hex) then query lines with
 * Lucene's answers. Inputs are hex byte strings (`-` empty); a missing answer is `null`.
 *
 * <p>Deterministic: fixed seeds. Usage: java GenFstUtil <outdir>
 */
public class GenFstUtil {
  static String hex(byte[] b) {
    if (b.length == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xFF));
    return sb.toString();
  }

  static String hexInts(IntsRef r) {
    byte[] b = new byte[r.length];
    for (int i = 0; i < r.length; i++) b[i] = (byte) r.ints[r.offset + i];
    return hex(b);
  }

  static <T> String save(FST<T> fst) throws Exception {
    ByteBuffersDataOutput o = new ByteBuffersDataOutput();
    fst.save(o, o);
    return hex(o.toArrayCopy());
  }

  interface Fmt<T> {
    String f(T t);
  }

  static <T> void queries(
      StringBuilder sb, FST<T> fst, byte[][] keys, Comparator<T> cmp, Fmt<T> fmt, Random r)
      throws Exception {
    Outputs<T> outs = fst.outputs;
    // get: keys, and near-misses.
    for (int i = 0; i < keys.length; i += 1 + keys.length / 60) {
      byte[] k = keys[i];
      T v = Util.get(fst, new BytesRef(k));
      sb.append("get ").append(hex(k)).append(' ').append(v == null ? "null" : fmt.f(v)).append('\n');
      byte[] miss = Arrays.copyOf(k, k.length + 1);
      miss[k.length] = (byte) r.nextInt(256);
      T m = Util.get(fst, new BytesRef(miss));
      sb.append("get ").append(hex(miss)).append(' ').append(m == null ? "null" : fmt.f(m)).append('\n');
    }
    FST.BytesReader in = fst.getBytesReader();
    // Prefix nodes: the root plus prefixes of some keys.
    for (int q = 0; q < 25; q++) {
      byte[] prefix =
          q == 0 ? new byte[0] : Arrays.copyOf(keys[r.nextInt(keys.length)], 0);
      if (q > 0) {
        byte[] k = keys[r.nextInt(keys.length)];
        prefix = Arrays.copyOf(k, r.nextInt(k.length + 1));
      }
      FST.Arc<T> arc = fst.getFirstArc(new FST.Arc<>());
      T out = outs.getNoOutput();
      boolean ok = true;
      for (byte b : prefix) {
        if (fst.findTargetArc(b & 0xff, arc, arc, in) == null) {
          ok = false;
          break;
        }
        out = outs.add(out, arc.output());
      }
      if (!ok) continue;
      for (int topN : new int[] {1, 3, 10}) {
        boolean allowEmpty = r.nextBoolean();
        Util.TopResults<T> res = Util.shortestPaths(fst, arc, out, cmp, topN, allowEmpty);
        sb.append("top ").append(hex(prefix)).append(' ').append(topN).append(' ').append(allowEmpty)
            .append(' ').append(res.isComplete).append(' ').append(res.topN.size());
        for (Util.Result<T> x : res) sb.append(' ').append(hexInts(x.input())).append(':').append(fmt.f(x.output()));
        sb.append('\n');
      }
      // A deeper queue with a result filter (rejects every other result by output).
      final int[] n = {0};
      Util.TopNSearcher<T> s =
          new Util.TopNSearcher<>(fst, 4, 12, cmp) {
            @Override
            protected boolean acceptResult(IntsRef input, T output) {
              return (n[0]++ & 1) == 0;
            }
          };
      s.addStartPaths(arc, out, true, new IntsRefBuilder());
      Util.TopResults<T> res = s.search();
      sb.append("filtered ").append(hex(prefix)).append(' ').append(res.isComplete).append(' ').append(res.topN.size());
      for (Util.Result<T> x : res) sb.append(' ').append(hexInts(x.input())).append(':').append(fmt.f(x.output()));
      sb.append('\n');
      // readCeilArc at a few labels, including END_LABEL.
      for (int label : new int[] {-1, 0, r.nextInt(256), r.nextInt(256), 'a', 255}) {
        FST.Arc<T> c = Util.readCeilArc(label, fst, arc, new FST.Arc<>(), in);
        sb.append("ceil ").append(hex(prefix)).append(' ').append(label).append(' ');
        if (c == null) sb.append("null");
        else sb.append(c.label()).append(' ').append(fmt.f(c.output())).append(' ').append(c.isFinal());
        sb.append('\n');
      }
    }
  }

  static <T> FST<T> build(Outputs<T> outputs, TreeMap<BytesRef, T> entries, boolean fixed)
      throws Exception {
    FSTCompiler<T> c =
        new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs).allowFixedLengthArcs(fixed).build();
    IntsRefBuilder scratch = new IntsRefBuilder();
    for (var e : entries.entrySet()) c.add(Util.toIntsRef(e.getKey(), scratch), e.getValue());
    return FST.fromFSTReader(c.compile(), c.getFSTReader());
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0], "fst_util");
    Files.createDirectories(out);
    PositiveIntOutputs pio = PositiveIntOutputs.getSingleton();
    Random r = new Random(0x5171);

    for (String kind : new String[] {"wide", "narrow", "narrow_nofixed", "dense"}) {
      TreeMap<BytesRef, Long> m = new TreeMap<>();
      int n = kind.equals("dense") ? 700 : 1500;
      while (m.size() < n) {
        int len = 1 + r.nextInt(kind.equals("wide") ? 6 : 9);
        byte[] k = new byte[len];
        for (int i = 0; i < len; i++) {
          k[i] =
              (byte)
                  (kind.equals("wide")
                      ? r.nextInt(256)
                      : kind.equals("dense") ? r.nextInt(40) + (i == 0 ? 0 : 'a') : 'a' + r.nextInt(5));
        }
        m.put(new BytesRef(k), r.nextInt(6) == 0 ? 0L : (long) r.nextInt(10000));
      }
      if (kind.equals("narrow")) m.put(new BytesRef(), 17L);
      FST<Long> fst = build(pio, m, !kind.equals("narrow_nofixed"));
      byte[][] keys = m.keySet().stream().map(b -> Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length)).toArray(byte[][]::new);
      StringBuilder sb = new StringBuilder("outputs long\nfst " + save(fst) + "\n");
      queries(sb, fst, keys, Long::compare, String::valueOf, r);
      Files.writeString(out.resolve("long_" + kind + ".txt"), sb.toString());
    }

    {
      PairOutputs<Long, BytesRef> po = new PairOutputs<>(pio, ByteSequenceOutputs.getSingleton());
      TreeMap<BytesRef, PairOutputs.Pair<Long, BytesRef>> m = new TreeMap<>();
      while (m.size() < 800) {
        byte[] k = new byte[1 + r.nextInt(7)];
        for (int i = 0; i < k.length; i++) k[i] = (byte) ('a' + r.nextInt(8));
        byte[] p = new byte[r.nextInt(3)];
        r.nextBytes(p);
        m.put(new BytesRef(k), po.newPair(r.nextInt(5) == 0 ? 0L : (long) r.nextInt(500), new BytesRef(p)));
      }
      FST<PairOutputs.Pair<Long, BytesRef>> fst = build(po, m, true);
      byte[][] keys = m.keySet().stream().map(b -> Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length)).toArray(byte[][]::new);
      StringBuilder sb = new StringBuilder("outputs pair\nfst " + save(fst) + "\n");
      queries(
          sb,
          fst,
          keys,
          (a, b) -> Long.compare(a.output1, b.output1),
          x -> x.output1 + "|" + hex(Arrays.copyOfRange(x.output2.bytes, x.output2.offset, x.output2.offset + x.output2.length)),
          r);
      Files.writeString(out.resolve("pair.txt"), sb.toString());
    }

    {
      // toDot over a small FST with every node encoding.
      TreeMap<BytesRef, Long> m = new TreeMap<>();
      for (String w : new String[] {"", "a", "ab", "abc", "abd", "b", "ba", "bb", "bc", "bd", "be", "bf", "c\"q", "d\\"}) {
        m.put(new BytesRef(w), (long) w.length() * 3);
      }
      for (int c = 'f'; c < 'w'; c += 2) m.put(new BytesRef("e" + (char) c), 1L);
      FST<Long> fst = build(pio, m, true);
      StringWriter w = new StringWriter();
      Util.toDot(fst, w, true, true);
      StringWriter w2 = new StringWriter();
      Util.toDot(fst, w2, false, false);
      Files.writeString(
          out.resolve("dot.txt"),
          "outputs long\nfst " + save(fst) + "\ndot1 " + hex(w.toString().getBytes(StandardCharsets.UTF_8))
              + "\ndot2 " + hex(w2.toString().getBytes(StandardCharsets.UTF_8)) + "\n");
    }
  }
}
