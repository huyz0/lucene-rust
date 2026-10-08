import com.ibm.icu.util.BytesTrie;
import com.ibm.icu.util.BytesTrieBuilder;
import com.ibm.icu.util.CharsTrie;
import com.ibm.icu.util.CharsTrieBuilder;
import com.ibm.icu.util.StringTrieBuilder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * M12 T12.4: ICU4J 77.1's {@code BytesTrie} and {@code CharsTrie} (what the dictionary break engines
 * match against), built by ICU4J's own builders over seeded key sets whose values and sizes reach
 * every value and jump-delta encoding, then walked unit by unit; compared by {@code
 * crates/lucene-analysis-icu/tests/icu_tries_fixtures.rs}.
 *
 * <p>{@code tries.txt}: per trie a {@code T kind option hex} line ({@code B} bytes or {@code C} UTF-16
 * units, {@code FAST}/{@code SMALL}), then one {@code Q} line per query -- the query's units in hex,
 * the result of {@code first} and each {@code next} as digits (Java's ordinals), and {@code
 * getValue()} after the last step when it has one ({@code -} otherwise). Code point queries ({@code
 * P}) walk a {@code CharsTrie} with {@code firstForCodePoint}/{@code nextForCodePoint}.
 */
public class GenAnalysisIcuTries {
  static final int[] VALUE_SIZES = {16, 1 << 10, 1 << 17, 1 << 25, Integer.MAX_VALUE};

  static String hex(byte[] b, int len) {
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < len; i++) s.append(String.format("%02x", b[i] & 0xff));
    return s.toString();
  }

  static String hexUnits(CharSequence s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) b.append(String.format("%04x", (int) s.charAt(i)));
    return b.toString();
  }

  static String key(Random rnd, int alphabet, int maxLen, boolean supplementary) {
    StringBuilder s = new StringBuilder();
    int len = 1 + rnd.nextInt(maxLen);
    for (int i = 0; i < len; i++) {
      if (supplementary && rnd.nextInt(8) == 0) {
        s.appendCodePoint(0x10400 + rnd.nextInt(4));
      } else {
        s.append((char) ('a' + rnd.nextInt(alphabet)));
      }
    }
    return s.toString();
  }

  static int value(Random rnd, int bound) {
    int v = rnd.nextInt(bound);
    return rnd.nextInt(6) == 0 ? -v - 1 : v;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_icu_tries");
    Files.createDirectories(out);
    StringBuilder sb = new StringBuilder();
    Random rnd = new Random(0x74726965L);
    // (key count, alphabet, max key length): small tries, deep linear runs, wide branches, and
    // tries big enough for three- and four-byte jump deltas.
    int[][] shapes = {{8, 3, 4}, {40, 26, 12}, {300, 6, 30}, {2000, 26, 8}, {6000, 20, 10}};
    for (int[] shape : shapes) {
      for (int vs : VALUE_SIZES) {
        for (StringTrieBuilder.Option option : StringTrieBuilder.Option.values()) {
          if (shape[0] >= 300 && vs != Integer.MAX_VALUE) continue;
          Map<String, Integer> keys = new LinkedHashMap<>();
          while (keys.size() < shape[0]) keys.put(key(rnd, shape[1], shape[2], false), value(rnd, vs));
          // bytes
          BytesTrieBuilder bb = new BytesTrieBuilder();
          for (Map.Entry<String, Integer> e : keys.entrySet()) {
            bb.add(e.getKey().getBytes(StandardCharsets.ISO_8859_1), e.getKey().length(), e.getValue());
          }
          ByteBuffer buf = bb.buildByteBuffer(option);
          byte[] bytes = new byte[buf.remaining()];
          buf.get(bytes);
          sb.append("T\tB\t").append(option).append('\t').append(hex(bytes, bytes.length)).append('\n');
          BytesTrie bt = new BytesTrie(bytes, 0);
          for (String q : queries(rnd, keys, shape)) {
            byte[] qb = q.getBytes(StandardCharsets.ISO_8859_1);
            StringBuilder r = new StringBuilder();
            BytesTrie.Result res = null;
            for (int i = 0; i < qb.length; i++) {
              res = i == 0 ? bt.first(qb[i] & 0xff) : bt.next(qb[i] & 0xff);
              r.append(res.ordinal());
              if (res == BytesTrie.Result.NO_MATCH) break;
            }
            sb.append("Q\t").append(hex(qb, qb.length)).append('\t').append(r).append('\t')
                .append(res != null && res.hasValue() ? Integer.toString(bt.getValue()) : "-")
                .append('\t').append(bt.current().ordinal()).append('\n');
          }
          if (shape[0] > 300) continue;
          // chars, with some supplementary keys
          Map<String, Integer> ckeys = new LinkedHashMap<>();
          while (ckeys.size() < shape[0]) ckeys.put(key(rnd, shape[1], shape[2], true), value(rnd, vs));
          CharsTrieBuilder cb = new CharsTrieBuilder();
          for (Map.Entry<String, Integer> e : ckeys.entrySet()) cb.add(e.getKey(), e.getValue());
          String chars = cb.buildCharSequence(option).toString();
          sb.append("T\tC\t").append(option).append('\t').append(hexUnits(chars)).append('\n');
          CharsTrie ct = new CharsTrie(chars, 0);
          for (String q : queries(rnd, ckeys, shape)) {
            StringBuilder r = new StringBuilder();
            BytesTrie.Result res = null;
            for (int i = 0; i < q.length(); i++) {
              res = i == 0 ? ct.first(q.charAt(i)) : ct.next(q.charAt(i));
              r.append(res.ordinal());
              if (res == BytesTrie.Result.NO_MATCH) break;
            }
            sb.append("Q\t").append(hexUnits(q)).append('\t').append(r).append('\t')
                .append(res != null && res.hasValue() ? Integer.toString(ct.getValue()) : "-")
                .append('\t').append(ct.current().ordinal()).append('\n');
            r.setLength(0);
            res = null;
            for (int i = 0; i < q.length(); ) {
              int cp = q.codePointAt(i);
              res = i == 0 ? ct.firstForCodePoint(cp) : ct.nextForCodePoint(cp);
              r.append(res.ordinal());
              i += Character.charCount(cp);
              if (res == BytesTrie.Result.NO_MATCH) break;
            }
            sb.append("P\t").append(hexUnits(q)).append('\t').append(r).append('\t')
                .append(res != null && res.hasValue() ? Integer.toString(ct.getValue()) : "-")
                .append('\n');
          }
        }
      }
    }
    Files.writeString(out.resolve("tries.txt"), sb.toString(), StandardCharsets.UTF_8);
  }

  /** Up to 100 keys, prefixes and extensions of keys, and random strings. */
  static List<String> queries(Random rnd, Map<String, Integer> keys, int[] shape) {
    List<String> q = new ArrayList<>();
    List<String> all = new ArrayList<>(keys.keySet());
    for (int i = 0; i < Math.min(100, all.size()); i++) {
      String k = all.get(rnd.nextInt(all.size()));
      q.add(k);
      if (k.length() > 1) q.add(k.substring(0, 1 + rnd.nextInt(k.length() - 1)));
      q.add(k + (char) ('a' + rnd.nextInt(shape[1])));
    }
    for (int i = 0; i < 20; i++) q.add(key(rnd, shape[1] + 1, shape[2], false));
    return q;
  }
}
