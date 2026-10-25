import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.IntsRefFSTEnum;
import org.apache.lucene.util.fst.PositiveIntOutputs;

/** Integer-label FSTEnum traces against Lucene 10.5.0, including reused cursors. */
public class GenIntsRefFstEnum {
  static String labels(int[] input) {
    if (input.length == 0) return "-";
    StringBuilder s = new StringBuilder();
    for (int label : input) {
      if (s.length() != 0) s.append(',');
      s.append(label);
    }
    return s.toString();
  }

  static String result(IntsRefFSTEnum.InputOutput<Long> pair) {
    if (pair == null) return "null";
    return labels(Arrays.copyOfRange(pair.input.ints, pair.input.offset,
        pair.input.offset + pair.input.length)) + ":" + pair.output;
  }

  static IntsRef ref(int[] input) {
    // A nonzero offset verifies the Java authority's IntsRef slice handling.
    int[] padded = new int[input.length + 2];
    System.arraycopy(input, 0, padded, 1, input.length);
    return new IntsRef(padded, 1, input.length);
  }

  static void generate(Path dir, String name, FST.INPUT_TYPE type,
      boolean fixed, float factor, boolean empty) throws Exception {
    TreeSet<int[]> keys = new TreeSet<>(Arrays::compare);
    if (empty) keys.add(new int[0]);
    int high = type == FST.INPUT_TYPE.BYTE2 ? 0x8000 : 0x10000;
    int max = type == FST.INPUT_TYPE.BYTE2 ? 0xffff : 0x10ffff;
    for (int label : new int[] {0, 255, 256, 65535, high, max}) {
      keys.add(new int[] {label});
      keys.add(new int[] {label, 0});
      keys.add(new int[] {label, 256, max});
    }
    for (int root = 0; root < 12; root++) {
      for (int child = 0; child < 24; child++) {
        int label = high + child * (name.equals("direct") ? 2 : 1);
        keys.add(new int[] {1000 + root * 31, label});
        keys.add(new int[] {1000 + root * 31, label, root});
      }
    }
    int[] deep = new int[30];
    Arrays.fill(deep, max);
    keys.add(deep);
    var outputs = PositiveIntOutputs.getSingleton();
    var compiler = new FSTCompiler.Builder<Long>(type, outputs)
        .allowFixedLengthArcs(fixed).directAddressingMaxOversizingFactor(factor).build();
    long ordinal = 0;
    for (int[] key : keys) compiler.add(ref(key), ordinal++ * 17 % 997);
    FST<Long> built = FST.fromFSTReader(compiler.compile(), compiler.getFSTReader());
    ByteBuffersDataOutput saved = new ByteBuffersDataOutput();
    built.save(saved, saved);
    byte[] bytes = saved.toArrayCopy();
    Files.write(dir.resolve(name + ".fst"), bytes);
    var in = new ByteArrayDataInput(bytes);
    FST<Long> fst = new FST<>(FST.readMetadata(in, outputs), in);
    var cursor = new IntsRefFSTEnum<>(fst);
    StringBuilder trace = new StringBuilder();
    ordinal = 0;
    for (int[] key : keys) {
      trace.append("key ").append(labels(key)).append(' ').append(ordinal++ * 17 % 997).append('\n');
      String got = result(cursor.next());
      if (!got.equals(labels(key) + ":" + ((ordinal - 1) * 17 % 997)))
        throw new AssertionError("Java save/read enumeration mismatch");
      trace.append("next - ").append(got).append('\n');
      trace.append("current - ").append(result(cursor.current())).append('\n');
    }
    trace.append("next - ").append(result(cursor.next())).append('\n');
    List<int[]> probes = new ArrayList<>();
    for (int[] key : keys) {
      probes.add(key);
      probes.add(Arrays.copyOf(key, key.length + 1));
      if (key.length > 0) {
        int[] bumped = key.clone();
        bumped[bumped.length - 1]++;
        probes.add(bumped);
      }
    }
    probes.add(new int[0]);
    probes.add(new int[] {max + 1});
    Collections.shuffle(probes, new Random(0x1a75));
    for (int[] probe : probes) {
      for (String op : List.of("ceil", "floor", "exact")) {
        var got = switch (op) {
          case "ceil" -> cursor.seekCeil(ref(probe));
          case "floor" -> cursor.seekFloor(ref(probe));
          default -> cursor.seekExact(ref(probe));
        };
        trace.append(op).append(' ').append(labels(probe)).append(' ').append(result(got)).append('\n');
        if (got != null) {
          trace.append("current - ").append(result(cursor.current())).append('\n');
          trace.append("next - ").append(result(cursor.next())).append('\n');
        }
      }
    }
    Files.writeString(dir.resolve(name + ".txt"), trace, StandardCharsets.UTF_8);
  }

  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[0], "ints_ref_fst_enum");
    Files.createDirectories(dir);
    generate(dir, "list", FST.INPUT_TYPE.BYTE4, false, 1, false);
    generate(dir, "binary", FST.INPUT_TYPE.BYTE4, true, 0, true);
    generate(dir, "direct", FST.INPUT_TYPE.BYTE4, true, 64, false);
    generate(dir, "continuous", FST.INPUT_TYPE.BYTE4, true, 1, true);
    generate(dir, "byte2", FST.INPUT_TYPE.BYTE2, true, 1, true);
  }
}
