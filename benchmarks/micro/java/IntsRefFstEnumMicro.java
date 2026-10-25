import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.IntsRefFSTEnum;
import org.apache.lucene.util.fst.PositiveIntOutputs;

/** Integer enumeration, over the identical Java fixture bytes as micro_ints_ref_fst_enum.rs. */
public class IntsRefFstEnumMicro {
  static long add(long hash, long word) {
    return (hash ^ word) * 0x100000001b3L;
  }

  static long result(long hash, IntsRefFSTEnum.InputOutput<Long> pair) {
    if (pair == null) return add(hash, -1);
    hash = add(hash, pair.input.length);
    for (int i = 0; i < pair.input.length; i++)
      hash = add(hash, pair.input.ints[pair.input.offset + i]);
    return add(hash, pair.output);
  }

  static long run(FST<Long> fst, List<int[]> probes, String mode) throws IOException {
    var cursor = new IntsRefFSTEnum<>(fst);
    long hash = 0xcbf29ce484222325L;
    if (mode.equals("next")) {
      IntsRefFSTEnum.InputOutput<Long> pair;
      while ((pair = cursor.next()) != null) hash = result(hash, pair);
    } else {
      for (int[] probe : probes) {
        var target = new IntsRef(probe, 0, probe.length);
        var pair = switch (mode) {
          case "exact" -> cursor.seekExact(target);
          case "ceil" -> cursor.seekCeil(target);
          case "floor" -> cursor.seekFloor(target);
          default -> throw new AssertionError(mode);
        };
        hash = result(hash, pair);
      }
    }
    return hash;
  }

  public static void main(String[] args) throws Exception {
    Path dir = Path.of("fixtures/data/ints_ref_fst_enum");
    var input = new ByteArrayDataInput(Files.readAllBytes(dir.resolve("continuous.fst")));
    var fst = new FST<Long>(FST.readMetadata(input, PositiveIntOutputs.getSingleton()), input);
    List<int[]> keys = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("continuous.txt"))) {
      if (!line.startsWith("key ")) continue;
      String key = line.split(" ")[1];
      keys.add(key.equals("-") ? new int[0] : Arrays.stream(key.split(",")).mapToInt(Integer::parseInt).toArray());
    }
    List<int[]> probes = new ArrayList<>();
    for (int[] key : keys) {
      probes.add(key);
      int[] longer = Arrays.copyOf(key, key.length + 1);
      longer[key.length] = 1;
      probes.add(longer);
    }
    Collections.reverse(probes);
    for (String mode : List.of("next", "exact", "ceil", "floor")) {
      String name = "ints_enum_" + mode;
      System.out.printf("#check\t%s\t%016x%n", name, run(fst, probes, mode));
      long units = mode.equals("next") ? keys.size() : probes.size();
      SweepMicro.measure(name, () -> {
        SweepMicro.sink = run(fst, probes, mode);
        return units;
      });
    }
  }
}
