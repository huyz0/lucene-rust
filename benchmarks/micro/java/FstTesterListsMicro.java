import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.misc.util.fst.ListOfOutputs;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.fst.FSTTester;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.fst.IntsRefFSTEnum;
import org.apache.lucene.util.fst.PositiveIntOutputs;

/** Complete actual tester, including list expansion, walks, seeks and output bytes. */
public class FstTesterListsMicro {
  static final ListOfOutputs<Long> OUTPUTS = new ListOfOutputs<>(PositiveIntOutputs.getSingleton());
  static volatile long sink;
  record Case(int mode, long seed, List<FSTTester.InputOutput<Object>> pairs) {}
  static List<Case> cases() throws Exception {
    List<Case> cases = new ArrayList<>();
    Case current = null;
    for (String row : Files.readAllLines(Path.of("fixtures/testframework/data/fst_tester_lists.tsv"))) {
      String[] fields = row.split("\t");
      switch (fields[0]) {
        case "case" -> current = fields[1].equals("mixed")
            ? new Case(Integer.parseInt(fields[2]), Long.parseLong(fields[3]), new ArrayList<>()) : null;
        case "pair" -> {
          if (current != null) {
            int[] input = fields[1].equals("-") ? new int[0]
                : java.util.Arrays.stream(fields[1].split(",")).mapToInt(Integer::parseInt).toArray();
            String[] output = fields[2].split(":");
            Object value;
            if (output[0].equals("S")) value = Long.valueOf(output[1]);
            else {
              List<Long> values = new ArrayList<>();
              for (String n : output[1].split(",")) values.add(Long.valueOf(n));
              value = values;
            }
            current.pairs.add(new FSTTester.InputOutput<>(new IntsRef(input, 0, input.length), value));
          }
        }
        case "end" -> { if (current != null) cases.add(current); current = null; }
        default -> {}
      }
    }
    if (cases.size() != 12) throw new AssertionError("cases");
    return cases;
  }
  static List<FSTTester.InputOutput<Object>> copy(List<FSTTester.InputOutput<Object>> pairs) {
    List<FSTTester.InputOutput<Object>> copy = new ArrayList<>(pairs.size());
    for (var pair : pairs) copy.add(new FSTTester.InputOutput<>(IntsRef.deepCopyOf(pair.input()),
        pair.output() instanceof List<?> values ? new ArrayList<>(values) : pair.output()));
    return copy;
  }
  static class Digest {
    long value;
    void mix(long n) { value = value * 31 + n; }
    void bytes(byte[] bytes) { mix(bytes.length); for (byte b : bytes) mix(b & 255); }
    void output(Object output) {
      if (output instanceof Long scalar) { mix(1); mix(scalar); }
      else { List<?> values = (List<?>) output; mix(2); mix(values.size()); for (Object value : values) mix((Long) value); }
    }
  }
  static long batch(List<Case> cases, int mode, Directory directory) throws Exception {
    Digest digest = new Digest();
    for (Case c : cases) if (c.mode == mode) {
      Random random = new Random(c.seed);
      FSTTester<Object> tester = new FSTTester<>(random, directory, mode, copy(c.pairs), OUTPUTS);
      var fst = tester.doTest();
      digest.mix(tester.nodeCount); digest.mix(tester.arcCount); digest.mix(random.nextInt());
      ByteBuffersDataOutput saved = new ByteBuffersDataOutput(); fst.save(saved, saved);
      digest.bytes(saved.toArrayCopy());
      var enumeration = new IntsRefFSTEnum<>(fst);
      for (var pair = enumeration.next(); pair != null; pair = enumeration.next()) {
        digest.mix(pair.input.length);
        for (int i = 0; i < pair.input.length; i++) digest.mix(pair.input.ints[pair.input.offset + i]);
        digest.output(pair.output);
      }
    }
    return digest.value;
  }
  static long[] run(List<Case> cases, int mode, Directory directory, int ms) throws Exception {
    long expected = batch(cases, mode, directory), start = System.nanoTime(), count = 0;
    do {
      long value = batch(cases, mode, directory);
      if (value != expected) throw new AssertionError("timed digest changed");
      sink = value; count += 6;
    } while (System.nanoTime() - start < ms * 1_000_000L);
    return new long[]{count, System.nanoTime() - start};
  }
  static long inputGraphBytes(FSTTester<Object> tester) {
    // copy() makes every ArrayList's backing capacity exactly its size. Long
    // references may share cached objects; count each reachable object once.
    long bytes = RamUsageEstimator.shallowSizeOf(tester) + RamUsageEstimator.shallowSizeOf(tester.pairs)
        + RamUsageEstimator.shallowSizeOf(new Object[tester.pairs.size()]);
    Set<Object> scalars = Collections.newSetFromMap(new IdentityHashMap<>());
    for (var pair : tester.pairs) {
      bytes += RamUsageEstimator.shallowSizeOf(pair) + RamUsageEstimator.shallowSizeOf(pair.input())
          + RamUsageEstimator.shallowSizeOf(pair.input().ints);
      if (pair.output() instanceof List<?> values) {
        bytes += RamUsageEstimator.shallowSizeOf(values) + RamUsageEstimator.shallowSizeOf(new Object[values.size()]);
        scalars.addAll(values);
      } else scalars.add(pair.output());
    }
    for (Object scalar : scalars) bytes += RamUsageEstimator.shallowSizeOf(scalar);
    return bytes;
  }
  public static void main(String[] args) throws Exception {
    if (!FstTesterListsMicro.class.desiredAssertionStatus() || LuceneTestCase.RANDOM_MULTIPLIER != 1)
      throw new AssertionError("run with -ea -Dtests.multiplier=1");
    int warmup = args.length > 0 ? Integer.parseInt(args[0]) : 1000;
    int measure = args.length > 1 ? Integer.parseInt(args[1]) : 1500;
    List<Case> cases = cases();
    try (Directory directory = new ByteBuffersDirectory()) {
      var sample = new FSTTester<>(new Random(0), directory, 0, copy(cases.get(0).pairs), OUTPUTS);
      System.out.printf("#memory\tfst_tester_lists_tester_input_graph\t%d%n", inputGraphBytes(sample));
      for (int mode : new int[]{0, 1}) {
        String name = mode == 0 ? "utf8" : "utf32";
        System.out.printf("#check\tfst_tester_lists_%s\t%d\t6%n", name, batch(cases, mode, directory));
        run(cases, mode, directory, warmup);
        long[] result = run(cases, mode, directory, measure);
        System.out.printf(java.util.Locale.ROOT, "fst_tester_lists_%s\t%.3f\t%d%n", name, (double) result[1] / result[0], result[0]);
      }
    }
  }
}
