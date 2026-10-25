import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import org.apache.lucene.misc.util.fst.ListOfOutputs;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.fst.FSTTester;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.IntsRefFSTEnum;
import org.apache.lucene.util.fst.PositiveIntOutputs;

/** Actual FSTTester list expansion, complete verification, saved bytes and RNG tails. */
public class GenFstTesterLists {
  static final StringBuilder OUT = new StringBuilder();
  static final long[] SEEDS = {0, 1, 42, -1, Long.MIN_VALUE, Long.MAX_VALUE};
  static final String[] CASES = {
    "empty", "scalars", "mixed", "empty-key", "single-key", "empty-list", "one-list", "unsorted", "duplicate-pairs"
  };
  static final ListOfOutputs<Long> OUTPUTS = new ListOfOutputs<>(PositiveIntOutputs.getSingleton());

  static FSTTester.InputOutput<Object> pair(int mode, String input, Object output) {
    return new FSTTester.InputOutput<>(FSTTester.toIntsRef(input, mode), output);
  }
  static List<FSTTester.InputOutput<Object>> pairs(String name, int mode) {
    List<FSTTester.InputOutput<Object>> pairs = new ArrayList<>();
    switch (name) {
      case "empty" -> {}
      case "scalars" -> {
        pairs.add(pair(mode, "", 0L)); pairs.add(pair(mode, "a", 7L));
        pairs.add(pair(mode, "ab", 0L)); pairs.add(pair(mode, "\u00e9", 13L));
        pairs.add(pair(mode, "\ud83d\ude00", 9L));
      }
      case "mixed" -> {
        pairs.add(pair(mode, "", List.of(0L, 7L, 0L)));
        pairs.add(pair(mode, "a", List.of(3L, 5L)));
        pairs.add(pair(mode, "ab", 0L));
        pairs.add(pair(mode, "abc", List.of(0L, 1L, 1L, 1024L)));
        pairs.add(pair(mode, "abd", List.of(11L, 2L, 11L)));
        pairs.add(pair(mode, "b", 1L));
        pairs.add(pair(mode, "\u00e9", List.of(9L, 0L, 13L)));
        pairs.add(pair(mode, "\u00e9a", List.of(1L << 40, 0L, 2L)));
        pairs.add(pair(mode, "\ud83d\ude00", List.of(0L, 4L)));
        pairs.add(pair(mode, "\ud83d\ude00a", List.of(7L, 0L, 7L)));
        pairs.add(pair(mode, "\ud800\udc00", 17L));
        pairs.add(pair(mode, "\udbff\udfff", List.of(0L, 0L, 1L)));
      }
      case "empty-key" -> pairs.add(pair(mode, "", List.of(0L, 8L, 0L)));
      case "single-key" -> pairs.add(pair(mode, "ab", List.of(0L, 2L, 1L << 40)));
      case "empty-list" -> pairs.add(pair(mode, "a", List.of()));
      case "one-list" -> pairs.add(pair(mode, "a", List.of(7L)));
      case "unsorted" -> {
        pairs.add(pair(mode, "b", List.of(1L, 2L)));
        pairs.add(pair(mode, "a", List.of(3L, 4L)));
      }
      case "duplicate-pairs" -> {
        pairs.add(pair(mode, "a", 1L)); pairs.add(pair(mode, "a", 2L));
      }
      default -> throw new AssertionError(name);
    }
    if (!name.equals("unsorted")) Collections.sort(pairs);
    return pairs;
  }
  static String input(IntsRef input) {
    if (input.length == 0) return "-";
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < input.length; i++) {
      if (i != 0) text.append(',');
      text.append(input.ints[input.offset + i]);
    }
    return text.toString();
  }
  static String output(Object value) {
    if (value instanceof Long scalar) return "S:" + scalar;
    StringBuilder text = new StringBuilder("M:");
    for (Object scalar : (List<?>) value) {
      if (text.length() != 2) text.append(',');
      text.append(scalar);
    }
    return text.toString();
  }
  static void run(String name, int mode, long seed) throws Exception {
    Random random = new Random(seed);
    var pairs = pairs(name, mode);
    OUT.append("case\t").append(name).append('\t').append(mode).append('\t').append(seed).append('\n');
    for (var pair : pairs) OUT.append("pair\t").append(input(pair.input())).append('\t').append(output(pair.output())).append('\n');
    try (Directory directory = new ByteBuffersDirectory()) {
      FSTTester<Object> tester = new FSTTester<>(random, directory, mode, pairs, OUTPUTS);
      FST<Object> fst = null;
      String status;
      try {
        fst = tester.doTest();
        status = fst == null ? "none" : "ok";
      } catch (AssertionError expected) {
        if (!List.of("empty-list", "one-list", "unsorted", "duplicate-pairs").contains(name)) throw expected;
        status = "assert";
      }
      OUT.append("result\t").append(status).append('\t').append(tester.nodeCount).append('\t')
          .append(tester.arcCount).append('\t').append(random.nextInt()).append('\n');
      if (fst != null) {
        ByteBuffersDataOutput saved = new ByteBuffersDataOutput();
        fst.save(saved, saved);
        OUT.append("fst\t").append(HexFormat.of().formatHex(saved.toArrayCopy())).append('\n');
        var enumeration = new IntsRefFSTEnum<>(fst);
        for (var item = enumeration.next(); item != null; item = enumeration.next())
          OUT.append("enum\t").append(input(item.input)).append('\t').append(output(item.output)).append('\n');
      }
      OUT.append("end\n");
    }
  }
  public static void main(String[] args) throws Exception {
    if (!GenFstTesterLists.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    if (LuceneTestCase.RANDOM_MULTIPLIER != 1) throw new AssertionError("tests.multiplier must be 1");
    for (long seed : SEEDS) for (int mode : new int[]{0, 1}) for (String name : CASES) run(name, mode, seed);
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("fst_tester_lists.tsv"), OUT, StandardCharsets.UTF_8);
  }
}
