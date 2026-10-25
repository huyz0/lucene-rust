import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeSet;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.tests.index.RandomCodec;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Observes the real RandomCodec; never computes an expected shuffle or selection. */
public class GenRandomCodec {
  static final StringBuilder OUT = new StringBuilder();
  static final String[] POSTINGS = {
    "Lucene104", "FST50", "Direct", "TestBloomFilteredLucenePostings", "MockRandom",
    "BlockTreeOrds", "LuceneFixedGap", "LuceneVarGapFixedInterval",
    "LuceneVarGapDocFreqInterval", "Asserting"
  };
  static final String[] FIELDS = {
    "", "a", "z", "Aa", "BB", "AaAa", "BBBB", "\u0000", "\u00e9",
    "\u4e2d", "\ud83d\ude00", "\ud800", "\udc00", "\uffff", "field", "field2"
  };

  static class TraceRandom extends Random {
    final List<Long> longDraws = new ArrayList<>();

    TraceRandom(long seed) { super(seed); }

    @Override
    public long nextLong() {
      long value = super.nextLong();
      longDraws.add(value);
      return value;
    }
  }

  static Field field(Class<?> type, String name) throws Exception {
    while (type != null) {
      try {
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        return field;
      } catch (NoSuchFieldException ignored) {
        type = type.getSuperclass();
      }
    }
    throw new NoSuchFieldException(name);
  }

  static Object get(Object owner, String name) throws Exception {
    return field(owner.getClass(), name).get(owner);
  }

  @SuppressWarnings("unchecked")
  static List<Object> formats(RandomCodec codec, boolean dv) throws Exception {
    return (List<Object>) get(codec, dv ? "dvFormats" : "formats");
  }

  static String name(Object format) {
    return format instanceof PostingsFormat p ? p.getName() : ((DocValuesFormat) format).getName();
  }

  static String text(String value) {
    StringBuilder result = new StringBuilder();
    for (int i = 0; i < value.length(); ++i) {
      if (i != 0) result.append(',');
      result.append(Integer.toHexString(value.charAt(i)));
    }
    return result.length() == 0 ? "-" : result.toString();
  }

  static void row(Object... cells) {
    StringJoiner line = new StringJoiner("\t");
    for (Object cell : cells) line.add(String.valueOf(cell));
    OUT.append(line).append('\n');
  }

  static String settings(Object format) throws Exception {
    StringJoiner result = new StringJoiner(",");
    // Only fields of the actual selected instance are read. In particular, the
    // two Lucene104 candidates may have different block sizes but the same SPI name.
    for (String key : new String[] {
        "minTermBlockSize", "maxTermBlockSize", "minSkipCount", "lowFreqCutoff",
        "termIndexInterval", "docFreqThreshold", "skipIndexIntervalSize"}) {
      try {
        result.add(key + "=" + get(format, key));
      } catch (NoSuchFieldException ignored) {
        // This setting belongs to another candidate class.
      }
    }
    return result.length() == 0 ? "-" : result.toString();
  }

  static Object select(RandomCodec codec, boolean dv, String field) {
    return dv ? codec.getDocValuesFormatForField(field) : codec.getPostingsFormatForField(field);
  }

  static String outcome(RandomCodec codec, boolean dv, String field) {
    try {
      return "ok:" + name(select(codec, dv, field));
    } catch (Throwable error) {
      return error.getClass().getSimpleName();
    }
  }

  static void capture(long seed, String avoidLabel, Set<String> avoid) throws Exception {
    TraceRandom random = new TraceRandom(seed);
    RandomCodec codec = new RandomCodec(random, avoid);
    String id = seed + ":" + avoidLabel;
    row("state", id, get(codec, "perFieldSeed"), get(codec, "maxPointsInLeafNode"),
        Long.toUnsignedString(Double.doubleToRawLongBits((double) get(codec, "maxMBSortInHeap"))),
        get(codec, "bkdSplitRandomSeed"), random.nextInt(), codec.avoidCodecs == avoid);
    // MockRandom is the sole constructor nextLong caller, including when avoided.
    if (random.longDraws.size() != 1) throw new AssertionError("constructor long draws changed");
    row("mock-seed", id, random.longDraws.get(0));
    row("names", id, new TreeSet<>(codec.formatNames), new TreeSet<>(codec.dvFormatNames));
    for (boolean dv : new boolean[] {false, true}) {
      IdentityHashMap<Object, Integer> identities = new IdentityHashMap<>();
      List<Object> retained = formats(codec, dv);
      for (int i = 0; i < retained.size(); ++i) {
        Object format = retained.get(i);
        if (identities.put(format, i) != null) throw new AssertionError("duplicate instance");
        row("format", id, dv, i, name(format), format.getClass().getSimpleName(), settings(format));
      }
      // Exercise both discovery orders; selected identity is checked on every repeat.
      List<String> fields = new ArrayList<>(Arrays.asList(FIELDS));
      if ((seed & 1) != 0) Collections.reverse(fields);
      for (String field : fields) {
        try {
          Object selected = select(codec, dv, field);
          Integer ordinal = identities.get(selected);
          if (ordinal == null || select(codec, dv, field) != selected) {
            throw new AssertionError("routing lost instance identity");
          }
          row("route", id, dv, text(field), field.hashCode(), ordinal);
        } catch (ArithmeticException | IndexOutOfBoundsException error) {
          row("route-error", id, dv, text(field), error.getClass().getSimpleName());
        }
      }
      row("null", id, dv, outcome(codec, dv, null));
    }
    row("tail", id, random.nextInt(), random.nextLong());
    row("points-fresh", id, codec.pointsFormat() != codec.pointsFormat());
  }

  static void edges() throws Exception {
    for (boolean dv : new boolean[] {false, true}) {
      RandomCodec codec = new RandomCodec(new Random(42));
      // Injection isolates the signed abs(MIN_VALUE) edge in the real method.
      // For postings use three retained real candidates; the normal list has four.
      if (!dv) field(RandomCodec.class, "formats").set(codec,
          new ArrayList<>(formats(codec, false).subList(0, 3)));
      field(RandomCodec.class, "perFieldSeed").setInt(codec, Integer.MIN_VALUE);
      row("min-abs", dv, outcome(codec, dv, ""));
      row("min-abs-retry", dv, outcome(codec, dv, ""));
      // With two real instances, abs(MIN_VALUE) has zero remainder and succeeds.
      field(RandomCodec.class, dv ? "dvFormats" : "formats").set(codec,
          new ArrayList<>(formats(codec, dv).subList(0, 2)));
      row("min-abs-zero-remainder", dv, outcome(codec, dv, ""));

      codec = new RandomCodec(new Random(42));
      // Cache entries are actual calls, not synthetic map inserts. In -ea mode,
      // the 10,000th insertion fails after putting the selected instance.
      for (int i = 0; i < 9999; ++i) select(codec, dv, "cache" + i);
      row("cache-limit", dv, outcome(codec, dv, "cache9999"));
      row("cache-retry", dv, outcome(codec, dv, "cache9999"));
      row("cache-next", dv, outcome(codec, dv, "cache10000"));
      row("cache-size", dv, ((Map<?, ?>) get(codec,
          dv ? "previousDVMappings" : "previousMappings")).size());
    }
    TreeSet<String> avoid = new TreeSet<>();
    RandomCodec codec = new RandomCodec(new Random(42), avoid);
    List<Object> before = new ArrayList<>(formats(codec, false));
    avoid.addAll(Arrays.asList(POSTINGS));
    row("avoid-mutation", codec.avoidCodecs == avoid,
        before.equals(formats(codec, false)), codec.avoidCodecs.size());
    Random random = new Random(42);
    try {
      new RandomCodec(random, null);
      throw new AssertionError("null avoid unexpectedly succeeded");
    } catch (NullPointerException expected) {
      row("null-avoid", expected.getClass().getSimpleName(), random.nextInt());
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    if (args.length == 4 && args[1].equals("--worker")) {
      boolean nightly = Boolean.parseBoolean(args[2]);
      int multiplier = Integer.parseInt(args[3]);
      if (!GenRandomCodec.class.desiredAssertionStatus()) throw new AssertionError("requires -ea");
      if (LuceneTestCase.TEST_NIGHTLY != nightly || LuceneTestCase.RANDOM_MULTIPLIER != multiplier) {
        throw new AssertionError("wrong randomized settings");
      }
      for (long seed = 0; seed < 256; ++seed) capture(seed, "none", Set.of());
      for (long seed : new long[] {42, -1, Long.MIN_VALUE, Long.MAX_VALUE}) {
        if (seed != 42) capture(seed, "none", Set.of());
        capture(seed, "unknown", Set.of("unknown"));
        for (String name : POSTINGS) capture(seed, "omit-" + name, Set.of(name));
        capture(seed, "all-postings", new TreeSet<>(Arrays.asList(POSTINGS)));
        capture(seed, "all-dv", Set.of("Lucene90", "Asserting"));
        capture(seed, "only-lucene104", new TreeSet<>(Arrays.stream(POSTINGS)
            .filter(name -> !name.equals("Lucene104")).toList()));
      }
      edges();
      Files.writeString(out.resolve("random_codec_" + nightly + "_" + multiplier + ".tsv"), OUT);
      return;
    }
    for (boolean nightly : new boolean[] {false, true}) {
      for (int multiplier : new int[] {1, 3}) {
        Process worker = new ProcessBuilder(
            Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "-ea",
            "-Dtests.seed=0", "-Dtests.nightly=" + nightly, "-Dtests.multiplier=" + multiplier,
            "-cp", System.getProperty("java.class.path"), "GenRandomCodec", out.toString(),
            "--worker", Boolean.toString(nightly), Integer.toString(multiplier))
            .inheritIO().start();
        if (worker.waitFor() != 0) throw new AssertionError("RandomCodec worker failed");
      }
    }
  }
}
