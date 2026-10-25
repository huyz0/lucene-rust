import java.lang.reflect.Field;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.tests.index.RandomCodec;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Actual Java RandomCodec routing; constructor and key preparation are untimed. */
public class RandomCodecMicro {
  static final int KEYS = 4096;
  static volatile long sink;
  static final class Digest {
    long value;
    void mix(long n) { value = value * 31 + n; }
    void text(String text) {
      mix(text.length());
      for (int i = 0; i < text.length(); ++i) mix(text.charAt(i));
    }
    void setting(String name, long n) { text(name); mix(n); }
  }
  static final class TraceRandom extends Random {
    long constructorLong;
    int longCalls;
    TraceRandom(long seed) { super(seed); }
    @Override public long nextLong() {
      long value = super.nextLong();
      constructorLong = value;
      ++longCalls;
      return value;
    }
  }
  static Object member(Object owner, String name) throws Exception {
    for (Class<?> type = owner.getClass(); type != null; type = type.getSuperclass()) {
      try {
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(owner);
      } catch (NoSuchFieldException ignored) {
        // Candidate-specific settings are intentionally absent on other formats.
      }
    }
    throw new NoSuchFieldException(name);
  }
  static String name(Object format) {
    return format instanceof PostingsFormat p ? p.getName() : ((DocValuesFormat) format).getName();
  }
  static long token(Object format, long mockSeed) throws Exception {
    Digest digest = new Digest();
    digest.text(name(format));
    for (String key : new String[] {"minTermBlockSize", "maxTermBlockSize", "minSkipCount",
        "lowFreqCutoff", "termIndexInterval", "docFreqThreshold", "skipIndexIntervalSize"}) {
      try { digest.setting(key, ((Number) member(format, key)).longValue()); }
      catch (NoSuchFieldException ignored) { /* Not a setting of this real candidate. */ }
    }
    if (name(format).equals("MockRandom")) digest.setting("seed", mockSeed);
    return digest.value;
  }
  static final class Prepared {
    final RandomCodec codec;
    final Object[] postings, docValues;
    final long[] postingsTokens, docValuesTokens;
    final int constructorTail;
    Prepared() throws Exception {
      TraceRandom random = new TraceRandom(42);
      codec = new RandomCodec(random);
      if (random.longCalls != 1) throw new AssertionError("constructor long draws changed");
      postings = ((List<?>) member(codec, "formats")).toArray();
      docValues = ((List<?>) member(codec, "dvFormats")).toArray();
      postingsTokens = new long[postings.length];
      docValuesTokens = new long[docValues.length];
      for (int i = 0; i < postings.length; ++i) postingsTokens[i] = token(postings[i], random.constructorLong);
      for (int i = 0; i < docValues.length; ++i) docValuesTokens[i] = token(docValues[i], random.constructorLong);
      constructorTail = random.nextInt();
    }
    long setupDigest(String[] fields) throws Exception {
      Digest digest = new Digest();
      digest.mix((int) member(codec, "perFieldSeed"));
      digest.mix(constructorTail);
      digest.mix((int) member(codec, "maxPointsInLeafNode"));
      digest.mix(Double.doubleToRawLongBits((double) member(codec, "maxMBSortInHeap")));
      digest.mix((int) member(codec, "bkdSplitRandomSeed"));
      for (var names : List.of(new TreeSet<>(codec.formatNames), new TreeSet<>(codec.dvFormatNames))) {
        digest.mix(names.size());
        for (String name : names) digest.text(name);
      }
      for (long[] tokens : new long[][] {postingsTokens, docValuesTokens}) {
        digest.mix(tokens.length);
        for (long value : tokens) digest.mix(value);
      }
      for (String field : fields) {
        digest.mix(field.length());
        for (int i = 0; i < field.length(); ++i) digest.mix(field.charAt(i));
        digest.mix(field.hashCode());
      }
      return digest.value;
    }
    long batch(String[] fields, boolean dv, boolean cached) {
      Digest digest = new Digest();
      Object[] candidates = dv ? docValues : postings;
      long[] tokens = dv ? docValuesTokens : postingsTokens;
      int rounds = cached ? 4 : 1;
      for (int round = 0; round < rounds; ++round) {
        for (int index = 0; index < fields.length; ++index) {
          Object selected = dv ? codec.getDocValuesFormatForField(fields[index])
              : codec.getPostingsFormatForField(fields[index]);
          int ordinal = 0;
          while (ordinal < candidates.length && candidates[ordinal] != selected) ++ordinal;
          if (ordinal == candidates.length) throw new AssertionError("unknown selected identity");
          digest.mix(index);
          digest.mix(ordinal);
          digest.mix(tokens[ordinal]);
        }
      }
      return digest.value;
    }
    int count(boolean dv) throws Exception {
      return ((Map<?, ?>) member(codec, dv ? "previousDVMappings" : "previousMappings")).size();
    }
    void checkBorrowedKeys(String[] fields, boolean dv) throws Exception {
      IdentityHashMap<String, Boolean> external = new IdentityHashMap<>();
      for (String field : fields) external.put(field, Boolean.TRUE);
      Map<?, ?> cache = (Map<?, ?>) member(codec, dv ? "previousDVMappings" : "previousMappings");
      if (cache.size() != KEYS) throw new AssertionError("cache key count");
      for (Object key : cache.keySet()) {
        if (!external.containsKey(key)) throw new AssertionError("cache copied an input String");
      }
    }
  }
  static String[] fields() {
    String[] fields = new String[KEYS];
    for (int index = 0; index < fields.length; ++index) {
      String prefix = switch (index % 8) {
        case 0 -> "Aa";
        case 1 -> "BB";
        case 2 -> "\ud83d\ude00";
        case 3 -> "\ud800";
        case 4 -> "\udc00";
        case 5 -> "\u4e2d\u00e9";
        case 6 -> "";
        default -> "\u0000";
      };
      fields[index] = prefix + ":" + (index / 8);
      fields[index].hashCode(); // Java String hash caching is warmed outside timing.
    }
    return fields;
  }
  static long[] runCached(Prepared prepared, String[] fields, boolean dv, long millis,
      long expected) throws Exception {
    long elapsed = 0, calls = 0;
    do {
      long start = System.nanoTime();
      long actual = prepared.batch(fields, dv, true);
      long nanos = System.nanoTime() - start;
      if (actual != expected) throw new AssertionError("cached digest changed");
      sink = actual;
      if (prepared.count(dv) != KEYS) throw new AssertionError("cached size changed");
      elapsed += nanos;
      calls += KEYS * 4;
    } while (elapsed < millis * 1_000_000L);
    return new long[] {elapsed, calls};
  }
  static long[] runFirst(String[] fields, boolean dv, long millis, long expected) throws Exception {
    long elapsed = 0, calls = 0;
    do {
      Prepared prepared = new Prepared();
      if (prepared.count(dv) != 0) throw new AssertionError("prepopulated cache");
      long start = System.nanoTime();
      long actual = prepared.batch(fields, dv, false);
      long nanos = System.nanoTime() - start;
      if (actual != expected) throw new AssertionError("first-lookup digest changed");
      sink = actual;
      if (prepared.count(dv) != KEYS) throw new AssertionError("missing cache entries");
      elapsed += nanos;
      calls += KEYS;
      prepared = null; // Release constructor graph/cache references outside timing.
    } while (elapsed < millis * 1_000_000L);
    return new long[] {elapsed, calls};
  }
  public static void main(String[] args) throws Exception {
    if (!RandomCodecMicro.class.desiredAssertionStatus()) throw new AssertionError("requires -ea");
    if (LuceneTestCase.TEST_NIGHTLY || LuceneTestCase.RANDOM_MULTIPLIER != 1)
      throw new AssertionError("requires nightly=false,multiplier=1");
    long warmup = args.length > 0 ? Long.parseLong(args[0]) : 500;
    long measure = args.length > 1 ? Long.parseLong(args[1]) : 750;
    String[] fields = fields();
    Prepared setup = new Prepared();
    long inputDigest = setup.setupDigest(fields);
    System.out.printf("#setup\trandom_codec_inputs\t%d\t%d%n", inputDigest, KEYS);
    long units = ArraysLength.units(fields);
    System.out.printf("#memory\trandom_codec_external_key_count\t%d%n", KEYS);
    System.out.printf("#memory\trandom_codec_external_logical_utf16_units\t%d%n", units);
    for (String kind : new String[] {"postings", "doc_values"}) {
      // The map stores the exact precomputed String objects: no copied key text.
      System.out.printf("#memory\trandom_codec_%s_cache_owned_utf16_payload_bytes\t0%n", kind);
      System.out.printf("#memory\trandom_codec_%s_cache_inline_key_box_bytes\t0%n", kind);
    }
    setup = null;
    for (boolean dv : new boolean[] {false, true}) {
      String kind = dv ? "doc_values" : "postings";
      for (boolean cached : new boolean[] {true, false}) {
        String mode = cached ? "cached" : "first_lookup";
        String name = "random_codec_" + kind + "_" + mode;
        Prepared prepared = new Prepared();
        long firstExpected = prepared.batch(fields, dv, false);
        prepared.checkBorrowedKeys(fields, dv);
        long expected = cached ? prepared.batch(fields, dv, true) : firstExpected;
        int lookups = cached ? KEYS * 4 : KEYS;
        System.out.printf("#check\t%s\t%d\t%d\t%d%n", name, expected, lookups, inputDigest);
        long[] result;
        if (cached) {
          runCached(prepared, fields, dv, warmup, expected);
          result = runCached(prepared, fields, dv, measure, expected);
        } else {
          prepared = null;
          runFirst(fields, dv, warmup, expected);
          result = runFirst(fields, dv, measure, expected);
        }
        System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result[0] / result[1], result[1]);
      }
    }
  }
  static final class ArraysLength {
    static long units(String[] fields) {
      long count = 0;
      for (String field : fields) count += field.length();
      return count;
    }
  }
}
