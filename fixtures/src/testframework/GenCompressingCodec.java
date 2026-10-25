import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.NormsFormat;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.SegmentInfoFormat;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.TermVectorsFormat;
import org.apache.lucene.codecs.compressing.CompressionMode;
import org.apache.lucene.codecs.lucene90.compressing.Lucene90CompressingStoredFieldsFormat;
import org.apache.lucene.codecs.lucene90.compressing.Lucene90CompressingTermVectorsFormat;
import org.apache.lucene.tests.codecs.compressing.CompressingCodec;
import org.apache.lucene.tests.codecs.compressing.DeflateWithPresetCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.FastCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.FastDecompressionCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.HighCompressionCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.LZ4WithPresetCompressingCodec;
import org.apache.lucene.tests.codecs.compressing.dummy.DummyCompressingCodec;

/**
 * Actual pinned Java CompressingCodec constructors, selected format objects,
 * inherited factories and random overloads. These are configuration/callback
 * fixtures only: no native stored-field/term-vector I/O is substituted here.
 */
public class GenCompressingCodec {
  static final String[] KINDS = {
    "fast", "fast_decompression", "high_compression", "dummy", "deflate_preset", "lz4_preset"
  };
  static final String[] ACCESSORS = {
    "docValuesFormat", "fieldInfosFormat", "liveDocsFormat", "normsFormat",
    "postingsFormat", "segmentInfoFormat", "compoundFormat", "pointsFormat", "knnVectorsFormat"
  };
  static final long[] SEEDS = {0, 1, 18, 42, -1, Long.MIN_VALUE, Long.MAX_VALUE};

  static void row(StringBuilder out, Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i != 0) out.append('\t');
      String value = String.valueOf(fields[i]);
      if (value.indexOf('\t') >= 0 || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0)
        throw new AssertionError("fixture field contains a separator");
      out.append(value);
    }
    out.append('\n');
  }
  static Object member(Object value, Class<?> owner, String name) throws Exception {
    var field = owner.getDeclaredField(name);
    field.setAccessible(true);
    return field.get(value);
  }
  static Codec delegate(CompressingCodec codec) throws Exception {
    return (Codec) member(codec, FilterCodec.class, "delegate");
  }
  static Object sf(CompressingCodec codec, String property) throws Exception {
    return member(codec.storedFieldsFormat(), Lucene90CompressingStoredFieldsFormat.class, property);
  }
  static Object tv(CompressingCodec codec, String property) throws Exception {
    return member(codec.termVectorsFormat(), Lucene90CompressingTermVectorsFormat.class, property);
  }
  static CompressingCodec defaults(int kind) {
    return switch (kind) {
      case 0 -> new FastCompressingCodec();
      case 1 -> new FastDecompressionCompressingCodec();
      case 2 -> new HighCompressionCompressingCodec();
      case 3 -> new DummyCompressingCodec();
      case 4 -> new DeflateWithPresetCompressingCodec();
      case 5 -> new LZ4WithPresetCompressingCodec();
      default -> throw new AssertionError("unknown kind");
    };
  }
  static CompressingCodec custom(int kind, int chunk, int docs, boolean suffix, int shift) {
    return switch (kind) {
      case 0 -> new FastCompressingCodec(chunk, docs, suffix, shift);
      case 1 -> new FastDecompressionCompressingCodec(chunk, docs, suffix, shift);
      case 2 -> new HighCompressionCompressingCodec(chunk, docs, suffix, shift);
      case 3 -> new DummyCompressingCodec(chunk, docs, suffix, shift);
      case 4 -> new DeflateWithPresetCompressingCodec(chunk, docs, suffix, shift);
      case 5 -> new LZ4WithPresetCompressingCodec(chunk, docs, suffix, shift);
      default -> throw new AssertionError("unknown kind");
    };
  }
  // Exercise both public constructors of the abstract base itself.
  static final class BaseCodec extends CompressingCodec {
    BaseCodec(String name, CompressionMode mode, int chunk, int docs, int shift) {
      super(name, mode, chunk, docs, shift);
    }
    BaseCodec(String name, String suffix, CompressionMode mode, int chunk, int docs, int shift) {
      super(name, suffix, mode, chunk, docs, shift);
    }
  }
  static void snapshot(StringBuilder out, String label, CompressingCodec codec) throws Exception {
    var stored = codec.storedFieldsFormat();
    var vectors = codec.termVectorsFormat();
    boolean storedStable = stored == codec.storedFieldsFormat();
    boolean vectorsStable = vectors == codec.termVectorsFormat();
    boolean modeShared = sf(codec, "compressionMode") == tv(codec, "compressionMode");
    if (!storedStable || !vectorsStable)
      throw new AssertionError("unstable selected format: " + label);
    if (!modeShared)
      throw new AssertionError("formats lost shared mode identity: " + label);
    row(out, "codec", label, codec.getClass().getSimpleName(), codec.getName(), delegate(codec).getName(), codec);
    row(out, "identity", label, storedStable, vectorsStable, modeShared);
    row(out, "stored", label, stored.getClass().getSimpleName(), sf(codec, "formatName"),
        sf(codec, "segmentSuffix"), sf(codec, "compressionMode"), sf(codec, "chunkSize"),
        sf(codec, "maxDocsPerChunk"), sf(codec, "blockShift"), stored);
    row(out, "vectors", label, vectors.getClass().getSimpleName(), tv(codec, "formatName"),
        tv(codec, "segmentSuffix"), tv(codec, "compressionMode"), tv(codec, "chunkSize"),
        tv(codec, "maxDocsPerChunk"), tv(codec, "blockSize"), vectors);
  }
  static String configuration() throws Exception {
    var out = new StringBuilder();
    for (int kind = 0; kind < KINDS.length; kind++) {
      var first = defaults(kind);
      var second = defaults(kind);
      snapshot(out, KINDS[kind] + "_default", first);
      snapshot(out, KINDS[kind] + "_custom", custom(kind, 37, 3, false, 2));
      snapshot(out, KINDS[kind] + "_suffix", custom(kind, 37, 3, true, 22));
      boolean modeShared = sf(first, "compressionMode") == sf(second, "compressionMode");
      if (modeShared != (kind < 4)) throw new AssertionError("unexpected mode allocation: " + kind);
      row(out, "between-codecs", KINDS[kind],
          first.storedFieldsFormat() == second.storedFieldsFormat(),
          first.termVectorsFormat() == second.termVectorsFormat(), modeShared);
    }
    snapshot(out, "base_empty_suffix", new BaseCodec("CustomCompressingData", CompressionMode.FAST, 11, 7, 2));
    snapshot(out, "base_suffix", new BaseCodec("CustomCompressingData", "privateS", CompressionMode.FAST, 11, 7, 22));
    snapshot(out, "base_maximum", new BaseCodec("CustomCompressingData", "privateS", CompressionMode.FAST,
        Integer.MAX_VALUE, Integer.MAX_VALUE, 22));
    // Codec's service-name check accepts the empty string; neither format rejects
    // a null compression mode during construction. Record, do not sanitize.
    snapshot(out, "base_empty_name", new BaseCodec("", CompressionMode.FAST, 1, 1, 2));
    snapshot(out, "base_null_mode", new BaseCodec("CustomCompressingData", null, 1, 1, 2));
    return out.toString();
  }

  static final class ProbeCodec extends FilterCodec {
    final List<String> events = new ArrayList<>();
    Object last;
    String fail = "";
    IllegalStateException failure;
    ProbeCodec(Codec delegate) { super("CompressingProbe", delegate); }
    <T> T call(String name, Supplier<T> action) {
      events.add(name);
      if (name.equals(fail)) throw failure;
      T result = action.get(); last = result; return result;
    }
    public DocValuesFormat docValuesFormat() { return call("docValuesFormat", super::docValuesFormat); }
    public FieldInfosFormat fieldInfosFormat() { return call("fieldInfosFormat", super::fieldInfosFormat); }
    public LiveDocsFormat liveDocsFormat() { return call("liveDocsFormat", super::liveDocsFormat); }
    public NormsFormat normsFormat() { return call("normsFormat", super::normsFormat); }
    public PostingsFormat postingsFormat() { return call("postingsFormat", super::postingsFormat); }
    public SegmentInfoFormat segmentInfoFormat() { return call("segmentInfoFormat", super::segmentInfoFormat); }
    public CompoundFormat compoundFormat() { return call("compoundFormat", super::compoundFormat); }
    public PointsFormat pointsFormat() { return call("pointsFormat", super::pointsFormat); }
    public KnnVectorsFormat knnVectorsFormat() { return call("knnVectorsFormat", super::knnVectorsFormat); }
    public StoredFieldsFormat storedFieldsFormat() { return call("storedFieldsFormat", super::storedFieldsFormat); }
    public TermVectorsFormat termVectorsFormat() { return call("termVectorsFormat", super::termVectorsFormat); }
    @Override public String toString() { events.add("delegate-display"); return "ProbeDisplay"; }
  }
  static String delegation() throws Exception {
    var out = new StringBuilder();
    for (int kind = 0; kind < KINDS.length; kind++) {
      var codec = defaults(kind);
      var original = delegate(codec);
      for (String name : ACCESSORS) {
        var accessor = Codec.class.getMethod(name);
        row(out, "delegate-identity", KINDS[kind], name,
            accessor.invoke(codec) == accessor.invoke(original));
      }
      var stored = codec.storedFieldsFormat();
      var vectors = codec.termVectorsFormat();
      var probe = new ProbeCodec(original);
      var delegateField = FilterCodec.class.getDeclaredField("delegate");
      delegateField.setAccessible(true); delegateField.set(codec, probe);
      for (String name : ACCESSORS) {
        var accessor = Codec.class.getMethod(name);
        probe.events.clear();
        Object first = accessor.invoke(codec), firstDelegate = probe.last;
        Object second = accessor.invoke(codec), secondDelegate = probe.last;
        boolean exact = first == firstDelegate && second == secondDelegate;
        if (!exact || !probe.events.equals(List.of(name, name)))
          throw new AssertionError("delegated identity/call count: " + name);
        row(out, "forward", KINDS[kind], name, exact, first == second, String.join(",", probe.events));
        probe.events.clear(); probe.fail = name;
        probe.failure = new IllegalStateException("delegate failure:" + name);
        try {
          accessor.invoke(codec);
          throw new AssertionError("missing delegate failure: " + name);
        } catch (InvocationTargetException error) {
          if (error.getCause() != probe.failure) throw error;
          if (!probe.events.equals(List.of(name))) throw new AssertionError("failure call count: " + name);
          row(out, "forward-error", KINDS[kind], name, error.getCause().getClass().getSimpleName(),
              error.getCause().getMessage(), true, String.join(",", probe.events));
        }
        probe.fail = "";
      }
      probe.events.clear();
      if (stored != codec.storedFieldsFormat() || vectors != codec.termVectorsFormat())
        throw new AssertionError("overrides changed after delegate injection");
      // toString must use the two selected formats, never delegate.toString.
      String display = codec.toString();
      if (!probe.events.isEmpty() || !display.startsWith(codec.getName() + "(storedFieldsFormat="))
        throw new AssertionError("selected accessor leaked to delegate");
      row(out, "override-events", KINDS[kind], String.join(",", probe.events));
    }
    return out.toString();
  }

  static final class RecordingRandom extends Random {
    final List<String> events = new ArrayList<>();
    RecordingRandom(long seed) { super(seed); }
    @Override public int nextInt(int bound) {
      int value = super.nextInt(bound); events.add("int(" + bound + ")=" + value); return value;
    }
    @Override public boolean nextBoolean() {
      boolean value = super.nextBoolean(); events.add("boolean=" + value); return value;
    }
    int tail() { return super.nextInt(); }
  }
  static String result(CompressingCodec codec) throws Exception {
    return codec.getClass().getSimpleName() + ";" + codec.getName() + ";"
        + sf(codec, "segmentSuffix") + ";" + sf(codec, "compressionMode") + ";"
        + sf(codec, "chunkSize") + ";" + sf(codec, "maxDocsPerChunk") + ";" + sf(codec, "blockShift");
  }
  static String randomness() throws Exception {
    var out = new StringBuilder();
    Set<String> selections = new HashSet<>();
    boolean suffixSuccess = false, suffixFailure = false;
    // Mixed overloads share the evolving stream; each row's trailing unbounded
    // nextInt also detects rejection-sampling differences in bounded draws.
    String[] overloads = {"explicit", "random", "reasonable", "suffix_false", "suffix_true"};
    for (long seed : SEEDS) {
      var random = new RecordingRandom(seed);
      for (int step = 0; step < 384; step++) {
        random.events.clear(); String overload = overloads[step % overloads.length];
        String status;
        try {
          CompressingCodec codec = switch (overload) {
            case "explicit" -> CompressingCodec.randomInstance(random, 37, 3, (step & 1) != 0, 10);
            case "random" -> CompressingCodec.randomInstance(random);
            case "reasonable" -> CompressingCodec.reasonableInstance(random);
            case "suffix_false" -> CompressingCodec.randomInstance(random, false);
            case "suffix_true" -> CompressingCodec.randomInstance(random, true);
            default -> throw new AssertionError(overload);
          };
          selections.add(codec.getClass().getSimpleName());
          if (overload.startsWith("suffix_")) suffixSuccess = true;
          status = "ok:" + result(codec);
        } catch (IllegalArgumentException error) {
          if (!overload.startsWith("suffix_")) throw error;
          suffixFailure = true;
          status = error.getClass().getSimpleName() + ":" + error.getMessage();
        }
        row(out, "random", seed, step, overload, status, String.join(",", random.events), random.tail());
      }
    }
    if (selections.size() != 6 || !suffixSuccess || !suffixFailure)
      throw new AssertionError("random fixture missed a selection or suffix success/error branch");
    return out.toString();
  }

  interface Construction { CompressingCodec create(); }
  static void invalid(StringBuilder out, String label, Construction action) {
    try { action.create(); throw new AssertionError("accepted invalid configuration: " + label); }
    catch (IllegalArgumentException error) {
      row(out, "invalid", label, error.getClass().getSimpleName(), error.getMessage());
    }
  }
  static String errors() {
    var out = new StringBuilder();
    int[][] invalid = {
      {0, 0, 0}, {-1, 1, 2}, {1, 0, 0}, {1, -1, 2},
      {1, 1, 0}, {1, 1, 1}, {1, 1, 23}, {1, 1, Integer.MAX_VALUE}
    };
    for (int kind = 0; kind < KINDS.length; kind++) for (int[] values : invalid) {
      final int selected = kind;
      invalid(out, KINDS[kind] + ":" + values[0] + ":" + values[1] + ":" + values[2],
          () -> custom(selected, values[0], values[1], true, values[2]));
    }
    for (String name : new String[] {"bad-name", "é", "a".repeat(128)}) {
      invalid(out, "base-name:" + name, () -> new BaseCodec(name, CompressionMode.FAST, 0, 0, 0));
    }
    // Selection happens before the chosen constructor rejects these parameters.
    for (long seed : SEEDS) for (int[] values : invalid) {
      var random = new RecordingRandom(seed);
      try {
        CompressingCodec.randomInstance(random, values[0], values[1], true, values[2]);
        throw new AssertionError("accepted invalid random explicit configuration");
      } catch (IllegalArgumentException error) {
        row(out, "random-invalid", seed, values[0], values[1], values[2],
            error.getClass().getSimpleName(), error.getMessage(), String.join(",", random.events), random.tail());
      }
    }
    return out.toString();
  }
  public static void main(String[] args) throws Exception {
    if (!GenCompressingCodec.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path output = Path.of(args[0]); Files.createDirectories(output);
    Files.writeString(output.resolve("compressing_codec_config.tsv"), configuration());
    Files.writeString(output.resolve("compressing_codec_delegation.tsv"), delegation());
    Files.writeString(output.resolve("compressing_codec_random.tsv"), randomness());
    Files.writeString(output.resolve("compressing_codec_errors.tsv"), errors());
  }
}
