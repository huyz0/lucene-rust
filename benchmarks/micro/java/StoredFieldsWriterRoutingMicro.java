import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Operational stored-fields writer routing; setup/readback are outside accumulated timing. */
public final class StoredFieldsWriterRoutingMicro {
  private static final int DOCS = 1024;
  private static final int SOURCES = 8;
  private static final long OFFSET = 0xcbf29ce484222325L;
  private static final long PRIME = 0x100000001b3L;
  private static volatile long sink;

  /** Intentional Lucene104 name alias: same wire format, only writer validation differs.
   * Reads resolve the native codec and use native readers, matching the Rust factory seam.
   * No SPI registration or fixture tracing is present in this benchmark.
   */
  private static final class WriterOnlyAssertingCodec extends FilterCodec {
    private final StoredFieldsFormat fields;
    private final AssertingStoredFieldsFormat asserting;
    WriterOnlyAssertingCodec() {
      super("Lucene104", new Lucene104Codec());
      var nativeFields = delegate.storedFieldsFormat();
      asserting = new AssertingStoredFieldsFormat();
      fields = new StoredFieldsFormat() {
        @Override public StoredFieldsWriter fieldsWriter(Directory dir, SegmentInfo info,
            IOContext context) throws IOException {
          return asserting.fieldsWriter(dir, info, context);
        }
        @Override public StoredFieldsReader fieldsReader(Directory dir, SegmentInfo info,
            FieldInfos infos, IOContext context) throws IOException {
          return nativeFields.fieldsReader(dir, info, infos, context);
        }
      };
    }
    @Override public StoredFieldsFormat storedFieldsFormat() { return fields; }
  }

  private static long mix(long digest, long value) { return (digest ^ value) * PRIME; }
  private static long bytes(long digest, byte[] values) {
    digest = mix(digest, values.length);
    for (byte value : values) digest = mix(digest, value & 255);
    return digest;
  }
  private static byte[] utf8(String value) { return value.getBytes(StandardCharsets.UTF_8); }

  // Shared with the Rust twin. Expected results come from these source values,
  // independently of IndexWriter and StoredFieldVisitor.
  private static byte[] binary(int doc) {
    byte[] result = new byte[128];
    for (int at = 0; at < result.length; at++) result[at] = (byte) (doc * 31 + at * 17);
    return result;
  }
  private static String text(int doc) { return "doc-" + doc; }
  private static int integer(int doc) { return doc * 31 - 16000; }
  private static long longer(int doc) { return -9007199254740993L + doc * 104729L; }
  private static float floating(int doc) { return doc % 16 == 1 ? -0.0f : doc * 0.25f; }
  private static double doubling(int doc) { return doc % 16 == 2 ? Double.NaN : -doc * 0.125; }

  private static Document[] corpus() {
    Document[] result = new Document[DOCS];
    for (int doc = 0; doc < DOCS; doc++) {
      var value = new Document();
      value.add(new StringField("id", Integer.toString(doc), Field.Store.NO));
      value.add(new NumericDocValuesField("rank", DOCS - 1 - doc));
      if (doc % 8 != 0) {
        value.add(new StoredField("binary", binary(doc)));
        value.add(new StoredField("text", text(doc)));
        value.add(new StoredField("int", integer(doc)));
        value.add(new StoredField("long", longer(doc)));
        value.add(new StoredField("float", floating(doc)));
        value.add(new StoredField("double", doubling(doc)));
      }
      result[doc] = value;
    }
    return result;
  }

  private static long header(long digest, int kind, String name) {
    return bytes(mix(digest, kind), utf8(name));
  }
  private static long expected(boolean merge, boolean sorted) {
    long digest = OFFSET;
    int count = 0;
    for (int at = 0; at < DOCS; at++) {
      int doc = sorted ? DOCS - 1 - at : at;
      if (merge && doc % 7 == 0) continue;
      digest = mix(mix(digest, 10), doc % 8 == 0 ? 0 : 6);
      if (doc % 8 != 0) {
        digest = bytes(header(digest, 1, "binary"), binary(doc));
        digest = bytes(header(digest, 2, "text"), utf8(text(doc)));
        digest = mix(header(digest, 3, "int"), integer(doc));
        digest = mix(header(digest, 4, "long"), longer(doc));
        digest = mix(header(digest, 5, "float"), Integer.toUnsignedLong(Float.floatToIntBits(floating(doc))));
        digest = mix(header(digest, 6, "double"), Double.doubleToLongBits(doubling(doc)));
      }
      digest = mix(digest, 11);
      count++;
    }
    return mix(digest, count);
  }

  private record Value(String name, int type, Object value) {}
  private static final class Visitor extends StoredFieldVisitor {
    final List<Value> values = new ArrayList<>();
    @Override public Status needsField(FieldInfo f) { return Status.YES; }
    @Override public void binaryField(FieldInfo f, byte[] v) { values.add(new Value(f.name, 1, v)); }
    @Override public void stringField(FieldInfo f, String v) { values.add(new Value(f.name, 2, v)); }
    @Override public void intField(FieldInfo f, int v) { values.add(new Value(f.name, 3, v)); }
    @Override public void longField(FieldInfo f, long v) { values.add(new Value(f.name, 4, v)); }
    @Override public void floatField(FieldInfo f, float v) { values.add(new Value(f.name, 5, v)); }
    @Override public void doubleField(FieldInfo f, double v) { values.add(new Value(f.name, 6, v)); }
    long consume(long digest) {
      digest = mix(mix(digest, 10), values.size());
      for (Value value : values) {
        digest = header(digest, value.type, value.name);
        digest = switch (value.type) {
          case 1 -> bytes(digest, (byte[]) value.value);
          case 2 -> bytes(digest, utf8((String) value.value));
          case 3 -> mix(digest, (Integer) value.value);
          case 4 -> mix(digest, (Long) value.value);
          // Check raw output bits: the independent expectation canonicalizes source NaNs.
          case 5 -> mix(digest, Integer.toUnsignedLong(Float.floatToRawIntBits((Float) value.value)));
          case 6 -> mix(digest, Double.doubleToRawLongBits((Double) value.value));
          default -> throw new AssertionError("stored type");
        };
      }
      return mix(digest, 11);
    }
  }

  private static IndexWriterConfig config(boolean asserting, boolean sorted) {
    var result = new IndexWriterConfig();
    result.setCodec(asserting ? new WriterOnlyAssertingCodec() : new Lucene104Codec());
    result.setUseCompoundFile(false);
    var policy = new LogDocMergePolicy();
    policy.setMergeFactor(1000);
    result.setMergePolicy(policy);
    result.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) result.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return result;
  }

  private record Batch(long nanos, long digest, int docs, long encodedBytes) {}
  private static long readback(Directory dir, boolean merge, long expected) throws IOException {
    long digest = OFFSET;
    int docs = 0;
    try (var reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1 || reader.maxDoc() != (merge ? 877 : DOCS)
          || reader.numDocs() != (merge ? 877 : DOCS)) throw new AssertionError("output shape differs");
      var stored = reader.storedFields();
      for (int doc = 0; doc < reader.maxDoc(); doc++) {
        var visitor = new Visitor();
        stored.document(doc, visitor);
        digest = visitor.consume(digest);
        docs++;
      }
    }
    digest = mix(digest, docs);
    if (digest != expected) throw new AssertionError("semantic digest differs: " + Long.toUnsignedString(digest));
    return digest;
  }

  private static Batch batch(Document[] corpus, boolean asserting, boolean merge, boolean sorted,
      long expected) throws Exception {
    try (var dir = new ByteBuffersDirectory(); var writer = new IndexWriter(dir, config(asserting, sorted))) {
      if (merge) {
        for (int doc = 0; doc < DOCS; doc++) {
          writer.addDocument(corpus[doc]);
          if ((doc + 1) % (DOCS / SOURCES) == 0) writer.flush();
        }
        for (int doc = 0; doc < DOCS; doc += 7) writer.deleteDocuments(new Term("id", Integer.toString(doc)));
        writer.commit();
        try (var before = DirectoryReader.open(dir)) {
          if (before.leaves().size() != SOURCES || before.maxDoc() != DOCS || before.numDocs() != 877) {
            throw new AssertionError("merge source preparation differs");
          }
        }
      }
      long started = System.nanoTime();
      if (merge) {
        writer.forceMerge(1);
      } else {
        for (Document doc : corpus) writer.addDocument(doc);
        writer.flush();
      }
      writer.commit();
      long elapsed = System.nanoTime() - started;
      long digest = readback(dir, merge, expected);
      long encoded = 0;
      for (String file : dir.listAll()) if (!file.equals("write.lock")) encoded += dir.fileLength(file);
      sink = digest;
      return new Batch(elapsed, digest, merge ? 877 : DOCS, encoded);
    }
  }

  private static void wrapperMemory() throws IOException {
    var codec = new WriterOnlyAssertingCodec();
    try (var dir = new ByteBuffersDirectory()) {
      var info = new SegmentInfo(dir, Version.LATEST, null, "_routing_memory", 0,
          false, false, null, Map.of(), new byte[16], Map.of(), null);
      var writer = codec.storedFieldsFormat().fieldsWriter(dir, info, IOContext.DEFAULT);
      long writerState = RamUsageEstimator.shallowSizeOf(writer);
      writer.finish(0);
      writer.close();
      System.out.println("#memory_scope\tincremental writer state is actual asserting wrapper shallow size; native delegate/buffers, allocator and directory excluded");
      System.out.println("sf_routing_asserting_writer_incremental_state_bytes\t" + writerState);
      System.out.println("#memory_scope\tconfig shallow sum covers writer-only FilterCodec, adapting format and AssertingStoredFieldsFormat; native codecs/formats excluded");
      System.out.println("sf_routing_asserting_codec_format_config_shallow_bytes\t"
          + (RamUsageEstimator.shallowSizeOf(codec) + RamUsageEstimator.shallowSizeOf(codec.fields)
              + RamUsageEstimator.shallowSizeOf(codec.asserting)));
    }
  }

  private record Measurement(long nanos, long iterations, Batch last) {}
  private static Measurement measure(Document[] corpus, boolean asserting, boolean merge,
      boolean sorted, long expected, long millis) throws Exception {
    long elapsed = 0, iterations = 0;
    Batch last;
    do {
      last = batch(corpus, asserting, merge, sorted, expected);
      elapsed += last.nanos;
      iterations++;
    } while (elapsed < millis * 1_000_000L);
    return new Measurement(elapsed, iterations, last);
  }

  public static void main(String[] args) throws Exception {
    if (!StoredFieldsWriterRoutingMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    if (args.length == 2 && args[0].equals("--verify-dir")) {
      try (var dir = FSDirectory.open(Path.of(args[1]))) {
        long digest = readback(dir, true, expected(true, true));
        System.out.println("stored-fields routing Rust output read by default Lucene104: " + Long.toUnsignedString(digest) + " docs=877 leaves=1");
      }
      return;
    }
    long warmup = Long.parseLong(args[0]), duration = Long.parseLong(args[1]);
    if (warmup < 0 || duration <= 0) throw new IllegalArgumentException("invalid durations");
    Document[] corpus = corpus();
    long sourcePayload = 0;
    for (int doc = 0; doc < DOCS; doc++) if (doc % 8 != 0) sourcePayload += 128 + utf8(text(doc)).length + 4 + 8 + 4 + 8;
    System.out.println("#memory_scope\tsource stored value payload excludes id/rank/names/objects; encoded index bytes exclude write.lock; no RSS/peak/whole-writer claim");
    System.out.println("sf_routing_source_stored_value_payload_bytes\t" + sourcePayload);
    wrapperMemory();
    String[] names = {"flush_unsorted", "merge_deleted_unsorted", "merge_deleted_sorted"};
    for (int kind = 0; kind < names.length; kind++) {
      boolean merge = kind != 0, sorted = kind == 2;
      long expected = expected(merge, sorted);
      for (boolean asserting : new boolean[] {false, true}) {
        String name = "sf_routing_" + (asserting ? "asserting_" : "default_") + names[kind];
        measure(corpus, asserting, merge, sorted, expected, warmup);
        Measurement result = measure(corpus, asserting, merge, sorted, expected, duration);
        System.out.printf("#check\t%s\t%s\t%d\t%d%n", name,
            Long.toUnsignedString(result.last.digest), result.last.docs, merge ? SOURCES : 1);
        System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name,
            (double) result.nanos / result.iterations, result.iterations);
        System.out.printf("%s_index_encoded_bytes\t%d%n", name, result.last.encodedBytes);
      }
    }
  }
}
