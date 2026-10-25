import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.List;
import org.apache.lucene.codecs.Codec;
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
import org.apache.lucene.index.StoredFieldDataInput;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingStoredFieldsFormat;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;

/** Real IndexWriter flush/import/delete/merge through only the asserting stored-fields format. */
public class GenStoredFieldsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<String> EVENTS = new ArrayList<>();
  private static int writerId;
  private static final int[] RANKS = {40, 10, 30, 20, 60, 50};
  private static final int[] VARIANTS = {0, -1, 1, 2, 3, -1};
  private static final int[][] ORDERS = {{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {2, 0, 5, 1, 4, 3}};

  private static String hex(byte[] bytes) { return HexFormat.of().formatHex(bytes); }
  private static String hex(String text) { return hex(text.getBytes(StandardCharsets.UTF_8)); }
  private static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i != 0) OUT.append('\t');
      OUT.append(fields[i]);
    }
    OUT.append('\n');
  }

  /** Public no-arg SPI provider: Lucene104Codec's storedFieldsFormat is final. */
  public static final class RoutingCodec extends FilterCodec {
    private final StoredFieldsFormat stored;
    public RoutingCodec() {
      super("StoredFieldsRoutingOracle", new Lucene104Codec());
      var asserting = new AssertingStoredFieldsFormat();
      try {
        var delegate = AssertingStoredFieldsFormat.class.getDeclaredField("in");
        delegate.setAccessible(true);
        delegate.set(asserting, new TraceFormat(this.delegate.storedFieldsFormat()));
      } catch (ReflectiveOperationException error) {
        throw new IllegalStateException(error);
      }
      stored = asserting;
    }
    @Override public StoredFieldsFormat storedFieldsFormat() { return stored; }
  }

  private static final class TraceFormat extends StoredFieldsFormat {
    private final StoredFieldsFormat delegate;
    TraceFormat(StoredFieldsFormat delegate) { this.delegate = delegate; }
    @Override public StoredFieldsReader fieldsReader(Directory dir, SegmentInfo si,
        FieldInfos fields, IOContext context) throws IOException {
      return delegate.fieldsReader(dir, si, fields, context);
    }
    @Override public StoredFieldsWriter fieldsWriter(Directory dir, SegmentInfo si,
        IOContext context) throws IOException {
      int id = writerId++;
      EVENTS.add(id + ":open");
      return new TraceWriter(id, delegate.fieldsWriter(dir, si, context));
    }
  }

  private static final class TraceWriter extends StoredFieldsWriter {
    private final int id;
    private final StoredFieldsWriter delegate;
    TraceWriter(int id, StoredFieldsWriter delegate) { this.id = id; this.delegate = delegate; }
    private void event(String event) { EVENTS.add(id + ":" + event); }
    private void field(FieldInfo field, String type, String value) {
      event("field:" + field.name + ":" + field.number + ":" + type + ":" + value);
    }
    @Override public void startDocument() throws IOException { event("start"); delegate.startDocument(); }
    @Override public void finishDocument() throws IOException { event("end"); delegate.finishDocument(); }
    @Override public void finish(int docs) throws IOException { event("finish:" + docs); delegate.finish(docs); }
    @Override public void close() throws IOException { event("close"); delegate.close(); }
    @Override public void writeField(FieldInfo f, int v) throws IOException { field(f, "i", "" + v); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, long v) throws IOException { field(f, "l", "" + v); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, float v) throws IOException { field(f, "f", Integer.toUnsignedString(Float.floatToRawIntBits(v))); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, double v) throws IOException { field(f, "d", Long.toUnsignedString(Double.doubleToRawLongBits(v))); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, BytesRef v) throws IOException { field(f, "b", hex(java.util.Arrays.copyOfRange(v.bytes, v.offset, v.offset + v.length))); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, String v) throws IOException { field(f, "s", hex(v)); delegate.writeField(f, v); }
    @Override public void writeField(FieldInfo f, StoredFieldDataInput v) throws IOException { field(f, "input", "" + v.length()); delegate.writeField(f, v); }
    @Override public long ramBytesUsed() { return delegate.ramBytesUsed(); }
    @Override public Collection<Accountable> getChildResources() { return delegate.getChildResources(); }
  }

  private static Document document(int doc) {
    Document result = new Document();
    result.add(new StringField("id", "" + (char) ('a' + doc), Field.Store.NO));
    result.add(new NumericDocValuesField("rank", RANKS[doc]));
    int v = VARIANTS[doc];
    if (v < 0) return result;
    byte[][] binaries = {{0, (byte) 255, 10, (byte) 128}, {}, {1, 2, 0, (byte) 254}, {(byte) 255, 0, 127, (byte) 128, 9}};
    String[] strings = {"", "hello λ\t\n", "tail\u0000end", "🙂"};
    int[] ints = {Integer.MIN_VALUE, Integer.MAX_VALUE, -17, 0};
    long[] longs = {Long.MIN_VALUE, Long.MAX_VALUE, -9007199254740993L, 0};
    int[] floats = {0x80000000, 0, 0x7fc00042, 0xff800000};
    long[] doubles = {0x8000000000000000L, 0x7ff8000000000042L, 0x7ff0000000000000L, 1};
    StoredField[] fields = {
      new StoredField("binary", binaries[v]), new StoredField("text", strings[v]),
      new StoredField("int", ints[v]), new StoredField("long", longs[v]),
      new StoredField("float", Float.intBitsToFloat(floats[v])),
      new StoredField("double", Double.longBitsToDouble(doubles[v]))
    };
    for (int at : ORDERS[doc / 2]) result.add(fields[at]);
    return result;
  }

  private static String inputFields(Document document) {
    List<String> values = new ArrayList<>();
    for (var field : document.getFields()) {
      if (!field.fieldType().stored()) continue;
      String value;
      if (field.binaryValue() != null) {
        var bytes = field.binaryValue();
        value = "b:" + hex(java.util.Arrays.copyOfRange(bytes.bytes, bytes.offset, bytes.offset + bytes.length));
      } else if (field.numericValue() instanceof Integer number) {
        value = "i:" + number;
      } else if (field.numericValue() instanceof Long number) {
        value = "l:" + number;
      } else if (field.numericValue() instanceof Float number) {
        value = "f:" + Integer.toUnsignedString(Float.floatToRawIntBits(number));
      } else if (field.numericValue() instanceof Double number) {
        value = "d:" + Long.toUnsignedString(Double.doubleToRawLongBits(number));
      } else {
        value = "s:" + hex(field.stringValue());
      }
      values.add(field.name() + ":" + value);
    }
    return values.isEmpty() ? "-" : String.join(";", values);
  }

  private static IndexWriterConfig config(boolean sorted) {
    var config = new IndexWriterConfig();
    config.setCodec(new RoutingCodec());
    config.setUseCompoundFile(false);
    var merges = new LogDocMergePolicy();
    merges.setMergeFactor(100);
    config.setMergePolicy(merges);
    config.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) config.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return config;
  }

  private static void snapshot(String name, String phase, IndexWriter writer) throws IOException {
    try (var reader = DirectoryReader.open(writer)) {
      row("CASE", name, phase, reader.maxDoc(), reader.numDocs(), reader.leaves().size());
      int ordinal = 0;
      for (var leaf : reader.leaves()) {
        var live = leaf.reader().getLiveDocs();
        var fields = leaf.reader().storedFields();
        for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
          if (live != null && !live.get(doc)) continue;
          List<String> values = new ArrayList<>();
          fields.document(doc, new StoredFieldVisitor() {
            @Override public Status needsField(FieldInfo field) { return Status.YES; }
            @Override public void binaryField(FieldInfo f, byte[] v) { values.add(f.name + ":b:" + hex(v)); }
            @Override public void stringField(FieldInfo f, String v) { values.add(f.name + ":s:" + hex(v)); }
            @Override public void intField(FieldInfo f, int v) { values.add(f.name + ":i:" + v); }
            @Override public void longField(FieldInfo f, long v) { values.add(f.name + ":l:" + v); }
            @Override public void floatField(FieldInfo f, float v) { values.add(f.name + ":f:" + Integer.toUnsignedString(Float.floatToRawIntBits(v))); }
            @Override public void doubleField(FieldInfo f, double v) { values.add(f.name + ":d:" + Long.toUnsignedString(Double.doubleToRawLongBits(v))); }
          });
          row("DOC", name, phase, ordinal++, leaf.docBase + doc,
              values.isEmpty() ? "-" : String.join(";", values));
        }
        for (var f : leaf.reader().getFieldInfos()) row("FIELD", name, phase, leaf.ord, f.name, f.number);
      }
    }
  }

  private static void scenario(boolean imported, boolean sorted) throws Exception {
    String name = (imported ? "imported" : "direct") + (sorted ? "_sorted" : "_unsorted");
    EVENTS.clear(); writerId = 0;
    try (var destination = new ByteBuffersDirectory(); var writer = new IndexWriter(destination, config(sorted))) {
      if (imported) {
        // Each fresh source first encounters the six stored names in a different order.
        // addIndexes preserves those source numbers until a real remapping merge occurs.
        for (int source = 0; source < 3; source++) {
          try (var dir = new ByteBuffersDirectory()) {
            try (var sourceWriter = new IndexWriter(dir, config(sorted))) {
              sourceWriter.addDocument(document(source * 2));
              sourceWriter.addDocument(document(source * 2 + 1));
              sourceWriter.flush();
            }
            writer.addIndexes(dir);
          }
        }
      } else {
        for (int doc = 0; doc < 6; doc++) {
          writer.addDocument(document(doc));
          if ((doc & 1) == 1) writer.flush();
        }
      }
      snapshot(name, "initial", writer);
      writer.deleteDocuments(new Term("id", "c"));
      writer.flush();
      snapshot(name, "before", writer);
      writer.forceMerge(1);
      writer.commit();
      snapshot(name, "after", writer);
    }
    for (int i = 0; i < EVENTS.size(); i++) row("TRACE", name, i, EVENTS.get(i));
  }

  public static void main(String[] args) throws Exception {
    if (!GenStoredFieldsWriterRouting.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Path providers = Files.createTempDirectory("stored-fields-routing-provider");
    Path service = providers.resolve("META-INF/services/org.apache.lucene.codecs.Codec");
    Files.createDirectories(service.getParent());
    Files.writeString(service, RoutingCodec.class.getName() + "\n");
    try (var loader = new URLClassLoader(new java.net.URL[] {providers.toUri().toURL()}, GenStoredFieldsWriterRouting.class.getClassLoader())) {
      Codec.reloadCodecs(loader);
      if (!(Codec.forName("StoredFieldsRoutingOracle") instanceof RoutingCodec)) throw new AssertionError("SPI registration");
      row("# Scope: stored-fields writer routing only; all other codec components delegate to Lucene104Codec");
      row("# CASE case phase maxDoc numDocs leaves; DOC case phase liveOrdinal physicalDoc fields; FIELD case phase leaf name number");
      row("# field=name:type:value; b/s values hex bytes/UTF8; f/d unsigned raw bits; empty fields=-; TRACE real delegate events");
      row("# INPUT ordinal id rank fields: raw source values before codec NaN canonicalization");
      for (int doc = 0; doc < 6; doc++) {
        row("INPUT", doc, (char) ('a' + doc), RANKS[doc], inputFields(document(doc)));
      }
      for (boolean imported : new boolean[] {false, true}) {
        for (boolean sorted : new boolean[] {false, true}) scenario(imported, sorted);
      }
      Files.writeString(out.resolve("stored_fields_writer_routing.tsv"), OUT);
    } finally {
      Files.deleteIfExists(service);
      Files.deleteIfExists(service.getParent());
      Files.deleteIfExists(service.getParent().getParent());
      Files.deleteIfExists(providers);
    }
  }
}
