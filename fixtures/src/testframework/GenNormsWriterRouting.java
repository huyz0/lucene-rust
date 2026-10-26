import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.NormsConsumer;
import org.apache.lucene.codecs.NormsFormat;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat;
import org.apache.lucene.util.SmallFloat;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.search.DocIdSetIterator;

/** Actual installed asserting norms writer, with native readers and independent readback. */
public class GenNormsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<String> EVENTS = new ArrayList<>(), READS = new ArrayList<>();
  private static final ThreadLocal<Integer> OPENING = new ThreadLocal<>();
  private static final int[] RANKS = {50, 10, 40, 20, 80, 60, 0, 30};
  private static final String[] VARYING = {"", "x", "x x", "x x x", "x x x x", "x x x x x", "", "x x"};
  private static final String[] SPARSE = {null, "", "s s", "s", null, "", null, "s s s"};
  private static int nextConsumer;
  private static String phase, readerPurpose = "writer";
  private static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  private static void event(String value) { EVENTS.add(phase + ":" + value); }
  private static String hex(String text) { return HexFormat.of().formatHex(text.getBytes(StandardCharsets.UTF_8)); }

  /** Registered custom identity; only norms consumer enters assertions. */
  public static final class RoutingCodec extends FilterCodec {
    private final NormsFormat norms;
    public RoutingCodec() {
      super("NormsRoutingOracle", new Lucene104Codec());
      var nativeFormat = delegate.normsFormat();
      var asserting = new AssertingNormsFormat();
      try {
        // Pinned Lucene 10.5 private final instance delegate, preserving the actual wrapper.
        var field = AssertingNormsFormat.class.getDeclaredField("in");
        field.setAccessible(true);
        field.set(asserting, new NativeTraceFormat(nativeFormat));
      } catch (ReflectiveOperationException error) { throw new IllegalStateException(error); }
      norms = new NormsFormat() {
        @Override public NormsConsumer normsConsumer(SegmentWriteState state) throws IOException {
          int id = nextConsumer++;
          OPENING.set(id);
          try { return new InputTraceConsumer(id, asserting.normsConsumer(state)); }
          finally { OPENING.remove(); }
        }
        @Override public NormsProducer normsProducer(SegmentReadState state) throws IOException {
          String identity = state.segmentInfo.name;
          READS.add(phase + ":open:" + readerPurpose + ":" + identity);
          return new ReadTraceProducer(nativeFormat.normsProducer(state), identity, readerPurpose);
        }
      };
    }
    @Override public NormsFormat normsFormat() { return norms; }
  }
  private static final class NativeTraceFormat extends NormsFormat {
    final NormsFormat delegate;
    NativeTraceFormat(NormsFormat delegate) { this.delegate = delegate; }
    @Override public NormsConsumer normsConsumer(SegmentWriteState state) throws IOException {
      int id = OPENING.get();
      event("open:" + id + ":" + state.segmentInfo.maxDoc());
      var consumer = delegate.normsConsumer(state);
      return new NormsConsumer() {
        @Override public void addNormsField(FieldInfo field, NormsProducer producer) throws IOException {
          event("native-add:" + id + ":" + field.name + ":" + field.number);
          consumer.addNormsField(field, producer);
        }
        @Override public void close() throws IOException { event("native-close:" + id); consumer.close(); }
      };
    }
    @Override public NormsProducer normsProducer(SegmentReadState state) throws IOException {
      throw new AssertionError("writer-only assertion format must never open a reader");
    }
  }
  private static final class InputTraceConsumer extends NormsConsumer {
    final int id;
    final NormsConsumer delegate;
    InputTraceConsumer(int id, NormsConsumer delegate) { this.id = id; this.delegate = delegate; }
    @Override public void addNormsField(FieldInfo field, NormsProducer producer) throws IOException {
      event("add:" + id + ":" + field.name + ":" + field.number);
      delegate.addNormsField(field, new NormsProducer() {
        int pass;
        @Override public NumericDocValues getNorms(FieldInfo requested) throws IOException {
          var values = producer.getNorms(requested);
          int acquisition = pass++;
          String prefix = id + ":" + requested.name + ":" + acquisition;
          event("get:" + prefix + ":" + values.docID());
          if (values.docID() != -1) throw new AssertionError("producer must return a fresh iterator");
          return new NumericDocValues() {
            @Override public int docID() { return values.docID(); }
            @Override public int nextDoc() throws IOException {
              int doc = values.nextDoc(); event("next:" + prefix + ":" + doc); return doc;
            }
            @Override public int advance(int target) throws IOException {
              int doc = values.advance(target); event("advance:" + prefix + ":" + target + ":" + doc); return doc;
            }
            @Override public boolean advanceExact(int target) throws IOException {
              boolean present = values.advanceExact(target); event("exact:" + prefix + ":" + target + ":" + present); return present;
            }
            @Override public long longValue() throws IOException {
              long value = values.longValue(); event("value:" + prefix + ":" + values.docID() + ":" + value); return value;
            }
            @Override public long cost() { return values.cost(); }
            @Override public int docIDRunEnd() throws IOException {
              int end = values.docIDRunEnd(); event("run-end:" + prefix + ":" + end); return end;
            }
            @Override public void intoBitSet(int upTo, FixedBitSet bits, int offset) throws IOException {
              int before = values.docID();
              values.intoBitSet(upTo, bits, offset);
              var documents = new ArrayList<String>();
              for (int doc = bits.nextSetBit(0); doc != DocIdSetIterator.NO_MORE_DOCS;
                  doc = doc + 1 < bits.length() ? bits.nextSetBit(doc + 1) : DocIdSetIterator.NO_MORE_DOCS) {
                documents.add(Integer.toString(doc + offset));
              }
              event("bits:" + prefix + ":" + before + ":" + upTo + ":" + offset + ":"
                  + values.docID() + ":" + String.join(",", documents));
            }
          };
        }
        @Override public void checkIntegrity() throws IOException { producer.checkIntegrity(); }
        @Override public void close() throws IOException { producer.close(); }
      });
    }
    @Override public void close() throws IOException { delegate.close(); }
  }
  private static final class ReadTraceProducer extends NormsProducer {
    final NormsProducer delegate;
    final String segment, purpose;
    ReadTraceProducer(NormsProducer delegate, String segment, String purpose) {
      this.delegate = delegate; this.segment = segment; this.purpose = purpose;
    }
    private void read(String operation) { READS.add(phase + ":" + operation + ":" + purpose + ":" + segment); }
    @Override public NumericDocValues getNorms(FieldInfo field) throws IOException {
      read("get:" + field.name + ":" + field.number); return delegate.getNorms(field);
    }
    @Override public void checkIntegrity() throws IOException { read("integrity"); delegate.checkIntegrity(); }
    @Override public void close() throws IOException { read("close"); delegate.close(); }
    @Override public NormsProducer getMergeInstance() {
      read("merge-instance"); return new ReadTraceProducer(delegate.getMergeInstance(), segment, purpose);
    }
  }
  private static Map<String, String> fields(String mode, int doc) {
    var fields = new LinkedHashMap<String, String>();
    fields.put("constant", "x"); fields.put("varying", VARYING[doc]);
    if (SPARSE[doc] != null) fields.put("sparse", SPARSE[doc]);
    if (doc == 1) fields.put("vanish", "v");
    if (doc == 4) fields.put("vanish", "v v");
    if (doc == 6) fields.put("vanish", "v v v");
    if (mode.equals("explicit") && doc >= 6) fields.put("late", doc == 6 ? "" : "late late");
    fields.put("omitted", "o o");
    return fields;
  }
  private static Document document(String mode, int ordinal) {
    var doc = new Document();
    doc.add(new StringField("id", "" + (char) ('a' + ordinal), Field.Store.YES));
    doc.add(new NumericDocValuesField("rank", RANKS[ordinal]));
    for (var entry : fields(mode, ordinal).entrySet()) {
      var type = new FieldType(TextField.TYPE_NOT_STORED);
      type.setOmitNorms(entry.getKey().equals("omitted"));
      if (mode.equals("general") && entry.getKey().equals("constant")) type.setStoreTermVectors(true);
      type.freeze();
      doc.add(new Field(entry.getKey(), entry.getValue(), type));
    }
    return doc;
  }
  private static IndexWriterConfig config(boolean sorted) {
    var config = new IndexWriterConfig(new StandardAnalyzer());
    config.setCodec(new RoutingCodec()); config.setUseCompoundFile(false);
    config.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy(); policy.setMergeFactor(1000);
    config.setMergePolicy(policy); config.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) config.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return config;
  }
  private static long norm(String text) {
    return text.isEmpty() ? 0 : Byte.toUnsignedLong(SmallFloat.intToByte4(text.split(" ").length));
  }
  private static void snapshot(String name, String mode, Directory directory) throws IOException {
    readerPurpose = "snapshot";
    try (var reader = DirectoryReader.open(directory)) {
      row("CASE", name, phase, reader.maxDoc(), reader.numDocs(), reader.leaves().size());
      for (var leaf : reader.leaves()) {
        var segment = leaf.reader();
        for (var field : segment.getFieldInfos()) row("FIELD", name, phase, leaf.ord, field.name,
            field.number, field.getIndexOptions(), field.omitsNorms(), field.hasNorms());
        var live = segment.getLiveDocs(); var stored = segment.storedFields();
        var all = new LinkedHashMap<String, NumericDocValues>();
        for (var field : segment.getFieldInfos()) if (field.hasNorms()) {
          var values = segment.getNormValues(field.name);
          if (values == null) throw new AssertionError("missing norm iterator " + field.name);
          all.put(field.name, values);
        }
        for (int doc = 0; doc < segment.maxDoc(); doc++) {
          String id = stored.document(doc).get("id");
          if (id == null || id.length() != 1) throw new AssertionError("missing stored identity");
          int ordinal = id.charAt(0) - 'a';
          row("DOC", name, phase, leaf.ord, doc, live == null || live.get(doc) ? 1 : 0, id);
          var input = fields(mode, ordinal);
          for (var entry : all.entrySet()) {
            boolean present = entry.getValue().advanceExact(doc);
            String expected = input.get(entry.getKey());
            if (present != (expected != null)) throw new AssertionError("norm presence " + id + ":" + entry.getKey());
            long value = present ? entry.getValue().longValue() : 0;
            if (present && value != norm(expected)) throw new AssertionError("norm value " + id + ":" + entry.getKey());
            row("NORM", name, phase, leaf.ord, doc, entry.getKey(), present ? 1 : 0, present ? value : "-");
          }
        }
      }
      var hits = new IndexSearcher(reader).search(new TermQuery(new Term("varying", "x")), reader.maxDoc());
      for (int ordinal = 0; ordinal < hits.scoreDocs.length; ordinal++) {
        var hit = hits.scoreDocs[ordinal];
        row("QUERY", name, phase, ordinal, hit.doc, reader.storedFields().document(hit.doc).get("id"),
            Integer.toUnsignedString(Float.floatToRawIntBits(hit.score)));
      }
    } finally { readerPurpose = "writer"; }
  }
  private static void commit(String name, String mode, String nextPhase, IndexWriter writer, Directory directory) throws IOException {
    phase = nextPhase; writer.commit(); snapshot(name, mode, directory);
  }
  private static void scenario(String mode, boolean sorted) throws Exception {
    String name = mode + (sorted ? "_sorted" : "_unsorted");
    EVENTS.clear(); READS.clear(); nextConsumer = 0; phase = "initial";
    for (int doc = 0; doc < RANKS.length; doc++) {
      var encoded = new ArrayList<String>();
      for (var field : fields(mode, doc).entrySet()) encoded.add(field.getKey() + ":" + hex(field.getValue()));
      row("INPUT", name, doc, (char) ('a' + doc), RANKS[doc], String.join(";", encoded));
    }
    try (var directory = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(directory, config(sorted))) {
        for (int doc = 0; doc < 6; doc++) {
          writer.addDocument(document(mode, doc));
          if (doc == 2 || doc == 5) writer.commit();
        }
        snapshot(name, mode, directory);
      }
      try (var writer = new IndexWriter(directory, config(sorted))) {
        phase = "append";
        writer.addDocument(document(mode, 6)); writer.addDocument(document(mode, 7));
        commit(name, mode, "append", writer, directory);
        phase = "deleted";
        writer.deleteDocuments(new Term("id", "b"), new Term("id", "e"), new Term("id", "g"));
        commit(name, mode, "deleted", writer, directory);
        phase = "merged"; writer.forceMerge(1);
        commit(name, mode, "merged", writer, directory);
      }
    }
    for (int i = 0; i < EVENTS.size(); i++) row("TRACE", name, i, EVENTS.get(i));
    // Reader-pool teardown order is not part of the writer consumer contract.
    // Retain every native acquisition/lifecycle event, with multiplicity.
    READS.sort(String::compareTo);
    for (int i = 0; i < READS.size(); i++) row("READ", name, i, READS.get(i));
  }
  public static void main(String[] args) throws Exception {
    if (!GenNormsWriterRouting.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path output = Path.of(args[0]); Files.createDirectories(output);
    Path providers = Files.createTempDirectory("norms-routing-provider");
    Path service = providers.resolve("META-INF/services/org.apache.lucene.codecs.Codec");
    Files.createDirectories(service.getParent()); Files.writeString(service, RoutingCodec.class.getName() + "\n");
    try (var loader = new URLClassLoader(new java.net.URL[] {providers.toUri().toURL()}, GenNormsWriterRouting.class.getClassLoader())) {
      Codec.reloadCodecs(loader);
      if (!(Codec.forName("NormsRoutingOracle") instanceof RoutingCodec)) throw new AssertionError("SPI registration");
      row("# Scope: actual AssertingNormsFormat consumer only; private final instance delegate replaced with native tracing; native readers retain original behavior");
      row("# INPUT case ordinal id rank name:utf8hex[;...]; omitted names are absent, empty hex is present empty text");
      row("# CASE case phase maxDoc numDocs leaves; FIELD case phase leaf name number indexOptions omitNorms hasNorms");
      row("# DOC case phase leaf localDoc live id; NORM case phase leaf localDoc name present value(- when absent)");
      row("# QUERY case phase hitOrdinal globalDoc storedId unsignedFloatRawBits; TermQuery varying:x default BM25");
      row("# TRACE case ordinal phase:event; open/add/get/next/value/native-add/native-close preserve actual producer passes and double native close");
      row("# READ case ordinal phase:event native producer acquisition provenance, sorted multiset with duplicates; segment names only, no random IDs; Rust retains impact columns rather than reopening norms for postings");
      for (String mode : new String[] {"fast", "general", "explicit"}) for (boolean sorted : new boolean[] {false, true}) scenario(mode, sorted);
      Files.writeString(output.resolve("norms_writer_routing.tsv"), OUT);
    } finally {
      Files.deleteIfExists(service); Files.deleteIfExists(service.getParent());
      Files.deleteIfExists(service.getParent().getParent()); Files.deleteIfExists(providers);
    }
  }
}
