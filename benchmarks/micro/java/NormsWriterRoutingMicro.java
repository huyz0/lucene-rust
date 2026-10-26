import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Locale;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.NormsConsumer;
import org.apache.lucene.codecs.NormsFormat;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.codecs.asserting.AssertingNormsFormat;
import org.apache.lucene.util.RamUsageEstimator;

/** Whole flush/merge commit twin; output close and publication occur inside timing. */
public final class NormsWriterRoutingMicro {
  private static final int DOCS = 1024, SOURCES = 8;
  private static final String[] FIELDS = {"constant", "varying", "sparse", "vanish"};
  private static volatile long sink;
  private static final class WriterOnlyAssertingCodec extends FilterCodec {
    final AssertingNormsFormat asserting = new AssertingNormsFormat();
    final NormsFormat format;
    WriterOnlyAssertingCodec() {
      super("Lucene104", new Lucene104Codec());
      var nativeFormat = delegate.normsFormat();
      format = new NormsFormat() {
        @Override public NormsConsumer normsConsumer(SegmentWriteState state) throws IOException {
          return asserting.normsConsumer(state);
        }
        @Override public NormsProducer normsProducer(SegmentReadState state) throws IOException {
          return nativeFormat.normsProducer(state);
        }
      };
    }
    @Override public NormsFormat normsFormat() { return format; }
  }
  private static Long value(int doc, int field) {
    return switch (field) {
      case 0 -> 1L;
      case 1 -> (long) (doc % 5);
      case 2 -> doc % 3 == 0 ? null : doc % 5 == 0 ? 0L : 3L;
      case 3 -> doc % 7 == 0 ? 2L : null;
      default -> throw new AssertionError("unknown field");
    };
  }
  private static Document document(int doc) {
    var result = new Document();
    result.add(new StringField("id", Integer.toString(doc), Field.Store.YES));
    for (int field = 0; field < FIELDS.length; field++) {
      Long length = value(doc, field);
      if (length != null) result.add(new TextField(FIELDS[field], "x ".repeat(length.intValue()).stripTrailing(), Field.Store.NO));
    }
    return result;
  }
  private static IndexWriterConfig config(boolean asserting) {
    var config = new IndexWriterConfig(new StandardAnalyzer());
    config.setCodec(asserting ? new WriterOnlyAssertingCodec() : new Lucene104Codec());
    config.setUseCompoundFile(false); config.setMaxBufferedDocs(DOCS * 2);
    config.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH); config.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy(); policy.setMergeFactor(1000);
    config.setMergePolicy(policy); config.setMergeScheduler(new SerialMergeScheduler());
    return config;
  }
  private static void prepare(IndexWriter writer, Directory directory, boolean merged) throws IOException {
    for (int doc = 0; doc < DOCS; doc++) {
      writer.addDocument(document(doc));
      if (merged && (doc + 1) % (DOCS / SOURCES) == 0) writer.commit();
    }
    if (merged) {
      var terms = new ArrayList<Term>();
      for (int doc = 0; doc < DOCS; doc++) if (doc % 7 == 0) terms.add(new Term("id", Integer.toString(doc)));
      writer.deleteDocuments(terms.toArray(Term[]::new)); writer.commit();
      if (SegmentInfos.readLatestCommit(directory).size() != SOURCES) throw new AssertionError("merge source count");
    }
    int normFiles = 0;
    for (String name : directory.listAll()) if (name.endsWith(".nvm") || name.endsWith(".nvd")) normFiles++;
    if (normFiles != (merged ? SOURCES * 2 : 0)) throw new AssertionError("norms outputs were created before the intended interval");
  }
  private static long mix(long digest, long value) { return (digest ^ value) * 0x100000001b3L; }
  private record Check(long digest, int docs, long bytes) {}
  private static Check verify(Directory directory, boolean merged) throws IOException {
    var expectedDocs = new ArrayList<Integer>();
    for (int doc = 0; doc < DOCS; doc++) if (!merged || doc % 7 != 0) expectedDocs.add(doc);
    long digest = 0xcbf29ce484222325L;
    try (var reader = DirectoryReader.open(directory)) {
      if (reader.maxDoc() != expectedDocs.size() || reader.numDocs() != expectedDocs.size()
          || reader.leaves().size() != 1) throw new AssertionError("output shape");
      var segment = reader.leaves().get(0).reader();
      var stored = segment.storedFields();
      var norms = new org.apache.lucene.index.NumericDocValues[FIELDS.length];
      for (int field = 0; field < FIELDS.length; field++) {
        norms[field] = segment.getNormValues(FIELDS[field]);
        if (norms[field] == null) throw new AssertionError("missing norms field " + FIELDS[field]);
      }
      for (int physical = 0; physical < expectedDocs.size(); physical++) {
        int id = Integer.parseInt(stored.document(physical).get("id"));
        if (id != expectedDocs.get(physical)) throw new AssertionError("stored identity/order");
        digest = mix(digest, id);
        for (int field = 0; field < FIELDS.length; field++) {
          Long expected = value(id, field);
          boolean present = norms[field].advanceExact(physical);
          if (present != (expected != null)) throw new AssertionError("norm presence");
          long actual = present ? norms[field].longValue() : 0;
          if (present && actual != expected.longValue()) throw new AssertionError("norm value");
          digest = mix(digest, present ? 1 : 0); digest = mix(digest, actual);
        }
      }
      segment.checkIntegrity();
    }
    var infos = SegmentInfos.readLatestCommit(directory);
    long bytes = 0;
    for (var info : infos) {
      bytes += directory.fileLength(info.info.name + ".nvm"); bytes += directory.fileLength(info.info.name + ".nvd");
    }
    sink = digest;
    return new Check(digest, expectedDocs.size(), bytes);
  }
  private record Measurement(long nanos, long iterations, Check check) {}
  private static Measurement run(boolean asserting, boolean merged, long millis) throws Exception {
    long nanos = 0, iterations = 0;
    Check check;
    do {
      try (var directory = new ByteBuffersDirectory(); var writer = new IndexWriter(directory, config(asserting))) {
        prepare(writer, directory, merged);
        long started = System.nanoTime();
        if (merged) writer.forceMerge(1);
        writer.commit();
        nanos += System.nanoTime() - started; iterations++;
        check = verify(directory, merged);
      }
    } while (nanos < millis * 1_000_000L);
    return new Measurement(nanos, iterations, check);
  }
  public static void main(String[] args) throws Exception {
    if (!NormsWriterRoutingMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    if (args[0].equals("--verify-dir")) {
      try (var directory = FSDirectory.open(Path.of(args[1]))) {
        var check = verify(directory, true);
        System.out.println("native Lucene104 read Rust norms: " + Long.toUnsignedString(check.digest) + " bytes=" + check.bytes);
      }
      return;
    }
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid durations");
    var codec = new WriterOnlyAssertingCodec();
    System.out.println("#memory_scope\tcurrent encoded norms files and shallow adapting codec/format/assertion only; writer/directory/allocator/RSS excluded");
    System.out.println("norms_routing_asserting_codec_shallow_bytes\t"
        + (RamUsageEstimator.shallowSizeOf(codec) + RamUsageEstimator.shallowSizeOf(codec.format) + RamUsageEstimator.shallowSizeOf(codec.asserting)));
    for (boolean merged : new boolean[] {false, true}) for (boolean asserting : new boolean[] {false, true}) {
      String name = "norms_routing_" + (asserting ? "asserting" : "default") + "_" + (merged ? "merge" : "flush") + "_commit";
      run(asserting, merged, warmup);
      var result = run(asserting, merged, measure);
      System.out.printf("#check\t%s\t%s\t%d\t%d\t%d%n", name, Long.toUnsignedString(result.check.digest), result.check.docs, FIELDS.length, result.check.bytes);
      System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result.nanos / result.iterations, result.iterations);
      System.out.printf("%s_encoded_bytes\t%d%n", name, result.check.bytes);
    }
  }
}
