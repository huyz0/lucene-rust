import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Locale;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.RamUsageEstimator;

/** Java twin of live_docs_writer_routing_micro; only deletion commit is timed. */
public final class LiveDocsWriterRoutingMicro {
  private static final int DOCS = 1024, SOURCES = 8;
  private static volatile long sink;

  /** Same native codec name and readers; only writes use assertions. */
  private static final class WriterOnlyAssertingCodec extends FilterCodec {
    final AssertingLiveDocsFormat asserting = new AssertingLiveDocsFormat();
    final LiveDocsFormat format;
    WriterOnlyAssertingCodec() {
      super("Lucene104", new Lucene104Codec());
      var nativeFormat = delegate.liveDocsFormat();
      format = new LiveDocsFormat() {
        @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context)
            throws IOException { return nativeFormat.readLiveDocs(dir, info, context); }
        @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info,
            int count, IOContext context) throws IOException {
          asserting.writeLiveDocs(bits, dir, info, count, context);
        }
        @Override public void files(SegmentCommitInfo info, Collection<String> files) throws IOException {
          nativeFormat.files(info, files);
        }
      };
    }
    @Override public LiveDocsFormat liveDocsFormat() { return format; }
  }

  private static boolean deleted(int doc, boolean subsequent) {
    return doc % 7 == 0 || (subsequent && doc % 11 == 0);
  }
  private static long mix(long digest, long doc) { return (digest ^ doc) * 0x100000001b3L; }
  private record Expected(long digest, int count) {}
  private static Expected expected(boolean subsequent) {
    long digest = 0xcbf29ce484222325L;
    int count = 0;
    for (int doc = 0; doc < DOCS; doc++) if (!deleted(doc, subsequent)) {
      digest = mix(digest, doc);
      count++;
    }
    return new Expected(digest, count);
  }
  private static IndexWriterConfig config(boolean asserting) {
    var result = new IndexWriterConfig();
    result.setCodec(asserting ? new WriterOnlyAssertingCodec() : new Lucene104Codec());
    result.setUseCompoundFile(false);
    result.setMaxBufferedDocs(DOCS / SOURCES);
    result.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    result.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy();
    policy.setMergeFactor(1000);
    result.setMergePolicy(policy);
    result.setMergeScheduler(new SerialMergeScheduler());
    return result;
  }
  private static void queueDeletes(IndexWriter writer, boolean subsequent) throws IOException {
    var terms = new ArrayList<Term>();
    for (int doc = 0; doc < DOCS; doc++) {
      boolean remove = subsequent ? doc % 11 == 0 && doc % 7 != 0 : doc % 7 == 0;
      if (remove) terms.add(new Term("id", Integer.toString(doc)));
    }
    writer.deleteDocuments(terms.toArray(Term[]::new));
  }
  private static void prepare(IndexWriter writer, Directory directory, boolean subsequent) throws IOException {
    for (int doc = 0; doc < DOCS; doc++) {
      var document = new Document();
      document.add(new StringField("id", Integer.toString(doc), Field.Store.NO));
      writer.addDocument(document);
      if ((doc + 1) % (DOCS / SOURCES) == 0) writer.flush();
    }
    writer.commit();
    if (subsequent) { queueDeletes(writer, false); writer.commit(); }
    queueDeletes(writer, subsequent);
    String suffix = subsequent ? "_2.liv" : "_1.liv";
    for (String file : directory.listAll()) if (file.endsWith(suffix)) {
      throw new AssertionError("pending deletion generation was written before timing");
    }
  }
  private static long verify(Directory directory, boolean subsequent) throws IOException {
    var expected = expected(subsequent);
    long digest = 0xcbf29ce484222325L;
    try (var reader = DirectoryReader.open(directory)) {
      if (reader.maxDoc() != DOCS || reader.numDocs() != expected.count
          || reader.leaves().size() != SOURCES) throw new AssertionError("output shape differs");
      for (var context : reader.leaves()) {
        var live = context.reader().getLiveDocs();
        if (live == null) throw new AssertionError("missing live docs");
        for (int doc = 0; doc < context.reader().maxDoc(); doc++) {
          int global = context.docBase + doc;
          if (live.get(doc) == deleted(global, subsequent)) throw new AssertionError("live bit differs");
          if (live.get(doc)) digest = mix(digest, global);
        }
      }
    }
    if (digest != expected.digest) throw new AssertionError("semantic digest differs");
    int files = 0;
    long bytes = 0;
    String suffix = subsequent ? "_2.liv" : "_1.liv";
    for (String file : directory.listAll()) if (file.endsWith(suffix)) {
      files++;
      bytes += directory.fileLength(file);
    }
    if (files != SOURCES) throw new AssertionError("live-doc generations differ");
    sink = digest;
    return bytes;
  }
  private record Measurement(long nanos, long iterations, long bytes) {}
  private static Measurement run(boolean asserting, boolean subsequent, long millis) throws Exception {
    long elapsed = 0, iterations = 0, bytes;
    do {
      try (var directory = new ByteBuffersDirectory();
          var writer = new IndexWriter(directory, config(asserting))) {
        prepare(writer, directory, subsequent);
        long started = System.nanoTime();
        writer.commit();
        elapsed += System.nanoTime() - started;
        iterations++;
        bytes = verify(directory, subsequent);
      }
    } while (elapsed < millis * 1_000_000L);
    return new Measurement(elapsed, iterations, bytes);
  }
  public static void main(String[] args) throws Exception {
    if (!LiveDocsWriterRoutingMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    if (args[0].equals("--verify-dir")) {
      try (var directory = FSDirectory.open(Path.of(args[1]))) {
        long bytes = verify(directory, true);
        System.out.println("native Lucene104 read Rust live-docs: "
            + Long.toUnsignedString(expected(true).digest) + " bytes=" + bytes);
      }
      return;
    }
    long warmup = Long.parseLong(args[0]), duration = Long.parseLong(args[1]);
    if (warmup < 0 || duration <= 0) throw new IllegalArgumentException("invalid durations");
    var codec = new WriterOnlyAssertingCodec();
    System.out.println("#memory_scope\tconfig shallow sum covers adapting codec/format/asserting format; native writer, directory, allocator, RSS excluded");
    System.out.println("liv_routing_asserting_codec_shallow_bytes\t"
        + (RamUsageEstimator.shallowSizeOf(codec) + RamUsageEstimator.shallowSizeOf(codec.format)
            + RamUsageEstimator.shallowSizeOf(codec.asserting)));
    for (boolean subsequent : new boolean[] {false, true}) {
      for (boolean asserting : new boolean[] {false, true}) {
        String name = "liv_routing_" + (asserting ? "asserting" : "default") + "_"
            + (subsequent ? "subsequent" : "first") + "_commit";
        run(asserting, subsequent, warmup);
        var result = run(asserting, subsequent, duration);
        var expected = expected(subsequent);
        System.out.printf("#check\t%s\t%s\t%d\t%d\t%d%n", name,
            Long.toUnsignedString(expected.digest), expected.count, SOURCES, result.bytes);
        System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name,
            (double) result.nanos / result.iterations, result.iterations);
        System.out.printf("%s_live_docs_encoded_bytes\t%d%n", name, result.bytes);
      }
    }
  }
}
