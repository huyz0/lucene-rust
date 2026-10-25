import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat;
import org.apache.lucene.util.Bits;

/** Real committed live-doc generations through only the asserting live-doc writer. */
public class GenLiveDocsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<String> EVENTS = new ArrayList<>();
  private static final int[] RANKS = {50, 10, 40, 20, 80, 60, 30, 70, 0};
  private static String phase;

  private static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i != 0) OUT.append('\t');
      OUT.append(fields[i]);
    }
    OUT.append('\n');
  }
  private static String mask(Bits bits, int length) {
    StringBuilder result = new StringBuilder();
    for (int doc = 0; doc < length; doc++) result.append(bits == null || bits.get(doc) ? '1' : '0');
    return result.toString();
  }

  /** Only writeLiveDocs enters the real assertion wrapper. Readers stay native. */
  public static final class RoutingCodec extends FilterCodec {
    private final LiveDocsFormat liveDocs;
    public RoutingCodec() {
      super("LiveDocsRoutingOracle", new Lucene104Codec());
      var nativeFormat = delegate.liveDocsFormat();
      var asserting = new AssertingLiveDocsFormat();
      try {
        var field = AssertingLiveDocsFormat.class.getDeclaredField("in");
        field.setAccessible(true);
        field.set(asserting, new TraceFormat(nativeFormat));
      } catch (ReflectiveOperationException error) {
        throw new IllegalStateException(error);
      }
      liveDocs = new LiveDocsFormat() {
        @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) throws IOException {
          return nativeFormat.readLiveDocs(dir, info, context);
        }
        @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info, int newDelCount, IOContext context) throws IOException {
          asserting.writeLiveDocs(bits, dir, info, newDelCount, context);
        }
        @Override public void files(SegmentCommitInfo info, Collection<String> files) throws IOException {
          nativeFormat.files(info, files);
        }
      };
    }
    @Override public LiveDocsFormat liveDocsFormat() { return liveDocs; }
  }

  private static final class TraceFormat extends LiveDocsFormat {
    private final LiveDocsFormat delegate;
    TraceFormat(LiveDocsFormat delegate) { this.delegate = delegate; }
    @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) throws IOException {
      return delegate.readLiveDocs(dir, info, context);
    }
    @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info, int count, IOContext context) throws IOException {
      // The trace is inside AssertingLiveDocsFormat, immediately before the native delegate.
      EVENTS.add(phase + ":" + info.info.name + ":" + info.info.maxDoc() + ":" + info.getDelGen()
          + ":" + info.getNextDelGen() + ":" + info.getDelCount() + ":" + count + ":" + mask(bits, bits.length()));
      delegate.writeLiveDocs(bits, dir, info, count, context);
    }
    @Override public void files(SegmentCommitInfo info, Collection<String> files) throws IOException {
      delegate.files(info, files);
    }
  }

  private static String group(int ordinal) { return ordinal == 2 || ordinal == 5 ? "drop" : "keep"; }
  private static Document document(int ordinal) {
    Document doc = new Document();
    doc.add(new StringField("id", "" + (char) ('a' + ordinal), Field.Store.YES));
    doc.add(new StringField("group", group(ordinal), Field.Store.NO));
    doc.add(new NumericDocValuesField("rank", RANKS[ordinal]));
    return doc;
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
  private static void snapshot(String name, Directory dir) throws IOException {
    var infos = SegmentInfos.readLatestCommit(dir);
    try (var reader = DirectoryReader.open(dir)) {
      row("CASE", name, phase, reader.maxDoc(), reader.numDocs(), reader.leaves().size());
      for (var leaf : reader.leaves()) {
        var segment = infos.info(leaf.ord);
        var live = leaf.reader().getLiveDocs();
        row("SEG", name, phase, leaf.ord, leaf.reader().maxDoc(), segment.getDelGen(), segment.getDelCount(), mask(live, leaf.reader().maxDoc()));
        var stored = leaf.reader().storedFields();
        for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
          row("DOC", name, phase, leaf.ord, doc, live == null || live.get(doc) ? 1 : 0, stored.document(doc).get("id"));
        }
      }
    }
  }
  private static void commit(String name, String nextPhase, IndexWriter writer, Directory dir) throws IOException {
    phase = nextPhase;
    writer.commit();
    snapshot(name, dir);
  }
  private static void scenario(boolean sorted) throws Exception {
    String name = sorted ? "sorted" : "unsorted";
    EVENTS.clear();
    try (var dir = new ByteBuffersDirectory()) {
      phase = "initial";
      try (var writer = new IndexWriter(dir, config(sorted))) {
        for (int doc = 0; doc < 8; doc++) {
          writer.addDocument(document(doc));
          if (doc == 3 || doc == 7) writer.commit();
        }
        snapshot(name, dir);
        phase = "term";
        writer.deleteDocuments(new Term("id", "b"));
        commit(name, "term", writer, dir);
        phase = "noop";
        writer.deleteDocuments(new Term("id", "b"), new Term("id", "missing"));
        writer.deleteDocuments(new TermQuery(new Term("group", "never")));
        commit(name, "noop", writer, dir);
        phase = "query";
        writer.deleteDocuments(new TermQuery(new Term("group", "drop")));
        commit(name, "query", writer, dir);
      }
      try (var writer = new IndexWriter(dir, config(sorted))) {
        phase = "append";
        writer.addDocument(document(8));
        commit(name, "append", writer, dir);
        phase = "append_delete";
        writer.deleteDocuments(new Term("id", "a"), new Term("id", "h"));
        commit(name, "append_delete", writer, dir);
        phase = "merged";
        writer.forceMerge(1);
        commit(name, "merged", writer, dir);
        phase = "merged_delete";
        writer.deleteDocuments(new Term("id", "g"));
        commit(name, "merged_delete", writer, dir);
      }
    }
    for (int ordinal = 0; ordinal < EVENTS.size(); ordinal++) row("TRACE", name, ordinal, EVENTS.get(ordinal));
  }
  public static void main(String[] args) throws Exception {
    if (!GenLiveDocsWriterRouting.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Path providers = Files.createTempDirectory("live-docs-routing-provider");
    Path service = providers.resolve("META-INF/services/org.apache.lucene.codecs.Codec");
    Files.createDirectories(service.getParent());
    Files.writeString(service, RoutingCodec.class.getName() + "\n");
    try (var loader = new URLClassLoader(new java.net.URL[] {providers.toUri().toURL()}, GenLiveDocsWriterRouting.class.getClassLoader())) {
      Codec.reloadCodecs(loader);
      if (!(Codec.forName("LiveDocsRoutingOracle") instanceof RoutingCodec)) throw new AssertionError("SPI registration");
      row("# Scope: real AssertingLiveDocsFormat.writeLiveDocs only; read/files and all other formats stay native Lucene104");
      row("# CASE case phase maxDoc numDocs leaves; SEG case phase leaf maxDoc delGen delCount mask; DOC case phase leaf localDoc live id");
      row("# TRACE case ordinal phase:segment:maxDoc:oldDelGen:nextDelGen:previousDelCount:newDelCount:mask; actual delegate provenance");
      for (int doc = 0; doc < RANKS.length; doc++) row("INPUT", doc, (char) ('a' + doc), RANKS[doc], group(doc));
      scenario(false);
      scenario(true);
      Files.writeString(out.resolve("live_docs_writer_routing.tsv"), OUT);
    } finally {
      Files.deleteIfExists(service);
      Files.deleteIfExists(service.getParent());
      Files.deleteIfExists(service.getParent().getParent());
      Files.deleteIfExists(providers);
    }
  }
}
