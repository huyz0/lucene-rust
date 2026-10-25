import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.cranky.CrankyCodec;
import org.apache.lucene.util.*;

/** Actual CrankyCodec composition; component I/O itself is tested by its own fixtures. */
public class GenCrankyCodec {
  static final String[] KINDS = {"docValues", "fieldInfos", "liveDocs", "norms", "postings",
      "segmentInfo", "storedFields", "termVectors", "compound", "points", "knnVectors"};
  static final FieldInfos EMPTY = new FieldInfos(new FieldInfo[0]);
  interface Action { void run() throws Exception; }
  static String outcome(Action action) throws Exception {
    try { action.run(); return "ok"; }
    catch (IOException error) { return "io:" + error.getMessage(); }
    catch (IllegalStateException error) { return "runtime:" + error.getMessage(); }
  }
  static void row(StringBuilder out, Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) out.append('\t'); out.append(fields[i]); }
    out.append('\n');
  }
  static Object member(Object object, String name) throws Exception {
    for (Class<?> type = object.getClass(); type != null; type = type.getSuperclass()) {
      try { var field = type.getDeclaredField(name); field.setAccessible(true); return field.get(object); }
      catch (NoSuchFieldException missing) { /* Search the actual superclass fields. */ }
    }
    throw new NoSuchFieldException(object.getClass().getName() + "." + name);
  }
  static Object component(Codec codec, int kind) {
    return switch (kind) {
      case 0 -> codec.docValuesFormat(); case 1 -> codec.fieldInfosFormat();
      case 2 -> codec.liveDocsFormat(); case 3 -> codec.normsFormat();
      case 4 -> codec.postingsFormat(); case 5 -> codec.segmentInfoFormat();
      case 6 -> codec.storedFieldsFormat(); case 7 -> codec.termVectorsFormat();
      case 8 -> codec.compoundFormat(); case 9 -> codec.pointsFormat();
      case 10 -> codec.knnVectorsFormat(); default -> throw new AssertionError(kind);
    };
  }
  static final class RecordingCodec extends FilterCodec {
    final List<String> events = new ArrayList<>();
    final Object[] last = new Object[KINDS.length];
    final boolean probes;
    int failFactory = -1;
    boolean failOperation, failDisplay;
    final FixedBitSet bits = new FixedBitSet(1);
    RecordingCodec(boolean probes) { super("CodecProbe", new Lucene104Codec()); this.probes = probes; }
    void before(int kind) {
      events.add("factory:" + KINDS[kind]);
      if (failFactory == kind) throw new IllegalStateException("factory:" + KINDS[kind]);
    }
    <T> T result(int kind, T format) { last[kind] = format; return format; }
    void call(String name) throws IOException {
      events.add(name);
      if (failOperation) throw new IOException("delegate:" + name);
    }
    // Probe metadata is intentionally ignored: this fixture tests composition,
    // guard/delegate order and shared RNG, not native format bytes or validation.
    final FieldInfosFormat fieldInfosProbe = new FieldInfosFormat() {
      @Override public FieldInfos read(Directory d, SegmentInfo s, String suffix, IOContext c) throws IOException { call("fieldInfos:read"); return EMPTY; }
      @Override public void write(Directory d, SegmentInfo s, String suffix, FieldInfos f, IOContext c) throws IOException { call("fieldInfos:write"); }
    };
    final LiveDocsFormat liveDocsProbe = new LiveDocsFormat() {
      @Override public Bits readLiveDocs(Directory d, SegmentCommitInfo s, IOContext c) throws IOException { call("liveDocs:read"); return bits; }
      @Override public void writeLiveDocs(Bits b, Directory d, SegmentCommitInfo s, int count, IOContext c) throws IOException { call("liveDocs:write"); }
      @Override public void files(SegmentCommitInfo s, Collection<String> files) throws IOException { call("liveDocs:files"); files.add("delegate-file"); }
    };
    final SegmentInfoFormat segmentInfoProbe = new SegmentInfoFormat() {
      @Override public SegmentInfo read(Directory d, String segment, byte[] id, IOContext c) throws IOException { throw new AssertionError("unneeded segment read"); }
      @Override public void write(Directory d, SegmentInfo s, IOContext c) throws IOException { call("segmentInfo:write"); }
    };
    final CompoundFormat compoundProbe = new CompoundFormat() {
      @Override public CompoundDirectory getCompoundReader(Directory d, SegmentInfo s) throws IOException { throw new AssertionError("unneeded compound read"); }
      @Override public void write(Directory d, SegmentInfo s, IOContext c) throws IOException { call("compound:write"); }
    };
    @Override public DocValuesFormat docValuesFormat() { before(0); return result(0, delegate.docValuesFormat()); }
    @Override public FieldInfosFormat fieldInfosFormat() { before(1); return result(1, probes ? fieldInfosProbe : delegate.fieldInfosFormat()); }
    @Override public LiveDocsFormat liveDocsFormat() { before(2); return result(2, probes ? liveDocsProbe : delegate.liveDocsFormat()); }
    @Override public NormsFormat normsFormat() { before(3); return result(3, delegate.normsFormat()); }
    @Override public PostingsFormat postingsFormat() { before(4); return result(4, delegate.postingsFormat()); }
    @Override public SegmentInfoFormat segmentInfoFormat() { before(5); return result(5, probes ? segmentInfoProbe : delegate.segmentInfoFormat()); }
    @Override public StoredFieldsFormat storedFieldsFormat() { before(6); return result(6, delegate.storedFieldsFormat()); }
    @Override public TermVectorsFormat termVectorsFormat() { before(7); return result(7, delegate.termVectorsFormat()); }
    @Override public CompoundFormat compoundFormat() { before(8); return result(8, probes ? compoundProbe : delegate.compoundFormat()); }
    @Override public PointsFormat pointsFormat() { before(9); return result(9, delegate.pointsFormat()); }
    @Override public KnnVectorsFormat knnVectorsFormat() { before(10); return result(10, delegate.knnVectorsFormat()); }
    @Override public String toString() {
      events.add("display");
      if (failDisplay) throw new IllegalStateException("display");
      return "DelegateDisplay";
    }
  }
  static void configuration(StringBuilder out) throws Exception {
    var delegate = new RecordingCodec(false); var random = new Random(42); var control = new Random(42);
    var codec = new CrankyCodec(delegate, random);
    if (member(codec, "delegate") != delegate || member(codec, "random") != random) throw new AssertionError("codec identity");
    row(out, "metadata", "name", codec.getName());
    row(out, "metadata", "display", codec.toString());
    if (!delegate.events.equals(List.of("display"))) throw new AssertionError(delegate.events);
    for (int kind = 0; kind < KINDS.length; kind++) {
      delegate.events.clear();
      Object first = component(codec, kind), firstDelegate = delegate.last[kind];
      Object second = component(codec, kind), secondDelegate = delegate.last[kind];
      boolean fresh = first != second;
      boolean exact = kind == 10 ? first == firstDelegate && second == secondDelegate
          : member(first, "delegate") == firstDelegate && member(second, "delegate") == secondDelegate;
      boolean sharedRandom = kind == 10 || member(first, "random") == random && member(second, "random") == random;
      boolean sameDelegate = firstDelegate == secondDelegate;
      if (fresh != (kind != 10) || !exact || !sharedRandom || sameDelegate != (kind != 9)) throw new AssertionError(KINDS[kind]);
      String name = kind == 0 ? ((DocValuesFormat)first).getName() : kind == 4 ? ((PostingsFormat)first).getName() : "-";
      int next = random.nextInt(); if (next != control.nextInt()) throw new AssertionError("accessor consumed random");
      row(out, "accessor", KINDS[kind], kind == 10 ? "delegate" : first.getClass().getSimpleName(), fresh, exact, kind == 10 ? "-" : sharedRandom, sameDelegate, name, next, String.join(",", delegate.events));
    }
    for (int kind = 0; kind < KINDS.length; kind++) {
      delegate.events.clear(); delegate.failFactory = kind; final int requested = kind;
      String status = outcome(() -> component(codec, requested));
      int next = random.nextInt(); if (next != control.nextInt()) throw new AssertionError("failed accessor consumed random");
      row(out, "factory-failure", KINDS[kind], status, next, String.join(",", delegate.events));
      delegate.failFactory = -1;
    }
    delegate.events.clear(); delegate.failDisplay = true;
    String status = outcome(() -> codec.toString());
    int next = random.nextInt(); if (next != control.nextInt()) throw new AssertionError("display consumed random");
    row(out, "display-failure", status, next, String.join(",", delegate.events));
  }
  static final String[] OPS = {"held-fieldInfos-write", "liveDocs-write", "segmentInfo-write", "compound-write",
      "fieldInfos-read", "liveDocs-read", "liveDocs-files", "all-accessors", "display", "failed-factory", "external-draw"};
  static void interleaved(StringBuilder out) throws Exception {
    for (long seed : new long[]{0, 1, 18, 42, -1, Long.MIN_VALUE, Long.MAX_VALUE}) {
      var delegate = new RecordingCodec(true); var random = new Random(seed); var codec = new CrankyCodec(delegate, random);
      FieldInfosFormat[] held = {codec.fieldInfosFormat(), codec.fieldInfosFormat()};
      if (held[0] == held[1] || member(held[0], "random") != random || member(held[1], "random") != random) throw new AssertionError("held wrapper identity");
      for (int step = 0; step < 384; step++) {
        delegate.events.clear(); delegate.failOperation = step % 17 == 0; final int operation = step % OPS.length;
        final int side = (step / OPS.length) & 1;
        String status = outcome(() -> {
          switch (operation) {
            case 0 -> held[side].write(null, null, "", EMPTY, IOContext.DEFAULT);
            case 1 -> codec.liveDocsFormat().writeLiveDocs(delegate.bits, null, null, 0, IOContext.DEFAULT);
            case 2 -> codec.segmentInfoFormat().write(null, null, IOContext.DEFAULT);
            case 3 -> codec.compoundFormat().write(null, null, IOContext.DEFAULT);
            case 4 -> { if (codec.fieldInfosFormat().read(null, null, "", IOContext.DEFAULT) != EMPTY) throw new AssertionError("field-info reader identity"); }
            case 5 -> { if (codec.liveDocsFormat().readLiveDocs(null, null, IOContext.DEFAULT) != delegate.bits) throw new AssertionError("live-doc reader identity"); }
            case 6 -> { var files = new ArrayList<String>(); files.add("existing"); codec.liveDocsFormat().files(null, files); if (!files.equals(List.of("existing", "delegate-file"))) throw new AssertionError("files forwarding"); }
            case 7 -> { for (int kind = 0; kind < KINDS.length; kind++) component(codec, kind); }
            case 8 -> { if (!codec.toString().equals("Cranky(DelegateDisplay)")) throw new AssertionError("display"); }
            case 9 -> { delegate.failFactory = 1; codec.fieldInfosFormat().write(null, null, "", EMPTY, IOContext.DEFAULT); }
            case 10 -> delegate.events.add("external:" + random.nextBoolean());
            default -> throw new AssertionError(operation);
          }
        });
        delegate.failFactory = -1;
        row(out, "mixed", seed, step, OPS[operation], delegate.failOperation, side, status, random.nextInt(), String.join(",", delegate.events));
      }
    }
  }
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]); Files.createDirectories(out);
    var result = new StringBuilder(); configuration(result); interleaved(result);
    Files.writeString(out.resolve("cranky_codec.tsv"), result);
  }
}