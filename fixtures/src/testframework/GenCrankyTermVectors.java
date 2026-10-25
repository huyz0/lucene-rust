import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executor;
import org.apache.lucene.codecs.TermVectorsFormat;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.codecs.TermVectorsWriter;
import org.apache.lucene.codecs.simpletext.SimpleTextTermVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

/** Actual cranky vector wrapper, including superclass addProx and merge dispatch. */
public class GenCrankyTermVectors {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static final byte[] POS = {3, 2, 0, 10, 4, 1, 0};
  static final byte[] OFF = {0, 2, 1, 3, 0, 0};
  static final StringBuilder OUT = new StringBuilder();
  static final FieldInfo FIELD = field(7);
  static FieldInfo field(int number) {
    return new FieldInfo("f", number, true, false, false,
        IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS,
        DocValuesType.NONE, DocValuesSkipIndexType.NONE, -1, Map.of(),
        0, 0, 0, 0, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, false, false);
  }
  static SegmentInfo info(Directory dir, int docs) {
    return new SegmentInfo(dir, Version.LATEST, Version.LATEST, "_ctv", docs,
        false, false, null, Map.of(), ID, Map.of(), null);
  }
  static void emit(Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  static String payload(BytesRef value) {
    if (value == null) return "-";
    if (value.length == 0) return "e";
    return HexFormat.of().formatHex(value.bytes, value.offset, value.offset + value.length);
  }
  static TermVectorsWriter wrap(TermVectorsWriter delegate, Random random) throws Exception {
    var ctor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyTermVectorsFormat$CrankyTermVectorsWriter")
        .getDeclaredConstructor(TermVectorsWriter.class, Random.class);
    ctor.setAccessible(true);
    return (TermVectorsWriter) ctor.newInstance(delegate, random);
  }
  static TermVectorsFormat format(TermVectorsFormat delegate, Random random) throws Exception {
    var ctor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyTermVectorsFormat")
        .getDeclaredConstructor(TermVectorsFormat.class, Random.class);
    ctor.setAccessible(true);
    return (TermVectorsFormat) ctor.newInstance(delegate, random);
  }
  static class Input extends DataInput {
    final byte[] bytes; int pos;
    Input(byte[] bytes) { this.bytes = bytes; }
    @Override public void skipBytes(long count) throws IOException {
      if (count < 0 || count > bytes.length - pos) throw new EOFException("input");
      pos += Math.toIntExact(count);
    }
    @Override public byte readByte() throws IOException {
      if (pos == bytes.length) throw new EOFException("input");
      return bytes[pos++];
    }
    @Override public void readBytes(byte[] out, int offset, int length) throws IOException {
      if (length > bytes.length - pos) throw new EOFException("input");
      System.arraycopy(bytes, pos, out, offset, length); pos += length;
    }
  }
  static class Probe extends TermVectorsWriter {
    final ArrayList<String> log;
    final String fail;
    int positions;
    Probe(ArrayList<String> log, String fail) { this.log = log; this.fail = fail; }
    void call(String name, String detail) throws IOException {
      log.add(name + detail);
      if (name.equals(fail) || (name.equals("position") && fail.equals("position#" + positions)))
        throw new IOException("delegate " + fail);
    }
    @Override public void startDocument(int fields) throws IOException { call("startDoc", ":" + fields); }
    @Override public void finishDocument() throws IOException { call("finishDoc", ""); }
    @Override public void startField(FieldInfo info, int terms, boolean positions, boolean offsets, boolean payloads) throws IOException {
      call("startField", ":" + info.name + ":" + info.number + ":" + terms + ":" + positions + ":" + offsets + ":" + payloads);
    }
    @Override public void finishField() throws IOException { call("finishField", ""); }
    @Override public void startTerm(BytesRef term, int freq) throws IOException { call("startTerm", ":" + payload(term) + ":" + freq); }
    @Override public void finishTerm() throws IOException { call("finishTerm", ""); }
    @Override public void addPosition(int position, int start, int end, BytesRef payload) throws IOException {
      positions++; call("position", ":" + position + ":" + start + ":" + end + ":" + payload(payload));
    }
    @Override public void addProx(int count, DataInput positions, DataInput offsets) throws IOException { call("BADprox", ""); }
    @Override public int merge(MergeState state) throws IOException { call("BADmerge", ""); return -1; }
    @Override public void finish(int docs) throws IOException { call("finish", ":" + docs); }
    @Override public void close() throws IOException { call("close", ""); }
    @Override public long ramBytesUsed() { log.add("ram"); return 37; }
    @Override public Collection<Accountable> getChildResources() { throw new AssertionError("delegate child resources called"); }
  }
  static class ProbeFormat extends TermVectorsFormat {
    final Probe writer;
    ProbeFormat(Probe writer) { this.writer = writer; }
    @Override public TermVectorsReader vectorsReader(Directory dir, SegmentInfo si, FieldInfos infos, IOContext context) throws IOException {
      writer.call("reader", ""); return null;
    }
    @Override public TermVectorsWriter vectorsWriter(Directory dir, SegmentInfo si, IOContext context) throws IOException {
      writer.call("open", ""); return writer;
    }
  }
  static long seed(int... bounds) {
    for (long seed = 0; seed < 10_000_000; seed++) {
      Random random = new Random(seed); boolean match = true;
      for (int i = 0; i < bounds.length; i++) {
        boolean zero = random.nextInt(bounds[i]) == 0;
        if (zero != (i == bounds.length - 1)) { match = false; break; }
      }
      if (match) return seed;
    }
    throw new AssertionError("seed not found");
  }
  static String outcome(Operation operation) throws Exception {
    try { operation.run(); return "ok"; } catch (IOException error) { return error.getMessage(); }
  }
  @FunctionalInterface interface Operation { void run() throws Exception; }
  static void record(String name, long seed, String fail, String status, ArrayList<String> log,
      Random random, Input positions, Input offsets, int merged) {
    emit("case", name, seed, fail, status, log.isEmpty() ? "-" : String.join(";", log),
        random.nextInt(), random.nextLong(), positions == null ? -1 : positions.pos,
        offsets == null ? -1 : offsets.pos, merged);
  }
  static void direct(String operation, String variant) throws Exception {
    int bound = operation.equals("finish") || operation.equals("close") || operation.equals("open") ? 100 : 10000;
    long seed = variant.equals("inject") ? seed(bound) : 42;
    String fail = variant.equals("delegate") ? operation : "-";
    Random random = new Random(seed); var log = new ArrayList<String>(); Probe probe = new Probe(log, fail);
    TermVectorsWriter writer = wrap(probe, random);
    try (Directory dir = new ByteBuffersDirectory()) {
      String status = outcome(() -> {
        switch (operation) {
          case "reader" -> {
            if (format(new ProbeFormat(probe), random).vectorsReader(dir, info(dir, 0), new FieldInfos(new FieldInfo[0]), IOContext.DEFAULT) != null)
              throw new AssertionError("reader identity");
          }
          case "open" -> format(new ProbeFormat(probe), random).vectorsWriter(dir, info(dir, 0), IOContext.DEFAULT);
          case "startDoc" -> writer.startDocument(1);
          case "finishDoc" -> writer.finishDocument();
          case "startField" -> writer.startField(FIELD, 1, true, true, true);
          case "finishField" -> writer.finishField();
          case "startTerm" -> writer.startTerm(new BytesRef(new byte[] {0, 10, 92, (byte)255}), 3);
          case "finishTerm" -> writer.finishTerm();
          case "position" -> writer.addPosition(7, 10, 13, new BytesRef(new byte[0]));
          case "finish" -> writer.finish(2);
          case "close" -> writer.close();
          case "resources" -> {
            if (writer.ramBytesUsed() != 37 || writer.getChildResources().size() != 1
                || writer.getChildResources().iterator().next() != probe) throw new AssertionError("resource delegation");
          }
          default -> throw new AssertionError(operation);
        }
      });
      record("direct:" + operation + ":" + variant, seed, fail, status, log, random, null, null, -1);
    }
  }
  static void prox(String variant) throws Exception {
    long seed = switch (variant) {
      case "inject" -> seed(10000);
      case "nested1" -> seed(10000, 10000);
      case "nested2" -> seed(10000, 10000, 10000);
      default -> 42;
    };
    String fail = variant.equals("delegate2") ? "position#2" : "-";
    var log = new ArrayList<String>(); Random random = new Random(seed);
    TermVectorsWriter writer = wrap(new Probe(log, fail), random);
    Input positions = variant.equals("none") || variant.equals("offsets") ? null
        : new Input(variant.equals("shortpos") ? new byte[] {3} : POS);
    Input offsets = variant.equals("none") || variant.equals("positions") ? null
        : new Input(variant.equals("shortoff") ? new byte[] {0} : OFF);
    int count = variant.equals("zero") ? 0 : variant.equals("negative") ? -1 : 3;
    String status = outcome(() -> writer.addProx(count, positions, offsets));
    record("prox:" + variant, seed, fail, status, log, random, positions, offsets, -1);
  }
  static void document(TermVectorsWriter writer, FieldInfo field, boolean rich) throws IOException {
    writer.startDocument(rich ? 1 : 0);
    if (rich) {
      writer.startField(field, 1, true, true, true);
      writer.startTerm(new BytesRef(new byte[] {0, 10, 92, (byte)255}), 3);
      writer.addProx(3, new Input(POS), new Input(OFF));
      writer.finishTerm(); writer.finishField();
    }
    writer.finishDocument();
  }
  static class Reader extends TermVectorsReader {
    final String name, fail; final TermVectorsReader delegate; final ArrayList<String> log;
    Reader(String name, String fail, TermVectorsReader delegate, ArrayList<String> log) {
      this.name = name; this.fail = fail; this.delegate = delegate; this.log = log;
    }
    @Override public Fields get(int doc) throws IOException {
      String operation = "get:" + name + ":" + doc; log.add(operation);
      if (fail.equals(operation)) throw new IOException("delegate " + fail);
      return delegate.get(doc);
    }
    @Override public void checkIntegrity() throws IOException {
      String operation = "integrity:" + name; log.add(operation);
      if (fail.equals(operation)) throw new IOException("delegate " + fail);
      delegate.checkIntegrity();
    }
    @Override public TermVectorsReader clone() { return new Reader(name, fail, delegate.clone(), log); }
    @Override public void close() {}

  }
  static void assign(MergeState state, String name, Object value) throws Exception {
    var field = MergeState.class.getField(name); field.setAccessible(true); field.set(state, value);
  }
  static MergeState state(Directory dir, TermVectorsReader[] readers, boolean sorted) throws Exception {
    var ctor = MergeState.class.getDeclaredConstructor(List.class, SegmentInfo.class, InfoStream.class, Executor.class, MergePolicy.OneMerge.class);
    ctor.setAccessible(true);
    MergeState state = ctor.newInstance(List.of(), info(dir, -1), InfoStream.NO_OUTPUT, (Executor) Runnable::run, null);
    int[][] maps = sorted ? new int[][] {{2, -1}, {0}, {1, 3}} : new int[][] {{0, -1}, {1}, {2, 3}};
    MergeState.DocMap[] docMaps = new MergeState.DocMap[maps.length];
    for (int i = 0; i < maps.length; i++) { final int[] map = maps[i]; docMaps[i] = new MergeState.DocMap() { @Override public int get(int doc) { return map[doc]; } }; }
    assign(state, "termVectorsReaders", readers);
    assign(state, "maxDocs", new int[] {2, 1, 2});
    assign(state, "docMaps", docMaps);
    state.mergeFieldInfos = new FieldInfos(new FieldInfo[] {FIELD}); state.needsIndexSort = sorted;
    return state;
  }
  static void merge(String variant, Directory dir, TermVectorsReader base) throws Exception {
    long seed = variant.equals("inject") ? seed(100) : variant.equals("nested") ? seed(100, 10000) : 42;
    String fail = switch (variant) {
      case "integrity" -> "integrity:b";
      case "get" -> "get:a:0";
      case "callback" -> "startTerm";
      default -> "-";
    };
    var log = new ArrayList<String>(); Random random = new Random(seed);
    var writer = wrap(new Probe(log, fail), random);
    var state = state(dir, new TermVectorsReader[] {new Reader("a", fail, base, log), null, new Reader("b", fail, base, log)}, variant.equals("sorted"));
    int[] count = {-1};
    String status = outcome(() -> count[0] = writer.merge(state));
    record("merge:" + variant, seed, fail, status, log, random, null, null, count[0]);
  }
  public static void main(String[] args) throws Exception {
    for (String operation : new String[] {"reader", "open", "startDoc", "finishDoc", "startField", "finishField", "startTerm", "finishTerm", "position", "finish", "close", "resources"}) {
      direct(operation, "pass");
      if (!operation.equals("reader") && !operation.equals("resources")) direct(operation, "inject");
      if (!operation.equals("resources")) direct(operation, "delegate");
    }
    for (String variant : new String[] {"both", "positions", "offsets", "none", "zero", "negative", "shortpos", "shortoff", "inject", "nested1", "nested2", "delegate2"}) prox(variant);
    Path out = Path.of(args[0]); Files.createDirectories(out);
    try (Directory dir = new ByteBuffersDirectory()) {
      TermVectorsFormat simple = new SimpleTextTermVectorsFormat();
      try (TermVectorsWriter writer = format(simple, new Random(42)).vectorsWriter(dir, info(dir, 2), IOContext.DEFAULT)) {
        document(writer, field(0), true); document(writer, field(0), false); writer.finish(2);
      }
      try (TermVectorsReader reader = simple.vectorsReader(dir, info(dir, 2), new FieldInfos(new FieldInfo[] {field(0)}), IOContext.DEFAULT)) {
        reader.checkIntegrity();
        if (reader.get(1) != null || reader.get(0).terms("f").size() != 1) throw new AssertionError("real readback");
        var terms = reader.get(0).terms("f").iterator(); terms.next();
        if (terms.totalTermFreq() != 3) throw new AssertionError("freq");
        var positions = terms.postings(null, org.apache.lucene.index.PostingsEnum.ALL); positions.nextDoc();
        int[] expectedPositions = {1, 3, 3}, expectedStarts = {0, 3, 6}, expectedEnds = {2, 6, 6};
        for (int i = 0; i < 3; i++) {
          if (positions.nextPosition() != expectedPositions[i] || positions.startOffset() != expectedStarts[i] || positions.endOffset() != expectedEnds[i]) throw new AssertionError("occurrence");
          BytesRef payload = positions.getPayload();
          if (i == 0 ? !new BytesRef(new byte[] {0, 10}).equals(payload) : payload != null) throw new AssertionError("payload");
        }
        for (String variant : new String[] {"sequential", "sorted", "integrity", "get", "callback", "inject", "nested"}) merge(variant, dir, reader);
      }
      try (var input = dir.openInput("_ctv.tvc", IOContext.DEFAULT)) {
        byte[] bytes = new byte[Math.toIntExact(input.length())]; input.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve("cranky_term_vectors.tvc"), bytes);
      }
    }
    Files.writeString(out.resolve("cranky_term_vectors.tsv"), OUT, StandardCharsets.UTF_8);
  }
}
