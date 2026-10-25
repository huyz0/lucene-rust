import java.lang.reflect.InvocationTargetException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.tests.codecs.vector.ConfigurableMCodec;
import org.apache.lucene.util.*;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;

/** Actual ConfigurableMCodec components, including its enabled tiny-segment threshold. */
public class GenConfigurableM {
  static final byte[] ID = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static final String SUFFIX = "s";
  static final String[] ACCESSORS = {
    "docValuesFormat", "fieldInfosFormat", "liveDocsFormat", "normsFormat",
    "postingsFormat", "segmentInfoFormat", "storedFieldsFormat", "termVectorsFormat",
    "compoundFormat", "pointsFormat"
  };

  static FieldInfo field(String name, int number, VectorEncoding encoding) {
    return new FieldInfo(name, number, false, false, false, IndexOptions.NONE,
        DocValuesType.NONE, DocValuesSkipIndexType.NONE, -1, Map.of(), 0, 0, 0, 4,
        encoding, VectorSimilarityFunction.EUCLIDEAN, false, false);
  }
  static final FieldInfos FIELDS = new FieldInfos(new FieldInfo[] {
    field("float_dense", 0, VectorEncoding.FLOAT32),
    field("byte_dense", 1, VectorEncoding.BYTE),
    field("float_sparse", 2, VectorEncoding.FLOAT32),
    field("byte_sparse", 3, VectorEncoding.BYTE),
    field("float_empty", 4, VectorEncoding.FLOAT32),
    field("byte_empty", 5, VectorEncoding.BYTE)
  });

  static Object member(Object value, Class<?> owner, String name) throws Exception {
    var field = owner.getDeclaredField(name);
    field.setAccessible(true);
    return field.get(value);
  }

  /** Probe only: inject a delegate to observe inherited FilterCodec calls and first errors. */
  static class ProbeCodec extends FilterCodec {
    final List<String> events = new ArrayList<>();
    String fail = "";
    ProbeCodec(Codec delegate) { super("ConfigurableMProbe", delegate); }
    <T> T call(String name, Supplier<T> action) {
      events.add(name);
      if (name.equals(fail)) throw new IllegalStateException("delegate failure:" + name);
      return action.get();
    }
    public DocValuesFormat docValuesFormat() { return call("docValuesFormat", super::docValuesFormat); }
    public FieldInfosFormat fieldInfosFormat() { return call("fieldInfosFormat", super::fieldInfosFormat); }
    public LiveDocsFormat liveDocsFormat() { return call("liveDocsFormat", super::liveDocsFormat); }
    public NormsFormat normsFormat() { return call("normsFormat", super::normsFormat); }
    public PostingsFormat postingsFormat() { return call("postingsFormat", super::postingsFormat); }
    public SegmentInfoFormat segmentInfoFormat() { return call("segmentInfoFormat", super::segmentInfoFormat); }
    public StoredFieldsFormat storedFieldsFormat() { return call("storedFieldsFormat", super::storedFieldsFormat); }
    public TermVectorsFormat termVectorsFormat() { return call("termVectorsFormat", super::termVectorsFormat); }
    public CompoundFormat compoundFormat() { return call("compoundFormat", super::compoundFormat); }
    public PointsFormat pointsFormat() { return call("pointsFormat", super::pointsFormat); }
  }

  static String configuration() throws Exception {
    var text = new StringBuilder();
    for (int m : new int[] {128, 4, 1, 512}) {
      var codec = m == 128 ? new ConfigurableMCodec() : new ConfigurableMCodec(m);
      var delegate = (Codec) member(codec, FilterCodec.class, "delegate");
      var format = codec.knnVectorsFormat();
      text.append("codec\t").append(m).append('\t').append(codec.getName())
          .append('\t').append(delegate.getName()).append('\t').append(format.getName())
          .append('\t').append(format == codec.knnVectorsFormat()).append('\n');
      for (String property : new String[] {
          "maxConn", "beamWidth", "numMergeWorkers", "tinySegmentsThreshold", "writeVersion", "mergeExec"}) {
        text.append("setting\t").append(m).append('\t').append(property).append('\t')
            .append(member(format, Lucene99HnswVectorsFormat.class, property)).append('\n');
      }
      if (m != 128 && m != 4) continue;
      for (String name : ACCESSORS) {
        var accessor = Codec.class.getMethod(name);
        text.append("delegate-identity\t").append(m).append('\t').append(name).append('\t')
            .append(accessor.invoke(codec) == accessor.invoke(delegate)).append('\n');
      }
      var probe = new ProbeCodec(delegate);
      var delegateField = FilterCodec.class.getDeclaredField("delegate");
      delegateField.setAccessible(true);
      delegateField.set(codec, probe);
      for (String name : ACCESSORS) {
        var accessor = Codec.class.getMethod(name);
        probe.events.clear();
        Object first = accessor.invoke(codec), second = accessor.invoke(codec);
        text.append("forward\t").append(m).append('\t').append(name).append('\t')
            .append(first == second).append('\t').append(String.join(",", probe.events)).append('\n');
        probe.events.clear();
        probe.fail = name;
        try {
          accessor.invoke(codec);
          throw new AssertionError("missing delegate failure: " + name);
        } catch (InvocationTargetException error) {
          if (!(error.getCause() instanceof IllegalStateException)) throw error;
          text.append("forward-error\t").append(m).append('\t').append(name).append('\t')
              .append(error.getCause().getMessage()).append('\t')
              .append(String.join(",", probe.events)).append('\n');
        }
        probe.fail = "";
      }
      probe.events.clear();
      if (codec.knnVectorsFormat() != format) throw new AssertionError("unstable override");
      text.append("override-events\t").append(m).append('\t')
          .append(String.join(",", probe.events)).append('\n');
    }
    for (int m : new int[] {-1, 0, 513, Integer.MAX_VALUE}) {
      try {
        new ConfigurableMCodec(m);
        throw new AssertionError("accepted invalid M: " + m);
      } catch (IllegalArgumentException error) {
        text.append("invalid\t").append(m).append('\t').append(error.getMessage()).append('\n');
      }
    }
    return text.toString();
  }

  static float[] floats(int doc) {
    return new float[] {doc * 0.25f, (doc % 17) * 0.5f, (doc % 31) * 0.125f, 1};
  }
  static byte[] bytes(int doc) {
    return new byte[] {(byte) (doc % 127 - 63), (byte) (doc / 127 - 5),
        (byte) (doc * 17 % 127 - 63), (byte) (doc * 29 % 127 - 63)};
  }
  static SegmentInfo segment(Directory directory, int maxDoc) {
    return new SegmentInfo(directory, Version.fromBits(10,5,0), null, "_cm", maxDoc,
        false, false, null, Map.of(), ID, Map.of(), null);
  }

  static String wrongId(KnnVectorsFormat format, Directory directory, int maxDoc) throws Exception {
    byte[] wrong = ID.clone();
    wrong[0] ^= 1;
    var info = new SegmentInfo(directory, Version.fromBits(10,5,0), null, "_cm", maxDoc,
        false, false, null, Map.of(), wrong, Map.of(), null);
    try (var reader = format.fieldsReader(new SegmentReadState(directory, info, FIELDS, IOContext.DEFAULT, SUFFIX))) {
      reader.checkIntegrity();
      throw new AssertionError("accepted wrong segment id");
    } catch (IOException error) {
      return "wrong-id\t" + error.getClass().getSimpleName() + "\n";
    }
  }

  static String snapshot(KnnVectorsReader reader, int count, int maxDoc) throws Exception {
    var text = new StringBuilder();
    for (var field : FIELDS) {
      boolean isFloat = field.getVectorEncoding() == VectorEncoding.FLOAT32;
      var values = isFloat ? reader.getFloatVectorValues(field.name) : reader.getByteVectorValues(field.name);
      int expectedCount = field.name.endsWith("empty") ? 0 : field.name.endsWith("sparse") ? count : maxDoc;
      text.append("field\t").append(field.name).append('\t').append(field.number).append('\t')
          .append(expectedCount).append('\t').append(values == null ? -1 : values.size()).append('\n');
      if (values == null) {
        if (expectedCount != 0) throw new AssertionError("missing values: " + field.name);
      } else {
        if (values.size() != expectedCount) throw new AssertionError("vector count: " + field.name);
        var iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
          text.append("value\t").append(field.name).append('\t').append(doc).append('\t').append(iterator.index());
          if (isFloat) {
            for (float value : ((FloatVectorValues) values).vectorValue(iterator.index()))
              text.append('\t').append(Float.floatToRawIntBits(value));
          } else {
            for (byte value : ((ByteVectorValues) values).vectorValue(iterator.index())) text.append('\t').append(value);
          }
          text.append('\n');
        }
      }
      var graph = ((HnswGraphProvider) reader).getGraph(field.name);
      text.append("graph\t").append(field.name).append('\t').append(graph.size()).append('\t')
          .append(graph.numLevels()).append('\t').append(graph.entryNode()).append('\n');
      boolean expectedGraph = expectedCount >= 648;
      if ((graph.numLevels() > 0) != expectedGraph) throw new AssertionError("graph threshold: " + field.name);
      for (int level = 0; level < graph.numLevels(); level++) {
        var nodes = graph.getNodesOnLevel(level);
        while (nodes.hasNext()) {
          int node = nodes.nextInt();
          text.append("neighbors\t").append(field.name).append('\t').append(level).append('\t').append(node);
          graph.seek(level, node);
          for (int neighbor = graph.nextNeighbor(); neighbor != DocIdSetIterator.NO_MORE_DOCS; neighbor = graph.nextNeighbor())
            text.append('\t').append(neighbor);
          text.append('\n');
        }
      }
      for (int query : new int[] {0, maxDoc / 2, maxDoc - 1}) for (boolean filtered : new boolean[] {false, true}) {
        Bits mask = filtered ? new Bits() {
          public boolean get(int doc) { return doc % 3 != 0; }
          public int length() { return maxDoc; }
        } : null;
        var collector = new TopKnnCollector(7, Integer.MAX_VALUE);
        if (isFloat) reader.search(field.name, floats(query), collector, AcceptDocs.fromLiveDocs(mask, maxDoc));
        else reader.search(field.name, bytes(query), collector, AcceptDocs.fromLiveDocs(mask, maxDoc));
        var hits = collector.topDocs();
        text.append("search\t").append(field.name).append('\t').append(query).append('\t').append(filtered)
            .append('\t').append(hits.totalHits.value()).append('\t').append(hits.totalHits.relation());
        for (var hit : hits.scoreDocs) text.append('\t').append(hit.doc).append(':').append(Float.floatToRawIntBits(hit.score));
        text.append('\n');
      }
    }
    return text.toString();
  }

  static void real(Path output, boolean defaults, int count) throws Exception {
    int maxDoc = count * 2;
    String prefix = "configurable_m_" + (defaults ? "default" : "4") + "_" + count;
    var codec = defaults ? new ConfigurableMCodec() : new ConfigurableMCodec(4);
    var format = codec.knnVectorsFormat();
    // The actual selected format retains threshold 100; no substitute format is constructed.
    long previousSeed = HnswGraphBuilder.randSeed;
    HnswGraphBuilder.randSeed = 42;
    try (var directory = new ByteBuffersDirectory()) {
      var info = segment(directory, maxDoc);
      try (var writer = format.fieldsWriter(new SegmentWriteState(
          InfoStream.NO_OUTPUT, directory, info, FIELDS, null, IOContext.DEFAULT, SUFFIX))) {
        for (var field : FIELDS) {
          var fieldWriter = writer.addField(field);
          if (field.name.endsWith("empty")) continue;
          int step = field.name.endsWith("sparse") ? 2 : 1;
          for (int doc = 0; doc < maxDoc; doc += step) {
            if (field.getVectorEncoding() == VectorEncoding.FLOAT32) {
              @SuppressWarnings("unchecked") var target = (KnnFieldVectorsWriter<float[]>) fieldWriter;
              target.addValue(doc, floats(doc));
            } else {
              @SuppressWarnings("unchecked") var target = (KnnFieldVectorsWriter<byte[]>) fieldWriter;
              target.addValue(doc, bytes(doc));
            }
          }
        }
        writer.flush(maxDoc, null);
        writer.finish();
      }
      try (var reader = format.fieldsReader(new SegmentReadState(directory, info, FIELDS, IOContext.DEFAULT, SUFFIX))) {
        reader.checkIntegrity();
        Files.writeString(output.resolve(prefix + ".tsv"), snapshot(reader, count, maxDoc));
      }
      Files.writeString(output.resolve(prefix + "_errors.tsv"), wrongId(format, directory, maxDoc));
      var manifest = new StringBuilder("segment\t_cm\nsuffix\ts\nmax-doc\t" + maxDoc + "\nsparse-count\t" + count + "\n");
      String[] files = directory.listAll();
      if (files.length != 4) throw new AssertionError("expected exactly four vector files");
      for (String extension : new String[] {"vec", "vemf", "vex", "vem"}) {
        String name = "_cm_s." + extension;
        try (var input = directory.openInput(name, IOContext.DEFAULT)) {
          byte[] data = new byte[Math.toIntExact(input.length())];
          input.readBytes(data, 0, data.length);
          Files.write(output.resolve(prefix + "." + extension), data);
          manifest.append("file\t").append(extension).append('\t').append(data.length).append('\n');
        }
      }
      Files.writeString(output.resolve(prefix + "_files.tsv"), manifest);
    } finally {
      HnswGraphBuilder.randSeed = previousSeed;
    }
  }

  public static void main(String[] args) throws Exception {
    if (!GenConfigurableM.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path output = Path.of(args[0]);
    Files.createDirectories(output);
    Files.writeString(output.resolve("configurable_m_config.tsv"), configuration());
    for (boolean defaults : new boolean[] {true, false})
      for (int count : new int[] {647, 648}) real(output, defaults, count);
  }
}
