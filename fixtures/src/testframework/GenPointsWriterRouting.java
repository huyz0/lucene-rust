import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene90.Lucene90PointsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat;
import org.apache.lucene.util.NumericUtils;

/** Installed asserting points writers, preserving native readers and merge algorithms. */
public class GenPointsWriterRouting {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<String> EVENTS = new ArrayList<>();
  private static final int[] RANKS = {50, 10, 40, 20, 80, 60, 0, 30, 70};
  private static final String[] FIELDS = {"one", "sparse", "multi", "two", "three", "vanish", "late"};
  private static final int LARGE_SOURCE_DOCS = 520, LARGE_POINT_DOCS = 2 * LARGE_SOURCE_DOCS;
  private static int nextWriter;
  private static String phase;

  private static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  private static void event(int writer, String operation) { EVENTS.add(phase + ":" + writer + ":" + operation); }
  private static String hex(byte[] bytes) { return bytes == null ? "-" : HexFormat.of().formatHex(bytes); }

  /** Only fieldsWriter enters assertions; fieldsReader returns the exact native reader. */
  public static final class RoutingCodec extends FilterCodec {
    private final PointsFormat points;
    public RoutingCodec() {
      super("PointsRoutingOracle", new Lucene104Codec());
      var nativeFormat = delegate.pointsFormat();
      var asserting = new AssertingPointsFormat(new TraceFormat(nativeFormat));
      points = new PointsFormat() {
        @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {
          return asserting.fieldsWriter(state);
        }
        @Override public PointsReader fieldsReader(SegmentReadState state) throws IOException {
          var reader = nativeFormat.fieldsReader(state);
          if (!(reader instanceof Lucene90PointsReader)) throw new AssertionError("native points reader");
          return reader;
        }
      };
    }
    @Override public PointsFormat pointsFormat() { return points; }
  }

  private static final class TraceFormat extends PointsFormat {
    final PointsFormat delegate;
    TraceFormat(PointsFormat delegate) { this.delegate = delegate; }
    @Override public PointsReader fieldsReader(SegmentReadState state) {
      throw new AssertionError("writer-only assertion format must never open a reader");
    }
    @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      int id = nextWriter++;
      var writer = delegate.fieldsWriter(state);
      event(id, "open:" + state.segmentInfo.name + ":" + state.segmentInfo.maxDoc());
      return new PointsWriter() {
        @Override public void writeField(FieldInfo field, PointsReader reader) throws IOException {
          event(id, "write:" + field.name + ":" + field.number + ":" + field.getPointDimensionCount()
              + ":" + field.getPointIndexDimensionCount() + ":" + field.getPointNumBytes());
          // Preserve the actual flush reader and its PointTree, including mutable native cursors.
          writer.writeField(field, reader);
        }
        @Override public void merge(MergeState state) throws IOException {
          event(id, "merge:" + state.pointsReaders.length + ":" + state.needsIndexSort);
          for (int source = 0; source < state.pointsReaders.length; source++) {
            var reader = state.pointsReaders[source];
            if (reader != null && !(reader instanceof Lucene90PointsReader)) {
              throw new AssertionError("merge reader must retain native instanceof dispatch");
            }
            event(id, "source:" + source + ":" + state.maxDocs[source] + ":"
                + (reader == null ? "none" : "native"));
            int liveCount = 0;
            for (int doc = 0; doc < state.maxDocs[source]; doc++) {
              if (state.liveDocs[source] == null || state.liveDocs[source].get(doc)) liveCount++;
            }
            event(id, "source-live:" + source + ":" + liveCount);
            for (var field : state.fieldInfos[source]) if (field.getPointDimensionCount() != 0) {
              var values = reader == null ? null : reader.getValues(field.name);
              event(id, "source-field:" + source + ":" + field.name + ":" + field.number + ":"
                  + field.getPointDimensionCount() + ":" + field.getPointIndexDimensionCount() + ":"
                  + field.getPointNumBytes() + ":" + (values == null ? 0 : values.size()) + ":"
                  + (values == null ? 0 : values.getDocCount()) + ":"
                  + hex(values == null ? null : values.getMinPackedValue()) + ":"
                  + hex(values == null ? null : values.getMaxPackedValue()));
            }
          }
          for (var field : state.mergeFieldInfos) if (field.getPointDimensionCount() != 0) {
            long maxPointCount = 0;
            for (int source = 0; source < state.pointsReaders.length; source++) {
              var info = state.fieldInfos[source].fieldInfo(field.name);
              var reader = state.pointsReaders[source];
              if (reader == null || info == null || info.getPointDimensionCount() == 0) continue;
              var values = reader.getValues(field.name);
              if (values != null) maxPointCount += values.size();
            }
            // Native sizing includes deleted points. Read metadata without replacing source objects.
            event(id, "max-point-count:" + field.name + ":" + maxPointCount);
          }
          // In particular, do not rebuild MergeState or replace its readers, maps or liveDocs.
          writer.merge(state);
          event(id, "merged");
        }
        @Override public void finish() throws IOException { event(id, "finish"); writer.finish(); }
        @Override public void close() throws IOException { event(id, "close"); writer.close(); }
      };
    }
  }

  private static int dimensions(String field) { return field.equals("two") ? 2 : field.equals("three") ? 3 : 1; }
  private static int indexDimensions(String field) { return field.equals("three") ? 2 : dimensions(field); }
  private static byte[] packed(int... values) {
    byte[] result = new byte[values.length * Integer.BYTES];
    for (int dim = 0; dim < values.length; dim++) NumericUtils.intToSortableBytes(values[dim], result, dim * Integer.BYTES);
    return result;
  }
  private static Map<String, List<byte[]>> values(int ordinal, boolean empty) {
    var fields = new LinkedHashMap<String, List<byte[]>>();
    if (empty) return fields;
    fields.put("one", List.of(packed(ordinal == 0 ? Integer.MIN_VALUE : ordinal == 8 ? Integer.MAX_VALUE : ordinal - 4)));
    if (ordinal % 3 != 0) fields.put("sparse", List.of(packed(ordinal == 1 ? Integer.MAX_VALUE : -ordinal)));
    fields.put("multi", List.of(packed(7), packed(7), packed(ordinal - 4)));
    fields.put("two", List.of(packed(ordinal - 4, 4 - ordinal)));
    fields.put("three", List.of(packed(ordinal % 2, ordinal - 4, Integer.MIN_VALUE),
        packed(ordinal % 2, ordinal - 4, Integer.MAX_VALUE)));
    if (ordinal == 1 || ordinal == 4 || ordinal == 6) fields.put("vanish", List.of(packed(-99)));
    if (ordinal >= 6) fields.put("late", List.of(packed(ordinal * 2)));
    return fields;
  }
  private static Document document(int ordinal, boolean empty) {
    var doc = new Document();
    doc.add(new StringField("id", "" + (char) ('a' + ordinal), Field.Store.YES));
    doc.add(new NumericDocValuesField("rank", RANKS[ordinal]));
    for (var entry : values(ordinal, empty).entrySet()) {
      var type = new FieldType();
      type.setDimensions(dimensions(entry.getKey()), indexDimensions(entry.getKey()), Integer.BYTES);
      type.freeze();
      for (byte[] value : entry.getValue()) doc.add(new Field(entry.getKey(), value, type));
    }
    return doc;
  }
  private static Map<String, List<byte[]>> largeValues(int ordinal) {
    var fields = new LinkedHashMap<String, List<byte[]>>();
    if (ordinal >= LARGE_POINT_DOCS) return fields;
    int value = ordinal - LARGE_SOURCE_DOCS;
    fields.put("multi", List.of(packed(value), packed(7), packed(7)));
    fields.put("two", List.of(packed(value, -value), packed(7, -7), packed(7, -7)));
    if (ordinal == 1 || ordinal == LARGE_SOURCE_DOCS + 1) fields.put("vanish", List.of(packed(-99)));
    return fields;
  }
  private static Document largeDocument(int ordinal) {
    var doc = new Document();
    doc.add(new StringField("id", "p" + ordinal, Field.Store.YES));
    doc.add(new NumericDocValuesField("rank", ordinal));
    for (var entry : largeValues(ordinal).entrySet()) {
      var type = new FieldType();
      type.setDimensions(dimensions(entry.getKey()), indexDimensions(entry.getKey()), Integer.BYTES);
      type.freeze();
      for (byte[] value : entry.getValue()) doc.add(new Field(entry.getKey(), value, type));
    }
    return doc;
  }
  private static IndexWriterConfig config(boolean sorted) {
    var config = new IndexWriterConfig();
    config.setCodec(new RoutingCodec()); config.setUseCompoundFile(false);
    config.setMaxFullFlushMergeWaitMillis(0);
    var policy = new LogDocMergePolicy(); policy.setMergeFactor(1000);
    config.setMergePolicy(policy); config.setMergeScheduler(new SerialMergeScheduler());
    if (sorted) config.setIndexSort(new Sort(new SortField("rank", SortField.Type.LONG)));
    return config;
  }
  private static byte[] bound(List<byte[]> points, int dims, boolean minimum) {
    byte[] result = Arrays.copyOf(points.get(0), dims * Integer.BYTES);
    for (byte[] point : points) for (int dim = 0; dim < dims; dim++) {
      int offset = dim * Integer.BYTES;
      int order = Arrays.compareUnsigned(point, offset, offset + Integer.BYTES, result, offset, offset + Integer.BYTES);
      if (minimum ? order < 0 : order > 0) System.arraycopy(point, offset, result, offset, Integer.BYTES);
    }
    return result;
  }
  private static void snapshot(String name, boolean empty, Directory directory) throws IOException {
    snapshot(name, empty, false, directory);
  }
  private static int leafCount(PointValues.PointTree tree) throws IOException {
    if (!tree.moveToChild()) return 1;
    int leaves = 0;
    do { leaves += leafCount(tree); } while (tree.moveToSibling());
    if (!tree.moveToParent()) throw new AssertionError("tree parent navigation");
    return leaves;
  }
  private static void snapshot(String name, boolean empty, boolean large, Directory directory) throws IOException {
    try (var reader = DirectoryReader.open(directory)) {
      row("CASE", name, phase, reader.maxDoc(), reader.numDocs(), reader.leaves().size());
      for (var leaf : reader.leaves()) {
        var segment = leaf.reader(); segment.checkIntegrity();
        var stored = segment.storedFields(); var live = segment.getLiveDocs();
        var ordinals = new ArrayList<Integer>();
        for (int doc = 0; doc < segment.maxDoc(); doc++) {
          String id = stored.document(doc).get("id");
          if (id == null || (large ? !id.startsWith("p") : id.length() != 1)) throw new AssertionError("stored identity");
          ordinals.add(large ? Integer.parseInt(id.substring(1)) : id.charAt(0) - 'a');
          row("DOC", name, phase, leaf.ord, doc, live == null || live.get(doc) ? 1 : 0, id);
        }
        for (String field : FIELDS) {
          var info = segment.getFieldInfos().fieldInfo(field);
          var actual = segment.getPointValues(field);
          var expected = new ArrayList<String>(); var packedValues = new ArrayList<byte[]>();
          int documents = 0;
          for (int doc = 0; doc < ordinals.size(); doc++) {
            var input = (large ? largeValues(ordinals.get(doc)) : values(ordinals.get(doc), empty)).get(field);
            if (input == null) continue;
            documents++;
            for (byte[] value : input) { expected.add(doc + ":" + hex(value)); packedValues.add(value); }
          }
          if (actual == null) {
            if (!expected.isEmpty()) throw new AssertionError("missing points " + field);
            if (!empty && phase.equals("merged") && field.equals("vanish")
                && (info == null || info.getPointDimensionCount() != 1)) {
              throw new AssertionError("all-deleted field must retain point FieldInfo");
            }
            row("FIELD", name, phase, leaf.ord, field, info == null ? -1 : info.number,
                info == null ? 0 : info.getPointDimensionCount(), info == null ? 0 : info.getPointIndexDimensionCount(),
                info == null ? 0 : info.getPointNumBytes(), 0, 0, "-", "-");
            continue;
          }
          if (expected.isEmpty() || actual.size() != expected.size() || actual.getDocCount() != documents
              || actual.getNumDimensions() != dimensions(field) || actual.getNumIndexDimensions() != indexDimensions(field)
              || actual.getBytesPerDimension() != Integer.BYTES
              || !Arrays.equals(actual.getMinPackedValue(), bound(packedValues, indexDimensions(field), true))
              || !Arrays.equals(actual.getMaxPackedValue(), bound(packedValues, indexDimensions(field), false))) {
            throw new AssertionError("point metadata " + name + ":" + phase + ":" + field);
          }
          row("FIELD", name, phase, leaf.ord, field, info.number, actual.getNumDimensions(),
              actual.getNumIndexDimensions(), actual.getBytesPerDimension(), actual.size(), actual.getDocCount(),
              hex(actual.getMinPackedValue()), hex(actual.getMaxPackedValue()));
          var points = new ArrayList<String>();
          actual.getPointTree().visitDocValues(new PointValues.IntersectVisitor() {
            @Override public void visit(int doc) { throw new AssertionError("packed value required"); }
            @Override public void visit(int doc, byte[] value) { points.add(doc + ":" + hex(value)); }
            @Override public PointValues.Relation compare(byte[] min, byte[] max) { return PointValues.Relation.CELL_CROSSES_QUERY; }
          });
          Collections.sort(points); Collections.sort(expected);
          if (!points.equals(expected)) throw new AssertionError("point multiset " + name + ":" + phase + ":" + field);
          for (String point : points) row("POINT", name, phase, leaf.ord, field, point);
          if (large) {
            int leaves = leafCount(actual.getPointTree());
            if (actual.size() > 1024 && leaves <= 1) throw new AssertionError("multi-leaf BKD required");
            row("TREE", name, phase, leaf.ord, field, actual.size(), leaves);
          }
        }
      }
      if (large) rangeQueries(name, reader);
    }
    row("CHECKPOINT", name, phase, nextWriter, EVENTS.size());
  }
  private static void rangeQueries(String name, DirectoryReader reader) throws IOException {
    int[][] ranges = {{Integer.MIN_VALUE, Integer.MAX_VALUE}, {-20, -10}, {7, 7}, {600, 700}};
    var searcher = new IndexSearcher(reader);
    for (int query = 0; query < ranges.length; query++) {
      int min = ranges[query][0], max = ranges[query][1];
      var expected = new ArrayList<Integer>();
      for (var leaf : reader.leaves()) {
        var live = leaf.reader().getLiveDocs(); var stored = leaf.reader().storedFields();
        for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
          if (live != null && !live.get(doc)) continue;
          int ordinal = Integer.parseInt(stored.document(doc).get("id").substring(1));
          // Independent integer predicate, with per-document deduplication and live-doc filtering.
          if (ordinal < LARGE_POINT_DOCS && ((ordinal - LARGE_SOURCE_DOCS >= min
              && ordinal - LARGE_SOURCE_DOCS <= max) || (7 >= min && 7 <= max))) expected.add(leaf.docBase + doc);
        }
      }
      var result = searcher.search(IntPoint.newRangeQuery("multi", min, max), reader.maxDoc());
      var actual = new ArrayList<Integer>();
      for (var hit : result.scoreDocs) actual.add(hit.doc);
      Collections.sort(actual); Collections.sort(expected);
      if (!actual.equals(expected)) throw new AssertionError("independent range oracle " + name + ":" + phase + ":" + query);
      row("RANGE", name, phase, query, "multi", min, max, actual.size());
      for (int doc : actual) row("RANGE_HIT", name, phase, query, doc, reader.storedFields().document(doc).get("id"));
    }
  }
  private static void commit(String name, boolean empty, String nextPhase, IndexWriter writer, Directory directory) throws IOException {
    phase = nextPhase; writer.commit(); snapshot(name, empty, directory);
  }
  private static void scenario(boolean sorted, boolean empty) throws Exception {
    String name = (empty ? "empty" : "points") + (sorted ? "_sorted" : "_unsorted");
    EVENTS.clear(); nextWriter = 0; phase = "initial";
    for (int ordinal = 0; ordinal < RANKS.length; ordinal++) {
      row("INPUT_DOC", name, ordinal, (char) ('a' + ordinal), RANKS[ordinal]);
      for (var entry : values(ordinal, empty).entrySet()) for (byte[] value : entry.getValue()) {
        row("INPUT_POINT", name, ordinal, entry.getKey(), dimensions(entry.getKey()), indexDimensions(entry.getKey()), Integer.BYTES, hex(value));
      }
    }
    try (var directory = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(directory, config(sorted))) {
        for (int ordinal = 0; ordinal < 6; ordinal++) {
          writer.addDocument(document(ordinal, empty));
          if (ordinal == 2 || ordinal == 5) writer.commit();
        }
        snapshot(name, empty, directory);
      }
      try (var writer = new IndexWriter(directory, config(sorted))) {
        phase = "append";
        for (int ordinal = 6; ordinal < RANKS.length; ordinal++) writer.addDocument(document(ordinal, empty));
        commit(name, empty, "append", writer, directory);
        phase = "deleted";
        writer.deleteDocuments(new Term("id", "b"), new Term("id", "e"), new Term("id", "g"));
        commit(name, empty, "deleted", writer, directory);
        phase = "merged"; writer.forceMerge(1);
        commit(name, empty, "merged", writer, directory);
      }
    }
    if (empty && !EVENTS.isEmpty()) throw new AssertionError("no points writer for a point-free index");
    for (int ordinal = 0; ordinal < EVENTS.size(); ordinal++) row("TRACE", name, ordinal, EVENTS.get(ordinal));
  }
  private static void largeScenario() throws Exception {
    String name = "multi_leaf_mixed_sources";
    EVENTS.clear(); nextWriter = 0; phase = "initial";
    for (int ordinal = 0; ordinal < LARGE_POINT_DOCS + 3; ordinal++) {
      row("INPUT_DOC", name, ordinal, "p" + ordinal, ordinal);
      for (var entry : largeValues(ordinal).entrySet()) for (byte[] value : entry.getValue()) {
        row("INPUT_POINT", name, ordinal, entry.getKey(), dimensions(entry.getKey()), indexDimensions(entry.getKey()), Integer.BYTES, hex(value));
      }
    }
    try (var directory = new ByteBuffersDirectory(); var writer = new IndexWriter(directory, config(false))) {
      for (int ordinal = 0; ordinal < LARGE_POINT_DOCS + 3; ordinal++) {
        writer.addDocument(largeDocument(ordinal));
        if (ordinal == LARGE_SOURCE_DOCS - 1 || ordinal == LARGE_POINT_DOCS - 1) {
          writer.commit(); row("FLUSH", name, phase, ordinal);
        }
      }
      writer.commit(); row("FLUSH", name, phase, LARGE_POINT_DOCS + 2);
      snapshot(name, false, true, directory);
      phase = "deleted";
      for (int ordinal : new int[] {1, 2, LARGE_SOURCE_DOCS + 1}) {
        row("DELETE", name, phase, "p" + ordinal);
        writer.deleteDocuments(new Term("id", "p" + ordinal));
      }
      writer.commit(); snapshot(name, false, true, directory);
      phase = "merged"; writer.forceMerge(1); writer.commit();
      snapshot(name, false, true, directory);
    }
    if (EVENTS.stream().noneMatch(value -> value.contains(":source:") && value.endsWith(":3:none"))
        || EVENTS.stream().noneMatch(value -> value.endsWith(":max-point-count:multi:3120"))
        || EVENTS.stream().noneMatch(value -> value.endsWith(":max-point-count:two:3120"))
        || EVENTS.stream().noneMatch(value -> value.endsWith(":max-point-count:vanish:2"))) {
      throw new AssertionError("mixed-source merge must expose unfiltered point budgets and point-free source");
    }
    for (int ordinal = 0; ordinal < EVENTS.size(); ordinal++) row("TRACE", name, ordinal, EVENTS.get(ordinal));
  }
  public static void main(String[] args) throws Exception {
    if (!GenPointsWriterRouting.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    Path output = Path.of(args[0]); Files.createDirectories(output);
    Path providers = Files.createTempDirectory("points-routing-provider");
    Path service = providers.resolve("META-INF/services/org.apache.lucene.codecs.Codec");
    Files.createDirectories(service.getParent()); Files.writeString(service, RoutingCodec.class.getName() + "\n");
    try (var loader = new URLClassLoader(new java.net.URL[] {providers.toUri().toURL()}, GenPointsWriterRouting.class.getClassLoader())) {
      Codec.reloadCodecs(loader);
      if (!(Codec.forName("PointsRoutingOracle") instanceof RoutingCodec)) throw new AssertionError("SPI registration");
      row("# Scope: actual AssertingPointsFormat writer only; native reader, PointTree and original MergeState retain native algorithm dispatch");
      row("# INPUT_DOC case ordinal storedId rank; INPUT_POINT case ordinal field dimensions indexDimensions bytesPerDimension packedHex; duplicate rows preserve multiplicity");
      row("# CASE case phase maxDoc numDocs leaves; DOC case phase leaf localDoc live storedId");
      row("# FIELD case phase leaf name number dimensions indexDimensions bytesPerDimension size docCount minHex maxHex; missing field number=-1, missing points bounds=-");
      row("# POINT case phase leaf field localDoc:packedHex; sorted multiset includes deleted documents before merge, preserves duplicate points");
      row("# TREE case phase leaf field pointCount leafCount; default native configuration, fields with >1024 points must have multiple leaves");
      row("# RANGE case phase queryOrdinal field inclusiveMin inclusiveMax exactHitCount; RANGE_HIT case phase queryOrdinal globalDoc storedId; independent integer predicate checks every live document");
      row("# FLUSH case phase afterInputOrdinal; DELETE case phase storedId; large case commits after ordinals 519,1039,1042: two 1560-point sources plus one point-free source");
      row("# CHECKPOINT case phase writersOpened eventCount; TRACE case ordinal phase:writer:event; native merge finishes internally, its finish is not a separate wrapper call");
      row("# TRACE source-field:source:name:number:dimensions:indexDimensions:bytes:size:docCount:min:max and source-live:source:liveCount expose unfiltered source metadata; max-point-count:name:sum includes deleted points");
      for (boolean empty : new boolean[] {false, true}) for (boolean sorted : new boolean[] {false, true}) scenario(sorted, empty);
      largeScenario();
      Files.writeString(output.resolve("points_writer_routing.tsv"), OUT);
    } finally {
      Files.deleteIfExists(service); Files.deleteIfExists(service.getParent());
      Files.deleteIfExists(service.getParent().getParent()); Files.deleteIfExists(providers);
    }
  }
}
