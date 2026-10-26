import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Locale;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.util.NumericUtils;
import java.util.Arrays;
import java.util.Comparator;
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
import org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat;
import org.apache.lucene.util.RamUsageEstimator;

/** Whole flush/merge commit twin: encoding, writer close and publication are timed.
 * Java writes fields incrementally; Rust buffers encoding until finish/merge. */
public final class PointsWriterRoutingBench {
  private static final int DOCS = 8192, SOURCES = 4;
  private static final String[] FIELDS = {"one", "sparse", "three", "vanish"};
  private static volatile long sink;
  private static final class WriterOnlyAssertingCodec extends FilterCodec {
    final AssertingPointsFormat asserting;
    final PointsFormat format;
    WriterOnlyAssertingCodec() {
      super("Lucene104", new Lucene104Codec());
      var nativeFormat = delegate.pointsFormat();
      asserting = new AssertingPointsFormat(nativeFormat);
      format = new PointsFormat() {
        @Override public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {
          return asserting.fieldsWriter(state);
        }
        @Override public PointsReader fieldsReader(SegmentReadState state) throws IOException {
          return nativeFormat.fieldsReader(state);
        }
      };
    }
    @Override public PointsFormat pointsFormat() { return format; }
  }
  private static int dimensions(int field) { return field == 2 ? 3 : 1; }
  private static int indexDimensions(int field) { return field == 2 ? 2 : 1; }
  private static byte[] packed(int... values) {
    byte[] packed = new byte[values.length * Integer.BYTES];
    for (int dim = 0; dim < values.length; dim++) NumericUtils.intToSortableBytes(values[dim], packed, dim * Integer.BYTES);
    return packed;
  }
  private static java.util.List<byte[]> values(int doc, int field) {
    return switch (field) {
      case 0 -> java.util.List.of(packed(doc == 0 ? Integer.MIN_VALUE : doc == DOCS - 1 ? Integer.MAX_VALUE : doc - 4096));
      case 1 -> doc % 3 == 0 ? java.util.List.of() : java.util.List.of(packed(7), packed(7), packed(doc % 97 - 48));
      case 2 -> java.util.List.of(packed(doc % 17 - 8, (doc * 13) % 103 - 51, Integer.MIN_VALUE),
          packed(doc % 17 - 8, (doc * 13) % 103 - 51, Integer.MAX_VALUE));
      case 3 -> doc % 7 == 0 ? java.util.List.of(packed(42)) : java.util.List.of();
      default -> throw new AssertionError("unknown field");
    };
  }
  private static Document document(int doc) {
    var result = new Document();
    result.add(new StringField("id", Integer.toString(doc), Field.Store.YES));
    for (int field = 0; field < FIELDS.length; field++) {
      var type = new FieldType(); type.setDimensions(dimensions(field), indexDimensions(field), Integer.BYTES); type.freeze();
      for (byte[] value : values(doc, field)) result.add(new Field(FIELDS[field], value, type));
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
    int pointFiles = 0;
    for (String name : directory.listAll()) if (name.endsWith(".kdm") || name.endsWith(".kdi") || name.endsWith(".kdd")) pointFiles++;
    if (pointFiles != (merged ? SOURCES * 3 : 0)) throw new AssertionError("points outputs were created before the intended interval");
  }
  private static long mix(long digest, long value) { return (digest ^ value) * 0x100000001b3L; }
  private record Point(int doc, byte[] value) {}
  private static final Comparator<Point> POINT_ORDER = Comparator.comparingInt(Point::doc)
      .thenComparing(Point::value, Arrays::compareUnsigned);
  private record Check(long digest, int docs, long encodedBytes, int points, int packedBytes) {}
  private static byte[] bound(java.util.List<Point> points, int dims, boolean minimum) {
    byte[] result = Arrays.copyOf(points.get(0).value, dims * Integer.BYTES);
    for (Point point : points) for (int dim = 0; dim < dims; dim++) {
      int offset = dim * Integer.BYTES;
      int cmp = Arrays.compareUnsigned(point.value, offset, offset + Integer.BYTES, result, offset, offset + Integer.BYTES);
      if (minimum ? cmp < 0 : cmp > 0) System.arraycopy(point.value, offset, result, offset, Integer.BYTES);
    }
    return result;
  }
  private static Check verify(Directory directory, boolean merged) throws IOException {
    var expectedDocs = new ArrayList<Integer>();
    for (int doc = 0; doc < DOCS; doc++) if (!merged || doc % 7 != 0) expectedDocs.add(doc);
    long digest = 0xcbf29ce484222325L;
    int pointCount = 0, packedBytes = 0;
    try (var reader = DirectoryReader.open(directory)) {
      if (reader.maxDoc() != expectedDocs.size() || reader.numDocs() != expectedDocs.size()
          || reader.leaves().size() != 1) throw new AssertionError("output shape");
      var segment = reader.leaves().get(0).reader();
      var stored = segment.storedFields();
      for (int physical = 0; physical < expectedDocs.size(); physical++) {
        int id = Integer.parseInt(stored.document(physical).get("id"));
        if (id != expectedDocs.get(physical)) throw new AssertionError("stored identity/order");
        digest = mix(digest, id);
      }
      for (int field = 0; field < FIELDS.length; field++) {
        var info = segment.getFieldInfos().fieldInfo(FIELDS[field]);
        if (info == null || info.getPointDimensionCount() != dimensions(field)
            || info.getPointIndexDimensionCount() != indexDimensions(field) || info.getPointNumBytes() != Integer.BYTES) {
          throw new AssertionError("point field schema");
        }
        var expected = new ArrayList<Point>();
        int documents = 0;
        for (int physical = 0; physical < expectedDocs.size(); physical++) {
          var input = values(expectedDocs.get(physical), field);
          if (!input.isEmpty()) documents++;
          for (byte[] value : input) expected.add(new Point(physical, value));
        }
        expected.sort(POINT_ORDER);
        var actual = new ArrayList<Point>();
        var points = segment.getPointValues(FIELDS[field]);
        if (points == null) {
          if (!expected.isEmpty()) throw new AssertionError("missing field points");
        } else {
          if (expected.isEmpty() || points.size() != expected.size() || points.getDocCount() != documents
              || points.getNumDimensions() != dimensions(field) || points.getNumIndexDimensions() != indexDimensions(field)
              || points.getBytesPerDimension() != Integer.BYTES
              || !Arrays.equals(points.getMinPackedValue(), bound(expected, indexDimensions(field), true))
              || !Arrays.equals(points.getMaxPackedValue(), bound(expected, indexDimensions(field), false))) {
            throw new AssertionError("point metadata");
          }
          if (!points.getPointTree().moveToChild()) throw new AssertionError("multi-leaf corpus required");
          points.getPointTree().visitDocValues(new PointValues.IntersectVisitor() {
            @Override public void visit(int doc) { throw new AssertionError("packed value required"); }
            @Override public void visit(int doc, byte[] value) { actual.add(new Point(doc, value.clone())); }
            @Override public PointValues.Relation compare(byte[] min, byte[] max) { return PointValues.Relation.CELL_CROSSES_QUERY; }
          });
          actual.sort(POINT_ORDER);
        }
        if (actual.size() != expected.size()) throw new AssertionError("point count");
        for (int point = 0; point < actual.size(); point++) {
          if (actual.get(point).doc != expected.get(point).doc || !Arrays.equals(actual.get(point).value, expected.get(point).value)) {
            throw new AssertionError("point multiset");
          }
        }
        digest = mix(digest, field); digest = mix(digest, actual.size());
        for (Point point : actual) {
          pointCount++; packedBytes += point.value.length; digest = mix(digest, point.doc);
          for (byte value : point.value) digest = mix(digest, Byte.toUnsignedInt(value));
        }
      }
      if (pointCount != (merged ? 35_106 : 42_130) || packedBytes != (merged ? 252_760 : 299_592)) {
        throw new AssertionError("analytic point count or packed payload bytes");
      }
      segment.checkIntegrity();
    }
    var infos = SegmentInfos.readLatestCommit(directory);
    long encodedBytes = 0;
    for (var info : infos) for (String extension : new String[] {"kdm", "kdi", "kdd"}) {
      encodedBytes += directory.fileLength(info.info.name + "." + extension);
    }
    sink = digest;
    return new Check(digest, expectedDocs.size(), encodedBytes, pointCount, packedBytes);
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
    if (!PointsWriterRoutingBench.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    if (args[0].equals("--verify-dir")) {
      try (var directory = FSDirectory.open(Path.of(args[1]))) {
        var check = verify(directory, true);
        System.out.println("native Lucene104 read Rust points: " + Long.toUnsignedString(check.digest) + " bytes=" + check.encodedBytes);
      }
      return;
    }
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid durations");
    var codec = new WriterOnlyAssertingCodec();
    System.out.println("#memory_scope\tencoded points files, decoded packed value payload and shallow adapting codec/format/assertion only; BKD working buffers/writer/directory/allocator/RSS excluded");
    System.out.println("points_routing_asserting_codec_shallow_bytes\t"
        + (RamUsageEstimator.shallowSizeOf(codec) + RamUsageEstimator.shallowSizeOf(codec.format) + RamUsageEstimator.shallowSizeOf(codec.asserting)));
    for (boolean merged : new boolean[] {false, true}) for (boolean asserting : new boolean[] {false, true}) {
      String name = "points_routing_" + (asserting ? "asserting" : "default") + "_" + (merged ? "merge" : "flush") + "_commit";
      run(asserting, merged, warmup);
      var result = run(asserting, merged, measure);
      System.out.printf("#check\t%s\t%s\t%d\t%d\t%d\t%d%n", name, Long.toUnsignedString(result.check.digest), result.check.docs, FIELDS.length, result.check.points, result.check.packedBytes);
      System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result.nanos / result.iterations, result.iterations);
      System.out.printf("%s_encoded_bytes\t%d%n", name, result.check.encodedBytes);
      System.out.printf("%s_packed_value_bytes\t%d%n", name, result.check.packedBytes);
    }
  }
}
