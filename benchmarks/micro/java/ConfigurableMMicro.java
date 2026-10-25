import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.vector.ConfigurableMCodec;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;

/** Actual ConfigurableMCodec's selected format, with actual generator inputs. */
public class ConfigurableMMicro {
  static final String[] EXTENSIONS = {"vec", "vemf", "vex", "vem"};
  // Published handles prevent constructor elimination. Rust uses black_box.
  static final Object[] CONSTRUCTED = new Object[128];
  static volatile long sink;
  static final class Digest {
    long value;
    void mix(long n) { value = value * 31 + n; }
    void bytes(byte[] values) {
      mix(values.length);
      for (byte value : values) mix(Byte.toUnsignedInt(value));
    }
    void text(String value) {
      mix(value.length());
      for (int i = 0; i < value.length(); i++) mix(value.charAt(i));
    }
  }
  static long construct() {
    Digest d = new Digest();
    int at = 0;
    for (int i = 0; i < 32; i++) for (int m : new int[] {128, 4, 1, 512}) {
      var codec = m == 128 ? new ConfigurableMCodec() : new ConfigurableMCodec(m);
      CONSTRUCTED[at++] = codec;
      var format = codec.knnVectorsFormat();
      d.text(codec.getName());
      d.text(format.getName());
      d.mix(format == codec.knnVectorsFormat() ? 1 : 0);
    }
    return d.value;
  }
  static SegmentWriteState writeState(Directory directory, int maxDoc) {
    return new SegmentWriteState(InfoStream.NO_OUTPUT, directory,
        GenConfigurableM.segment(directory, maxDoc), GenConfigurableM.FIELDS,
        null, IOContext.DEFAULT, GenConfigurableM.SUFFIX);
  }
  static KnnVectorsReader open(KnnVectorsFormat format, Directory directory, int maxDoc)
      throws Exception {
    return format.fieldsReader(new SegmentReadState(directory,
        GenConfigurableM.segment(directory, maxDoc), GenConfigurableM.FIELDS,
        IOContext.DEFAULT, GenConfigurableM.SUFFIX));
  }
  static long read(KnnVectorsReader reader) throws Exception {
    reader.checkIntegrity();
    Digest d = new Digest();
    for (FieldInfo field : GenConfigurableM.FIELDS) {
      d.text(field.name);
      d.mix(field.number);
      if (field.getVectorEncoding() == VectorEncoding.FLOAT32) {
        var values = reader.getFloatVectorValues(field.name);
        d.mix(values.dimension()); d.mix(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
          d.mix(values.ordToDoc(ord)); d.mix(ord);
          for (float value : values.vectorValue(ord)) d.mix(Float.floatToRawIntBits(value));
        }
      } else {
        var values = reader.getByteVectorValues(field.name);
        d.mix(values.dimension()); d.mix(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
          d.mix(values.ordToDoc(ord)); d.mix(ord);
          for (byte value : values.vectorValue(ord)) d.mix(value);
        }
      }
      var graph = ((HnswGraphProvider) reader).getGraph(field.name);
      d.mix(graph.size()); d.mix(graph.numLevels()); d.mix(graph.entryNode());
      for (int level = 0; level < graph.numLevels(); level++) {
        d.mix(level);
        var nodes = graph.getNodesOnLevel(level);
        while (nodes.hasNext()) {
          int node = nodes.nextInt(); d.mix(node);
          graph.seek(level, node);
          int count = 0;
          for (int neighbor = graph.nextNeighbor(); neighbor != DocIdSetIterator.NO_MORE_DOCS;
              neighbor = graph.nextNeighbor()) {
            d.mix(neighbor); count++;
          }
          d.mix(count);
        }
      }
    }
    return d.value;
  }
  static long search(KnnVectorsReader reader, int maxDoc, boolean filtered, Bits mask)
      throws Exception {
    Digest d = new Digest();
    var accept = AcceptDocs.fromLiveDocs(filtered ? mask : null, maxDoc);
    for (FieldInfo field : GenConfigurableM.FIELDS) for (int doc : new int[] {0, maxDoc / 2, maxDoc - 1}) {
      var collector = new TopKnnCollector(7, Integer.MAX_VALUE);
      if (field.getVectorEncoding() == VectorEncoding.FLOAT32)
        reader.search(field.name, GenConfigurableM.floats(doc), collector, accept);
      else reader.search(field.name, GenConfigurableM.bytes(doc), collector, accept);
      var hits = collector.topDocs();
      d.mix(hits.totalHits.value());
      d.mix(hits.totalHits.relation() == TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO ? 1 : 0);
      d.mix(hits.scoreDocs.length);
      for (var hit : hits.scoreDocs) { d.mix(hit.doc); d.mix(Float.floatToRawIntBits(hit.score)); }
    }
    return d.value;
  }
  static final class Source {
    final int[][] docs = new int[6][];
    final float[][] floats;
    final byte[][] bytes;
    Source(int maxDoc) {
      floats = new float[maxDoc][]; bytes = new byte[maxDoc][];
      for (int doc = 0; doc < maxDoc; doc++) {
        floats[doc] = GenConfigurableM.floats(doc); bytes[doc] = GenConfigurableM.bytes(doc);
      }
      for (FieldInfo field : GenConfigurableM.FIELDS) {
        int step = field.name.endsWith("sparse") ? 2 : 1;
        int count = field.name.endsWith("empty") ? 0 : maxDoc / step;
        docs[field.number] = new int[count];
        for (int ord = 0; ord < count; ord++) docs[field.number][ord] = ord * step;
      }
    }
  }
  static byte[][] write(KnnVectorsFormat format, int maxDoc, Source source) throws Exception {
    long previousSeed = HnswGraphBuilder.randSeed;
    HnswGraphBuilder.randSeed = 42;
    try (Directory directory = new ByteBuffersDirectory()) {
      try (var writer = format.fieldsWriter(writeState(directory, maxDoc))) {
        for (FieldInfo field : GenConfigurableM.FIELDS) {
          var target = writer.addField(field);
          if (field.getVectorEncoding() == VectorEncoding.FLOAT32) {
            @SuppressWarnings("unchecked") var floats = (KnnFieldVectorsWriter<float[]>) target;
            for (int doc : source.docs[field.number]) floats.addValue(doc, source.floats[doc]);
          } else {
            @SuppressWarnings("unchecked") var bytes = (KnnFieldVectorsWriter<byte[]>) target;
            for (int doc : source.docs[field.number]) bytes.addValue(doc, source.bytes[doc]);
          }
        }
        writer.flush(maxDoc, null); writer.finish();
      }
      byte[][] output = new byte[4][];
      for (int i = 0; i < EXTENSIONS.length; i++) {
        try (var input = directory.openInput("_cm_s." + EXTENSIONS[i], IOContext.DEFAULT)) {
          output[i] = new byte[Math.toIntExact(input.length())];
          input.readBytes(output[i], 0, output[i].length);
        }
      }
      return output;
    } finally {
      HnswGraphBuilder.randSeed = previousSeed;
    }
  }
  static long writeDigest(KnnVectorsFormat format, int maxDoc, Source source) throws Exception {
    Digest d = new Digest();
    for (byte[] output : write(format, maxDoc, source)) d.bytes(output);
    return d.value;
  }
  interface Work { long run() throws Exception; }
  static long[] run(Work work, long millis, long expected, int units) throws Exception {
    long start = System.nanoTime(), count = 0;
    do {
      long actual = work.run();
      if (actual != expected) throw new AssertionError("timed digest changed");
      sink = actual; count += units;
    } while (System.nanoTime() - start < millis * 1_000_000L);
    return new long[] {System.nanoTime() - start, count};
  }
  static void report(String name, long warmup, long measure, int units, Work work) throws Exception {
    long expected = work.run();
    System.out.printf("#check\t%s\t%d\t%d%n", name, expected, units);
    run(work, warmup, expected, units);
    long[] result = run(work, measure, expected, units);
    System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, (double) result[0] / result[1], result[1]);
  }
  public static void main(String[] args) throws Exception {
    if (!ConfigurableMMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    long warmup = args.length > 0 ? Long.parseLong(args[0]) : 500;
    long measure = args.length > 1 ? Long.parseLong(args[1]) : 750;
    report("configurable_m_construct", warmup, measure, 32, ConfigurableMMicro::construct);
    for (String label : new String[] {"default_647", "default_648", "4_647", "4_648"}) {
      int m = label.startsWith("default") ? 128 : 4;
      int count = label.endsWith("647") ? 647 : 648, maxDoc = count * 2;
      var codec = m == 128 ? new ConfigurableMCodec() : new ConfigurableMCodec(m);
      var format = codec.knnVectorsFormat();
      Source source = new Source(maxDoc);
      byte[][] expected = new byte[4][];
      long payload = 0;
      try (Directory directory = new ByteBuffersDirectory()) {
        for (int i = 0; i < EXTENSIONS.length; i++) {
          expected[i] = Files.readAllBytes(Path.of("fixtures/testframework/data/configurable_m_" + label + "." + EXTENSIONS[i]));
          payload += expected[i].length;
          try (var output = directory.createOutput("_cm_s." + EXTENSIONS[i], IOContext.DEFAULT)) {
            output.writeBytes(expected[i], 0, expected[i].length);
          }
        }
        byte[][] actual = write(format, maxDoc, source);
        for (int i = 0; i < EXTENSIONS.length; i++) if (!Arrays.equals(actual[i], expected[i]))
          throw new AssertionError(label + "." + EXTENSIONS[i] + ": differs from actual Java fixture");
        System.out.printf("#memory\tconfigurable_m_%s_serialized_payload_bytes\t%d%n", label, payload);
        report("configurable_m_" + label + "_read", warmup, measure, 1, () -> {
          try (var reader = open(format, directory, maxDoc)) { return read(reader); }
        });
        boolean[] bits = new boolean[maxDoc];
        for (int doc = 0; doc < maxDoc; doc++) bits[doc] = doc % 3 != 0;
        Bits mask = new Bits() {
          public boolean get(int doc) { return bits[doc]; }
          public int length() { return maxDoc; }
        };
        try (var reader = open(format, directory, maxDoc)) {
          for (boolean filtered : new boolean[] {false, true}) {
            String kind = filtered ? "search_filtered" : "search_unfiltered";
            report("configurable_m_" + label + "_" + kind, warmup, measure, 1,
                () -> search(reader, maxDoc, filtered, mask));
          }
        }
        report("configurable_m_" + label + "_write", warmup, measure, 1,
            () -> writeDigest(format, maxDoc, source));
      }
    }
  }
}
