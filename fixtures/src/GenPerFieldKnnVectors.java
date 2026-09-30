import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

/**
 * {@code PerFieldKnnVectorsFormat} write routing across flush and merge, with the formats in
 * lucene-core: {@code v_small} and {@code v_bytes} go to {@code Lucene99HnswVectorsFormat(8, 40)},
 * {@code v_default} to the default {@code Lucene99HnswVectorsFormat()}, {@code v_sq} to {@code
 * Lucene104HnswScalarQuantizedVectorsFormat(UNSIGNED_BYTE, 16, 100)} and {@code v_flat} to the
 * graph-less {@code Lucene104ScalarQuantizedVectorsFormat(SEVEN_BIT)}.
 *
 * <p>Every document carries its fields in the order {@code id}, {@code v_sq} (missing from every
 * seventh document), {@code v_default} (missing from the first), {@code v_small} (missing from
 * the first of the second segment), {@code v_bytes}, {@code v_flat}. A flush numbers each format
 * name's instances in the order its documents first carry a field of each ({@code addField}), so
 * the first segment's default HNSW instance is {@code _1} and the second's {@code _0}; the merge
 * numbers them in field-number order.
 *
 * <p>{@code per_field_knn_vectors/flushed}: two flushed segments of {@link #PER_SEGMENT}
 * documents. {@code per_field_knn_vectors/merged}: the same, two documents deleted by {@code id}
 * (so the quantized centroid is recomputed), force-merged into one. {@code manifest.txt}: per
 * segment its vector files and each vector field's attributes, then, for a query per field, the
 * top ten hits of a {@code Knn*VectorQuery} over the whole index with their scores' bits.
 */
public class GenPerFieldKnnVectors {
  static final int PER_SEGMENT = 300;
  static final int DIM = 16;

  /** The Rust test computes the same. */
  static long value(int i, int k) {
    long x = (i + 1) * 2654435761L + k * 40503L;
    return (x ^ (x >>> 13)) % 100000;
  }

  static float[] floats(int i, int salt) {
    float[] v = new float[DIM];
    for (int j = 0; j < DIM; j++) {
      v[j] = (value(i, salt * 100 + j) % 2001 - 1000) / 100f;
    }
    return v;
  }

  static byte[] bytes(int i) {
    byte[] v = new byte[DIM];
    for (int j = 0; j < DIM; j++) {
      v[j] = (byte) (value(i, 500 + j) % 255 - 127);
    }
    return v;
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.NO));
    if (i % 7 != 0) {
      d.add(new KnnFloatVectorField("v_sq", floats(i, 1), VectorSimilarityFunction.EUCLIDEAN));
    }
    if (i != 0) {
      d.add(new KnnFloatVectorField("v_default", floats(i, 2), VectorSimilarityFunction.COSINE));
    }
    if (i != PER_SEGMENT) {
      d.add(new KnnFloatVectorField("v_small", floats(i, 3), VectorSimilarityFunction.EUCLIDEAN));
    }
    d.add(new KnnByteVectorField("v_bytes", bytes(i), VectorSimilarityFunction.EUCLIDEAN));
    d.add(new KnnFloatVectorField("v_flat", floats(i, 4), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
    return d;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("per_field_knn_vectors");
    if (Files.exists(root)) {
      try (var walk = Files.walk(root)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    write(root.resolve("flushed"), false);
    write(root.resolve("merged"), true);
    System.out.println("wrote per_field_knn_vectors/");
  }

  static void write(Path out, boolean merge) throws IOException {
    Files.createDirectories(out);
    KnnVectorsFormat small = new Lucene99HnswVectorsFormat(8, 40);
    KnnVectorsFormat standard = new Lucene99HnswVectorsFormat();
    KnnVectorsFormat sq =
        new Lucene104HnswScalarQuantizedVectorsFormat(ScalarEncoding.UNSIGNED_BYTE, 16, 100);
    KnnVectorsFormat flat = new Lucene104ScalarQuantizedVectorsFormat(ScalarEncoding.SEVEN_BIT);
    Lucene104Codec codec =
        new Lucene104Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            switch (field) {
              case "v_small":
              case "v_bytes":
                return small;
              case "v_sq":
                return sq;
              case "v_flat":
                return flat;
              default:
                return standard;
            }
          }
        };
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setCodec(codec);
      cfg.setUseCompoundFile(false);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxFullFlushMergeWaitMillis(0);
      MergePolicy policy;
      if (merge) {
        TieredMergePolicy tmp = new TieredMergePolicy();
        tmp.setNoCFSRatio(0.0);
        policy = tmp;
      } else {
        policy = NoMergePolicy.INSTANCE;
      }
      cfg.setMergePolicy(policy);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < 2; seg++) {
          for (int i = seg * PER_SEGMENT; i < (seg + 1) * PER_SEGMENT; i++) {
            w.addDocument(doc(i));
          }
          w.commit();
        }
        if (merge) {
          w.deleteDocuments(new Term("id", "d5"), new Term("id", "d302"));
          w.commit();
          w.forceMerge(1);
          w.commit();
        }
      }
      List<String> lines = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext leaf : reader.leaves()) {
          SegmentCommitInfo sci = ((SegmentReader) leaf.reader()).getSegmentInfo();
          StringBuilder files = new StringBuilder("segment " + sci.info.name + " files");
          new TreeSet<>(sci.files()).stream()
              .filter(f -> f.contains("Vectors"))
              .forEach(f -> files.append(' ').append(f));
          lines.add(files.toString());
          for (FieldInfo fi : leaf.reader().getFieldInfos()) {
            if (fi.getVectorDimension() == 0) {
              continue;
            }
            lines.add(
                "field "
                    + sci.info.name
                    + " "
                    + fi.name
                    + " "
                    + fi.number
                    + " "
                    + fi.getAttribute("PerFieldKnnVectorsFormat.format")
                    + " "
                    + fi.getAttribute("PerFieldKnnVectorsFormat.suffix"));
          }
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        String[] fields = {"v_sq", "v_default", "v_small", "v_bytes", "v_flat"};
        for (int q = 0; q < 3; q++) {
          for (String field : fields) {
            Query query =
                field.equals("v_bytes")
                    ? new KnnByteVectorQuery(field, bytes(1000 + q), 10)
                    : new KnnFloatVectorQuery(field, floats(1000 + q, 7), 10);
            TopDocs top = searcher.search(query, 10);
            StringBuilder sb = new StringBuilder("query " + q + " " + field);
            for (ScoreDoc sd : top.scoreDocs) {
              sb.append(' ').append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
            }
            lines.add(sb.toString());
          }
        }
      }
      Files.write(out.resolve("manifest.txt"), lines, StandardCharsets.UTF_8);
      try (CheckIndex check = new CheckIndex(dir)) {
        if (!check.checkIndex().clean) {
          throw new AssertionError("CheckIndex failed on " + out);
        }
      }
    }
  }
}
