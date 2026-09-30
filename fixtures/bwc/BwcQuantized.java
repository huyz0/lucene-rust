import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/**
 * Writes one index of per-field quantized vector fields with whichever Lucene version is on the
 * classpath, for the M8 quantized-format differential tests: {@code
 * Lucene99HnswScalarQuantizedVectorsFormat} (9.9+: format version 0 on 9.9, version 1 with 4-bit
 * and compressed variants from 9.10), the flat {@code Lucene99ScalarQuantizedVectorsFormat} (usable
 * per field from 9.11), and {@code Lucene102(Hnsw)BinaryQuantizedVectorsFormat} (10.2-10.3). The
 * formats are built by reflection because their constructors changed across the releases.
 *
 * <p>Two segments (1,000 and 200 documents), every 13th document deleted. Some fields are sparse
 * (every fourth document has none). The codec is the release's default with only {@code
 * knnVectorsFormat()} overridden, under the default codec's name, so Lucene 10.5.0 opens the index
 * through backward-codecs as it would a real one.
 */
public class BwcQuantized {
  record Spec(String name, KnnVectorsFormat format, VectorSimilarityFunction sim, int dim, boolean isByte, boolean sparse) {}

  static Object make(String cls, Class<?>[] types, Object... args) {
    try {
      Constructor<?> c = Class.forName(cls).getConstructor(types);
      return c.newInstance(args);
    } catch (ReflectiveOperationException e) {
      return null;
    }
  }

  static final String SQ = "org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat";
  static final String SQ_FLAT = "org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat";
  static final String BQ = "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat";
  static final String BQ_FLAT = "org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat";

  /** An HNSW scalar-quantized format: the 9.10+ constructor, else 9.9's (7 bits only). */
  static KnnVectorsFormat sq(int bits, boolean compress, Float confidence) {
    Object f = make(SQ,
        new Class<?>[] {int.class, int.class, int.class, int.class, boolean.class, Float.class, ExecutorService.class},
        16, 100, 1, bits, compress, confidence, null);
    if (f == null && bits == 7 && !compress) {
      f = make(SQ, new Class<?>[] {int.class, int.class, int.class, Float.class, ExecutorService.class},
          16, 100, 1, confidence, null);
    }
    return f instanceof KnnVectorsFormat k ? k : null;
  }

  static KnnVectorsFormat sqFlat(int bits, boolean compress, Float confidence) {
    Object f = make(SQ_FLAT, new Class<?>[] {Float.class, int.class, boolean.class}, confidence, bits, compress);
    return f instanceof KnnVectorsFormat k ? k : null;
  }

  static KnnVectorsFormat plain(String cls) {
    Object f = make(cls, new Class<?>[] {});
    return f instanceof KnnVectorsFormat k ? k : null;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Spec[] all = {
      new Spec("sq7_l2", sq(7, false, null), VectorSimilarityFunction.EUCLIDEAN, 16, false, true),
      new Spec("sq7_dot", sq(7, false, 0.95f), VectorSimilarityFunction.DOT_PRODUCT, 16, false, false),
      new Spec("sq7_cos", sq(7, false, null), VectorSimilarityFunction.COSINE, 16, false, false),
      new Spec("sq7_mip", sq(7, false, 0.95f), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 16, false, false),
      new Spec("sq4c_dot", sq(4, true, null), VectorSimilarityFunction.DOT_PRODUCT, 16, false, false),
      new Spec("sq4_l2", sq(4, false, 0f), VectorSimilarityFunction.EUCLIDEAN, 16, false, true),
      new Spec("sq4c_mip", sq(4, true, 0f), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 16, false, false),
      new Spec("sqflat_l2", sqFlat(7, false, null), VectorSimilarityFunction.EUCLIDEAN, 16, false, true),
      new Spec("sqbyte", sq(7, false, null), VectorSimilarityFunction.EUCLIDEAN, 16, true, false),
      new Spec("bq_l2", plain(BQ), VectorSimilarityFunction.EUCLIDEAN, 40, false, true),
      new Spec("bq_cos", plain(BQ), VectorSimilarityFunction.COSINE, 40, false, false),
      new Spec("bq_dot", plain(BQ), VectorSimilarityFunction.DOT_PRODUCT, 40, false, false),
      new Spec("bq_mip", plain(BQ), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 40, false, false),
      new Spec("bqflat_l2", plain(BQ_FLAT), VectorSimilarityFunction.EUCLIDEAN, 40, false, true),
      new Spec("bqflat_dot", plain(BQ_FLAT), VectorSimilarityFunction.DOT_PRODUCT, 40, false, false),
    };
    java.util.List<Spec> specs = new java.util.ArrayList<>();
    for (Spec s : all) if (s.format() != null) specs.add(s);

    Codec base = Codec.getDefault();
    KnnVectorsFormat perField = new PerFieldKnnVectorsFormat() {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        for (Spec s : specs) if (s.name().equals(field)) return s.format();
        throw new IllegalArgumentException(field);
      }
    };
    Codec codec = new FilterCodec(base.getName(), base) {
      @Override
      public KnnVectorsFormat knnVectorsFormat() {
        return perField;
      }
    };

    Random r = new Random(20260930L);
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setCodec(codec);
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int docs = 0;
        for (int seg = 0; seg < 2; seg++) {
          int n = seg == 0 ? 1000 : 200;
          for (int i = 0; i < n; i++, docs++) {
            Document d = new Document();
            d.add(new StringField("id", Integer.toString(docs), Field.Store.YES));
            for (Spec s : specs) {
              if (s.sparse() && i % 4 == 0) continue;
              if (s.isByte()) {
                byte[] v = new byte[s.dim()];
                for (int k = 0; k < v.length; k++) v[k] = (byte) (r.nextInt(256) - 128);
                d.add(new KnnByteVectorField(s.name(), v, s.sim()));
              } else {
                float[] v = new float[s.dim()];
                double norm = 0;
                for (int k = 0; k < v.length; k++) {
                  v[k] = r.nextFloat() * 2 - 1;
                  norm += v[k] * v[k];
                }
                if (s.sim() == VectorSimilarityFunction.DOT_PRODUCT) {
                  for (int k = 0; k < v.length; k++) v[k] = (float) (v[k] / Math.sqrt(norm));
                }
                d.add(new KnnFloatVectorField(s.name(), v, s.sim()));
              }
            }
            w.addDocument(d);
          }
          w.flush();
        }
        for (int id = 0; id < docs; id += 13) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
        }
        w.commit();
      }
    }
    Files.writeString(out.resolve("written_by.txt"), Version.LATEST.toString() + "\n");
  }
}
