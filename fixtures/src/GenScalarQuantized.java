import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;
import org.apache.lucene.util.quantization.ScalarQuantizer;

/**
 * Differential fixtures for scalar vector quantization (`util/quantization`) and the
 * `Lucene104HnswScalarQuantizedVectorsFormat` (`codecs/lucene104`).
 *
 * <ul>
 *   <li>{@code scalar_quantized/quantizer.txt}: {@code OptimizedScalarQuantizer.scalarQuantize}/
 *       {@code multiScalarQuantize}/{@code deQuantize} and the packing helpers on random inputs,
 *       and the legacy {@code ScalarQuantizer} (quantiles from vectors, including the
 *       reservoir-sampled auto-interval search, and its quantized similarity scores).
 *   <li>{@code scalar_quantized_index/}: one flushed segment, two fields per
 *       {@code ScalarEncoding} (one format instance per encoding, so each encoding has its own
 *       {@code .veq}/{@code .vemq} pair), a dense and a sparse field each; {@code manifest.txt}
 *       lists the fields and the top-10 of KNN queries run by Lucene.
 *   <li>{@code scalar_quantized_merge_plain/} and {@code scalar_quantized_merge_deletes/}: a
 *       {@code sources/} index of two segments and a {@code merged/} copy of it force-merged to
 *       one, without and with a deletion (the two centroid-merge paths).
 * </ul>
 *
 * <p>Dimensions are at most 16 so that {@code VectorUtil.dotProduct} (used for the centroid's
 * squared norm and COSINE normalization) takes the same sequential fused-multiply-add loop in
 * Lucene's default and Panama implementations and in the port's kernel.
 *
 * <p>Usage: java GenScalarQuantized &lt;outdir&gt;
 */
public class GenScalarQuantized {
  static final StringBuilder Q = new StringBuilder();

  static void line(StringBuilder sb, Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) sb.append(' ');
      sb.append(parts[i]);
    }
    sb.append('\n');
  }

  static String fbits(float[] v) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < v.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(Float.floatToIntBits(v[i]));
    }
    return v.length == 0 ? "-" : sb.toString();
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x & 0xFF));
    return b.length == 0 ? "-" : sb.toString();
  }

  static float[] randomVector(Random r, int dim, float scale) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) v[i] = (r.nextFloat() - 0.4f) * scale;
    return v;
  }

  static String result(OptimizedScalarQuantizer.QuantizationResult q) {
    return Float.floatToIntBits(q.lowerInterval())
        + " "
        + Float.floatToIntBits(q.upperInterval())
        + " "
        + Float.floatToIntBits(q.additionalCorrection())
        + " "
        + q.quantizedComponentSum();
  }

  static void quantizer(Path out) throws Exception {
    Random r = new Random(0x5CA1A2_2026L);
    VectorSimilarityFunction[] sims = VectorSimilarityFunction.values();
    byte[] allBits = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int c = 0; c < 80; c++) {
      VectorSimilarityFunction sim = sims[c % sims.length];
      int dim = 1 + r.nextInt(c < 40 ? 16 : 300);
      float scale = c % 5 == 0 ? 100f : 1f;
      float[] vec = randomVector(r, dim, scale);
      float[] centroid = c % 7 == 0 ? new float[dim] : randomVector(r, dim, scale / 4);
      if (sim == VectorSimilarityFunction.COSINE && dim <= 16) {
        VectorUtil.l2normalize(vec);
        if (c % 7 != 0) VectorUtil.l2normalize(centroid);
      }
      byte bits = allBits[c % allBits.length];
      OptimizedScalarQuantizer q = new OptimizedScalarQuantizer(sim);
      byte[] dest = new byte[dim];
      float[] copy = vec.clone();
      OptimizedScalarQuantizer.QuantizationResult res = q.scalarQuantize(copy, dest, bits, centroid);
      line(Q, "osq", sim.ordinal(), bits, fbits(vec), fbits(centroid));
      line(Q, "result", result(res), hex(dest), fbits(copy));
      float[] deq =
          OptimizedScalarQuantizer.deQuantize(
              dest, new float[dim], bits, res.lowerInterval(), res.upperInterval(), centroid);
      line(Q, "dequantized", fbits(deq));
      // multi
      byte[] mbits = {bits, (byte) (1 + (bits % 8))};
      byte[][] dests = {new byte[dim], new byte[dim]};
      OptimizedScalarQuantizer.QuantizationResult[] multi =
          q.multiScalarQuantize(vec.clone(), dests, mbits, centroid);
      line(Q, "multi", mbits[0], mbits[1], result(multi[0]), hex(dests[0]), result(multi[1]), hex(dests[1]));
    }
    // Packing helpers.
    for (int c = 0; c < 20; c++) {
      int dim = 8 * (1 + r.nextInt(8));
      byte[] q4 = new byte[dim];
      byte[] q1 = new byte[dim];
      byte[] q2 = new byte[dim];
      for (int i = 0; i < dim; i++) {
        q4[i] = (byte) r.nextInt(16);
        q1[i] = (byte) r.nextInt(2);
        q2[i] = (byte) r.nextInt(4);
      }
      byte[] t = new byte[dim / 2];
      OptimizedScalarQuantizer.transposeHalfByte(q4, t);
      byte[] b = new byte[dim / 8];
      OptimizedScalarQuantizer.packAsBinary(q1, b);
      byte[] d = new byte[dim / 4];
      OptimizedScalarQuantizer.transposeDibit(q2, d);
      long bitDot = VectorUtil.int4BitDotProduct(t, b);
      long dibitDot = VectorUtil.int4DibitDotProduct(t, d);
      line(Q, "pack", hex(q4), hex(q1), hex(q2), hex(t), hex(b), hex(d), bitDot, dibitDot);
    }
    // Byte kernels.
    for (int c = 0; c < 20; c++) {
      int dim = 1 + r.nextInt(300);
      byte[] a = new byte[dim];
      byte[] bb = new byte[dim];
      r.nextBytes(a);
      r.nextBytes(bb);
      byte[] a4 = new byte[dim * 2];
      byte[] p4 = new byte[dim];
      for (int i = 0; i < dim * 2; i++) a4[i] = (byte) r.nextInt(16);
      for (int i = 0; i < dim; i++) p4[i] = (byte) r.nextInt(256);
      line(
          Q,
          "kernels",
          hex(a),
          hex(bb),
          VectorUtil.dotProduct(a, bb),
          VectorUtil.uint8DotProduct(a, bb),
          VectorUtil.squareDistance(a, bb),
          VectorUtil.uint8SquareDistance(a, bb),
          VectorUtil.xorBitCount(a, bb),
          hex(a4),
          hex(p4),
          VectorUtil.int4DotProductSinglePacked(a4, p4),
          VectorUtil.int4SquareDistanceSinglePacked(a4, p4),
          VectorUtil.int4DotProductBothPacked(p4, p4),
          VectorUtil.int4SquareDistanceBothPacked(p4, a));
    }
    // Legacy ScalarQuantizer. Order matters: the auto-interval reservoir sample draws from a
    // process-wide Random(42), and the Rust test replays these records in this order.
    for (int c = 0; c < 12; c++) {
      int dim = 2 + r.nextInt(15);
      int n = c == 11 ? 1100 : 10 + r.nextInt(300);
      byte bits = (byte) (c % 2 == 0 ? 7 : 4);
      VectorSimilarityFunction sim =
          new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
          }[c % 3];
      List<float[]> vs = new ArrayList<>();
      StringBuilder all = new StringBuilder();
      for (int i = 0; i < n; i++) {
        float[] v = randomVector(r, dim, 2f);
        vs.add(v);
        if (i > 0) all.append(';');
        all.append(fbits(v));
      }
      float ci = c < 4 ? 1f : (c < 8 ? 0.9f + 0.01f * c : 0f);
      ScalarQuantizer sq;
      String kind;
      if (ci == 0f) {
        kind = "auto";
        sq =
            ScalarQuantizer.fromVectorsAutoInterval(
                org.apache.lucene.index.FloatVectorValues.fromFloats(vs, dim), sim, n, bits);
      } else {
        kind = "ci";
        sq =
            ScalarQuantizer.fromVectors(
                org.apache.lucene.index.FloatVectorValues.fromFloats(vs, dim), ci, n, bits);
      }
      line(Q, "legacy", kind, sim.ordinal(), bits, Float.floatToIntBits(ci), dim, all);
      byte[] q0 = new byte[dim];
      byte[] q1 = new byte[dim];
      float c0 = sq.quantize(vs.get(0), q0, sim);
      float c1 = sq.quantize(vs.get(1), q1, sim);
      ScalarQuantizer other = new ScalarQuantizer(sq.getLowerQuantile() - 0.5f, sq.getUpperQuantile() + 0.5f, bits);
      float re = other.recalculateCorrectiveOffset(q0, sq, sim);
      ScalarQuantizedVectorSimilarity s =
          ScalarQuantizedVectorSimilarity.fromVectorSimilarity(sim, sq.getConstantMultiplier(), bits);
      float score = s.score(q0, c0, q1, c1);
      line(
          Q,
          "legacyresult",
          Float.floatToIntBits(sq.getLowerQuantile()),
          Float.floatToIntBits(sq.getUpperQuantile()),
          hex(q0),
          Float.floatToIntBits(c0),
          hex(q1),
          Float.floatToIntBits(c1),
          Float.floatToIntBits(re),
          Float.floatToIntBits(score));
    }
    Files.createDirectories(out.resolve("scalar_quantized"));
    Files.writeString(out.resolve("scalar_quantized").resolve("quantizer.txt"), Q.toString());
  }

  // -------------------------------------------------------------------------------------------
  // Index fixtures
  // -------------------------------------------------------------------------------------------

  record FieldSpec(
      String name,
      ScalarEncoding encoding,
      int dim,
      VectorSimilarityFunction sim,
      int every,
      long seed) {}

  static final int NUM_DOCS = 1000;
  static final int K = 10;
  static final int NUM_QUERIES = 6;

  static List<FieldSpec> indexFields() {
    List<FieldSpec> fs = new ArrayList<>();
    ScalarEncoding[] encs = ScalarEncoding.values();
    int[] dims = {16, 13, 16, 9, 11};
    VectorSimilarityFunction[] denseSims = {
      VectorSimilarityFunction.EUCLIDEAN,
      VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
      VectorSimilarityFunction.DOT_PRODUCT,
      VectorSimilarityFunction.EUCLIDEAN,
      VectorSimilarityFunction.COSINE
    };
    VectorSimilarityFunction[] sparseSims = {
      VectorSimilarityFunction.COSINE,
      VectorSimilarityFunction.EUCLIDEAN,
      VectorSimilarityFunction.COSINE,
      VectorSimilarityFunction.DOT_PRODUCT,
      VectorSimilarityFunction.EUCLIDEAN
    };
    for (int i = 0; i < encs.length; i++) {
      String e = encs[i].name().toLowerCase();
      fs.add(new FieldSpec(e + "_dense", encs[i], dims[i], denseSims[i], 1, 1000L * (i + 1)));
      fs.add(new FieldSpec(e + "_sparse", encs[i], 16 - i, sparseSims[i], 3, 77_000L * (i + 1)));
    }
    return fs;
  }

  static float[] vectorFor(FieldSpec f, int doc) {
    float[] v = GenVectorsMulti.floatVector(f.dim(), f.seed() + doc);
    if (f.sim() == VectorSimilarityFunction.DOT_PRODUCT) {
      VectorUtil.l2normalize(v);
    }
    return v;
  }

  static IndexWriterConfig config(Map<String, KnnVectorsFormat> formats) {
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(100_000);
    cfg.setRAMBufferSizeMB(1024);
    cfg.setCodec(
        new Lucene104Codec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            return formats.get(field);
          }
        });
    return cfg;
  }

  static void addDocs(IndexWriter w, List<FieldSpec> fields, int from, int to) throws IOException {
    for (int d = from; d < to; d++) {
      Document doc = new Document();
      doc.add(new StringField("id", Integer.toString(d), Field.Store.NO));
      doc.add(new StringField("g", "g" + (d % 10), Field.Store.NO));
      for (FieldSpec f : fields) {
        if (d % f.every() == 0) {
          doc.add(new KnnFloatVectorField(f.name(), vectorFor(f, d), f.sim()));
        }
      }
      w.addDocument(doc);
    }
  }

  static void describe(StringBuilder m, Directory dir, List<FieldSpec> fields, boolean search)
      throws IOException {
    SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      for (int s = 0; s < sis.size(); s++) {
        SegmentCommitInfo sci = sis.info(s);
        line(
            m,
            "segment",
            sci.info.name,
            GenVectorsMulti.hex(sci.info.getId()),
            sci.info.maxDoc(),
            sci.getDelCount());
        LeafReader leaf = reader.leaves().get(s).reader();
        for (FieldSpec f : fields) {
          FieldInfo fi = leaf.getFieldInfos().fieldInfo(f.name());
          if (fi == null) continue;
          line(
              m,
              "field",
              f.name(),
              fi.number,
              fi.getAttribute("PerFieldKnnVectorsFormat.suffix"),
              f.dim(),
              f.sim().ordinal(),
              f.encoding().getWireNumber());
        }
      }
      if (!search) return;
      IndexSearcher searcher = new IndexSearcher(reader);
      Random r = new Random(0xFEED);
      for (FieldSpec f : fields) {
        for (int q = 0; q < NUM_QUERIES; q++) {
          float[] query = GenVectorsMulti.floatVector(f.dim(), 5_000_000L + 97L * q + f.seed());
          if (f.sim() == VectorSimilarityFunction.DOT_PRODUCT) VectorUtil.l2normalize(query);
          TopDocs td = searcher.search(new KnnFloatVectorQuery(f.name(), query, K), K);
          StringBuilder hits = new StringBuilder();
          for (ScoreDoc sd : td.scoreDocs) {
            if (hits.length() > 0) hits.append(',');
            hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
          }
          line(m, "query", f.name(), fbits(query), hits.length() == 0 ? "-" : hits);
          if (f.every() == 1) {
            // The same query filtered to a tenth of the documents, with a strategy threshold
            // (60%) that picks FilteredHnswGraphSearcher.
            TopDocs ftd =
                searcher.search(
                    new KnnFloatVectorQuery(
                        f.name(),
                        query,
                        K,
                        new org.apache.lucene.search.TermQuery(new Term("g", "g0")),
                        new org.apache.lucene.search.knn.KnnSearchStrategy.Hnsw(60)),
                    K);
            StringBuilder fh = new StringBuilder();
            for (ScoreDoc sd : ftd.scoreDocs) {
              if (fh.length() > 0) fh.append(',');
              fh.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
            }
            line(m, "fquery", f.name(), fbits(query), fh.length() == 0 ? "-" : fh);
          }
        }
      }
    }
  }

  static Map<String, KnnVectorsFormat> formats(List<FieldSpec> fields) {
    Map<ScalarEncoding, KnnVectorsFormat> perEncoding = new LinkedHashMap<>();
    Map<String, KnnVectorsFormat> formats = new HashMap<>();
    for (FieldSpec f : fields) {
      KnnVectorsFormat fmt =
          perEncoding.computeIfAbsent(
              f.encoding(), e -> new Lucene104HnswScalarQuantizedVectorsFormat(e, 16, 100));
      formats.put(f.name(), fmt);
    }
    return formats;
  }

  static void index(Path out) throws Exception {
    Path dirPath = out.resolve("scalar_quantized_index");
    if (Files.exists(dirPath)) GenVectorsMulti.deleteRecursive(dirPath);
    Files.createDirectories(dirPath);
    List<FieldSpec> fields = indexFields();
    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(dirPath)) {
      try (IndexWriter w = new IndexWriter(dir, config(formats(fields)))) {
        addDocs(w, fields, 0, NUM_DOCS);
        w.commit();
      }
      describe(m, dir, fields, true);
    }
    Files.writeString(dirPath.resolve("manifest.txt"), m.toString());
  }

  static void merge(Path out, String name, boolean deletes) throws Exception {
    Path base = out.resolve(name);
    if (Files.exists(base)) GenVectorsMulti.deleteRecursive(base);
    Path sources = base.resolve("sources");
    Path merged = base.resolve("merged");
    Files.createDirectories(sources);
    Files.createDirectories(merged);
    List<FieldSpec> fields = new ArrayList<>();
    fields.add(
        new FieldSpec("byte_dense", ScalarEncoding.UNSIGNED_BYTE, 16, VectorSimilarityFunction.EUCLIDEAN, 1, 31));
    fields.add(
        new FieldSpec(
            "bit_sparse", ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE, 12, VectorSimilarityFunction.COSINE, 2, 41));
    fields.add(
        new FieldSpec("nibble_dense", ScalarEncoding.PACKED_NIBBLE, 10, VectorSimilarityFunction.DOT_PRODUCT, 1, 51));
    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(sources)) {
      try (IndexWriter w = new IndexWriter(dir, config(formats(fields)))) {
        addDocs(w, fields, 0, 700);
        w.commit();
        addDocs(w, fields, 700, 1200);
        w.commit();
        if (deletes) {
          w.deleteDocuments(new Term("id", "5"));
          w.deleteDocuments(new Term("id", "702"));
          w.commit();
        }
      }
      line(m, "sources");
      describe(m, dir, fields, false);
    }
    for (Path p : Files.list(sources).toList()) {
      if (!p.getFileName().toString().equals("write.lock")) {
        Files.copy(p, merged.resolve(p.getFileName()));
      }
    }
    try (Directory dir = FSDirectory.open(merged)) {
      IndexWriterConfig cfg = config(formats(fields));
      cfg.setMergePolicy(new org.apache.lucene.index.TieredMergePolicy());
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        w.forceMerge(1);
        w.commit();
      }
      line(m, "merged");
      describe(m, dir, fields, true);
    }
    Files.writeString(base.resolve("manifest.txt"), m.toString());
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    quantizer(out);
    index(out);
    merge(out, "scalar_quantized_merge_plain", false);
    merge(out, "scalar_quantized_merge_deletes", true);
  }
}
