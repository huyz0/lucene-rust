import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Generates a real `Lucene99HnswVectorsFormat` fixture -- the `.vec`/`.vemf`
 * flat store plus the `.vem`/`.vex` HNSW graph -- through an actual
 * {@link IndexWriter}, and records enough ground truth for the Rust side to
 * check three separate things against it:
 *
 * <ol>
 *   <li><b>The flat format.</b> Four fields exercise every branch a
 *       {@code Lucene99FlatVectorsReader} has: a <i>dense</i> FLOAT32 field
 *       (every document has a value, so no ord-to-doc structures are written
 *       at all), two <i>sparse</i> ones (an {@code IndexedDISI} bitset plus a
 *       {@code DirectMonotonicWriter} ord-to-doc mapping), a BYTE-encoded
 *       field, and all four {@link VectorSimilarityFunction}s.</li>
 *   <li><b>The graph format.</b> Per level: node count, total arc count and an
 *       order-sensitive hash over every (node, neighbours...) pair, so a
 *       mis-decoded `.vex` offset or delta cannot pass. A fifth field is
 *       deliberately tiny, so Lucene skips graph construction entirely
 *       ({@code numLevels == 0}) -- the branch a reader is most likely to get
 *       wrong because no normal fixture reaches it.</li>
 *   <li><b>The search.</b> For twenty queries: Lucene's own HNSW top-k (via
 *       {@link KnnFloatVectorQuery}, i.e. the real
 *       {@code Lucene99HnswVectorsReader.search}) <i>and</i> the exact
 *       brute-force top-k over the same vectors. The first is what a faithful
 *       port must reproduce exactly; the second is the denominator of the
 *       recall figure both engines are measured on.</li>
 * </ol>
 *
 * <p>Also records the first ten {@code new SplittableRandom(42).nextDouble()}
 * draws. Level assignment during graph construction is
 * {@code (int)(-ln(U) * ml)} over exactly that stream, so a port whose
 * generator drifts builds a differently-shaped graph for reasons that would
 * otherwise look like an algorithmic difference.
 */
public class GenVectors {

  static final int MAX_DOC = 4000;
  static final int SPARSE_COUNT_LIMIT = MAX_DOC; // docs 0..MAX_DOC step 3
  static final int BYTE_DOC_LIMIT = 2000;
  static final int TINY_DOC_LIMIT = 5;
  static final int NUM_QUERIES = 20;
  static final int K = 10;

  record FieldSpec(String name, int dim, VectorEncoding encoding, VectorSimilarityFunction sim) {}

  static final FieldSpec DENSE =
      new FieldSpec("dense_f32", 16, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
  static final FieldSpec SPARSE =
      new FieldSpec("sparse_f32", 8, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE);
  static final FieldSpec MIP =
      new FieldSpec(
          "mip_f32", 6, VectorEncoding.FLOAT32, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
  static final FieldSpec BYTES =
      new FieldSpec("byte_dot", 8, VectorEncoding.BYTE, VectorSimilarityFunction.DOT_PRODUCT);
  static final FieldSpec TINY =
      new FieldSpec("tiny_f32", 4, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);

  static final FieldSpec[] SPECS = {DENSE, SPARSE, MIP, BYTES, TINY};

  /**
   * Deterministic pseudo-random components. A plain 64-bit LCG rather than
   * {@code java.util.Random}, so the numbers depend only on this file.
   */
  static long lcg(long state) {
    return state * 6364136223846793005L + 1442695040888963407L;
  }

  static float[] floatVector(int dim, long seed) {
    float[] v = new float[dim];
    long s = seed;
    for (int i = 0; i < dim; i++) {
      s = lcg(s);
      v[i] = ((s >>> 40) / (float) (1 << 24)) - 0.5f;
    }
    return v;
  }

  static byte[] byteVector(int dim, long seed) {
    byte[] v = new byte[dim];
    long s = seed;
    for (int i = 0; i < dim; i++) {
      s = lcg(s);
      v[i] = (byte) ((s >>> 40) & 0xFF);
    }
    return v;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("vectors_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(MAX_DOC + 1);
      cfg.setRAMBufferSizeMB(1024);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < MAX_DOC; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(DENSE.name(), floatVector(DENSE.dim(), i + 1), DENSE.sim()));
          if (i % 3 == 0) {
            doc.add(
                new KnnFloatVectorField(
                    SPARSE.name(), floatVector(SPARSE.dim(), 1_000_003L + i), SPARSE.sim()));
          }
          if (i % 7 == 1) {
            doc.add(
                new KnnFloatVectorField(
                    MIP.name(), floatVector(MIP.dim(), 5_000_011L + i), MIP.sim()));
          }
          if (i < BYTE_DOC_LIMIT) {
            doc.add(
                new KnnByteVectorField(
                    BYTES.name(), byteVector(BYTES.dim(), 7_000_019L + i), BYTES.sim()));
          }
          if (i < TINY_DOC_LIMIT) {
            doc.add(
                new KnnFloatVectorField(
                    TINY.name(), floatVector(TINY.dim(), 9_000_037L + i), TINY.sim()));
          }
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      String segmentName = sci.info.name;

      String vecFile = null, vemfFile = null, vemFile = null, vexFile = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".vec")) vecFile = f;
        if (f.endsWith(".vemf")) vemfFile = f;
        if (f.endsWith(".vem")) vemFile = f;
        if (f.endsWith(".vex")) vexFile = f;
      }
      if (vecFile == null || vemfFile == null || vemFile == null || vexFile == null) {
        throw new AssertionError("missing a vector file: " + sci.info.files());
      }
      // `_0_Lucene99HnswVectorsFormat_0.vec` -> `Lucene99HnswVectorsFormat_0`
      String suffix = vecFile.substring(0, vecFile.length() - ".vec".length());
      suffix = suffix.substring(segmentName.length());
      if (suffix.startsWith("_")) {
        suffix = suffix.substring(1);
      }

      m.append("segment_name=").append(segmentName).append('\n');
      m.append("id_hex=").append(hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("segment_suffix=").append(suffix).append('\n');
      m.append("vec_file=").append(vecFile).append('\n');
      m.append("vemf_file=").append(vemfFile).append('\n');
      m.append("vem_file=").append(vemFile).append('\n');
      m.append("vex_file=").append(vexFile).append('\n');
      m.append("default_max_conn=").append(Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN).append('\n');
      m.append("default_beam_width=")
          .append(Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH)
          .append('\n');
      m.append("hnsw_graph_threshold=")
          .append(Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD)
          .append('\n');

      SplittableRandom sr = new SplittableRandom(42);
      StringBuilder srBits = new StringBuilder();
      for (int i = 0; i < 10; i++) {
        if (i > 0) srBits.append(',');
        srBits.append(Double.doubleToRawLongBits(sr.nextDouble()));
      }
      m.append("splittable_random_42=").append(srBits).append('\n');

      FieldInfos fieldInfos =
          sci.info.getCodec().fieldInfosFormat().read(dir, sci.info, "", IOContext.READONCE);
      SegmentReadState state =
          new SegmentReadState(dir, sci.info, fieldInfos, IOContext.READONCE, suffix);
      try (KnnVectorsReader reader = new Lucene99HnswVectorsFormat().fieldsReader(state)) {
        m.append("field_count=").append(SPECS.length).append('\n');
        for (int fi = 0; fi < SPECS.length; fi++) {
          describeField(m, "f" + fi, SPECS[fi], fieldInfos, reader);
        }
      }

      // Search ground truth needs a DirectoryReader, so that the "HNSW" rows
      // come out of the same code path a real query takes.
      try (DirectoryReader dr = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(dr);
        floatQueries(m, "f0", DENSE, searcher, dr);
        floatQueries(m, "f1", SPARSE, searcher, dr);
        floatQueries(m, "f2", MIP, searcher, dr);
        byteQueries(m, "f3", BYTES, searcher, dr);
      }
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }

  static void describeField(
      StringBuilder m, String key, FieldSpec spec, FieldInfos fieldInfos, KnnVectorsReader reader)
      throws IOException {
    FieldInfo info = fieldInfos.fieldInfo(spec.name());
    m.append(key).append(".name=").append(spec.name()).append('\n');
    m.append(key).append(".number=").append(info.number).append('\n');
    m.append(key).append(".dim=").append(spec.dim()).append('\n');
    m.append(key).append(".encoding=").append(spec.encoding().name()).append('\n');
    m.append(key).append(".similarity=").append(spec.sim().name()).append('\n');

    int size;
    StringBuilder ordToDoc = new StringBuilder();
    if (spec.encoding() == VectorEncoding.FLOAT32) {
      FloatVectorValues values = reader.getFloatVectorValues(spec.name());
      size = values.size();
      appendOrdToDoc(ordToDoc, values, size);
      // Every component of a handful of ordinals, as raw float bits.
      for (int ord : spotOrds(size)) {
        m.append(key).append(".spot.").append(ord).append('=');
        float[] v = values.vectorValue(ord);
        for (int i = 0; i < v.length; i++) {
          if (i > 0) m.append(',');
          m.append(Float.floatToRawIntBits(v[i]));
        }
        m.append('\n');
      }
    } else {
      ByteVectorValues values = reader.getByteVectorValues(spec.name());
      size = values.size();
      appendOrdToDoc(ordToDoc, values, size);
      for (int ord : spotOrds(size)) {
        m.append(key).append(".spot.").append(ord).append('=');
        byte[] v = values.vectorValue(ord);
        for (int i = 0; i < v.length; i++) {
          if (i > 0) m.append(',');
          m.append(v[i]);
        }
        m.append('\n');
      }
    }
    m.append(key).append(".count=").append(size).append('\n');
    m.append(key).append(".ord_to_doc=").append(ordToDoc).append('\n');

    HnswGraph graph = ((HnswGraphProvider) reader).getGraph(spec.name());
    int numLevels = graph == null ? 0 : graph.numLevels();
    // `HnswGraph.EMPTY` reports one (empty) level; the on-disk `numLevels` for
    // a field with no graph is 0, which is what `size() == 0` distinguishes.
    if (graph != null && graph.size() == 0) {
      numLevels = 0;
    }
    m.append(key).append(".num_levels=").append(numLevels).append('\n');
    if (numLevels == 0) {
      return;
    }
    m.append(key).append(".max_conn=").append(graph.maxConn()).append('\n');
    m.append(key).append(".entry_node=").append(graph.entryNode()).append('\n');
    for (int level = 0; level < numLevels; level++) {
      List<Integer> nodes = new ArrayList<>();
      HnswGraph.NodesIterator it = graph.getSortedNodes(level);
      while (it.hasNext()) {
        nodes.add(it.nextInt());
      }
      long arcTotal = 0;
      long hash = 0;
      StringBuilder sample = new StringBuilder();
      for (int idx = 0; idx < nodes.size(); idx++) {
        int node = nodes.get(idx);
        graph.seek(level, node);
        hash = hash * 31 + node;
        StringBuilder nbrs = new StringBuilder();
        for (int n = graph.nextNeighbor();
            n != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
            n = graph.nextNeighbor()) {
          hash = hash * 31 + n;
          arcTotal++;
          if (nbrs.length() > 0) nbrs.append(',');
          nbrs.append(n);
        }
        // Eight readable samples, spread across the level, so a hash mismatch
        // has something to look at.
        if (nodes.size() <= 8 || idx % Math.max(1, nodes.size() / 8) == 0) {
          if (sample.length() > 0) sample.append(';');
          sample.append(node).append(':').append(nbrs);
        }
      }
      String lk = key + ".level" + level;
      m.append(lk).append(".node_count=").append(nodes.size()).append('\n');
      m.append(lk).append(".arc_total=").append(arcTotal).append('\n');
      m.append(lk).append(".arc_hash=").append(hash).append('\n');
      m.append(lk).append(".sample=").append(sample).append('\n');
      if (level > 0) {
        StringBuilder ns = new StringBuilder();
        for (int i = 0; i < nodes.size(); i++) {
          if (i > 0) ns.append(',');
          ns.append(nodes.get(i));
        }
        m.append(lk).append(".nodes=").append(ns).append('\n');
      }
    }
  }

  static int[] spotOrds(int size) {
    if (size == 0) {
      return new int[0];
    }
    return new int[] {0, 1, size / 2, size - 1};
  }

  static void appendOrdToDoc(StringBuilder sb, KnnVectorValues values, int size)
      throws IOException {
    // Sampled, not exhaustive: a wrong DirectMonotonicReader shows up on any
    // ordinal, and the exact-search rows below check every doc id anyway.
    int step = Math.max(1, size / 32);
    boolean first = true;
    for (int ord = 0; ord < size; ord += step) {
      if (!first) sb.append(';');
      first = false;
      sb.append(ord).append(':').append(values.ordToDoc(ord));
    }
    if (size > 0) {
      sb.append(';').append(size - 1).append(':').append(values.ordToDoc(size - 1));
    }
  }

  static void floatQueries(
      StringBuilder m, String key, FieldSpec spec, IndexSearcher searcher, DirectoryReader dr)
      throws IOException {
    m.append("q.").append(key).append(".count=").append(NUM_QUERIES).append('\n');
    FloatVectorValues values = dr.leaves().get(0).reader().getFloatVectorValues(spec.name());
    for (int q = 0; q < NUM_QUERIES; q++) {
      float[] target = floatVector(spec.dim(), 900_000_007L + q * 31L);
      StringBuilder tv = new StringBuilder();
      for (int i = 0; i < target.length; i++) {
        if (i > 0) tv.append(',');
        tv.append(Float.floatToRawIntBits(target[i]));
      }
      m.append("q.").append(key).append('.').append(q).append(".vec=").append(tv).append('\n');

      TopDocs hnsw = searcher.search(new KnnFloatVectorQuery(spec.name(), target, K), K);
      m.append("q.")
          .append(key)
          .append('.')
          .append(q)
          .append(".hnsw=")
          .append(renderTopDocs(hnsw))
          .append('\n');

      m.append("q.")
          .append(key)
          .append('.')
          .append(q)
          .append(".exact=")
          .append(exactFloat(values, target, spec.sim()))
          .append('\n');
    }
  }

  static void byteQueries(
      StringBuilder m, String key, FieldSpec spec, IndexSearcher searcher, DirectoryReader dr)
      throws IOException {
    m.append("q.").append(key).append(".count=").append(NUM_QUERIES).append('\n');
    ByteVectorValues values = dr.leaves().get(0).reader().getByteVectorValues(spec.name());
    for (int q = 0; q < NUM_QUERIES; q++) {
      byte[] target = byteVector(spec.dim(), 800_000_011L + q * 37L);
      StringBuilder tv = new StringBuilder();
      for (int i = 0; i < target.length; i++) {
        if (i > 0) tv.append(',');
        tv.append(target[i]);
      }
      m.append("q.").append(key).append('.').append(q).append(".vec=").append(tv).append('\n');
      TopDocs hnsw = searcher.search(new KnnByteVectorQuery(spec.name(), target, K), K);
      m.append("q.")
          .append(key)
          .append('.')
          .append(q)
          .append(".hnsw=")
          .append(renderTopDocs(hnsw))
          .append('\n');
      m.append("q.")
          .append(key)
          .append('.')
          .append(q)
          .append(".exact=")
          .append(exactByte(values, target, spec.sim()))
          .append('\n');
    }
  }

  static String renderTopDocs(TopDocs td) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < td.scoreDocs.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(td.scoreDocs[i].doc).append(':').append(Float.floatToRawIntBits(td.scoreDocs[i].score));
    }
    return sb.toString();
  }

  /** Brute force over every stored vector; the recall denominator. */
  static String exactFloat(FloatVectorValues values, float[] target, VectorSimilarityFunction sim)
      throws IOException {
    int size = values.size();
    float[] scores = new float[size];
    int[] docs = new int[size];
    for (int ord = 0; ord < size; ord++) {
      scores[ord] = sim.compare(target, values.vectorValue(ord));
      docs[ord] = values.ordToDoc(ord);
    }
    return topK(scores, docs, size);
  }

  static String exactByte(ByteVectorValues values, byte[] target, VectorSimilarityFunction sim)
      throws IOException {
    int size = values.size();
    float[] scores = new float[size];
    int[] docs = new int[size];
    for (int ord = 0; ord < size; ord++) {
      scores[ord] = sim.compare(target, values.vectorValue(ord));
      docs[ord] = values.ordToDoc(ord);
    }
    return topK(scores, docs, size);
  }

  static String topK(float[] scores, int[] docs, int size) {
    Integer[] order = new Integer[size];
    for (int i = 0; i < size; i++) {
      order[i] = i;
    }
    java.util.Arrays.sort(
        order,
        (a, b) -> {
          int c = Float.compare(scores[b], scores[a]);
          return c != 0 ? c : Integer.compare(docs[a], docs[b]);
        });
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < Math.min(K, size); i++) {
      if (i > 0) sb.append(';');
      sb.append(docs[order[i]]).append(':').append(Float.floatToRawIntBits(scores[order[i]]));
    }
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        for (Path child : (Iterable<Path>) entries::iterator) {
          deleteRecursive(child);
        }
      }
    }
    Files.deleteIfExists(p);
  }

  static String hex(byte[] b) {
    return HexFormat.of().formatHex(b);
  }
}
