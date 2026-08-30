import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/**
 * Generates a **multi-segment** `Lucene99HnswVectorsFormat` fixture and
 * records what real Lucene's {@link KnnFloatVectorQuery}/
 * {@link KnnByteVectorQuery} return over it, so a port's *query-level*
 * fan-out can be checked doc-for-doc rather than only its per-segment search
 * (which {@code GenVectors} already covers).
 *
 * <p>Three things about {@code AbstractKnnVectorQuery.rewrite} are invisible
 * to a single-segment fixture and are the whole point of this one:
 *
 * <ol>
 *   <li><b>Per-leaf {@code k} is pro-rata, not {@code k}.</b>
 *       {@code TopKnnCollectorManager.isOptimistic()} is true, so each leaf is
 *       searched with a collector of
 *       {@code perLeafTopKCalculation(k, leafMaxDoc/indexMaxDoc)}. The four
 *       segments here are deliberately <i>unequal</i> (2000/1000/960/40), so
 *       four different collector sizes are exercised in one query.</li>
 *   <li><b>The optimistic re-entry pass.</b> A leaf whose worst phase-1 hit is
 *       still at or above the merged top-k's worst gets searched again with a
 *       full-{@code k} collector. That can only happen when a leaf's
 *       {@code perLeafTopK} is smaller than {@code k} <i>and</i> its hits are
 *       genuinely competitive, so the 40-document segment's
 *       {@code dense_f32} vectors are a tight cluster near the origin --
 *       close to every query target and therefore competitive for every
 *       query, while {@code perLeafTopK} for that leaf is 5 against
 *       {@code k = 10}.</li>
 *   <li><b>Filtered KNN.</b> Two {@link StringField}s give a <i>selective</i>
 *       filter (fewer accepted documents per leaf than {@code perLeafTopK}, so
 *       every leaf takes {@code exactSearch}) and a <i>permissive</i> one (a
 *       quarter of the index, so the graph is walked with {@code acceptOrds}
 *       and {@code visitedLimit = cost + 1}). The accepted <b>local</b> doc
 *       ids are recorded per leaf, so the port under test is checked on the
 *       KNN policy and not on its own term-query resolution.</li>
 * </ol>
 *
 * <p>The 40-document segment carries no HNSW graph at all (Lucene's
 * {@code shouldCreateGraph} threshold), so the fan-out also has to merge an
 * exact leaf with three approximate ones.
 */
public class GenVectorsMulti {

  /** Deliberately unequal, so four different `perLeafTopK` values apply. */
  static final int[] SEGMENT_SIZES = {2000, 1000, 960, 40};

  static final int NUM_QUERIES = 20;
  static final int K = 10;

  /** Buckets for the selective filter: ~20 documents in the whole index. */
  static final int BUCKETS = 200;
  /** Groups for the permissive filter: a quarter of the index. */
  static final int GROUPS = 4;

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

  static final FieldSpec[] SPECS = {DENSE, SPARSE, MIP, BYTES};

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

  /** A tight cluster near the origin: closer to every query target than a
   * uniformly random vector is, which is what makes the last segment
   * competitive enough for the re-entry pass to trigger. */
  static float[] clusteredVector(int dim, long seed) {
    float[] v = floatVector(dim, seed);
    for (int i = 0; i < dim; i++) {
      v[i] *= 0.04f;
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
    Path out = Path.of(args[0]).resolve("vectors_multi_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    int totalDocs = 0;
    for (int size : SEGMENT_SIZES) {
      totalDocs += size;
    }

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(totalDocs + 1);
      cfg.setRAMBufferSizeMB(1024);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int i = 0;
        for (int s = 0; s < SEGMENT_SIZES.length; s++) {
          boolean clustered = s == SEGMENT_SIZES.length - 1;
          for (int n = 0; n < SEGMENT_SIZES[s]; n++, i++) {
            Document doc = new Document();
            float[] dense =
                clustered
                    ? clusteredVector(DENSE.dim(), 31L + i)
                    : floatVector(DENSE.dim(), i + 1);
            doc.add(new KnnFloatVectorField(DENSE.name(), dense, DENSE.sim()));
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
            doc.add(
                new KnnByteVectorField(
                    BYTES.name(), byteVector(BYTES.dim(), 7_000_019L + i), BYTES.sim()));
            doc.add(new StringField("bucket", "b" + (i % BUCKETS), Field.Store.NO));
            doc.add(new StringField("group", "g" + (i % GROUPS), Field.Store.NO));
            w.addDocument(doc);
          }
          // NoMergePolicy plus a commit per batch is what pins the segment
          // count: one flushed segment per commit, never merged afterwards.
          w.commit();
        }
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENT_SIZES.length) {
        throw new AssertionError("expected " + SEGMENT_SIZES.length + " segments, got " + sis.size());
      }

      m.append("segment_count=").append(sis.size()).append('\n');
      m.append("index_max_doc=").append(totalDocs).append('\n');
      m.append("k=").append(K).append('\n');
      m.append("bucket_count=").append(BUCKETS).append('\n');
      m.append("group_count=").append(GROUPS).append('\n');
      m.append("selective_term=b0\n");
      m.append("permissive_term=g0\n");
      for (int fi = 0; fi < SPECS.length; fi++) {
        m.append("f").append(fi).append(".name=").append(SPECS[fi].name()).append('\n');
        m.append("f").append(fi).append(".dim=").append(SPECS[fi].dim()).append('\n');
        m.append("f").append(fi).append(".encoding=").append(SPECS[fi].encoding().name()).append('\n');
        m.append("f").append(fi).append(".similarity=").append(SPECS[fi].sim().name()).append('\n');
      }

      int docBase = 0;
      for (int s = 0; s < sis.size(); s++) {
        SegmentCommitInfo sci = sis.info(s);
        String name = sci.info.name;
        String vec = null, vemf = null, vem = null, vex = null;
        for (String f : sci.info.files()) {
          if (f.endsWith(".vec")) vec = f;
          if (f.endsWith(".vemf")) vemf = f;
          if (f.endsWith(".vem")) vem = f;
          if (f.endsWith(".vex")) vex = f;
        }
        if (vec == null || vemf == null || vem == null || vex == null) {
          throw new AssertionError("segment " + name + " is missing a vector file: " + sci.info.files());
        }
        String suffix = vec.substring(0, vec.length() - ".vec".length()).substring(name.length());
        if (suffix.startsWith("_")) {
          suffix = suffix.substring(1);
        }
        String p = "s" + s + ".";
        m.append(p).append("segment_name=").append(name).append('\n');
        m.append(p).append("id_hex=").append(hex(sci.info.getId())).append('\n');
        m.append(p).append("max_doc=").append(sci.info.maxDoc()).append('\n');
        m.append(p).append("doc_base=").append(docBase).append('\n');
        m.append(p).append("segment_suffix=").append(suffix).append('\n');
        m.append(p).append("vec_file=").append(vec).append('\n');
        m.append(p).append("vemf_file=").append(vemf).append('\n');
        m.append(p).append("vem_file=").append(vem).append('\n');
        m.append(p).append("vex_file=").append(vex).append('\n');
        docBase += sci.info.maxDoc();
      }

      try (DirectoryReader dr = DirectoryReader.open(dir)) {
        if (dr.leaves().size() != sis.size()) {
          throw new AssertionError("leaf count != segment count");
        }
        IndexSearcher searcher = new IndexSearcher(dr);
        // The accepted **local** doc ids per leaf, straight out of Lucene's
        // own postings, so the Rust side is checked on the KNN policy rather
        // than on its own term-query resolution.
        for (int s = 0; s < dr.leaves().size(); s++) {
          LeafReaderContext ctx = dr.leaves().get(s);
          m.append("s").append(s).append(".selective_docs=")
              .append(localDocs(ctx, "bucket", "b0")).append('\n');
          m.append("s").append(s).append(".permissive_docs=")
              .append(localDocs(ctx, "group", "g0")).append('\n');
        }

        Query selective = new TermQuery(new Term("bucket", "b0"));
        Query permissive = new TermQuery(new Term("group", "g0"));

        // `nearOrigin` pulls every fifth dense query toward the clustered
        // 40-document segment, which is what makes the optimistic re-entry
        // pass fire (that leaf's `perLeafTopK` is 5 against `k = 10`, so a
        // query it dominates needs a second pass to fill `k` from it).
        floatQueries(m, "f0", DENSE, searcher, 900_000_007L, selective, permissive, true);
        floatQueries(m, "f1", SPARSE, searcher, 700_000_009L, null, null, false);
        floatQueries(m, "f2", MIP, searcher, 500_000_003L, null, null, false);
        byteQueries(m, "f3", BYTES, searcher, 800_000_011L);
      }
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }

  static String localDocs(LeafReaderContext ctx, String field, String value) throws IOException {
    Terms terms = ctx.reader().terms(field);
    StringBuilder sb = new StringBuilder();
    if (terms == null) {
      return sb.toString();
    }
    TermsEnum te = terms.iterator();
    if (!te.seekExact(new BytesRef(value))) {
      return sb.toString();
    }
    PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
    for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
      if (sb.length() > 0) sb.append(',');
      sb.append(doc);
    }
    return sb.toString();
  }

  static void floatQueries(
      StringBuilder m,
      String key,
      FieldSpec spec,
      IndexSearcher searcher,
      long seedBase,
      Query selective,
      Query permissive,
      boolean nearOrigin)
      throws IOException {
    m.append("q.").append(key).append(".count=").append(NUM_QUERIES).append('\n');
    for (int q = 0; q < NUM_QUERIES; q++) {
      float[] target = floatVector(spec.dim(), seedBase + q * 31L);
      if (nearOrigin && q % 5 == 3) {
        for (int i = 0; i < target.length; i++) {
          target[i] *= 0.15f;
        }
      }
      StringBuilder tv = new StringBuilder();
      for (int i = 0; i < target.length; i++) {
        if (i > 0) tv.append(',');
        tv.append(Float.floatToRawIntBits(target[i]));
      }
      String qk = "q." + key + "." + q;
      m.append(qk).append(".vec=").append(tv).append('\n');
      m.append(qk)
          .append(".hnsw=")
          .append(render(searcher.search(new KnnFloatVectorQuery(spec.name(), target, K), K)))
          .append('\n');
      if (selective != null) {
        m.append(qk)
            .append(".selective=")
            .append(
                render(
                    searcher.search(
                        new KnnFloatVectorQuery(spec.name(), target, K, selective), K)))
            .append('\n');
        m.append(qk)
            .append(".permissive=")
            .append(
                render(
                    searcher.search(
                        new KnnFloatVectorQuery(spec.name(), target, K, permissive), K)))
            .append('\n');
      }
    }
  }

  static void byteQueries(
      StringBuilder m, String key, FieldSpec spec, IndexSearcher searcher, long seedBase)
      throws IOException {
    m.append("q.").append(key).append(".count=").append(NUM_QUERIES).append('\n');
    for (int q = 0; q < NUM_QUERIES; q++) {
      byte[] target = byteVector(spec.dim(), seedBase + q * 37L);
      StringBuilder tv = new StringBuilder();
      for (int i = 0; i < target.length; i++) {
        if (i > 0) tv.append(',');
        tv.append(target[i]);
      }
      String qk = "q." + key + "." + q;
      m.append(qk).append(".vec=").append(tv).append('\n');
      m.append(qk)
          .append(".hnsw=")
          .append(render(searcher.search(new KnnByteVectorQuery(spec.name(), target, K), K)))
          .append('\n');
    }
  }

  static String render(TopDocs td) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < td.scoreDocs.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(td.scoreDocs[i].doc)
          .append(':')
          .append(Float.floatToRawIntBits(td.scoreDocs[i].score));
    }
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    if (Files.isDirectory(p)) {
      try (var entries = Files.list(p)) {
        List<Path> children = new ArrayList<>();
        entries.forEach(children::add);
        for (Path child : children) {
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
