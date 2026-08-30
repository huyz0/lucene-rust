import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A multi-segment `Lucene99HnswVectorsFormat` fixture built for exactly one
 * thing {@link GenVectorsMulti}'s cannot reach: the optimistic re-entry pass
 * firing on a leaf that <b>has an HNSW graph</b>, so that
 * {@code ReentrantKnnCollectorManager}'s {@code KnnSearchStrategy.Seeded} and
 * therefore {@code SeededHnswGraphSearcher} are actually exercised.
 *
 * <p>Why a second index rather than more queries over the first one. Two
 * constraints have to hold at once, and {@code GenVectorsMulti}'s segment
 * sizes cannot satisfy both:
 *
 * <ol>
 *   <li><b>The leaf must be re-enterable.</b>
 *       {@code AbstractKnnVectorQuery.rewrite} re-enters a leaf whose worst
 *       phase-1 hit is at or above the merged top-{@code k}'s worst. With
 *       {@code perLeafTopK >= k} that needs a score tie, so in practice it
 *       needs {@code perLeafTopK < k}: {@code k*p + 16*sqrt(k*p*(1-p)) < k}.
 *       At {@code k = 10} that means {@code p < 0.039}, i.e. a leaf under
 *       ~156 documents of a 4000-document index.</li>
 *   <li><b>The leaf must carry a graph.</b>
 *       {@code Lucene99HnswVectorsWriter.shouldCreateGraph} needs
 *       {@code n > ln(n) * 100}, i.e. about 660 vectors.</li>
 * </ol>
 *
 * <p>The two are only compatible at a larger {@code k}: at {@code k = 100} a
 * leaf holding a quarter of the index has {@code perLeafTopK = 93 < 100} and
 * comfortably clears the graph threshold. So this index is deliberately
 * <b>1400 / 700 / 700 / 40</b> and is queried at {@code k = 100}, and the
 * <b>second</b> 700-document segment holds a tight cluster around a fixed
 * point that every query target sits next to -- which is what makes its 93
 * phase-1 hits dominate the merged top 100 and its re-entry condition true.
 *
 * <p>{@code GenVectorsMulti}'s index reaches the re-entry pass too, but only
 * on its 40-document segment, which is below {@code shouldCreateGraph} and so
 * takes the exhaustive branch -- where Java ignores the search strategy
 * entirely. That is why its 80 recorded queries agree with an unseeded port.
 *
 * <p>The 40-document leaf here is kept for the same reason it exists there:
 * the fan-out has to merge one graphless leaf with three graph-bearing ones.
 * Vector generation is {@link GenVectorsMulti}'s, reused rather than copied.
 */
public class GenVectorsSeeded {

  /**
   * 1400 / 700 / 700 / 40. At `k = 100` the four `perLeafTopK` values are
   * 129, 93, 93 and 20, so two graph-bearing leaves sit below `k`.
   */
  static final int[] SEGMENT_SIZES = {1400, 700, 700, 40};

  /** The clustered one: segment index 1. */
  static final int CLUSTERED_SEGMENT = 1;

  static final int NUM_QUERIES = 20;
  /** Large enough that `perLeafTopK < k` for a quarter-sized leaf. */
  static final int K = 100;
  /** Recorded too, as the control: at `k = 10` no leaf is re-enterable. */
  static final int SMALL_K = 10;

  static final String FIELD = "dense_f32";
  static final int DIM = 16;
  static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;

  /**
   * The cluster centre. Query targets are pulled onto it so the clustered
   * leaf owns the top of every result list.
   */
  static float[] centre() {
    float[] c = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      c[i] = 0.25f - 0.5f * (i % 3);
    }
    return c;
  }

  /** A tight ball around {@link #centre()}. */
  static float[] clustered(long seed) {
    float[] v = GenVectorsMulti.floatVector(DIM, seed);
    float[] c = centre();
    for (int i = 0; i < DIM; i++) {
      v[i] = c[i] + v[i] * 0.02f;
    }
    return v;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("vectors_seeded_index");
    if (Files.exists(out)) {
      GenVectorsMulti.deleteRecursive(out);
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
          for (int n = 0; n < SEGMENT_SIZES[s]; n++, i++) {
            Document doc = new Document();
            float[] v =
                s == CLUSTERED_SEGMENT
                    ? clustered(97L + i)
                    : GenVectorsMulti.floatVector(DIM, i + 1);
            doc.add(new KnnFloatVectorField(FIELD, v, SIM));
            w.addDocument(doc);
          }
          w.commit();
        }
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENT_SIZES.length) {
        throw new AssertionError(
            "expected " + SEGMENT_SIZES.length + " segments, got " + sis.size());
      }

      m.append("segment_count=").append(sis.size()).append('\n');
      m.append("index_max_doc=").append(totalDocs).append('\n');
      m.append("k=").append(K).append('\n');
      m.append("small_k=").append(SMALL_K).append('\n');
      m.append("clustered_segment=").append(CLUSTERED_SEGMENT).append('\n');
      m.append("f0.name=").append(FIELD).append('\n');
      m.append("f0.dim=").append(DIM).append('\n');
      m.append("f0.encoding=FLOAT32\n");
      m.append("f0.similarity=").append(SIM.name()).append('\n');

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
          throw new AssertionError(
              "segment " + name + " is missing a vector file: " + sci.info.files());
        }
        String suffix = vec.substring(0, vec.length() - ".vec".length()).substring(name.length());
        if (suffix.startsWith("_")) {
          suffix = suffix.substring(1);
        }
        String p = "s" + s + ".";
        m.append(p).append("segment_name=").append(name).append('\n');
        m.append(p).append("id_hex=").append(GenVectorsMulti.hex(sci.info.getId())).append('\n');
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
        IndexSearcher searcher = new IndexSearcher(dr);
        m.append("q.f0.count=").append(NUM_QUERIES).append('\n');
        for (int q = 0; q < NUM_QUERIES; q++) {
          // Every target sits inside the cluster's neighbourhood, a little
          // further out than the cluster's own radius, so the clustered leaf
          // owns the top of the list without the ordering being degenerate.
          float[] target = GenVectorsMulti.floatVector(DIM, 300_000_007L + q * 41L);
          float[] c = centre();
          for (int i = 0; i < DIM; i++) {
            target[i] = c[i] + target[i] * 0.06f;
          }
          StringBuilder tv = new StringBuilder();
          for (int i = 0; i < target.length; i++) {
            if (i > 0) tv.append(',');
            tv.append(Float.floatToRawIntBits(target[i]));
          }
          String qk = "q.f0." + q;
          m.append(qk).append(".vec=").append(tv).append('\n');
          m.append(qk)
              .append(".hnsw=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(new KnnFloatVectorQuery(FIELD, target, K), K)))
              .append('\n');
          m.append(qk)
              .append(".hnsw_small_k=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(new KnnFloatVectorQuery(FIELD, target, SMALL_K), SMALL_K)))
              .append('\n');
        }
        recordReentryShape(m, dr);
      }
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }

  /**
   * The per-leaf `perLeafTopK` values Lucene will use at {@link #K}, recorded
   * so the Rust test can assert the fixture still has the shape it was built
   * for -- a fixture that silently stops reaching the branch it exists for
   * proves nothing, and these four numbers are the branch's precondition.
   *
   * <p><b>These are re-derived, not ground truth.</b>
   * {@code AbstractKnnVectorQuery.perLeafTopKCalculation} is private, so the
   * formula is copied here rather than called. A Rust test comparing against
   * these numbers therefore checks its own formula against a hand-copy of the
   * same formula and cannot catch a shared misreading of Java. What carries
   * real weight is the *result*: at {@link #K} every recorded query takes
   * more hits from the clustered leaf than its {@code perLeafTopK}, which one
   * pass cannot produce. Treat this row as a tripwire on the fixture's shape,
   * not as evidence about the formula.
   */
  static void recordReentryShape(StringBuilder m, DirectoryReader dr) throws IOException {
    int indexMaxDoc = dr.maxDoc();
    for (int s = 0; s < dr.leaves().size(); s++) {
      int leafMaxDoc = dr.leaves().get(s).reader().maxDoc();
      float p = leafMaxDoc / (float) indexMaxDoc;
      int perLeafTopK = (int) Math.max(1, K * p + 16 * Math.sqrt(K * p * (1 - p)));
      m.append("s").append(s).append(".per_leaf_top_k=").append(perLeafTopK).append('\n');
    }
  }
}
