import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A <b>single-segment</b> `Lucene99HnswVectorsFormat` fixture that also
 * carries a term dictionary, so a filtered KNN query can be expressed as a
 * real {@code TermQuery} and checked end to end -- including over the C ABI,
 * where the filter arrives as a clause array and has to be resolved through
 * this port's own postings reader before it can become an accept set.
 *
 * <p>Why single-segment matters, and why {@link GenVectorsMulti}'s index
 * cannot stand in. {@code lucene-ffi} opens <b>one</b> segment's vector files
 * per handle, so its ground truth has to be what
 * {@code IndexSearcher.search(query, k)} returns over a one-leaf index. On a
 * one-leaf index {@code leafProportion == 1}, so
 * {@code perLeafTopKCalculation(k, 1) == k} and
 * {@code perLeafResults.size() > 1} is false -- the collector is exactly
 * {@code k} and there is no re-entry pass. Running the same query against one
 * leaf of a four-leaf index is a different search (that leaf's collector is
 * pro-rata sized), so the multi-segment fixture's recorded results are not
 * usable as single-segment ground truth.
 *
 * <p>Both of Java's filtered branches are reached, by construction:
 *
 * <ul>
 *   <li>{@code bucket:b0} accepts {@value #BUCKETS}<sup>-1</sup> of the
 *       index -- 6 documents against {@code k = 10} -- so
 *       {@code cost <= perLeafTopK} and {@code getLeafResults} short-circuits
 *       to {@code exactSearch}.</li>
 *   <li>{@code group:g0} accepts a quarter of it, so the graph is walked with
 *       {@code acceptOrds} and {@code visitedLimit = cost + 1}.</li>
 * </ul>
 *
 * <p>The accepted <b>local</b> doc ids are recorded straight out of Lucene's
 * own postings, so a test that wants to check only the KNN policy can supply
 * them directly, while the FFI test resolves the same term through this
 * port's block-tree reader and must arrive at the same set.
 *
 * <p>1200 documents is above {@code shouldCreateGraph}'s
 * {@code n > ln(n) * 100} (~660), so the field really does carry a graph.
 * Vector generation is {@link GenVectorsMulti}'s, reused rather than copied.
 */
public class GenVectorsFiltered {

  static final int NUM_DOCS = 1200;
  static final int NUM_QUERIES = 20;
  static final int K = 10;

  /** ~6 accepted documents: below `perLeafTopK`, so `exactSearch`. */
  static final int BUCKETS = 200;
  /** A quarter of the index: the filtered graph walk. */
  static final int GROUPS = 4;

  static final String FLOAT_FIELD = "dense_f32";
  static final int FLOAT_DIM = 16;
  static final VectorSimilarityFunction FLOAT_SIM = VectorSimilarityFunction.EUCLIDEAN;

  static final String BYTE_FIELD = "byte_dot";
  static final int BYTE_DIM = 8;
  static final VectorSimilarityFunction BYTE_SIM = VectorSimilarityFunction.DOT_PRODUCT;

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("vectors_filter_index");
    if (Files.exists(out)) {
      GenVectorsMulti.deleteRecursive(out);
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();

    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(NUM_DOCS + 1);
      cfg.setRAMBufferSizeMB(1024);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          Document doc = new Document();
          doc.add(
              new KnnFloatVectorField(
                  FLOAT_FIELD, GenVectorsMulti.floatVector(FLOAT_DIM, i + 1), FLOAT_SIM));
          doc.add(
              new KnnByteVectorField(
                  BYTE_FIELD, GenVectorsMulti.byteVector(BYTE_DIM, 7_000_019L + i), BYTE_SIM));
          doc.add(new StringField("bucket", "b" + (i % BUCKETS), Field.Store.NO));
          doc.add(new StringField("group", "g" + (i % GROUPS), Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected exactly one segment, got " + sis.size());
      }
      SegmentCommitInfo sci = sis.info(0);
      String name = sci.info.name;
      String vec = null, vemf = null, vem = null, vex = null;
      String tim = null, tip = null, tmd = null, docFile = null;
      for (String f : sci.info.files()) {
        if (f.endsWith(".vec")) vec = f;
        if (f.endsWith(".vemf")) vemf = f;
        if (f.endsWith(".vem")) vem = f;
        if (f.endsWith(".vex")) vex = f;
        if (f.endsWith(".tim")) tim = f;
        if (f.endsWith(".tip")) tip = f;
        if (f.endsWith(".tmd")) tmd = f;
        if (f.endsWith(".doc")) docFile = f;
      }
      if (vec == null || vemf == null || vem == null || vex == null) {
        throw new AssertionError("missing a vector file: " + sci.info.files());
      }
      if (tim == null || tip == null || tmd == null || docFile == null) {
        throw new AssertionError("missing a postings file: " + sci.info.files());
      }
      String vectorSuffix =
          vec.substring(0, vec.length() - ".vec".length()).substring(name.length());
      if (vectorSuffix.startsWith("_")) {
        vectorSuffix = vectorSuffix.substring(1);
      }
      String postingsSuffix = tim.substring(0, tim.length() - ".tim".length()).substring(name.length());
      if (postingsSuffix.startsWith("_")) {
        postingsSuffix = postingsSuffix.substring(1);
      }

      m.append("segment_name=").append(name).append('\n');
      m.append("id_hex=").append(GenVectorsMulti.hex(sci.info.getId())).append('\n');
      m.append("max_doc=").append(sci.info.maxDoc()).append('\n');
      m.append("k=").append(K).append('\n');
      m.append("bucket_count=").append(BUCKETS).append('\n');
      m.append("group_count=").append(GROUPS).append('\n');
      m.append("selective_field=bucket\n");
      m.append("selective_term=b0\n");
      m.append("permissive_field=group\n");
      m.append("permissive_term=g0\n");
      m.append("segment_suffix=").append(vectorSuffix).append('\n');
      m.append("postings_suffix=").append(postingsSuffix).append('\n');
      m.append("vec_file=").append(vec).append('\n');
      m.append("vemf_file=").append(vemf).append('\n');
      m.append("vem_file=").append(vem).append('\n');
      m.append("vex_file=").append(vex).append('\n');
      m.append("tim_file=").append(tim).append('\n');
      m.append("tip_file=").append(tip).append('\n');
      m.append("tmd_file=").append(tmd).append('\n');
      m.append("doc_file=").append(docFile).append('\n');
      m.append("f0.name=").append(FLOAT_FIELD).append('\n');
      m.append("f0.dim=").append(FLOAT_DIM).append('\n');
      m.append("f0.encoding=FLOAT32\n");
      m.append("f0.similarity=").append(FLOAT_SIM.name()).append('\n');
      m.append("f1.name=").append(BYTE_FIELD).append('\n');
      m.append("f1.dim=").append(BYTE_DIM).append('\n');
      m.append("f1.encoding=BYTE\n");
      m.append("f1.similarity=").append(BYTE_SIM.name()).append('\n');

      try (DirectoryReader dr = DirectoryReader.open(dir)) {
        if (dr.leaves().size() != 1) {
          throw new AssertionError("expected one leaf, got " + dr.leaves().size());
        }
        m.append("selective_docs=")
            .append(GenVectorsMulti.localDocs(dr.leaves().get(0), "bucket", "b0"))
            .append('\n');
        m.append("permissive_docs=")
            .append(GenVectorsMulti.localDocs(dr.leaves().get(0), "group", "g0"))
            .append('\n');

        IndexSearcher searcher = new IndexSearcher(dr);
        Query selective = new TermQuery(new Term("bucket", "b0"));
        Query permissive = new TermQuery(new Term("group", "g0"));

        m.append("q.f0.count=").append(NUM_QUERIES).append('\n');
        for (int q = 0; q < NUM_QUERIES; q++) {
          float[] target = GenVectorsMulti.floatVector(FLOAT_DIM, 400_000_009L + q * 29L);
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
                      searcher.search(new KnnFloatVectorQuery(FLOAT_FIELD, target, K), K)))
              .append('\n');
          m.append(qk)
              .append(".selective=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(
                          new KnnFloatVectorQuery(FLOAT_FIELD, target, K, selective), K)))
              .append('\n');
          m.append(qk)
              .append(".permissive=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(
                          new KnnFloatVectorQuery(FLOAT_FIELD, target, K, permissive), K)))
              .append('\n');
        }

        m.append("q.f1.count=").append(NUM_QUERIES).append('\n');
        for (int q = 0; q < NUM_QUERIES; q++) {
          byte[] target = GenVectorsMulti.byteVector(BYTE_DIM, 600_000_017L + q * 43L);
          StringBuilder tv = new StringBuilder();
          for (int i = 0; i < target.length; i++) {
            if (i > 0) tv.append(',');
            tv.append(target[i]);
          }
          String qk = "q.f1." + q;
          m.append(qk).append(".vec=").append(tv).append('\n');
          m.append(qk)
              .append(".hnsw=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(new KnnByteVectorQuery(BYTE_FIELD, target, K), K)))
              .append('\n');
          m.append(qk)
              .append(".selective=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(
                          new KnnByteVectorQuery(BYTE_FIELD, target, K, selective), K)))
              .append('\n');
          m.append(qk)
              .append(".permissive=")
              .append(
                  GenVectorsMulti.render(
                      searcher.search(
                          new KnnByteVectorQuery(BYTE_FIELD, target, K, permissive), K)))
              .append('\n');
        }
      }
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }
}
