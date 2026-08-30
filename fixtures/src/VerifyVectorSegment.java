import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for a whole index whose
 * documents carry <b>vector fields</b>, written by a real
 * {@code IndexWriter.add_document_with_vectors} session (see
 * {@code crates/lucene-index/examples/write_vector_segment_fixture.rs}).
 *
 * <p>{@code VerifyVectors} already checks the four vector files themselves
 * against a hand-built {@code SegmentInfo}. What it cannot see is everything
 * that binds them into a segment, and each of those fails <em>silently</em>:
 *
 * <ul>
 *   <li>{@code PerFieldKnnVectorsFormat.format}/{@code .suffix} missing from
 *       the {@code .fnm}, or files not written under the suffixed segment name
 *       -- the field reads back with no vectors at all rather than as an
 *       error, exactly the shape c4 found for postings and doc values.</li>
 *   <li>The four files missing from {@code SegmentInfo.files} -- invisible to
 *       a reader, fatal to {@code IndexFileDeleter} and {@code CheckIndex}.</li>
 *   <li>A {@code .fnm} claiming {@code vectorDimension > 0} for a field the
 *       flush wrote no vectors for: {@code FieldInfo.hasVectorValues()} is then
 *       true -- which is what {@code IncrementalHnswGraphMerger} and
 *       {@code CheckIndex} key off -- while {@code PerFieldKnnVectorsFormat}
 *       registers no reader for it, so the field reads back as vector-capable
 *       and yields nothing. Lucene raises no error for that combination
 *       (measured), which is why the fixture declares one such field on purpose
 *       and this verifier asserts on the dimension directly.</li>
 * </ul>
 *
 * <p>Checked here, in order of how specific the diagnosis is:
 *
 * <ol>
 *   <li>Every ordinal's components, as an order-sensitive hash over raw bits,
 *       and every ordinal's document id -- against values the Rust fixture
 *       derived from its own generator, not read back out of the files it
 *       wrote.</li>
 *   <li>The declared-but-never-written vector field is absent.</li>
 *   <li>For every query: Lucene's own brute-force top-k over the vectors it
 *       just read, compared against what {@code KnnFloatVectorQuery} /
 *       {@code KnnByteVectorQuery} return over the Rust-built graph (recall,
 *       with a floor). The small field, which has no graph, must match
 *       <em>exactly</em> -- Lucene falls back to an exhaustive scan there, so
 *       anything short of exact is a flat-store bug, not a graph one.</li>
 *   <li>Postings and stored fields still work alongside.</li>
 *   <li>{@code CheckIndex} at {@code MIN_LEVEL_FOR_SLOW_CHECKS}, which runs
 *       {@code testVectors} and {@code testHnswGraphs} over every field.</li>
 * </ol>
 *
 * <p>Usage: {@code java VerifyVectorSegment <index-dir>}.
 */
public class VerifyVectorSegment {

  /** Recall floor for Lucene's KNN query over the Rust-built graph. */
  static final double MIN_RECALL = 0.80;

  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    Map<String, String> m = readManifest(path.resolve("manifest.properties"));
    int numDocs = Integer.parseInt(m.get("num_docs"));
    int k = Integer.parseInt(m.get("k"));
    int fieldCount = Integer.parseInt(m.get("field_count"));
    int failures = 0;

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != numDocs || reader.numDocs() != numDocs) {
        System.out.println(
            "MISMATCH doc count: maxDoc=" + reader.maxDoc() + " numDocs=" + reader.numDocs());
        failures++;
      }
      if (reader.leaves().size() != 1) {
        System.out.println("MISMATCH expected a single segment, got " + reader.leaves().size());
        failures++;
      }
      LeafReader leaf = reader.leaves().get(0).reader();
      IndexSearcher searcher = new IndexSearcher(reader);

      // A field whose FieldInfo declared a dimension but which no document
      // ever carried must read back with no vectors and, crucially, with
      // hasVectorValues() false -- otherwise the segment would not have opened
      // at all, and this line would never be reached.
      FieldInfo never = leaf.getFieldInfos().fieldInfo("never_written");
      if (never == null) {
        System.out.println("MISMATCH field \"never_written\" missing from the .fnm");
        failures++;
      } else if (never.hasVectorValues() || never.getVectorDimension() != 0) {
        System.out.println(
            "MISMATCH field \"never_written\" claims vectorDimension "
                + never.getVectorDimension()
                + "; a field no document carried must record 0");
        failures++;
      }
      if (leaf.getFloatVectorValues("never_written") != null) {
        System.out.println("MISMATCH field \"never_written\" has vector values");
        failures++;
      }

      for (int f = 0; f < fieldCount; f++) {
        failures += checkField(leaf, searcher, m, "f" + f, k);
      }

      // The rest of the segment must still work: vectors are written after the
      // stored fields and postings and patch the same `.si`.
      int matched = searcher.count(new TermQuery(new Term("body", "shared")));
      if (matched != numDocs) {
        System.out.println("MISMATCH body:shared matched " + matched + ", expected " + numDocs);
        failures++;
      }
      String id = reader.storedFields().document(numDocs - 1).get("id");
      if (!("doc" + (numDocs - 1)).equals(id)) {
        System.out.println("MISMATCH last document's id=" + id);
        failures++;
      }
    }

    // CheckIndex last: broadest, least specific. At this level it runs
    // testVectors (every ordinal decoded, ord<->doc cross-checked) and
    // testHnswGraphs (every level, every node's neighbours) itself.
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("MISMATCH CheckIndex reported the index unclean:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        failures++;
      }
    }

    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Vector segment verified against real Lucene. PASS");
  }

  static int checkField(
      LeafReader leaf, IndexSearcher searcher, Map<String, String> m, String key, int k)
      throws IOException {
    String name = m.get(key + ".name");
    int dim = Integer.parseInt(m.get(key + ".dim"));
    int count = Integer.parseInt(m.get(key + ".count"));
    int[] docs = parseInts(m.get(key + ".docs"));
    long wantHash = Long.parseLong(m.get(key + ".value_hash"));
    VectorEncoding encoding = VectorEncoding.valueOf(m.get(key + ".encoding"));
    VectorSimilarityFunction sim = VectorSimilarityFunction.valueOf(m.get(key + ".similarity"));
    int failures = 0;

    FieldInfo fi = leaf.getFieldInfos().fieldInfo(name);
    if (fi == null || !fi.hasVectorValues()) {
      System.out.println(
          name
              + ": MISMATCH no vector values -- typically a missing "
              + "PerFieldKnnVectorsFormat.format/.suffix attribute in .fnm, or the "
              + "vector files not written under the suffixed segment name");
      return 1;
    }
    failures += expect(name, "dimension", fi.getVectorDimension(), dim);
    if (fi.getVectorEncoding() != encoding) {
      System.out.println(name + ": MISMATCH encoding " + fi.getVectorEncoding());
      failures++;
    }
    if (fi.getVectorSimilarityFunction() != sim) {
      System.out.println(name + ": MISMATCH similarity " + fi.getVectorSimilarityFunction());
      failures++;
    }

    long gotHash = 0;
    if (encoding == VectorEncoding.FLOAT32) {
      FloatVectorValues values = leaf.getFloatVectorValues(name);
      if (values == null) {
        System.out.println(name + ": MISMATCH getFloatVectorValues returned null");
        return failures + 1;
      }
      failures += expect(name, "size", values.size(), count);
      for (int ord = 0; ord < Math.min(values.size(), count); ord++) {
        failures += expect(name, "ordToDoc(" + ord + ")", values.ordToDoc(ord), docs[ord]);
        for (float c : values.vectorValue(ord)) {
          gotHash = gotHash * 31 + Float.floatToRawIntBits(c);
        }
      }
    } else {
      ByteVectorValues values = leaf.getByteVectorValues(name);
      if (values == null) {
        System.out.println(name + ": MISMATCH getByteVectorValues returned null");
        return failures + 1;
      }
      failures += expect(name, "size", values.size(), count);
      for (int ord = 0; ord < Math.min(values.size(), count); ord++) {
        failures += expect(name, "ordToDoc(" + ord + ")", values.ordToDoc(ord), docs[ord]);
        for (byte b : values.vectorValue(ord)) {
          gotHash = gotHash * 31 + b;
        }
      }
    }
    failures += expect(name, "value hash", gotHash, wantHash);
    if (failures > 0) {
      return failures;
    }

    return failures + checkSearch(leaf, searcher, m, key, name, dim, encoding, sim, k, count);
  }

  /**
   * Lucene's own brute-force top-k over the vectors it just read, against what
   * {@link KnnFloatVectorQuery}/{@link KnnByteVectorQuery} return over the
   * Rust-built graph. Both sides are Lucene's, so a disagreement is about the
   * graph, not about float arithmetic differing across languages.
   */
  static int checkSearch(
      LeafReader leaf,
      IndexSearcher searcher,
      Map<String, String> m,
      String key,
      String name,
      int dim,
      VectorEncoding encoding,
      VectorSimilarityFunction sim,
      int k,
      int count)
      throws IOException {
    int queries = Integer.parseInt(m.get("q." + key + ".count"));
    // Lucene builds no graph below HNSW_GRAPH_THRESHOLD, so its query falls
    // back to an exhaustive scan and must be exact. Derived from Lucene's own
    // rule (Lucene99HnswVectorsWriter.shouldCreateGraph + HnswGraphSearcher's
    // expectedVisitedNodes) rather than from its crossing point, so that a
    // change to either does not silently downgrade the exact assertion below
    // into the weaker recall one.
    int expectedVisitedNodes = (int) (Math.log(count) * Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD);
    boolean hasGraph = count > expectedVisitedNodes && expectedVisitedNodes > 0;
    int hits = 0;
    int total = 0;
    for (int q = 0; q < queries; q++) {
      int[] raw = parseInts(m.get("q." + key + "." + q + ".vec"));
      Set<Integer> exact;
      Query query;
      if (encoding == VectorEncoding.FLOAT32) {
        float[] target = new float[dim];
        for (int i = 0; i < dim; i++) {
          target[i] = Float.intBitsToFloat(raw[i]);
        }
        exact = bruteForceFloat(leaf, name, sim, target, k);
        query = new KnnFloatVectorQuery(name, target, k);
      } else {
        byte[] target = new byte[dim];
        for (int i = 0; i < dim; i++) {
          target[i] = (byte) raw[i];
        }
        exact = bruteForceByte(leaf, name, sim, target, k);
        query = new KnnByteVectorQuery(name, target, k);
      }
      ScoreDoc[] got = searcher.search(query, k).scoreDocs;
      Set<Integer> gotDocs = new HashSet<>();
      for (ScoreDoc sd : got) {
        gotDocs.add(sd.doc);
        if (exact.contains(sd.doc)) {
          hits++;
        }
      }
      total += exact.size();
      if (!hasGraph && !gotDocs.equals(exact)) {
        System.out.println(
            name
                + ": MISMATCH query "
                + q
                + " over a graphless field must be exact; got "
                + gotDocs
                + " want "
                + exact);
        return 1;
      }
    }
    double recall = total == 0 ? 1.0 : (double) hits / total;
    System.out.println(
        String.format(
            Locale.ROOT,
            "  %-12s KnnVectorQuery over the Rust-written segment: recall@%d = %.4f over %d queries%s",
            name, k, recall, queries, hasGraph ? "" : " (exhaustive, exact)"));
    if (recall < MIN_RECALL) {
      System.out.println(name + ": MISMATCH recall " + recall + " < " + MIN_RECALL);
      return 1;
    }
    return 0;
  }

  static Set<Integer> bruteForceFloat(
      LeafReader leaf, String name, VectorSimilarityFunction sim, float[] target, int k)
      throws IOException {
    FloatVectorValues values = leaf.getFloatVectorValues(name);
    List<ScoreDoc> all = new ArrayList<>();
    KnnVectorValues.DocIndexIterator it = values.iterator();
    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
      all.add(new ScoreDoc(doc, sim.compare(target, values.vectorValue(it.index()))));
    }
    return topK(all, k);
  }

  static Set<Integer> bruteForceByte(
      LeafReader leaf, String name, VectorSimilarityFunction sim, byte[] target, int k)
      throws IOException {
    ByteVectorValues values = leaf.getByteVectorValues(name);
    List<ScoreDoc> all = new ArrayList<>();
    KnnVectorValues.DocIndexIterator it = values.iterator();
    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
      all.add(new ScoreDoc(doc, sim.compare(target, values.vectorValue(it.index()))));
    }
    return topK(all, k);
  }

  static Set<Integer> topK(List<ScoreDoc> all, int k) {
    all.sort(
        Comparator.comparingDouble((ScoreDoc sd) -> -sd.score).thenComparingInt(sd -> sd.doc));
    Set<Integer> out = new HashSet<>();
    for (int i = 0; i < Math.min(k, all.size()); i++) {
      out.add(all.get(i).doc);
    }
    return out;
  }

  static int expect(String field, String what, long got, long want) {
    if (got != want) {
      System.out.println(field + ": MISMATCH " + what + " " + got + " != " + want);
      return 1;
    }
    return 0;
  }

  static int[] parseInts(String spec) {
    if (spec == null || spec.isEmpty()) {
      return new int[0];
    }
    String[] parts = spec.split(",");
    int[] out = new int[parts.length];
    for (int i = 0; i < parts.length; i++) {
      out[i] = Integer.parseInt(parts[i]);
    }
    return out;
  }

  static Map<String, String> readManifest(Path path) throws IOException {
    Map<String, String> map = new HashMap<>();
    for (String line : Files.readAllLines(path)) {
      int eq = line.indexOf('=');
      if (eq > 0) {
        map.put(line.substring(0, eq), line.substring(eq + 1));
      }
    }
    return map;
  }
}
