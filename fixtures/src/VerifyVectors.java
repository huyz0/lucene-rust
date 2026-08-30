import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens the
 * {@code .vec}/{@code .vemf} flat store and the {@code .vem}/{@code .vex} HNSW
 * graph written by this port's {@code vectors::write_flat_vectors} and
 * {@code hnsw_vectors::write_hnsw_vectors} (see
 * {@code crates/lucene-codecs/examples/write_vectors_fixture.rs}) through real
 * Lucene's own {@link Lucene99HnswVectorsFormat}, with a hand-built
 * {@link SegmentInfo} (as {@code VerifyPoints.java} does) but with the
 * {@link FieldInfos} read back out of a Rust-written {@code .fnm} -- see (1)
 * below for why that difference matters.
 *
 * <p>Five things are checked, because a vector segment can be wrong in five
 * independent ways:
 *
 * <ol>
 *   <li><b>The `.fnm`.</b> The `FieldInfos` are read back through real
 *       {@link Lucene94FieldInfosFormat} rather than hand-built, so Lucene's own
 *       cross-checks between `.vemf` and `.fnm` (dimension and similarity must
 *       agree) run against two files this port wrote independently.</li>
 *   <li><b>The flat store.</b> Every ordinal's components are hashed
 *       (order-sensitive, over raw float bits, so float summation order cannot
 *       hide a difference) and compared with the hash Rust recorded; the
 *       ordinal-to-document mapping is compared for every ordinal, which is
 *       where a mis-written {@code IndexedDISI} bitset or
 *       {@code DirectMonotonicWriter} block shows up.</li>
 *   <li><b>The graph.</b> Level count, entry node, and per level the node
 *       count, total arc count and an order-sensitive hash over every node's
 *       neighbour list. This is what proves the {@code .vex} node offsets and
 *       group-varint neighbour deltas are what Lucene expects, rather than
 *       merely what this port's own reader expects.</li>
 *   <li><b>The no-graph field.</b> One field is too small for Lucene to build
 *       a graph over ({@code numLevels == 0}, zero-length {@code .vex}
 *       region), and Lucene must accept it and fall back to an exhaustive
 *       scan rather than reading a graph that is not there.</li>
 *   <li><b>Search.</b> Lucene runs its own {@link TopKnnCollector} search over
 *       the Rust-built graph and the recall of the result is measured against
 *       the exact top-k Rust recorded. A graph that decodes cleanly but is
 *       badly *built* passes (1)-(3) and fails here.</li>
 * </ol>
 *
 * <p>Usage: {@code java VerifyVectors <fixture-dir>}.
 */
public class VerifyVectors {

  /** Recall floor for Lucene's search over the Rust-built graph. */
  static final double MIN_RECALL = 0.80;

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));
    int failures = 0;

    try (Directory directory = FSDirectory.open(dir)) {
      String segmentName = manifest.get("segment_name");
      byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
      int maxDoc = Integer.parseInt(manifest.get("max_doc"));
      int k = Integer.parseInt(manifest.get("k"));
      int fieldCount = Integer.parseInt(manifest.get("field_count"));

      SegmentInfo si =
          new SegmentInfo(
              directory,
              org.apache.lucene.util.Version.LATEST,
              org.apache.lucene.util.Version.LATEST,
              segmentName,
              maxDoc,
              false,
              false,
              null,
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      // Read the FieldInfos back through real Lucene rather than hand-building
      // them: that is what puts Lucene's own `.vemf`-vs-`.fnm` cross-checks
      // (`FieldEntry`'s "Inconsistent vector similarity function" and
      // "Inconsistent vector dimension") in front of two files this port wrote
      // independently. A hand-built FieldInfos cannot see a disagreement
      // between them -- the exact blind spot that let a merged `.fnm` missing
      // its postings-format attributes pass thirteen write-path verifiers.
      FieldInfos fis = new Lucene94FieldInfosFormat().read(directory, si, "", IOContext.DEFAULT);
      if (fis.size() != fieldCount) {
        System.out.println("MISMATCH field_count: expected=" + fieldCount + " got=" + fis.size());
        failures++;
      }
      for (int i = 0; i < fieldCount; i++) {
        failures += checkFieldInfo(fis, manifest, "f" + i);
      }
      SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT);

      try (KnnVectorsReader reader = new Lucene99HnswVectorsFormat().fieldsReader(state)) {
        for (int i = 0; i < fieldCount; i++) {
          failures += verifyField(reader, manifest, "f" + i, k);
        }
      }
    } catch (Throwable t) {
      System.out.println("FAILED TO OPEN: " + t);
      t.printStackTrace(System.out);
      System.exit(1);
    }

    if (failures > 0) {
      System.out.println(failures + " mismatch(es)");
      System.exit(1);
    }
    System.out.println("All Rust-written vectors verified against real Lucene. PASS");
  }

  static int verifyField(KnnVectorsReader reader, Map<String, String> m, String key, int k)
      throws IOException {
    String name = m.get(key + ".name");
    int dim = Integer.parseInt(m.get(key + ".dim"));
    int count = Integer.parseInt(m.get(key + ".count"));
    boolean isByte = m.get(key + ".encoding").equals("BYTE");
    int[] docs = parseInts(m.get(key + ".docs"));
    int failures = 0;

    long valueHash = 0;
    int size;
    int[] gotDocs;
    if (isByte) {
      ByteVectorValues values = reader.getByteVectorValues(name);
      size = values.size();
      gotDocs = new int[size];
      for (int ord = 0; ord < size; ord++) {
        byte[] v = values.vectorValue(ord);
        for (byte b : v) {
          valueHash = valueHash * 31 + b;
        }
        gotDocs[ord] = values.ordToDoc(ord);
      }
      if (values.dimension() != dim) {
        System.out.println(name + ": MISMATCH dimension " + values.dimension() + " != " + dim);
        failures++;
      }
    } else {
      FloatVectorValues values = reader.getFloatVectorValues(name);
      size = values.size();
      gotDocs = new int[size];
      for (int ord = 0; ord < size; ord++) {
        float[] v = values.vectorValue(ord);
        for (float c : v) {
          valueHash = valueHash * 31 + Float.floatToRawIntBits(c);
        }
        gotDocs[ord] = values.ordToDoc(ord);
      }
      if (values.dimension() != dim) {
        System.out.println(name + ": MISMATCH dimension " + values.dimension() + " != " + dim);
        failures++;
      }
    }

    if (size != count) {
      System.out.println(name + ": MISMATCH size " + size + " != " + count);
      failures++;
    }
    long expectedValueHash = Long.parseLong(m.get(key + ".value_hash"));
    if (valueHash != expectedValueHash) {
      System.out.println(
          name + ": MISMATCH value_hash " + valueHash + " != " + expectedValueHash);
      failures++;
    }
    for (int ord = 0; ord < Math.min(size, docs.length); ord++) {
      if (gotDocs[ord] != docs[ord]) {
        System.out.println(
            name + ": MISMATCH ordToDoc(" + ord + ") " + gotDocs[ord] + " != " + docs[ord]);
        failures++;
        break;
      }
    }

    failures += verifyGraph(reader, m, key, name);
    failures += verifySearch(reader, m, key, name, dim, isByte, k);
    return failures;
  }

  static int verifyGraph(KnnVectorsReader reader, Map<String, String> m, String key, String name)
      throws IOException {
    int numLevels = Integer.parseInt(m.get(key + ".num_levels"));
    HnswGraph graph = ((HnswGraphProvider) reader).getGraph(name);
    int failures = 0;
    if (numLevels == 0) {
      // `HnswGraph.EMPTY` is what Lucene returns for a field with no graph.
      if (graph != null && graph.size() != 0) {
        System.out.println(name + ": MISMATCH expected no graph, got size=" + graph.size());
        failures++;
      }
      return failures;
    }
    if (graph == null || graph.numLevels() != numLevels) {
      System.out.println(
          name
              + ": MISMATCH numLevels "
              + (graph == null ? "null" : graph.numLevels())
              + " != "
              + numLevels);
      return failures + 1;
    }
    int entryNode = Integer.parseInt(m.get(key + ".entry_node"));
    if (graph.entryNode() != entryNode) {
      System.out.println(name + ": MISMATCH entryNode " + graph.entryNode() + " != " + entryNode);
      failures++;
    }
    int maxConn = Integer.parseInt(m.get(key + ".max_conn"));
    if (graph.maxConn() != maxConn) {
      System.out.println(name + ": MISMATCH maxConn " + graph.maxConn() + " != " + maxConn);
      failures++;
    }
    for (int level = 0; level < numLevels; level++) {
      List<Integer> nodes = new ArrayList<>();
      HnswGraph.NodesIterator it = graph.getSortedNodes(level);
      while (it.hasNext()) {
        nodes.add(it.nextInt());
      }
      long arcTotal = 0;
      long hash = 0;
      for (int node : nodes) {
        graph.seek(level, node);
        hash = hash * 31 + node;
        for (int n = graph.nextNeighbor();
            n != DocIdSetIterator.NO_MORE_DOCS;
            n = graph.nextNeighbor()) {
          hash = hash * 31 + n;
          arcTotal++;
        }
      }
      String lk = key + ".level" + level;
      failures += expect(name, lk + ".node_count", nodes.size(), Long.parseLong(m.get(lk + ".node_count")));
      failures += expect(name, lk + ".arc_total", arcTotal, Long.parseLong(m.get(lk + ".arc_total")));
      failures += expect(name, lk + ".arc_hash", hash, Long.parseLong(m.get(lk + ".arc_hash")));
    }
    return failures;
  }

  static int verifySearch(
      KnnVectorsReader reader,
      Map<String, String> m,
      String key,
      String name,
      int dim,
      boolean isByte,
      int k)
      throws IOException {
    String countKey = "q." + key + ".count";
    if (!m.containsKey(countKey)) {
      return 0;
    }
    int queries = Integer.parseInt(m.get(countKey));
    int maxDoc = Integer.parseInt(m.get("max_doc"));
    int hits = 0;
    int total = 0;
    for (int q = 0; q < queries; q++) {
      String qk = "q." + key + "." + q;
      Set<Integer> exact = new HashSet<>();
      for (int d : parseInts(m.get(qk + ".exact"))) {
        exact.add(d);
      }
      TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, null);
      AcceptDocs accept = AcceptDocs.fromLiveDocs(null, maxDoc);
      if (isByte) {
        byte[] target = new byte[dim];
        int[] raw = parseInts(m.get(qk + ".vec"));
        for (int i = 0; i < dim; i++) {
          target[i] = (byte) raw[i];
        }
        reader.search(name, target, collector, accept);
      } else {
        float[] target = new float[dim];
        int[] raw = parseInts(m.get(qk + ".vec"));
        for (int i = 0; i < dim; i++) {
          target[i] = Float.intBitsToFloat(raw[i]);
        }
        reader.search(name, target, collector, accept);
      }
      ScoreDoc[] got = collector.topDocs().scoreDocs;
      for (ScoreDoc sd : got) {
        if (exact.contains(sd.doc)) {
          hits++;
        }
      }
      total += exact.size();
    }
    double recall = total == 0 ? 1.0 : (double) hits / total;
    System.out.println(
        String.format(
            java.util.Locale.ROOT,
            "  %-12s Lucene search over the Rust-built graph: recall@%d = %.4f over %d queries",
            name, k, recall, queries));
    if (recall < MIN_RECALL) {
      System.out.println(name + ": MISMATCH recall " + recall + " < " + MIN_RECALL);
      return 1;
    }
    return 0;
  }

  static int expect(String field, String what, long got, long want) {
    if (got != want) {
      System.out.println(field + ": MISMATCH " + what + " " + got + " != " + want);
      return 1;
    }
    return 0;
  }

  /** The Rust-written `.fnm` must describe each vector field the way the manifest says. */
  static int checkFieldInfo(FieldInfos fis, Map<String, String> m, String key) {
    String name = m.get(key + ".name");
    FieldInfo fi = fis.fieldInfo(name);
    if (fi == null) {
      System.out.println(name + ": MISMATCH missing from the Rust-written .fnm");
      return 1;
    }
    int failures = 0;
    failures += expect(name, "field number", fi.number, Long.parseLong(m.get(key + ".number")));
    failures +=
        expect(name, "vector dimension", fi.getVectorDimension(), Long.parseLong(m.get(key + ".dim")));
    if (fi.getVectorEncoding() != VectorEncoding.valueOf(m.get(key + ".encoding"))) {
      System.out.println(name + ": MISMATCH vector encoding " + fi.getVectorEncoding());
      failures++;
    }
    if (fi.getVectorSimilarityFunction()
        != VectorSimilarityFunction.valueOf(m.get(key + ".similarity"))) {
      System.out.println(
          name + ": MISMATCH vector similarity " + fi.getVectorSimilarityFunction());
      failures++;
    }
    return failures;
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
