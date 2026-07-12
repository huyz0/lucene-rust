import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.lucene90.Lucene90PointsFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens
 * `.kdm`/`.kdi`/`.kdd` triples written by this port's `points::write`
 * (single-dimension `LongPoint`-style fields -- see
 * `crates/lucene-codecs/examples/write_points_fixture.rs`) directly through
 * real Lucene's {@link Lucene90PointsFormat}, using a hand-built
 * {@link SegmentInfo}/{@link FieldInfos} the same way
 * {@code VerifyStoredFields.java}/{@code VerifyFieldInfos.java} do -- this
 * keeps the slice scoped to exactly the points/BKD format itself, no
 * `.si`/`.fnm` writer needed.
 *
 * <p>Uses real {@link PointValues#intersect} with a visitor whose {@code
 * compare} always returns {@code CELL_CROSSES_QUERY} (the same technique
 * {@code GenPoints.java} uses) to force a full decode of every point rather
 * than relying on any bounding-box pruning.
 *
 * <p>Usage: {@code java VerifyPoints <fixture-dir>}, where
 * {@code <fixture-dir>} contains {@code _0.kdm}/{@code _0.kdi}/
 * {@code _0.kdd} (single-leaf) plus {@code _1.kdm}/{@code _1.kdi}/
 * {@code _1.kdd} (multi-leaf, small {@code maxPointsInLeafNode}) and a
 * {@code manifest.properties} describing the expected `(docID, value)`
 * pairs for both segments (unprefixed keys for {@code _0}, {@code
 * segment1_}-prefixed keys for {@code _1}). Exits nonzero and prints a diff
 * on any mismatch in either segment.
 */
public class VerifyPoints {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    int failures = 0;
    try (Directory directory = FSDirectory.open(dir)) {
      failures += verifySegment(directory, manifest, "_0", "");
      failures += verifySegment(directory, manifest, "_1", "segment1_");
    } catch (CorruptIndexException e) {
      System.out.println("FAILED TO OPEN: " + e);
      System.exit(1);
    }

    if (failures > 0) {
      System.out.println(failures + " mismatch(es)");
      System.exit(1);
    }
    System.out.println("All points verified against real Lucene across both segments. PASS");
  }

  /** Verifies one segment (named by {@code segmentName}) against its manifest section. */
  static int verifySegment(
      Directory directory, Map<String, String> manifest, String segmentName, String prefix)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(prefix + "max_doc"));
    byte[] id = HexFormat.of().parseHex(manifest.get(prefix + "id_hex"));
    int fieldNumber = Integer.parseInt(manifest.get(prefix + "field_number"));
    int bytesPerDim = Integer.parseInt(manifest.get(prefix + "bytes_per_dim"));
    int expectedPointCount = Integer.parseInt(manifest.get(prefix + "point_count"));
    String pointsSpec = manifest.getOrDefault(prefix + "points", "");

    List<long[]> expected = new ArrayList<>();
    if (!pointsSpec.isEmpty()) {
      for (String entry : pointsSpec.split(";")) {
        String[] parts = entry.split(":", 2);
        expected.add(new long[] {Long.parseLong(parts[0]), Long.parseLong(parts[1])});
      }
    }

    FieldInfo fieldInfo =
        new FieldInfo(
            "val",
            fieldNumber,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            new HashMap<>(),
            1,
            1,
            bytesPerDim,
            0,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false);
    FieldInfos fis = new FieldInfos(new FieldInfo[] {fieldInfo});

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

    PointsFormat format = new Lucene90PointsFormat();
    SegmentReadState readState = new SegmentReadState(directory, si, fis, IOContext.DEFAULT);
    PointsReader reader = format.fieldsReader(readState);

    int failures = 0;
    PointValues values = reader.getValues("val");
    if (values == null) {
      System.out.println(segmentName + ": MISMATCH: field 'val' has no PointValues");
      reader.close();
      return 1;
    }

    if (values.size() != expectedPointCount) {
      System.out.println(
          segmentName
              + ": MISMATCH point_count: expected="
              + expectedPointCount
              + " got="
              + values.size());
      failures++;
    }

    List<long[]> got = new ArrayList<>();
    values.intersect(
        new PointValues.IntersectVisitor() {
          @Override
          public void visit(int docID) {
            throw new AssertionError("should not be called: compare always returns CROSSES");
          }

          @Override
          public void visit(int docID, byte[] packedValue) {
            long decoded = NumericUtils.sortableBytesToLong(packedValue, 0);
            got.add(new long[] {docID, decoded});
          }

          @Override
          public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
            return PointValues.Relation.CELL_CROSSES_QUERY;
          }
        });
    got.sort((a, b) -> Long.compare(a[0], b[0]));
    expected.sort((a, b) -> Long.compare(a[0], b[0]));

    if (got.size() != expected.size()) {
      System.out.println(
          segmentName
              + ": MISMATCH decoded point count: expected="
              + expected.size()
              + " got="
              + got.size());
      failures++;
    } else {
      for (int i = 0; i < expected.size(); i++) {
        long[] e = expected.get(i);
        long[] g = got.get(i);
        if (e[0] != g[0] || e[1] != g[1]) {
          System.out.println(
              segmentName
                  + ": MISMATCH point "
                  + i
                  + ": expected doc="
                  + e[0]
                  + " value="
                  + e[1]
                  + " got doc="
                  + g[0]
                  + " value="
                  + g[1]);
          failures++;
        }
      }
    }

    if (segmentName.equals("_1")) {
      // _1 is the multi-leaf segment (see write_points_fixture.rs):
      // `compare()` above always returns CELL_CROSSES_QUERY, which is
      // exactly the case that never exercises real BKDReader's
      // pruning path (`BKDPointTree.moveToChild`/`readNodeData`) -- the
      // one thing that actually consumes the packed index's
      // reconstructed split values to decide whether to skip a subtree.
      // Run a second query with a real, narrow bounding box here so a
      // wrong split value (e.g. this port's now-fixed `pack_index`
      // delta-coding bug) has an observable effect.
      failures += verifyPruningQuery(reader, expected);
    }

    reader.close();
    if (failures == 0) {
      System.out.println(segmentName + ": all " + expected.size() + " points verified. OK");
    }
    return failures;
  }

  /**
   * Runs a narrow-bounding-box {@link PointValues#intersect} query against
   * {@code reader}'s "val" field, using a real {@code compare()} that returns
   * {@code CELL_OUTSIDE_QUERY}/{@code CELL_INSIDE_QUERY}/{@code
   * CELL_CROSSES_QUERY} based on an actual byte-wise comparison against the
   * cell's own min/max packed value -- the same relation real range-query
   * fields compute. Asserts both a pruned-outside cell and a fully-inside
   * cell were actually seen (otherwise this query wouldn't be proving
   * anything about the pruning path), then asserts the returned point set
   * matches the expected values that fall inside the query's value range.
   */
  static int verifyPruningQuery(PointsReader reader, List<long[]> expected) throws IOException {
    PointValues values = reader.getValues("val");

    // Middle third of the value range, by rank -- guaranteed to leave some
    // points strictly below and some strictly above, so the query really
    // does prune both a left and a right subtree, not just one side.
    List<Long> sortedValues = new ArrayList<>();
    for (long[] p : expected) {
      sortedValues.add(p[1]);
    }
    Collections.sort(sortedValues);
    long queryMinValue = sortedValues.get(sortedValues.size() / 3);
    long queryMaxValue = sortedValues.get(2 * sortedValues.size() / 3);
    byte[] queryMin = new byte[Long.BYTES];
    byte[] queryMax = new byte[Long.BYTES];
    NumericUtils.longToSortableBytes(queryMinValue, queryMin, 0);
    NumericUtils.longToSortableBytes(queryMaxValue, queryMax, 0);

    List<long[]> expectedInRange = new ArrayList<>();
    for (long[] p : expected) {
      if (p[1] >= queryMinValue && p[1] <= queryMaxValue) {
        expectedInRange.add(p);
      }
    }

    boolean[] sawOutside = new boolean[1];
    boolean[] sawInside = new boolean[1];
    List<long[]> got = new ArrayList<>();
    Map<Integer, Long> expectedByDoc = new HashMap<>();
    for (long[] p : expected) {
      expectedByDoc.put((int) p[0], p[1]);
    }

    values.intersect(
        new PointValues.IntersectVisitor() {
          @Override
          public void visit(int docID) {
            // Only called for cells real Lucene has already determined are
            // fully CELL_INSIDE_QUERY -- no packedValue is handed back, so
            // recover it from the manifest (this fixture controls the data).
            got.add(new long[] {docID, expectedByDoc.get(docID)});
          }

          @Override
          public void visit(int docID, byte[] packedValue) {
            long decoded = NumericUtils.sortableBytesToLong(packedValue, 0);
            if (decoded >= queryMinValue && decoded <= queryMaxValue) {
              got.add(new long[] {docID, decoded});
            }
          }

          @Override
          public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
            if (Arrays.compareUnsigned(maxPackedValue, queryMin) < 0
                || Arrays.compareUnsigned(minPackedValue, queryMax) > 0) {
              sawOutside[0] = true;
              return PointValues.Relation.CELL_OUTSIDE_QUERY;
            }
            if (Arrays.compareUnsigned(minPackedValue, queryMin) >= 0
                && Arrays.compareUnsigned(maxPackedValue, queryMax) <= 0) {
              sawInside[0] = true;
              return PointValues.Relation.CELL_INSIDE_QUERY;
            }
            return PointValues.Relation.CELL_CROSSES_QUERY;
          }
        });

    int failures = 0;
    if (!sawOutside[0]) {
      System.out.println(
          "_1: MISMATCH pruning query: expected at least one CELL_OUTSIDE_QUERY cell"
              + " (query never pruned anything -- this proves nothing about split-value"
              + " reconstruction)");
      failures++;
    }
    if (!sawInside[0]) {
      System.out.println(
          "_1: MISMATCH pruning query: expected at least one CELL_INSIDE_QUERY cell");
      failures++;
    }

    got.sort((a, b) -> Long.compare(a[0], b[0]));
    expectedInRange.sort((a, b) -> Long.compare(a[0], b[0]));
    if (got.size() != expectedInRange.size()) {
      System.out.println(
          "_1: MISMATCH pruning query point count: expected="
              + expectedInRange.size()
              + " got="
              + got.size());
      failures++;
    } else {
      for (int i = 0; i < expectedInRange.size(); i++) {
        long[] e = expectedInRange.get(i);
        long[] g = got.get(i);
        if (e[0] != g[0] || e[1] != g[1]) {
          System.out.println(
              "_1: MISMATCH pruning query point "
                  + i
                  + ": expected doc="
                  + e[0]
                  + " value="
                  + e[1]
                  + " got doc="
                  + g[0]
                  + " value="
                  + g[1]);
          failures++;
        }
      }
    }

    if (failures == 0) {
      System.out.println(
          "_1: pruning query verified ("
              + expectedInRange.size()
              + " points in range ["
              + queryMinValue
              + ", "
              + queryMaxValue
              + "], real BKDReader pruning exercised). OK");
    }
    return failures;
  }

  static Map<String, String> readManifest(Path path) throws IOException {
    Map<String, String> m = new HashMap<>();
    for (String line : Files.readAllLines(path)) {
      int eq = line.indexOf('=');
      if (eq < 0) continue;
      m.put(line.substring(0, eq), line.substring(eq + 1));
    }
    return m;
  }
}
