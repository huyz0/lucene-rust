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
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens a
 * `.kdm`/`.kdi`/`.kdd` triple written by this port's `points::write` (a
 * single BKD leaf, single-dimension `LongPoint`-style field -- see
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
 * {@code _0.kdd} and a {@code manifest.properties} describing the expected
 * `(docID, value)` pairs. Exits nonzero and prints a diff on any mismatch.
 */
public class VerifyPoints {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    int maxDoc = Integer.parseInt(manifest.get("max_doc"));
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int fieldNumber = Integer.parseInt(manifest.get("field_number"));
    int bytesPerDim = Integer.parseInt(manifest.get("bytes_per_dim"));
    int expectedPointCount = Integer.parseInt(manifest.get("point_count"));
    String pointsSpec = manifest.getOrDefault("points", "");

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

    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              org.apache.lucene.util.Version.LATEST,
              org.apache.lucene.util.Version.LATEST,
              "_0",
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
        System.out.println("MISMATCH: field 'val' has no PointValues");
        System.exit(1);
      }

      if (values.size() != expectedPointCount) {
        System.out.println(
            "MISMATCH point_count: expected=" + expectedPointCount + " got=" + values.size());
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
            "MISMATCH decoded point count: expected=" + expected.size() + " got=" + got.size());
        failures++;
      } else {
        for (int i = 0; i < expected.size(); i++) {
          long[] e = expected.get(i);
          long[] g = got.get(i);
          if (e[0] != g[0] || e[1] != g[1]) {
            System.out.println(
                "MISMATCH point " + i + ": expected doc=" + e[0] + " value=" + e[1]
                    + " got doc=" + g[0] + " value=" + g[1]);
            failures++;
          }
        }
      }

      reader.close();

      if (failures > 0) {
        System.out.println(failures + " mismatch(es)");
        System.exit(1);
      }
      System.out.println(
          "All " + expected.size() + " points verified against real Lucene. PASS");
    } catch (CorruptIndexException e) {
      System.out.println("FAILED TO OPEN: " + e);
      System.exit(1);
    }
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
