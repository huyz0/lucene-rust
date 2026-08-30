import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PointValues.IntersectVisitor;
import org.apache.lucene.index.PointValues.Relation;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;

/**
 * Cross-engine ground truth for {@link PointValues#estimatePointCount} and
 * {@link PointValues#estimateDocCount}, appended to the already-checked-in
 * {@code points_index} fixture's {@code manifest.properties} <b>without
 * regenerating it</b> -- same technique, and same reason, as
 * {@link AppendCountManifest} (re-running {@code GenPoints} would stamp a fresh
 * random segment id into an index whose id other suites pin).
 *
 * <p>The estimate is not the match count: it adds a whole subtree's
 * {@code BKDPointTree.size()} for a cell entirely inside the query without
 * descending, and assumes {@code (size + 1) / 2} at a leaf it cannot descend
 * past. So a "wrong but plausible" port -- one that walks the same tree but
 * gets the per-node size arithmetic wrong -- produces numbers in the right
 * ballpark, which is exactly why the exact figures have to come from Lucene
 * rather than from a re-derivation here. {@code .exact} records the true
 * matching point count alongside, so a reader can see how far off the estimate
 * is meant to be.
 *
 * <p>The three fields cover the shapes that make the arithmetic differ:
 * {@code val} is 1 333 points over three leaves (so the tree is unbalanced and
 * one subtree is a level deeper than its sibling), {@code multi} is two indexed
 * dimensions over four leaves, and {@code shape} indexes only the first two of
 * its four dimensions.
 *
 * <p>Idempotent: re-running replaces any previously-appended
 * {@code point_estimate.*} lines rather than duplicating them.
 */
public class AppendPointEstimateManifest {

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("points_index");
    StringBuilder out = new StringBuilder();

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      LeafReader leaf = reader.leaves().get(0).reader();

      // --- "val": one 8-byte dimension, LongPoint-encoded. ------------------
      // GenPoints writes `i * 7919 - 1_000_000` for every doc with `i % 3 != 0`,
      // i.e. -992 081 .. 14 833 161.
      List<Case> valCases = new ArrayList<>();
      valCases.add(longCase("val.all", Long.MIN_VALUE, Long.MAX_VALUE));
      valCases.add(longCase("val.none_below", Long.MIN_VALUE, -2_000_000L));
      valCases.add(longCase("val.none_above", 100_000_000L, Long.MAX_VALUE));
      valCases.add(longCase("val.lower_half", Long.MIN_VALUE, 7_000_000L));
      valCases.add(longCase("val.narrow", 0L, 100_000L));
      valCases.add(longCase("val.single", 7919L - 1_000_000L, 7919L - 1_000_000L));
      valCases.add(longCase("val.from_middle", 7_000_000L, Long.MAX_VALUE));
      record(out, leaf, "val", valCases);

      // --- "multi": two indexed 4-byte dimensions. --------------------------
      List<Case> multiCases = new ArrayList<>();
      multiCases.add(
          intCase(
              "multi.all",
              new int[] {Integer.MIN_VALUE, Integer.MIN_VALUE},
              new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE}));
      multiCases.add(
          intCase(
              "multi.dim1_only",
              new int[] {Integer.MIN_VALUE, 1},
              new int[] {Integer.MAX_VALUE, 2}));
      multiCases.add(
          intCase("multi.corner", new int[] {0, 0}, new int[] {Integer.MAX_VALUE, 0}));
      multiCases.add(
          intCase(
              "multi.empty",
              new int[] {Integer.MIN_VALUE, 5},
              new int[] {Integer.MAX_VALUE, 9}));
      record(out, leaf, "multi", multiCases);

      // --- "shape": four dimensions, only the first two indexed. ------------
      List<Case> shapeCases = new ArrayList<>();
      shapeCases.add(
          intCase(
              "shape.all",
              new int[] {Integer.MIN_VALUE, Integer.MIN_VALUE},
              new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE}));
      shapeCases.add(
          intCase("shape.quadrant", new int[] {0, 0}, new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE}));
      shapeCases.add(
          intCase(
              "shape.strip",
              new int[] {-1_000_000, Integer.MIN_VALUE},
              new int[] {1_000_000, Integer.MAX_VALUE}));
      record(out, leaf, "shape", shapeCases);
    }

    Path manifestPath = indexDir.resolve("manifest.properties");
    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("point_estimate.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended point_estimate.* ground truth to " + manifestPath);
  }

  private record Case(String key, byte[] lower, byte[] upper) {}

  private static Case longCase(String key, long lo, long hi) {
    byte[] lower = new byte[Long.BYTES];
    byte[] upper = new byte[Long.BYTES];
    NumericUtils.longToSortableBytes(lo, lower, 0);
    NumericUtils.longToSortableBytes(hi, upper, 0);
    return new Case(key, lower, upper);
  }

  private static Case intCase(String key, int[] lo, int[] hi) {
    byte[] lower = new byte[lo.length * Integer.BYTES];
    byte[] upper = new byte[hi.length * Integer.BYTES];
    for (int d = 0; d < lo.length; d++) {
      NumericUtils.intToSortableBytes(lo[d], lower, d * Integer.BYTES);
      NumericUtils.intToSortableBytes(hi[d], upper, d * Integer.BYTES);
    }
    return new Case(key, lower, upper);
  }

  private static void record(StringBuilder out, LeafReader leaf, String field, List<Case> cases)
      throws IOException {
    PointValues values = leaf.getPointValues(field);
    HexFormat hex = HexFormat.of();
    for (Case c : cases) {
      RangeVisitor visitor =
          new RangeVisitor(c.lower(), c.upper(), values.getNumIndexDimensions(), values.getBytesPerDimension());
      long points = values.estimatePointCount(visitor);
      long docs = values.estimateDocCount(visitor);
      CountingVisitor exact =
          new CountingVisitor(c.lower(), c.upper(), values.getNumIndexDimensions(), values.getBytesPerDimension());
      values.intersect(exact);
      String p = "point_estimate." + c.key();
      out.append(p).append(".lower_hex=").append(hex.formatHex(c.lower())).append('\n');
      out.append(p).append(".upper_hex=").append(hex.formatHex(c.upper())).append('\n');
      out.append(p).append(".points=").append(points).append('\n');
      out.append(p).append(".docs=").append(docs).append('\n');
      out.append(p).append(".exact=").append(exact.count).append('\n');
    }
    out.append("point_estimate.").append(field).append(".size=").append(values.size()).append('\n');
    out.append("point_estimate.")
        .append(field)
        .append(".doc_count=")
        .append(values.getDocCount())
        .append('\n');
  }

  /**
   * {@code PointRangeQuery.relate}, verbatim: an inclusive per-dimension box
   * compared unsigned byte-wise. {@code estimatePointCount} only ever calls
   * {@code compare}, so the two visit methods are unreachable here and say so.
   */
  private static class RangeVisitor implements IntersectVisitor {
    final byte[] lower;
    final byte[] upper;
    final int numIndexDims;
    final int bytesPerDim;

    RangeVisitor(byte[] lower, byte[] upper, int numIndexDims, int bytesPerDim) {
      this.lower = lower;
      this.upper = upper;
      this.numIndexDims = numIndexDims;
      this.bytesPerDim = bytesPerDim;
    }

    @Override
    public void visit(int docID) {
      throw new AssertionError("estimatePointCount must never visit a document");
    }

    @Override
    public void visit(int docID, byte[] packedValue) {
      throw new AssertionError("estimatePointCount must never decode a point");
    }

    @Override
    public Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
      boolean crosses = false;
      for (int dim = 0, offset = 0; dim < numIndexDims; dim++, offset += bytesPerDim) {
        int end = offset + bytesPerDim;
        if (Arrays.compareUnsigned(minPackedValue, offset, end, upper, offset, end) > 0
            || Arrays.compareUnsigned(maxPackedValue, offset, end, lower, offset, end) < 0) {
          return Relation.CELL_OUTSIDE_QUERY;
        }
        if (crosses == false) {
          crosses =
              Arrays.compareUnsigned(minPackedValue, offset, end, lower, offset, end) < 0
                  || Arrays.compareUnsigned(maxPackedValue, offset, end, upper, offset, end) > 0;
        }
      }
      return crosses ? Relation.CELL_CROSSES_QUERY : Relation.CELL_INSIDE_QUERY;
    }
  }

  /** The same box, but actually counting the matching points. */
  private static class CountingVisitor extends RangeVisitor {
    long count;

    CountingVisitor(byte[] lower, byte[] upper, int numIndexDims, int bytesPerDim) {
      super(lower, upper, numIndexDims, bytesPerDim);
    }

    @Override
    public void visit(int docID) {
      count++;
    }

    @Override
    public void visit(int docID, byte[] packedValue) {
      for (int dim = 0, offset = 0; dim < numIndexDims; dim++, offset += bytesPerDim) {
        int end = offset + bytesPerDim;
        if (Arrays.compareUnsigned(packedValue, offset, end, lower, offset, end) < 0
            || Arrays.compareUnsigned(packedValue, offset, end, upper, offset, end) > 0) {
          return;
        }
      }
      count++;
    }
  }
}
