import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.NumericUtils;

/**
 * Reverse-direction verifier for points written by this port's {@code IndexWriter} -- at flush,
 * at a merge, and at an index-sorted merge ({@code write_points_segment_fixture}).
 *
 * <p>For each of the three indexes, every point of every field is read back through {@link
 * PointValues#intersect} with a visitor that accepts everything, grouped by document (resolved to
 * the generator's number through the stored {@code id}), and compared, as a sorted list per
 * document, with what the generator put there -- so a point attached to the wrong document after
 * the sorted merge's doc-id remap is caught, not only a malformed tree. Then range queries through
 * {@code LongPoint/IntPoint/DoublePoint.newRangeQuery} and a two-dimensional box, whose counts are
 * computed from the generator, and {@link CheckIndex} (which walks every BKD tree and, for the
 * sorted index, re-checks the sort). A points field declared but carried by no document must
 * come back with no point shape at all, flushed or merged.
 *
 * <p>Usage: {@code java VerifyPointsSegment <out-dir>}.
 */
public class VerifyPointsSegment {
  /** Must match {@code write_points_segment_fixture.rs}. */
  private static final int NUM_DOCS = 20_000;

  private static int failures = 0;

  private static void fail(String message) {
    if (++failures <= 30) {
      System.out.println("MISMATCH " + message);
    }
  }

  /** Document {@code i}'s values for {@code field}, decoded, sorted. */
  static List<String> expected(String field, int i) {
    List<String> out = new ArrayList<>();
    switch (field) {
      case "lp":
        out.add(Long.toString(7L * i - 1000));
        if (i % 4 == 0) out.add(Long.toString(-i));
        break;
      case "ip":
        if (i % 3 != 0) out.add(Integer.toString(i % 1000));
        break;
      case "dp":
        out.add(Double.toString(i / 8.0 - 100.0));
        break;
      case "xy":
        out.add((i % 97) + "," + (i % 89 - 44));
        break;
      default:
        throw new IllegalArgumentException(field);
    }
    Collections.sort(out);
    return out;
  }

  static String decode(String field, byte[] packed) {
    switch (field) {
      case "lp":
        return Long.toString(LongPoint.decodeDimension(packed, 0));
      case "ip":
        return Integer.toString(IntPoint.decodeDimension(packed, 0));
      case "dp":
        return Double.toString(DoublePoint.decodeDimension(packed, 0));
      case "xy":
        return IntPoint.decodeDimension(packed, 0) + "," + IntPoint.decodeDimension(packed, 4);
      default:
        throw new IllegalArgumentException(field);
    }
  }

  static void verify(Path path, int wantSegments) throws IOException {
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != NUM_DOCS) {
        fail(path + ": maxDoc=" + reader.maxDoc());
      }
      if (wantSegments == 1 ? reader.leaves().size() != 1 : reader.leaves().size() < 3) {
        fail(path + ": " + reader.leaves().size() + " segment(s)");
      }
      for (String field : new String[] {"lp", "ip", "dp", "xy"}) {
        int checked = 0;
        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          StoredFields stored = leaf.storedFields();
          Map<Integer, List<String>> got = new HashMap<>();
          PointValues values = leaf.getPointValues(field);
          if (values != null) {
            values.intersect(
                new PointValues.IntersectVisitor() {
                  @Override
                  public void visit(int docID) {
                    throw new IllegalStateException("every cell crosses the query");
                  }

                  @Override
                  public void visit(int docID, byte[] packedValue) {
                    got.computeIfAbsent(docID, d -> new ArrayList<>())
                        .add(decode(field, packedValue));
                  }

                  @Override
                  public PointValues.Relation compare(byte[] min, byte[] max) {
                    return PointValues.Relation.CELL_CROSSES_QUERY;
                  }
                });
          }
          for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            int i = Integer.parseInt(stored.document(doc).get("id").substring(3));
            List<String> have = got.getOrDefault(doc, new ArrayList<>());
            Collections.sort(have);
            List<String> want = expected(field, i);
            if (!have.equals(want)) {
              fail(path.getFileName() + " doc" + i + " " + field + ": got " + have + ", want " + want);
            }
            checked++;
          }
        }
        if (checked != NUM_DOCS) {
          fail(path + ": " + field + " checked " + checked + " documents");
        }
      }

      // A points field no document carries must not claim a point shape:
      // a `.fnm` that did would make the segment demand a `.kdm` entry that
      // does not exist.
      for (LeafReaderContext ctx : reader.leaves()) {
        if (ctx.reader().getPointValues("none") != null) {
          fail(path.getFileName() + ": field 'none' has point values");
        }
        var info = ctx.reader().getFieldInfos().fieldInfo("none");
        if (info != null && info.getPointDimensionCount() != 0) {
          fail(path.getFileName() + ": field 'none' claims " + info.getPointDimensionCount() + " point dimension(s)");
        }
      }

      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      checkCount(searcher, path, LongPoint.newRangeQuery("lp", -500, 50_000), i -> {
        int n = (7L * i - 1000 >= -500 && 7L * i - 1000 <= 50_000) ? 1 : 0;
        if (i % 4 == 0 && -i >= -500) n = Math.max(n, 1);
        return n > 0;
      });
      checkCount(searcher, path, IntPoint.newRangeQuery("ip", 100, 250),
          i -> i % 3 != 0 && i % 1000 >= 100 && i % 1000 <= 250);
      checkCount(searcher, path, DoublePoint.newRangeQuery("dp", -50.5, 900.25),
          i -> i / 8.0 - 100.0 >= -50.5 && i / 8.0 - 100.0 <= 900.25);
      checkCount(searcher, path,
          IntPoint.newRangeQuery("xy", new int[] {10, -5}, new int[] {40, 20}),
          i -> i % 97 >= 10 && i % 97 <= 40 && i % 89 - 44 >= -5 && i % 89 - 44 <= 20);
    }
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8));
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      if (!checker.checkIndex().clean) {
        fail(path + ": CheckIndex reported the index unclean");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
      }
    }
  }

  interface DocPredicate {
    boolean test(int i);
  }

  static void checkCount(IndexSearcher searcher, Path path, Query q, DocPredicate matches)
      throws IOException {
    int want = 0;
    for (int i = 0; i < NUM_DOCS; i++) {
      if (matches.test(i)) want++;
    }
    int got = searcher.count(q);
    if (got != want) {
      fail(path.getFileName() + " " + q + ": count " + got + ", want " + want);
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    verify(out.resolve("flushed"), 3);
    verify(out.resolve("merged"), 1);
    verify(out.resolve("sorted"), 1);
    if (failures > 0) {
      System.out.println(failures + " check(s) failed");
      System.exit(1);
    }
    System.out.println("Points written at flush, merge and sorted merge verified against real Lucene. PASS");
  }
}
