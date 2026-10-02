import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.LatLonShape;
import org.apache.lucene.document.ShapeAccess;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.document.XYShape;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M9 T9.3's write-path proof: real Lucene opens the shape index this port's {@code IndexWriter}
 * wrote ({@code write_geo_shapes_fixture}, from {@code GenGeoShapes}' documents), finds it clean
 * ({@link CheckIndex}), and answers every query of {@code fixtures/data/geo_shapes/queries.tsv} --
 * every geometry under every relation, indexed ({@code LatLonShape}, {@code XYShape}) and over the
 * shape doc values -- exactly as it answered over its own index of the same documents.
 *
 * <p>{@link #run} is also how {@code GenGeoShapes} answers each query in the first place, so the
 * two cannot build a query differently.
 *
 * <p>Usage: {@code java VerifyGeoShapes <index-dir> <fixtures/data/geo_shapes>}.
 */
public class VerifyGeoShapes {
  interface QuerySupplier {
    Query get();
  }

  static double d(String s) {
    return Double.parseDouble(s);
  }

  static float f(String s) {
    return Float.parseFloat(s);
  }

  static String hexBits(int maxDoc, ScoreDoc[] docs) {
    byte[] b = new byte[(maxDoc + 7) / 8];
    for (ScoreDoc sd : docs) b[sd.doc >> 3] |= (byte) (1 << (sd.doc & 7));
    int end = b.length;
    while (end > 0 && b[end - 1] == 0) end--;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < end; i++) sb.append(String.format("%02x", b[i] & 0xff));
    return sb.toString();
  }

  /** {@code C total hexbits scorebits}, or {@code E message} if building or running it throws. */
  static String constant(IndexSearcher s, QuerySupplier q) throws Exception {
    Query query;
    TopDocs td;
    int maxDoc = s.getIndexReader().maxDoc();
    try {
      query = q.get();
      td = s.search(query, maxDoc);
    } catch (IllegalArgumentException e) {
      return "E\t" + e.getMessage();
    }
    float score = td.scoreDocs.length == 0 ? 0f : td.scoreDocs[0].score;
    for (ScoreDoc sd : td.scoreDocs) {
      if (sd.score != score) throw new AssertionError("not constant: " + query + " " + sd.score);
    }
    return "C\t" + td.scoreDocs.length + "\t" + hexBits(maxDoc, td.scoreDocs) + "\t" + Integer.toHexString(Float.floatToRawIntBits(score));
  }

  /** Answers one {@code queries.tsv} query (its tokens before {@code =>}). */
  static String run(IndexSearcher s, String[] a) throws Exception {
    String field = a[1];
    ShapeField.QueryRelation rel = ShapeField.QueryRelation.valueOf(a[2]);
    switch (a[0]) {
      case "geom":
        return constant(s, () -> LatLonShape.newGeometryQuery(field, rel, VerifyGeoPoints.latLon(a[3])));
      case "dvgeom":
        return constant(s, () -> ShapeAccess.latLonDocValuesQuery(field, rel, VerifyGeoPoints.latLon(a[3])));
      case "box":
        return constant(s, () -> LatLonShape.newBoxQuery(field, rel, d(a[3]), d(a[4]), d(a[5]), d(a[6])));
      case "dvbox":
        return constant(s, () -> LatLonShape.newSlowDocValuesBoxQuery(field, rel, d(a[3]), d(a[4]), d(a[5]), d(a[6])));
      case "xygeom":
        return constant(s, () -> XYShape.newGeometryQuery(field, rel, VerifyGeoPoints.xy(a[3])));
      case "xydvgeom":
        return constant(s, () -> ShapeAccess.xyDocValuesQuery(field, rel, VerifyGeoPoints.xy(a[3])));
      case "xybox":
        return constant(s, () -> XYShape.newBoxQuery(field, rel, f(a[3]), f(a[4]), f(a[5]), f(a[6])));
      case "xydvbox":
        return constant(s, () -> XYShape.newSlowDocValuesBoxQuery(field, rel, f(a[3]), f(a[4]), f(a[5]), f(a[6])));
      default:
        throw new IllegalStateException(a[0]);
    }
  }

  public static void main(String[] args) throws Exception {
    Path index = Path.of(args[0]);
    Path fixtures = Path.of(args[1]);
    try (Directory dir = FSDirectory.open(index)) {
      try (CheckIndex check = new CheckIndex(dir)) {
        check.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
        CheckIndex.Status status = check.checkIndex();
        if (!status.clean) throw new AssertionError("CheckIndex: not clean");
      }
      int n = 0;
      List<String> failures = new ArrayList<>();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 4) throw new AssertionError("segments: " + reader.leaves().size());
        IndexSearcher s = new IndexSearcher(reader);
        s.setQueryCache(null);
        for (String line : Files.readAllLines(fixtures.resolve("queries.tsv"))) {
          int at = line.indexOf("\t=>\t");
          String[] a = line.substring(0, at).split("\t");
          String want = line.substring(at + 4);
          String got = run(s, a);
          if (!got.equals(want)) {
            failures.add(line.substring(0, Math.min(at, 200)) + "\n  want " + want.substring(0, Math.min(200, want.length())) + "\n  got  " + got.substring(0, Math.min(200, got.length())));
          }
          n++;
        }
      }
      if (n < 1000) throw new AssertionError("only " + n + " queries");
      if (!failures.isEmpty()) {
        throw new AssertionError(failures.size() + " of " + n + " differ:\n" + String.join("\n", failures.subList(0, Math.min(10, failures.size()))));
      }
      System.out.println("VerifyGeoShapes: " + n + " queries answer as over Lucene's own index");
    }
  }
}
