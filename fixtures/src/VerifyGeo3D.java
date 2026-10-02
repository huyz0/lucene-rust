import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.spatial3d.Geo3DDocValuesField;
import org.apache.lucene.spatial3d.Geo3DPoint;
import org.apache.lucene.spatial3d.geom.PlanetModel;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M9 T9.4's write-path proof: real Lucene opens the {@code Geo3DPoint}/{@code
 * Geo3DDocValuesField} index this port's {@code IndexWriter} wrote ({@code
 * write_geo3d_points_fixture}, from {@code GenGeo3dPoints}' documents), finds it clean ({@link
 * CheckIndex}), and answers every query and sort of {@code fixtures/data/geo3d_points/queries.tsv}
 * exactly as it answers over its own index of the same documents ({@code index/} beside it) --
 * both run here, in one JVM, so the comparison holds whatever the JVM's trig intrinsics do to the
 * shapes (the recorded answers were made with them off).
 *
 * <p>Usage: {@code java VerifyGeo3D <index-dir> <fixtures/data/geo3d_points>}.
 */
public class VerifyGeo3D {
  static double d(String s) {
    return Double.parseDouble(s);
  }

  static double[] list(String s) {
    String[] p = s.split(";");
    double[] out = new double[p.length];
    for (int i = 0; i < p.length; i++) out[i] = d(p[i]);
    return out;
  }

  static double[][] ring(String s) {
    String[] pts = s.split(";");
    double[] a = new double[pts.length];
    double[] b = new double[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] xy = pts[i].split(" ");
      a[i] = d(xy[0]);
      b[i] = d(xy[1]);
    }
    return new double[][] {a, b};
  }

  static Polygon[] polygons(String spec) {
    List<Polygon> out = new ArrayList<>();
    for (String g : spec.split(" \\+ ")) {
      String[] rings = g.substring(2).split("\\|");
      Polygon[] holes = new Polygon[rings.length - 1];
      for (int i = 1; i < rings.length; i++) {
        double[][] r = ring(rings[i]);
        holes[i - 1] = new Polygon(r[0], r[1]);
      }
      double[][] s = ring(rings[0]);
      out.add(new Polygon(s[0], s[1], holes));
    }
    return out.toArray(new Polygon[0]);
  }

  static PlanetModel model(String field) {
    return field.equals("s") ? PlanetModel.SPHERE : PlanetModel.WGS84;
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

  interface Q {
    Query get();
  }

  interface S {
    SortField get();
  }

  static String constant(IndexSearcher s, Q q) throws Exception {
    Query query;
    TopDocs td;
    try {
      query = q.get();
      td = s.search(query, s.getIndexReader().maxDoc());
    } catch (RuntimeException e) {
      return "E\t" + e.getClass().getName() + "\t" + e.getMessage();
    }
    return "C\t" + td.scoreDocs.length + "\t" + hexBits(s.getIndexReader().maxDoc(), td.scoreDocs);
  }

  static String sorted(IndexSearcher s, Query q, S sf, int n) throws Exception {
    TopFieldDocs td;
    try {
      td = s.search(q, n, new Sort(sf.get()));
    } catch (RuntimeException e) {
      return "E\t" + e.getClass().getName() + "\t" + e.getMessage();
    }
    StringBuilder sb = new StringBuilder("T\t").append(td.totalHits.value()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      FieldDoc fd = (FieldDoc) td.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(fd.doc).append(':').append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0])));
    }
    return sb.toString();
  }

  static S sortField(String field, PlanetModel pm, String[] a, int at) {
    switch (a[at]) {
      case "dist":
        return () -> Geo3DDocValuesField.newDistanceSort(field, d(a[at + 1]), d(a[at + 2]), d(a[at + 3]), pm);
      case "path":
        return () -> Geo3DDocValuesField.newPathSort(field, list(a[at + 1]), list(a[at + 2]), d(a[at + 3]), pm);
      case "odist":
        return () -> Geo3DDocValuesField.newOutsideDistanceSort(field, d(a[at + 1]), d(a[at + 2]), d(a[at + 3]), pm);
      case "obox":
        return () -> Geo3DDocValuesField.newOutsideBoxSort(field, d(a[at + 1]), d(a[at + 2]), d(a[at + 3]), d(a[at + 4]), pm);
      case "opoly":
        return () -> Geo3DDocValuesField.newOutsidePolygonSort(field, pm, polygons(a[at + 1]));
      case "olpoly":
        return () -> Geo3DDocValuesField.newOutsideLargePolygonSort(field, pm, polygons(a[at + 1]));
      case "opath":
        return () -> Geo3DDocValuesField.newOutsidePathSort(field, list(a[at + 1]), list(a[at + 2]), d(a[at + 3]), pm);
      default:
        throw new IllegalStateException(a[at]);
    }
  }

  static String run(IndexSearcher s, String[] a) throws Exception {
    String field = a[1];
    PlanetModel pm = model(field);
    switch (a[0]) {
      case "dist":
        return constant(s, () -> Geo3DPoint.newDistanceQuery(field, pm, d(a[2]), d(a[3]), d(a[4])));
      case "box":
        return constant(s, () -> Geo3DPoint.newBoxQuery(field, pm, d(a[2]), d(a[3]), d(a[4]), d(a[5])));
      case "poly":
        return constant(s, () -> Geo3DPoint.newPolygonQuery(field, pm, polygons(a[2])));
      case "lpoly":
        return constant(s, () -> Geo3DPoint.newLargePolygonQuery(field, pm, polygons(a[2])));
      case "path":
        return constant(s, () -> Geo3DPoint.newPathQuery(field, list(a[2]), list(a[3]), d(a[4]), pm));
      case "sort":
        {
          int n = Integer.parseInt(a[2]);
          Query base;
          if (a[3].equals("all")) {
            base = new MatchAllDocsQuery();
          } else {
            String[] v = a[3].substring(4).split(",");
            base = Geo3DPoint.newBoxQuery(field.equals("pd") ? "p" : field, pm, d(v[0]), d(v[1]), d(v[2]), d(v[3]));
          }
          return sorted(s, base, sortField(field, pm, a, 4), n);
        }
      default:
        throw new IllegalStateException(a[0]);
    }
  }

  public static void main(String[] args) throws Exception {
    Path rust = Path.of(args[0]);
    Path fixture = Path.of(args[1]);
    try (Directory rd = FSDirectory.open(rust);
        CheckIndex ci = new CheckIndex(rd)) {
      CheckIndex.Status st = ci.checkIndex();
      if (!st.clean) throw new AssertionError("CheckIndex failed on the Rust index");
    }
    int n = 0;
    try (Directory rd = FSDirectory.open(rust);
        DirectoryReader rr = DirectoryReader.open(rd);
        Directory jd = FSDirectory.open(fixture.resolve("index"));
        DirectoryReader jr = DirectoryReader.open(jd)) {
      IndexSearcher rs = new IndexSearcher(rr);
      IndexSearcher js = new IndexSearcher(jr);
      rs.setQueryCache(null);
      js.setQueryCache(null);
      for (String line : Files.readAllLines(fixture.resolve("queries.tsv"))) {
        String query = line.substring(0, line.indexOf("\t=>\t"));
        String[] a = query.split("\t");
        String got = run(rs, a);
        String want = run(js, a);
        if (!got.equals(want)) {
          throw new AssertionError("differs: " + query + "\n  lucene's index: " + want + "\n  rust's index:   " + got);
        }
        n++;
      }
    }
    if (n < 250) throw new AssertionError("only " + n + " queries");
    System.out.println("VerifyGeo3D: CheckIndex clean, " + n + " queries answered as over Lucene's index");
  }
}
