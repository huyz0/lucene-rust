import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.LatLonDocValuesField;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.document.XYDocValuesField;
import org.apache.lucene.document.XYPointField;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Point;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.geo.XYCircle;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.geo.XYLine;
import org.apache.lucene.geo.XYPoint;
import org.apache.lucene.geo.XYPolygon;
import org.apache.lucene.geo.XYRectangle;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M9 T9.2's write-path proof: real Lucene opens the geo point index this port's {@code
 * IndexWriter} wrote ({@code write_geo_points_fixture}, from {@code GenGeoPoints}' documents),
 * finds it clean ({@link CheckIndex}), and answers every query of {@code
 * fixtures/data/geo_points/queries.tsv} -- boxes, distances, polygons, geometry relations, the
 * doc-values forms, the cartesian queries, distance features, distance sorts and {@code
 * LatLonPoint.nearest} -- exactly as it answered over its own index of the same documents.
 *
 * <p>Usage: {@code java VerifyGeoPoints <index-dir> <fixtures/data/geo_points>}.
 */
public class VerifyGeoPoints {
  interface QuerySupplier {
    Query get();
  }

  static double d(String s) {
    return Double.parseDouble(s);
  }

  static float f(String s) {
    return Float.parseFloat(s);
  }

  static double[][] ringD(String s) {
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

  static float[][] ringF(String s) {
    String[] pts = s.split(";");
    float[] a = new float[pts.length];
    float[] b = new float[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] xy = pts[i].split(" ");
      a[i] = f(xy[0]);
      b[i] = f(xy[1]);
    }
    return new float[][] {a, b};
  }

  static LatLonGeometry[] latLon(String spec) {
    List<LatLonGeometry> out = new ArrayList<>();
    for (String g : spec.split(" \\+ ")) {
      String body = g.substring(2);
      switch (g.substring(0, 2)) {
        case "P:" -> {
          String[] v = body.split(",");
          out.add(new Point(d(v[0]), d(v[1])));
        }
        case "L:" -> {
          double[][] r = ringD(body);
          out.add(new Line(r[0], r[1]));
        }
        case "G:" -> {
          String[] rings = body.split("\\|");
          double[][] shell = ringD(rings[0]);
          Polygon[] holes = new Polygon[rings.length - 1];
          for (int i = 1; i < rings.length; i++) {
            double[][] h = ringD(rings[i]);
            holes[i - 1] = new Polygon(h[0], h[1]);
          }
          out.add(new Polygon(shell[0], shell[1], holes));
        }
        case "C:" -> {
          String[] v = body.split(",");
          out.add(new Circle(d(v[0]), d(v[1]), d(v[2])));
        }
        case "R:" -> {
          String[] v = body.split(",");
          out.add(new Rectangle(d(v[0]), d(v[1]), d(v[2]), d(v[3])));
        }
        default -> throw new IllegalStateException(g);
      }
    }
    return out.toArray(new LatLonGeometry[0]);
  }

  static XYGeometry[] xy(String spec) {
    List<XYGeometry> out = new ArrayList<>();
    for (String g : spec.split(" \\+ ")) {
      String body = g.substring(2);
      switch (g.substring(0, 2)) {
        case "P:" -> {
          String[] v = body.split(",");
          out.add(new XYPoint(f(v[0]), f(v[1])));
        }
        case "L:" -> {
          float[][] r = ringF(body);
          out.add(new XYLine(r[0], r[1]));
        }
        case "G:" -> {
          String[] rings = body.split("\\|");
          float[][] shell = ringF(rings[0]);
          XYPolygon[] holes = new XYPolygon[rings.length - 1];
          for (int i = 1; i < rings.length; i++) {
            float[][] h = ringF(rings[i]);
            holes[i - 1] = new XYPolygon(h[0], h[1]);
          }
          out.add(new XYPolygon(shell[0], shell[1], holes));
        }
        case "C:" -> {
          String[] v = body.split(",");
          out.add(new XYCircle(f(v[0]), f(v[1]), f(v[2])));
        }
        case "R:" -> {
          String[] v = body.split(",");
          out.add(new XYRectangle(f(v[0]), f(v[1]), f(v[2]), f(v[3])));
        }
        default -> throw new IllegalStateException(g);
      }
    }
    return out.toArray(new XYGeometry[0]);
  }

  static Polygon[] polygons(String spec) {
    LatLonGeometry[] g = latLon(spec);
    Polygon[] p = new Polygon[g.length];
    for (int i = 0; i < g.length; i++) p[i] = (Polygon) g[i];
    return p;
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

  static String constant(IndexSearcher s, QuerySupplier q) throws Exception {
    Query query;
    try {
      query = q.get();
    } catch (IllegalArgumentException e) {
      return "E\t" + e.getMessage();
    }
    int maxDoc = s.getIndexReader().maxDoc();
    TopDocs td = s.search(query, maxDoc);
    return "C\t" + td.scoreDocs.length + "\t" + hexBits(maxDoc, td.scoreDocs);
  }

  static String fieldDocs(String tag, TopFieldDocs td) {
    StringBuilder sb = new StringBuilder(tag).append('\t').append(td.totalHits.value()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      FieldDoc fd = (FieldDoc) td.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(fd.doc).append(':').append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0])));
    }
    return sb.toString();
  }

  static String run(IndexSearcher s, String[] a) throws Exception {
    String field = a[1];
    switch (a[0]) {
      case "box":
        return constant(s, () -> LatLonPoint.newBoxQuery(field, d(a[2]), d(a[3]), d(a[4]), d(a[5])));
      case "dvbox":
        return constant(s, () -> LatLonDocValuesField.newSlowBoxQuery(field, d(a[2]), d(a[3]), d(a[4]), d(a[5])));
      case "dist":
        return constant(s, () -> LatLonPoint.newDistanceQuery(field, d(a[2]), d(a[3]), d(a[4])));
      case "dvdist":
        return constant(s, () -> LatLonDocValuesField.newSlowDistanceQuery(field, d(a[2]), d(a[3]), d(a[4])));
      case "poly":
        return constant(s, () -> LatLonPoint.newPolygonQuery(field, polygons(a[2])));
      case "dvpoly":
        return constant(s, () -> LatLonDocValuesField.newSlowPolygonQuery(field, polygons(a[2])));
      case "geom":
        return constant(s, () -> LatLonPoint.newGeometryQuery(field, ShapeField.QueryRelation.valueOf(a[2]), latLon(a[3])));
      case "dvgeom":
        return constant(s, () -> LatLonDocValuesField.newSlowGeometryQuery(field, ShapeField.QueryRelation.valueOf(a[2]), latLon(a[3])));
      case "xygeom":
        return constant(s, () -> XYPointField.newGeometryQuery(field, xy(a[2])));
      case "xydvgeom":
        return constant(s, () -> XYDocValuesField.newSlowGeometryQuery(field, xy(a[2])));
      case "xybox":
        return constant(s, () -> XYPointField.newBoxQuery(field, f(a[2]), f(a[3]), f(a[4]), f(a[5])));
      case "xydvbox":
        return constant(s, () -> XYDocValuesField.newSlowBoxQuery(field, f(a[2]), f(a[3]), f(a[4]), f(a[5])));
      case "xydist":
        return constant(s, () -> XYPointField.newDistanceQuery(field, f(a[2]), f(a[3]), f(a[4])));
      case "xydvdist":
        return constant(s, () -> XYDocValuesField.newSlowDistanceQuery(field, f(a[2]), f(a[3]), f(a[4])));
      case "feature":
        {
          Query q = LatLonPoint.newDistanceFeatureQuery(field, f(a[2]), d(a[3]), d(a[4]), d(a[5]));
          TopDocs td = s.search(q, Integer.parseInt(a[6]));
          StringBuilder sb = new StringBuilder("S\t").append(td.totalHits.value()).append('\t').append(td.totalHits.relation()).append('\t');
          for (int i = 0; i < td.scoreDocs.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(td.scoreDocs[i].doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(td.scoreDocs[i].score)));
          }
          return sb.toString();
        }
      case "sort":
      case "xysort":
        {
          Query q;
          if (a[5].equals("all")) {
            q = new MatchAllDocsQuery();
          } else {
            String[] b = a[5].substring(4).split(",");
            q = LatLonPoint.newBoxQuery("ll", d(b[0]), d(b[1]), d(b[2]), d(b[3]));
          }
          SortField sf =
              a[0].equals("sort")
                  ? LatLonDocValuesField.newDistanceSort(field, d(a[2]), d(a[3]))
                  : XYDocValuesField.newDistanceSort(field, f(a[2]), f(a[3]));
          return fieldDocs("T", s.search(q, Integer.parseInt(a[4]), new Sort(sf)));
        }
      case "nearest":
        return fieldDocs("N", LatLonPoint.nearest(s, field, d(a[2]), d(a[3]), Integer.parseInt(a[4])));
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
      System.out.println("VerifyGeoPoints: " + n + " queries answer as over Lucene's own index");
    }
  }
}
