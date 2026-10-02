import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LatLonShape;
import org.apache.lucene.document.LatLonShapeDocValuesField;
import org.apache.lucene.document.ShapeAccess;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.XYShape;
import org.apache.lucene.document.XYShapeDocValuesField;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.Component2D;
import org.apache.lucene.geo.GeoEncodingUtils;
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
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * The shape fields and queries (M9 T9.3), differentially: indexes a seeded random corpus of {@link
 * LatLonShape} and {@link XYShape} shapes -- points, lines, polygons with holes, several shapes per
 * document, shapes on the poles and the dateline, slivers and collinear runs, exact duplicates,
 * documents without the field, several segments with deletions -- with their shape doc values, and
 * records Lucene's answer to many random queries of every geometry under every relation, indexed
 * and doc-values forms.
 *
 * <p>Fields: {@code shape} (0-3 lat/lon shapes per document, triangles and one doc value),
 * {@code one} (exactly one line or polygon in every document of the first three segments: the
 * dense scorers' "every document has a value" path), {@code pt} (exactly one point in those
 * documents: the single-valued sparse and inverse paths), {@code xy} (0-2 cartesian shapes and a
 * doc value); {@code id} is the delete key.
 *
 * <p>Outputs, under {@code geo_shapes/}:
 *
 * <ul>
 *   <li>{@code index/}, {@code docs.tsv} ({@code id}, then one {@code field<TAB>spec} pair per
 *       shape, {@link GeoCorpus} specs), {@code deletes.tsv};
 *   <li>{@code queries.tsv}: the query's tokens, {@code =>}, then {@code C total hexbits
 *       scorebits} (bit d of the hex byte string is global doc d; every hit has the one score) or
 *       {@code E message};
 *   <li>{@code triangles.tsv}: {@code ShapeField.encodeTriangle} on random triangles ({@code
 *       aX,aY,ab,bX,bY,bc,cX,cY,ca}, then the 28 bytes in hex or {@code E}), and {@code
 *       decodeTriangle} of the bytes ({@code aX,aY,bX,bY,cX,cY,ab,bc,ca,TYPE});
 *   <li>{@code doc_values.tsv}: a shape's doc value ({@code latlon|xy}, how it was built, the spec,
 *       the bytes in hex, then the header and centroid/bounding box) and, on following {@code rel}
 *       lines, {@code ShapeDocValues.relate} against random query geometries.
 * </ul>
 */
public class GenGeoShapes {
  static final int SEGMENTS = 3;
  static final int DOCS_PER_SEGMENT = 500;
  static final int SMALL_SEGMENT = 40;

  static final double[][] CLUSTERS = {
    {0, 0}, {89.5, 10}, {-89.7, -170}, {10, 179.5}, {-30, -179.6}, {45.5, 7.25}, {40.7, -74.0},
  };

  static double rd(double v) {
    return Math.rint(v * 1e5) / 1e5;
  }

  static double lat(double v) {
    return rd(Math.max(-90, Math.min(90, v)));
  }

  static double lon(double v) {
    return rd(Math.max(-180, Math.min(180, v)));
  }

  static double[] center(Random r) {
    if (r.nextInt(4) > 0) {
      double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
      return new double[] {lat(c[0] + r.nextGaussian() * 0.5), lon(c[1] + r.nextGaussian() * 0.5)};
    }
    double[] c = GeoCorpus.center(r);
    return new double[] {lat(c[0]), lon(c[1])};
  }

  /** A star ring around (clat, clon), rounded: simple unless rounding folds it. */
  static double[][] star(Random r, double clat, double clon, double radius, int n, double minFrac) {
    double[] angles = new double[n];
    for (int i = 0; i < n; i++) angles[i] = r.nextDouble() * 2 * Math.PI;
    java.util.Arrays.sort(angles);
    double[] lats = new double[n + 1];
    double[] lons = new double[n + 1];
    for (int i = 0; i < n; i++) {
      double rad = radius * (minFrac + (1 - minFrac) * r.nextDouble());
      lats[i] = lat(clat + rad * Math.sin(angles[i]));
      lons[i] = lon(clon + rad * Math.cos(angles[i]));
    }
    lats[n] = lats[0];
    lons[n] = lons[0];
    return new double[][] {lats, lons};
  }

  static Polygon starPolygon(Random r, double[] c, double radius, int n, int holes) {
    double[][] ring = star(r, c[0], c[1], radius, n, holes > 0 ? 0.85 : 0.4);
    if (r.nextBoolean()) ring = GeoCorpus.reverse(ring);
    Polygon[] hs = new Polygon[holes];
    for (int h = 0; h < holes; h++) {
      double a = 2 * Math.PI * h / holes;
      double off = holes == 1 ? 0 : radius * 0.45;
      double[][] hole = star(r, c[0] + off * Math.sin(a), c[1] + off * Math.cos(a), radius * 0.18, 3 + r.nextInt(6), 0.5);
      if (r.nextBoolean()) hole = GeoCorpus.reverse(hole);
      hs[h] = new Polygon(hole[0], hole[1]);
    }
    return new Polygon(ring[0], ring[1], hs);
  }

  static final List<LatLonGeometry> SEEN = new ArrayList<>();

  /** One random lat/lon shape; polygons the tessellator rejects are retried by the caller. */
  static LatLonGeometry shape(Random r) {
    double[] c = center(r);
    double radius = Math.pow(10, -3 + r.nextDouble() * 3.3);
    switch (r.nextInt(16)) {
      case 0:
      case 1:
        return new Point(c[0], c[1]);
      case 2:
        // the poles and the dateline exactly
        return new Point(r.nextBoolean() ? (r.nextBoolean() ? 90 : -90) : c[0], r.nextBoolean() ? (r.nextBoolean() ? 180 : -180) : c[1]);
      case 3:
      case 4:
        {
          int n = 2 + r.nextInt(10);
          double[] lats = new double[n];
          double[] lons = new double[n];
          boolean meridian = r.nextInt(5) == 0;
          for (int k = 0; k < n; k++) {
            lats[k] = lat(c[0] + (r.nextDouble() * 2 - 1) * radius);
            lons[k] = meridian ? c[1] : lon(c[1] + (r.nextDouble() * 2 - 1) * radius);
          }
          return new Line(lats, lons);
        }
      case 5:
        {
          // a line along the dateline or through a pole
          double l0 = r.nextBoolean() ? 180 : -180;
          return new Line(new double[] {lat(c[0] - radius), lat(c[0] + radius)}, new double[] {l0, l0});
        }
      case 6:
      case 7:
      case 8:
        return starPolygon(r, c, radius, 3 + r.nextInt(20), 0);
      case 9:
        return starPolygon(r, c, radius, 6 + r.nextInt(20), 1 + r.nextInt(3));
      case 10:
        {
          // touching a pole or the dateline exactly
          double lat0 = r.nextBoolean() ? 90 : -90;
          double lon0 = r.nextBoolean() ? 180 : -180;
          double w = rd(0.5 + r.nextDouble() * 5);
          double sLat = Math.signum(lat0);
          double sLon = Math.signum(lon0);
          if (r.nextBoolean()) {
            double[] lats = {lat0, lat0 - sLat * w, lat0 - sLat * w, lat0, lat0};
            double[] lons = {c[1], c[1], lon(c[1] + w), lon(c[1] + w), c[1]};
            return new Polygon(lats, lons);
          } else {
            double[] lats = {c[0], c[0], lat(c[0] + w), lat(c[0] + w), c[0]};
            double[] lons = {lon0, lon0 - sLon * w, lon0 - sLon * w, lon0, lon0};
            return new Polygon(lats, lons);
          }
        }
      case 11:
        {
          // a sliver: nearly degenerate
          double d = Math.max(1e-5, rd(radius * 1e-3));
          double[] lats = {c[0], lat(c[0] + d), lat(c[0] + radius), c[0]};
          double[] lons = {c[1], lon(c[1] + radius), lon(c[1] + radius + d), c[1]};
          return new Polygon(lats, lons);
        }
      case 12:
        {
          // a box with collinear runs and a duplicate vertex
          double[] lats = {c[0], c[0], c[0], lat(c[0] + radius), lat(c[0] + radius), lat(c[0] + radius), c[0]};
          double[] lons = {c[1], lon(c[1] + radius / 2), lon(c[1] + radius), lon(c[1] + radius), lon(c[1] + radius), c[1], c[1]};
          return new Polygon(lats, lons);
        }
      case 13:
        // many vertices: the tessellator's morton path, a deep doc-value tree
        return starPolygon(r, c, radius, 81 + r.nextInt(12), r.nextInt(3) == 0 ? 1 : 0);
      case 14:
        if (!SEEN.isEmpty()) return SEEN.get(r.nextInt(SEEN.size()));
        return new Point(c[0], c[1]);
      default:
        {
          // exactly on encoded values
          double la = GeoEncodingUtils.decodeLatitude(r.nextInt());
          double lo = GeoEncodingUtils.decodeLongitude(r.nextInt());
          if (r.nextBoolean()) return new Point(la, lo);
          double la2 = GeoEncodingUtils.decodeLatitude(GeoEncodingUtils.encodeLatitude(la) + 1 + r.nextInt(1000));
          double lo2 = GeoEncodingUtils.decodeLongitude(GeoEncodingUtils.encodeLongitude(lo) + 1 + r.nextInt(1000));
          return new Line(new double[] {la, Math.min(90, la2)}, new double[] {lo, Math.min(180, lo2)});
        }
    }
  }

  static Field[] fields(String name, LatLonGeometry g) {
    if (g instanceof Point p) return LatLonShape.createIndexableFields(name, p.getLat(), p.getLon());
    if (g instanceof Line l) return LatLonShape.createIndexableFields(name, l);
    return LatLonShape.createIndexableFields(name, (Polygon) g);
  }

  /** A shape that indexes (the tessellator accepts it). */
  static LatLonGeometry indexable(Random r, boolean linesOrPolygons) {
    for (int attempt = 0; ; attempt++) {
      LatLonGeometry g = shape(r);
      if (linesOrPolygons && (g instanceof Point || (g instanceof Polygon p && p.numPoints() > 40))) continue;
      try {
        fields("x", g);
        SEEN.add(g);
        return g;
      } catch (IllegalArgumentException e) {
        if (attempt > 100) throw e;
      }
    }
  }

  static final float[][] XY_CLUSTERS = {{0, 0}, {1000, -1000}, {-3.5e6f, 2e6f}, {1e-3f, 1e-3f}};

  static float fr(double v) {
    return (float) v;
  }

  static XYGeometry xyShape(Random r) {
    float[] c = XY_CLUSTERS[r.nextInt(XY_CLUSTERS.length)];
    double s = Math.pow(10, r.nextDouble() * 4 - 2) * (Math.abs(c[0]) * 1e-3 + 1);
    double cx = c[0] + r.nextGaussian() * s;
    double cy = c[1] + r.nextGaussian() * s;
    switch (r.nextInt(6)) {
      case 0:
        return new XYPoint(fr(cx), fr(cy));
      case 1:
        {
          int n = 2 + r.nextInt(8);
          float[] x = new float[n];
          float[] y = new float[n];
          for (int k = 0; k < n; k++) {
            x[k] = fr(cx + (r.nextDouble() * 2 - 1) * s);
            y[k] = fr(cy + (r.nextDouble() * 2 - 1) * s);
          }
          return new XYLine(x, y);
        }
      default:
        {
          int holes = r.nextInt(4) == 0 ? 1 : 0;
          double[][] ring = GeoCorpus.starRing(r, 0, 0, 1, 3 + r.nextInt(20), holes > 0 ? 0.85 : 0.4, false);
          float[] x = new float[ring[0].length];
          float[] y = new float[ring[0].length];
          for (int i = 0; i < x.length; i++) {
            x[i] = fr(cx + ring[1][i] * s);
            y[i] = fr(cy + ring[0][i] * s);
          }
          if (holes == 0) return new XYPolygon(x, y);
          double[][] h = GeoCorpus.starRing(r, 0, 0, 0.2, 3 + r.nextInt(5), 0.5, false);
          float[] hx = new float[h[0].length];
          float[] hy = new float[h[0].length];
          for (int i = 0; i < hx.length; i++) {
            hx[i] = fr(cx + h[1][i] * s);
            hy[i] = fr(cy + h[0][i] * s);
          }
          return new XYPolygon(x, y, new XYPolygon(hx, hy));
        }
    }
  }

  static Field[] xyFields(String name, XYGeometry g) {
    if (g instanceof XYPoint p) return XYShape.createIndexableFields(name, p.getX(), p.getY());
    if (g instanceof XYLine l) return XYShape.createIndexableFields(name, l);
    return XYShape.createIndexableFields(name, (XYPolygon) g);
  }

  static XYGeometry xyIndexable(Random r) {
    for (int attempt = 0; ; attempt++) {
      XYGeometry g = xyShape(r);
      try {
        xyFields("x", g);
        return g;
      } catch (IllegalArgumentException e) {
        if (attempt > 100) throw e;
      }
    }
  }

  // --- documents ----------------------------------------------------------------------------

  /** Java's way to the doc value of one shape. */
  static LatLonShapeDocValuesField latLonDocValue(String name, LatLonGeometry g) {
    if (g instanceof Point p) return LatLonShape.createDocValueField(name, p.getLat(), p.getLon());
    if (g instanceof Line l) return LatLonShape.createDocValueField(name, l);
    return LatLonShape.createDocValueField(name, (Polygon) g);
  }

  static XYShapeDocValuesField xyDocValue(String name, XYGeometry g) {
    if (g instanceof XYPoint p) return XYShape.createDocValueField(name, p.getX(), p.getY());
    if (g instanceof XYLine l) return XYShape.createDocValueField(name, l);
    return XYShape.createDocValueField(name, (XYPolygon) g);
  }

  /** Several shapes' doc value: their triangle fields decoded. */
  static List<ShapeField.DecodedTriangle> decoded(List<Field> fs) {
    List<ShapeField.DecodedTriangle> out = new ArrayList<>();
    for (Field f : fs) {
      BytesRef br = f.binaryValue();
      byte[] b = new byte[br.length];
      System.arraycopy(br.bytes, br.offset, b, 0, br.length);
      ShapeField.DecodedTriangle t = new ShapeField.DecodedTriangle();
      ShapeField.decodeTriangle(b, t);
      out.add(t);
    }
    return out;
  }

  /**
   * The document {@code GenGeoShapes} indexes for one {@code docs.tsv} line, and the order its
   * fields are added in -- which {@code VerifyGeoShapes} and the Rust test repeat: {@code id}, each
   * shape's triangles in line order, then the {@code shape} doc value, then the {@code xy} one.
   */
  static Document document(String line) {
    String[] p = line.split("\t");
    Document doc = new Document();
    doc.add(new StringField("id", p[0], Field.Store.YES));
    List<Field> shape = new ArrayList<>();
    List<LatLonGeometry> shapes = new ArrayList<>();
    List<Field> xy = new ArrayList<>();
    List<XYGeometry> xys = new ArrayList<>();
    for (int i = 1; i < p.length; i += 2) {
      String field = p[i];
      if (field.equals("xy")) {
        XYGeometry g = parseXY(p[i + 1]);
        Field[] fs = xyFields(field, g);
        for (Field f : fs) doc.add(f);
        xy.addAll(List.of(fs));
        xys.add(g);
      } else {
        LatLonGeometry g = parseLatLon(p[i + 1]);
        Field[] fs = fields(field, g);
        for (Field f : fs) doc.add(f);
        if (field.equals("shape")) {
          shape.addAll(List.of(fs));
          shapes.add(g);
        }
      }
    }
    if (shapes.size() == 1) {
      doc.add(latLonDocValue("shape", shapes.get(0)));
    } else if (shapes.size() > 1) {
      doc.add(LatLonShape.createDocValueField("shape", shape.toArray(new Field[0])));
    }
    if (xys.size() == 1) {
      doc.add(xyDocValue("xy", xys.get(0)));
    } else if (xys.size() > 1) {
      doc.add(XYShape.createDocValueField("xy", decoded(xy)));
    }
    return doc;
  }

  static LatLonGeometry parseLatLon(String spec) {
    return VerifyGeoPoints.latLon(spec)[0];
  }

  static XYGeometry parseXY(String spec) {
    return VerifyGeoPoints.xy(spec)[0];
  }

  static String docLine(Random r, int id, boolean small) {
    StringBuilder sb = new StringBuilder().append(id);
    if (id == SEGMENTS * DOCS_PER_SEGMENT) {
      // two polygons either side of the dateline: a CONTAINS box across it
      // is the conjunction of its halves. Lucene matches neither half: each
      // reaches +-180, which the polygons' boundaries touch (NOTWITHIN)
      sb.append("\tshape\tG:0.0 170.0;0.0 180.0;10.0 180.0;10.0 170.0;0.0 170.0");
      sb.append("\tshape\tG:0.0 -180.0;0.0 -170.0;10.0 -170.0;10.0 -180.0;0.0 -180.0");
      return sb.toString();
    }
    int n = r.nextInt(5) == 0 ? 0 : 1 + (r.nextInt(4) == 0 ? 1 + r.nextInt(2) : 0);
    for (int i = 0; i < n; i++) sb.append("\tshape\t").append(GeoCorpus.spec(indexable(r, false)));
    if (!small) {
      sb.append("\tone\t").append(GeoCorpus.spec(indexable(r, true)));
      double[] c = center(r);
      sb.append("\tpt\t").append(GeoCorpus.spec(new Point(c[0], c[1])));
      int nxy = r.nextInt(2) == 0 ? 0 : 1 + (r.nextInt(5) == 0 ? 1 : 0);
      for (int i = 0; i < nxy; i++) sb.append("\txy\t").append(GeoCorpus.spec(xyIndexable(r)));
    }
    return sb.toString();
  }

  // --- random queries ---------------------------------------------------------------------------

  static double[] box(Random r) {
    double[] c = center(r);
    double h = Math.pow(10, -2 + r.nextDouble() * 3);
    double w = Math.pow(10, -2 + r.nextDouble() * 3);
    double minLat = lat(c[0] - h), maxLat = lat(c[0] + h);
    double minLon = lon(c[1] - w), maxLon = lon(c[1] + w);
    switch (r.nextInt(10)) {
      case 0:
        return new double[] {-90, 90, -180, 180};
      case 1:
      case 2:
        // across the dateline
        return new double[] {minLat, maxLat, rd(170 + r.nextDouble() * 10), rd(-180 + r.nextDouble() * 10)};
      case 3:
        return new double[] {minLat, maxLat, 180, r.nextBoolean() ? 180 : -170};
      case 4:
        return new double[] {rd(80 + r.nextDouble() * 10), 90, -180, 180};
      case 5:
        return new double[] {minLat, maxLat, -180, maxLon};
      default:
        return new double[] {minLat, maxLat, minLon, maxLon};
    }
  }

  static Polygon queryPolygon(Random r) {
    for (int attempt = 0; ; attempt++) {
      try {
        double[] c = center(r);
        double rad = Math.pow(10, -2 + r.nextDouble() * 2.5);
        return starPolygon(r, c, rad, 3 + r.nextInt(25), r.nextInt(3) == 0 ? 1 + r.nextInt(2) : 0);
      } catch (IllegalArgumentException e) {
        if (attempt > 50) throw e;
      }
    }
  }

  static LatLonGeometry geometry(Random r) {
    switch (r.nextInt(7)) {
      case 0:
        {
          double[] c = center(r);
          double radius = r.nextInt(6) == 0 ? rd(r.nextDouble() * 50) : rd(Math.pow(10, 2 + r.nextDouble() * 4.5));
          return new Circle(c[0], c[1], radius);
        }
      case 1:
        {
          double[] b = box(r);
          return new Rectangle(b[0], b[1], b[2], b[3]);
        }
      case 2:
        {
          int n = 2 + r.nextInt(8);
          double[] c = center(r);
          double radius = Math.pow(10, -2 + r.nextDouble() * 2.5);
          double[] lats = new double[n];
          double[] lons = new double[n];
          for (int k = 0; k < n; k++) {
            lats[k] = lat(c[0] + (r.nextDouble() * 2 - 1) * radius);
            lons[k] = lon(c[1] + (r.nextDouble() * 2 - 1) * radius);
          }
          return new Line(lats, lons);
        }
      case 3:
        {
          if (r.nextInt(3) == 0 && !SEEN.isEmpty()) {
            // a vertex of an indexed shape
            LatLonGeometry g = SEEN.get(r.nextInt(SEEN.size()));
            if (g instanceof Point p) return p;
            if (g instanceof Line l) return new Point(l.getLat(0), l.getLon(0));
            Polygon p = (Polygon) g;
            return new Point(p.getPolyLat(0), p.getPolyLon(0));
          }
          double[] c = center(r);
          return new Point(c[0], c[1]);
        }
      case 4:
        if (!SEEN.isEmpty() && r.nextInt(3) == 0) {
          // an indexed polygon itself
          LatLonGeometry g = SEEN.get(r.nextInt(SEEN.size()));
          if (g instanceof Polygon p) return p;
        }
        return queryPolygon(r);
      default:
        return queryPolygon(r);
    }
  }

  static LatLonGeometry[] geometries(Random r) {
    int n = r.nextInt(4) == 0 ? 2 + r.nextInt(2) : 1;
    LatLonGeometry[] g = new LatLonGeometry[n];
    for (int i = 0; i < n; i++) g[i] = geometry(r);
    return g;
  }

  static XYGeometry xyGeometry(Random r) {
    float[] c = XY_CLUSTERS[r.nextInt(XY_CLUSTERS.length)];
    double s = Math.pow(10, r.nextDouble() * 4 - 2) * (Math.abs(c[0]) * 1e-3 + 1);
    switch (r.nextInt(5)) {
      case 0:
        return new XYCircle(c[0] + (float) r.nextGaussian(), c[1] + (float) r.nextGaussian(), (float) (s * 2));
      case 1:
        return new XYRectangle((float) (c[0] - s), (float) (c[0] + s * r.nextDouble()), (float) (c[1] - s * r.nextDouble()), (float) (c[1] + s));
      case 2:
        return new XYPoint(c[0] + (float) (r.nextGaussian() * s), c[1] + (float) (r.nextGaussian() * s));
      case 3:
        {
          int n = 2 + r.nextInt(6);
          float[] x = new float[n];
          float[] y = new float[n];
          for (int k = 0; k < n; k++) {
            x[k] = fr(c[0] + (r.nextDouble() * 2 - 1) * s);
            y[k] = fr(c[1] + (r.nextDouble() * 2 - 1) * s);
          }
          return new XYLine(x, y);
        }
      default:
        {
          double[][] ring = GeoCorpus.starRing(r, 0, 0, 1, 3 + r.nextInt(20), 0.4, false);
          float[] x = new float[ring[0].length];
          float[] y = new float[ring[0].length];
          for (int i = 0; i < x.length; i++) {
            x[i] = (float) (c[0] + ring[1][i] * s);
            y[i] = (float) (c[1] + ring[0][i] * s);
          }
          return new XYPolygon(x, y);
        }
    }
  }

  static XYGeometry[] xyGeometries(Random r) {
    int n = r.nextInt(4) == 0 ? 2 + r.nextInt(2) : 1;
    XYGeometry[] g = new XYGeometry[n];
    for (int i = 0; i < n; i++) g[i] = xyGeometry(r);
    return g;
  }

  static final ShapeField.QueryRelation[] RELATIONS = ShapeField.QueryRelation.values();

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
  }

  static String d(double v) {
    return Double.toString(v);
  }

  static String f(float v) {
    return Float.toString(v);
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("geo_shapes");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261003L);
    StringBuilder docsOut = new StringBuilder();
    StringBuilder deletesOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        for (int seg = 0; seg <= SEGMENTS; seg++) {
          boolean small = seg == SEGMENTS;
          int n = small ? SMALL_SEGMENT : DOCS_PER_SEGMENT;
          for (int i = 0; i < n; i++, id++) {
            String line = docLine(r, id, small);
            docsOut.append(line).append('\n');
            w.addDocument(document(line));
          }
          w.commit();
        }
        for (int d = 0; d < 2 * DOCS_PER_SEGMENT; d += 1 + r.nextInt(25)) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
          deletesOut.append(d).append('\n');
        }
        w.commit();
      }
    }
    Files.writeString(root.resolve("docs.tsv"), docsOut.toString());
    Files.writeString(root.resolve("deletes.tsv"), deletesOut.toString());

    StringBuilder q = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != SEGMENTS + 1) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] fields = {"shape", "one", "pt"};
      List<String> lines = new ArrayList<>();
      for (int i = 0; i < 640; i++) {
        String field = fields[i % 3];
        ShapeField.QueryRelation rel = RELATIONS[(i / 3) % RELATIONS.length];
        String spec = GeoCorpus.spec(geometries(r));
        lines.add("geom\t" + field + "\t" + rel + "\t" + spec);
        if (field.equals("shape")) lines.add("dvgeom\t" + field + "\t" + rel + "\t" + spec);
      }
      for (int i = 0; i < 160; i++) {
        String field = fields[i % 3];
        ShapeField.QueryRelation rel = RELATIONS[(i / 3) % RELATIONS.length];
        double[] b = box(r);
        String qa = field + "\t" + rel + "\t" + d(b[0]) + "\t" + d(b[1]) + "\t" + d(b[2]) + "\t" + d(b[3]);
        lines.add("box\t" + qa);
        if (field.equals("shape")) lines.add("dvbox\t" + qa);
      }
      for (String rel : new String[] {"CONTAINS", "INTERSECTS", "WITHIN", "DISJOINT"}) {
        lines.add("box\tshape\t" + rel + "\t2.0\t8.0\t175.0\t-175.0");
        lines.add("dvbox\tshape\t" + rel + "\t2.0\t8.0\t175.0\t-175.0");
      }
      for (int i = 0; i < 240; i++) {
        ShapeField.QueryRelation rel = RELATIONS[i % RELATIONS.length];
        String spec = GeoCorpus.spec(xyGeometries(r));
        lines.add("xygeom\txy\t" + rel + "\t" + spec);
        lines.add("xydvgeom\txy\t" + rel + "\t" + spec);
      }
      for (int i = 0; i < 80; i++) {
        ShapeField.QueryRelation rel = RELATIONS[i % RELATIONS.length];
        XYGeometry g = xyGeometry(r);
        if (!(g instanceof XYRectangle rect)) continue;
        String qa = "xy\t" + rel + "\t" + f(rect.minX) + "\t" + f(rect.maxX) + "\t" + f(rect.minY) + "\t" + f(rect.maxY);
        lines.add("xybox\t" + qa);
        lines.add("xydvbox\t" + qa);
      }
      for (String line : lines) {
        q.append(line).append("\t=>\t").append(VerifyGeoShapes.run(s, line.split("\t"))).append('\n');
      }
    }
    Files.writeString(root.resolve("queries.tsv"), q.toString());
    triangles(root, r);
    docValues(root, r);
  }

  // --- the encodings on their own -----------------------------------------------------------------

  static int coord(Random r, int base) {
    switch (r.nextInt(4)) {
      case 0:
        return r.nextInt();
      case 1:
        return base + r.nextInt(5) - 2;
      case 2:
        return r.nextBoolean() ? Integer.MAX_VALUE : Integer.MIN_VALUE;
      default:
        return base + r.nextInt(2001) - 1000;
    }
  }

  static void triangles(Path root, Random r) throws IOException {
    StringBuilder sb = new StringBuilder();
    byte[] bytes = new byte[7 * Integer.BYTES];
    for (int i = 0; i < 1500; i++) {
      int bx = r.nextInt(), by = r.nextInt();
      int[] v = new int[6];
      for (int k = 0; k < 6; k++) v[k] = coord(r, k % 2 == 0 ? bx : by);
      if (i % 7 == 0) {
        // two vertices the same, or all three
        v[2] = v[0];
        v[3] = v[1];
        if (i % 21 == 0) {
          v[4] = v[0];
          v[5] = v[1];
        }
      } else if (i % 11 == 0) {
        // one meridian
        v[2] = v[0];
        v[4] = v[0];
      }
      boolean ab = r.nextBoolean(), bc = r.nextBoolean(), ca = r.nextBoolean();
      sb.append(v[0]).append(',').append(v[1]).append(',').append(ab).append(',')
          .append(v[2]).append(',').append(v[3]).append(',').append(bc).append(',')
          .append(v[4]).append(',').append(v[5]).append(',').append(ca).append('\t');
      try {
        ShapeField.encodeTriangle(bytes, v[1], v[0], ab, v[3], v[2], bc, v[5], v[4], ca);
      } catch (IllegalArgumentException e) {
        sb.append("E\t").append(e.getMessage()).append('\n');
        continue;
      }
      for (byte b : bytes) sb.append(String.format("%02x", b & 0xff));
      ShapeField.DecodedTriangle t = new ShapeField.DecodedTriangle();
      ShapeField.decodeTriangle(bytes, t);
      sb.append('\t').append(t.aX).append(',').append(t.aY).append(',').append(t.bX).append(',').append(t.bY)
          .append(',').append(t.cX).append(',').append(t.cY).append(',').append(t.ab).append(',').append(t.bc)
          .append(',').append(t.ca).append(',').append(t.type).append('\n');
    }
    Files.writeString(root.resolve("triangles.tsv"), sb.toString());
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return sb.toString();
  }

  static void docValues(Path root, Random r) throws IOException {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 140; i++) {
      boolean xy = i % 4 == 3;
      String how;
      String spec;
      org.apache.lucene.document.ShapeDocValuesField field;
      if (xy) {
        int n = i % 8 == 7 ? 2 + r.nextInt(2) : 1;
        XYGeometry[] gs = new XYGeometry[n];
        List<Field> fs = new ArrayList<>();
        for (int k = 0; k < n; k++) {
          gs[k] = xyIndexable(r);
          fs.addAll(List.of(xyFields("f", gs[k])));
        }
        spec = GeoCorpus.spec(gs);
        how = n == 1 ? "geometry" : "fields";
        field = n == 1 ? xyDocValue("f", gs[0]) : XYShape.createDocValueField("f", decoded(fs));
      } else {
        int n = i % 6 == 5 ? 2 + r.nextInt(3) : 1;
        LatLonGeometry[] gs = new LatLonGeometry[n];
        List<Field> fs = new ArrayList<>();
        for (int k = 0; k < n; k++) {
          gs[k] = indexable(r, false);
          fs.addAll(List.of(fields("f", gs[k])));
        }
        spec = GeoCorpus.spec(gs);
        how = n == 1 ? "geometry" : "fields";
        field = n == 1 ? latLonDocValue("f", gs[0]) : LatLonShape.createDocValueField("f", fs.toArray(new Field[0]));
      }
      BytesRef value = field.binaryValue();
      sb.append("dv\t").append(xy ? "xy" : "latlon").append('\t').append(how).append('\t').append(spec).append('\t')
          .append(hex(value)).append('\t').append(ShapeAccess.header(ShapeAccess.docValues(field))).append('\t');
      if (xy) {
        XYPoint c = ((XYShapeDocValuesField) field).getCentroid();
        XYRectangle b = ((XYShapeDocValuesField) field).getBoundingBox();
        sb.append(f(c.getX())).append(',').append(f(c.getY())).append('\t')
            .append(f(b.minX)).append(',').append(f(b.maxX)).append(',').append(f(b.minY)).append(',').append(f(b.maxY));
      } else {
        Point c = ((LatLonShapeDocValuesField) field).getCentroid();
        Rectangle b = ((LatLonShapeDocValuesField) field).getBoundingBox();
        sb.append(d(c.getLat())).append(',').append(d(c.getLon())).append('\t')
            .append(d(b.minLat)).append(',').append(d(b.maxLat)).append(',').append(d(b.minLon)).append(',').append(d(b.maxLon));
      }
      sb.append('\n');
      for (int k = 0; k < 5; k++) {
        String gspec;
        Component2D component;
        if (xy) {
          XYGeometry[] gs = xyGeometries(r);
          gspec = GeoCorpus.spec(gs);
          component = XYGeometry.create(gs);
          sb.append("rel\t").append(gspec).append('\t').append(ShapeAccess.relateXY(value, component)).append('\n');
        } else {
          LatLonGeometry[] gs = geometries(r);
          gspec = GeoCorpus.spec(gs);
          component = LatLonGeometry.create(gs);
          sb.append("rel\t").append(gspec).append('\t').append(ShapeAccess.relateLatLon(value, component)).append('\n');
        }
      }
    }
    Files.writeString(root.resolve("doc_values.tsv"), sb.toString());
  }
}
