import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.Component2D;
import org.apache.lucene.geo.GeoEncodingUtils;
import org.apache.lucene.geo.GeoUtils;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Point;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.geo.XYCircle;
import org.apache.lucene.geo.XYEncodingUtils;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.geo.XYLine;
import org.apache.lucene.geo.XYPoint;
import org.apache.lucene.geo.XYPolygon;
import org.apache.lucene.geo.XYRectangle;
import org.apache.lucene.index.PointValues.Relation;
import org.apache.lucene.util.SloppyMath;

/**
 * Cross-engine ground truth for {@code lucene-util}'s {@code geo} module, {@code sloppy_math} and
 * {@code strict_math}: plain-text files under {@code geo/} that {@code
 * crates/lucene-util/tests/geo_fixtures.rs} replays.
 *
 * <ul>
 *   <li>{@code strict_math.tsv}: {@code StrictMath.sin/cos/asin/acos} on random arguments, and an
 *       FNV-1a digest over the exact argument sweeps SloppyMath builds its tables from.
 *   <li>{@code sloppy_math.tsv}: {@code SloppyMath} haversin, cos, sin, asin.
 *   <li>{@code encoding.tsv}: {@code GeoEncodingUtils} and {@code XYEncodingUtils}, including the
 *       validation errors.
 *   <li>{@code geo_utils.tsv}: {@code GeoUtils} predicates, {@code Rectangle.fromPointDistance},
 *       {@code axisLat}, {@code XYRectangle.fromPointDistance}, the distance predicate.
 *   <li>{@code component2d.tsv}: shapes (polygons with holes, lines, circles, rectangles, points,
 *       multi-shapes; lat/lon and cartesian) and every {@code Component2D} query on random boxes,
 *       points, lines and triangles, plus {@code createComponentPredicate}.
 * </ul>
 *
 * Doubles in the math files are raw bits in hex, so NaN payloads and signed zeros are exact;
 * geometry and query numbers are {@code Double.toString}/{@code Float.toString}, which round-trip.
 */
public class GenGeo {
  static final Random R = new Random(0x6E0_9A11L);

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String hf(float v) {
    return Integer.toHexString(Float.floatToRawIntBits(v));
  }

  static long fnv = 0xcbf29ce484222325L;

  static void fnvAdd(double v) {
    fnv ^= Double.doubleToRawLongBits(v);
    fnv *= 0x100000001b3L;
  }

  static String err(Throwable t) {
    return "ERR\t" + t.getClass().getName() + "\t" + GeoCorpus.esc(String.valueOf(t.getMessage()));
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo");
    Files.createDirectories(out);
    Files.writeString(out.resolve("strict_math.tsv"), strictMath());
    Files.writeString(out.resolve("sloppy_math.tsv"), sloppyMath());
    Files.writeString(out.resolve("encoding.tsv"), encoding());
    Files.writeString(out.resolve("geo_utils.tsv"), geoUtils());
    Files.writeString(out.resolve("component2d.tsv"), component2d());
  }

  // ------------------------------------------------------------------ math

  static double randomArg(double scale) {
    switch (R.nextInt(5)) {
      case 0:
        return (R.nextDouble() * 2 - 1) * 1e-9;
      case 1:
        return Math.PI / 2 * (R.nextInt(40) - 20) + (R.nextDouble() - 0.5) * 1e-12;
      default:
        return (R.nextDouble() * 2 - 1) * scale;
    }
  }

  static String strictMath() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 1500; i++) {
      double x = randomArg(i < 1000 ? 8 : 1e5);
      sb.append("sin\t").append(h(x)).append('\t').append(h(StrictMath.sin(x))).append('\n');
      sb.append("cos\t").append(h(x)).append('\t').append(h(StrictMath.cos(x))).append('\n');
    }
    for (int i = 0; i < 1500; i++) {
      double x = R.nextInt(10) == 0 ? R.nextDouble() * 3 - 1.5 : R.nextDouble() * 2 - 1;
      sb.append("asin\t").append(h(x)).append('\t').append(h(StrictMath.asin(x))).append('\n');
      sb.append("acos\t").append(h(x)).append('\t').append(h(StrictMath.acos(x))).append('\n');
    }
    // SloppyMath's table sweeps (same arithmetic as its static initializer).
    double PIO2_HI = Double.longBitsToDouble(0x3FF921FB54400000L);
    double PIO2_LO = Double.longBitsToDouble(0x3DD0B4611A626331L);
    double TWOPI_HI = 4 * PIO2_HI;
    double TWOPI_LO = 4 * PIO2_LO;
    int SIN_COS_TABS_SIZE = (1 << 11) + 1;
    double DELTA_HI = TWOPI_HI / (SIN_COS_TABS_SIZE - 1);
    double DELTA_LO = TWOPI_LO / (SIN_COS_TABS_SIZE - 1);
    fnv = 0xcbf29ce484222325L;
    for (int i = 0; i < SIN_COS_TABS_SIZE; i++) {
      double angle = i * DELTA_HI + i * DELTA_LO;
      fnvAdd(StrictMath.sin(angle));
      fnvAdd(StrictMath.cos(angle));
    }
    sb.append("sweep_sincos\t").append(Long.toHexString(fnv)).append('\n');
    double asinMax = StrictMath.sin(Math.toRadians(73.0));
    int ASIN_TABS_SIZE = (1 << 13) + 1;
    double ASIN_DELTA = asinMax / (ASIN_TABS_SIZE - 1);
    fnv = 0xcbf29ce484222325L;
    for (int i = 0; i < ASIN_TABS_SIZE; i++) {
      fnvAdd(StrictMath.asin(i * ASIN_DELTA));
    }
    sb.append("sweep_asin\t").append(Long.toHexString(fnv)).append('\n');
    return sb.toString();
  }

  static double lat() {
    return R.nextInt(20) == 0 ? (R.nextBoolean() ? 90 : -90) : R.nextDouble() * 180 - 90;
  }

  static double lon() {
    return R.nextInt(20) == 0 ? (R.nextBoolean() ? 180 : -180) : R.nextDouble() * 360 - 180;
  }

  static String sloppyMath() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 2000; i++) {
      double lat1 = lat(), lon1 = lon();
      double lat2, lon2;
      if (i % 3 == 0) {
        // nearby points
        lat2 = GeoCorpus.clampLat(lat1 + (R.nextDouble() - 0.5) * 1e-3);
        lon2 = GeoCorpus.clampLon(lon1 + (R.nextDouble() - 0.5) * 1e-3);
      } else {
        lat2 = lat();
        lon2 = lon();
      }
      sb.append("hav\t")
          .append(h(lat1))
          .append(',')
          .append(h(lon1))
          .append(',')
          .append(h(lat2))
          .append(',')
          .append(h(lon2))
          .append('\t')
          .append(h(SloppyMath.haversinSortKey(lat1, lon1, lat2, lon2)))
          .append('\t')
          .append(h(SloppyMath.haversinMeters(lat1, lon1, lat2, lon2)))
          .append('\n');
    }
    for (int i = 0; i < 1500; i++) {
      double x = randomArg(i < 1000 ? 10 : 1e9);
      sb.append("cos\t").append(h(x)).append('\t').append(h(SloppyMath.cos(x))).append('\n');
      sb.append("sin\t").append(h(x)).append('\t').append(h(SloppyMath.sin(x))).append('\n');
    }
    for (int i = 0; i < 1500; i++) {
      double x = R.nextInt(10) == 0 ? R.nextDouble() * 2.2 - 1.1 : R.nextDouble() * 2 - 1;
      if (i < 20) x = new double[] {0, -0.0, 1, -1, Double.NaN, 0.956, 0.9563047559630354, 0.99}[i % 8];
      sb.append("asin\t").append(h(x)).append('\t').append(h(SloppyMath.asin(x))).append('\n');
    }
    for (int i = 0; i < 200; i++) {
      double key = i < 4 ? new double[] {0, 1, 2, Double.MAX_VALUE}[i] : R.nextDouble() * 2.5;
      sb.append("hmkey\t").append(h(key)).append('\t').append(h(SloppyMath.haversinMeters(key))).append('\n');
    }
    return sb.toString();
  }

  // ------------------------------------------------------------------ encoding

  static String encoding() {
    StringBuilder sb = new StringBuilder();
    double[] special = {
      0, -0.0, 90, -90, 180, -180, 90.0000001, -90.0000001, 180.0000001, Double.NaN,
      Double.POSITIVE_INFINITY, Math.nextDown(90.0), Math.nextUp(-90.0), 1e-300, -1e-300
    };
    for (int i = 0; i < 3000; i++) {
      double v;
      if (i < special.length) v = special[i];
      else if (i % 3 == 0) v = GeoEncodingUtils.decodeLatitude(R.nextInt()) + (R.nextInt(3) - 1) * 1e-12;
      else v = R.nextDouble() * 400 - 200;
      sb.append("lat\t").append(h(v));
      for (int k = 0; k < 2; k++) {
        try {
          int e = k == 0 ? GeoEncodingUtils.encodeLatitude(v) : GeoEncodingUtils.encodeLatitudeCeil(v);
          sb.append('\t').append(e);
        } catch (IllegalArgumentException e) {
          sb.append('\t').append(err(e));
          break;
        }
      }
      sb.append('\n');
      sb.append("lon\t").append(h(v));
      for (int k = 0; k < 2; k++) {
        try {
          int e = k == 0 ? GeoEncodingUtils.encodeLongitude(v) : GeoEncodingUtils.encodeLongitudeCeil(v);
          sb.append('\t').append(e);
        } catch (IllegalArgumentException e) {
          sb.append('\t').append(err(e));
          break;
        }
      }
      sb.append('\n');
    }
    for (int i = 0; i < 1000; i++) {
      int e = i < 4 ? new int[] {Integer.MIN_VALUE, Integer.MAX_VALUE, 0, -1}[i] : R.nextInt();
      sb.append("dec\t")
          .append(e)
          .append('\t')
          .append(h(GeoEncodingUtils.decodeLatitude(e)))
          .append('\t')
          .append(h(GeoEncodingUtils.decodeLongitude(e)))
          .append('\n');
    }
    float[] fspecial = {
      0f, -0f, Float.MAX_VALUE, -Float.MAX_VALUE, Float.MIN_VALUE, Float.NaN, Float.POSITIVE_INFINITY
    };
    for (int i = 0; i < 1000; i++) {
      float v = i < fspecial.length ? fspecial[i] : Float.intBitsToFloat(R.nextInt());
      sb.append("xy\t").append(hf(v)).append('\t');
      try {
        int e = XYEncodingUtils.encode(v);
        sb.append(e).append('\t').append(hf(XYEncodingUtils.decode(e)));
      } catch (IllegalArgumentException ex) {
        sb.append(err(ex));
      }
      sb.append('\n');
    }
    return sb.toString();
  }

  // ------------------------------------------------------------------ geo utils

  static String geoUtils() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 300; i++) {
      double radius =
          i < 5 ? new double[] {0, 1, 1e10, 2.0015114070321853E7, Double.MAX_VALUE}[i]
              : Math.pow(10, R.nextDouble() * 8) * R.nextDouble();
      sb.append("dqsk\t").append(h(radius)).append('\t').append(h(GeoUtils.distanceQuerySortKey(radius))).append('\n');
    }
    for (int i = 0; i < 800; i++) {
      double lat = i % 7 == 0 ? (R.nextBoolean() ? 1 : -1) * (85 + R.nextDouble() * 5) : lat();
      double lon = i % 5 == 0 ? (R.nextBoolean() ? 1 : -1) * (175 + R.nextDouble() * 5) : lon();
      double radius = Math.pow(10, R.nextDouble() * 7.5);
      if (i == 3) lat = 95;
      sb.append("fpd\t").append(h(lat)).append(',').append(h(lon)).append(',').append(h(radius)).append('\t');
      try {
        Rectangle r = Rectangle.fromPointDistance(lat, lon, radius);
        sb.append(h(r.minLat)).append(',').append(h(r.maxLat)).append(',').append(h(r.minLon)).append(',').append(h(r.maxLon));
      } catch (IllegalArgumentException e) {
        sb.append(err(e));
      }
      sb.append('\t').append(h(Rectangle.axisLat(lat, radius))).append('\n');
    }
    for (int i = 0; i < 800; i++) {
      double lat = lat(), lon = lon();
      double radius = Math.pow(10, R.nextDouble() * 6.5);
      double key = GeoUtils.distanceQuerySortKey(radius);
      double axis = Rectangle.axisLat(lat, radius);
      double minLat = GeoCorpus.clampLat(lat + (R.nextDouble() - 0.6) * radius / 50000);
      double maxLat = GeoCorpus.clampLat(minLat + R.nextDouble() * radius / 50000);
      double minLon = GeoCorpus.clampLon(lon + (R.nextDouble() - 0.6) * radius / 50000);
      double maxLon = GeoCorpus.clampLon(minLon + R.nextDouble() * radius / 50000);
      if (i % 50 == 0) {
        double t = minLon;
        minLon = maxLon + 1e-9;
        maxLon = t;
      }
      sb.append("relate\t")
          .append(h(minLat)).append(',').append(h(maxLat)).append(',')
          .append(h(minLon)).append(',').append(h(maxLon)).append(',')
          .append(h(lat)).append(',').append(h(lon)).append(',')
          .append(h(key)).append(',').append(h(axis)).append('\t');
      try {
        sb.append(GeoUtils.relate(minLat, maxLat, minLon, maxLon, lat, lon, key, axis).ordinal());
      } catch (IllegalArgumentException e) {
        sb.append(err(e));
      }
      sb.append('\n');
    }
    for (int i = 0; i < 1000; i++) {
      double[] p = new double[8];
      for (int k = 0; k < 8; k++) {
        p[k] = R.nextInt(4) == 0 ? R.nextInt(5) : R.nextDouble() * 4;
      }
      sb.append("seg\t");
      for (int k = 0; k < 8; k++) sb.append(k == 0 ? "" : ",").append(h(p[k]));
      sb.append('\t')
          .append(GeoUtils.orient(p[0], p[1], p[2], p[3], p[4], p[5]))
          .append('\t')
          .append(GeoUtils.lineCrossesLine(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) ? 1 : 0)
          .append('\t')
          .append(GeoUtils.lineOverlapLine(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) ? 1 : 0)
          .append('\t')
          .append(GeoUtils.lineCrossesLineWithBoundary(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) ? 1 : 0)
          .append('\n');
    }
    for (int i = 0; i < 500; i++) {
      float x = (float) ((R.nextDouble() * 2 - 1) * Math.pow(10, R.nextInt(40) - 5));
      float y = (float) ((R.nextDouble() * 2 - 1) * Math.pow(10, R.nextInt(40) - 5));
      float r = i == 0 ? -1f : i == 1 ? Float.POSITIVE_INFINITY : (float) (R.nextDouble() * Math.pow(10, R.nextInt(40) - 5));
      if (i == 2) x = Float.MAX_VALUE;
      sb.append("xyfpd\t").append(hf(x)).append(',').append(hf(y)).append(',').append(hf(r)).append('\t');
      try {
        XYRectangle rect = XYRectangle.fromPointDistance(x, y, r);
        sb.append(hf(rect.minX)).append(',').append(hf(rect.maxX)).append(',').append(hf(rect.minY)).append(',').append(hf(rect.maxY));
      } catch (IllegalArgumentException e) {
        sb.append(err(e));
      }
      sb.append('\n');
    }
    // distance predicates over encoded points
    for (int i = 0; i < 40; i++) {
      double lat = i % 6 == 0 ? (R.nextBoolean() ? 89.5 : -89.5) : lat();
      double lon = i % 4 == 0 ? (R.nextBoolean() ? 179.9 : -179.9) : lon();
      double radius = Math.pow(10, 1 + R.nextDouble() * 6);
      GeoEncodingUtils.DistancePredicate pred = GeoEncodingUtils.createDistancePredicate(lat, lon, radius);
      Rectangle box = Rectangle.fromPointDistance(lat, lon, radius);
      sb.append("dpred\t").append(h(lat)).append(',').append(h(lon)).append(',').append(h(radius)).append('\n');
      for (int k = 0; k < 40; k++) {
        double qlat = GeoCorpus.clampLat(box.minLat + R.nextDouble() * (box.maxLat - box.minLat) * 1.2 - (box.maxLat - box.minLat) * 0.1);
        double qlon = k % 2 == 0 ? lon() : GeoCorpus.clampLon(box.minLon + R.nextDouble() * 2 - 1);
        int elat = GeoEncodingUtils.encodeLatitude(qlat);
        int elon = GeoEncodingUtils.encodeLongitude(qlon);
        sb.append("t\t").append(elat).append(',').append(elon).append('\t').append(pred.test(elat, elon) ? 1 : 0).append('\n');
      }
    }
    return sb.toString();
  }

  // ------------------------------------------------------------------ component2d

  static String rel(Component2D.WithinRelation w) {
    return switch (w) {
      case CANDIDATE -> "C";
      case NOTWITHIN -> "N";
      case DISJOINT -> "D";
    };
  }

  interface Q {
    String run();
  }

  static String safe(Q q) {
    try {
      return q.run();
    } catch (IllegalArgumentException e) {
      return err(e);
    }
  }

  static String component2d() {
    StringBuilder sb = new StringBuilder();
    List<LatLonGeometry[]> latlon = new ArrayList<>();
    List<Polygon> polys = GeoCorpus.polygons(R, 30);
    for (Polygon p : polys) latlon.add(new LatLonGeometry[] {p});
    for (Line l : GeoCorpus.lines(R, 8)) latlon.add(new LatLonGeometry[] {l});
    for (int i = 0; i < 10; i++) {
      double[] c = GeoCorpus.center(R);
      latlon.add(new LatLonGeometry[] {new Circle(c[0], c[1], Math.pow(10, 1 + R.nextDouble() * 6.3))});
    }
    // circles whose boxes cross the dateline or reach a pole
    latlon.add(new LatLonGeometry[] {new Circle(10, 179.5, 300_000)});
    latlon.add(new LatLonGeometry[] {new Circle(-20, -179.9, 50_000)});
    latlon.add(new LatLonGeometry[] {new Circle(89.5, 30, 200_000)});
    latlon.add(new LatLonGeometry[] {new Circle(0, 180, 1_000_000)});
    for (int i = 0; i < 8; i++) {
      double minLat = R.nextDouble() * 170 - 85;
      double maxLat = Math.min(90, minLat + R.nextDouble() * 20);
      double minLon = R.nextDouble() * 360 - 180;
      double maxLon = i % 3 == 0 ? minLon - R.nextDouble() * 20 : Math.min(180, minLon + R.nextDouble() * 30);
      if (maxLon < -180) maxLon += 360;
      if (i == 1) {
        minLon = 180;
        maxLon = -170;
      }
      latlon.add(new LatLonGeometry[] {new Rectangle(minLat, maxLat, minLon, maxLon)});
    }
    for (int i = 0; i < 4; i++) {
      latlon.add(new LatLonGeometry[] {new Point(i == 0 ? 90 : R.nextDouble() * 180 - 90, i == 1 ? 180 : R.nextDouble() * 360 - 180)});
    }
    // multi-shapes: ComponentTree
    for (int i = 0; i < 6; i++) {
      int n = 2 + R.nextInt(6);
      LatLonGeometry[] gs = new LatLonGeometry[n];
      for (int k = 0; k < n; k++) {
        double[] c = GeoCorpus.center(R);
        switch (R.nextInt(4)) {
          case 0 -> gs[k] = new Point(c[0], c[1]);
          case 1 -> gs[k] = new Circle(c[0], c[1], Math.pow(10, 2 + R.nextDouble() * 4));
          case 2 -> gs[k] = polys.get(R.nextInt(polys.size()));
          default -> gs[k] = new Rectangle(Math.min(c[0], 89), Math.min(c[0] + 1, 90), c[1], Math.min(c[1] + 1, 180));
        }
      }
      latlon.add(gs);
    }
    int id = 0;
    for (LatLonGeometry[] gs : latlon) {
      sb.append("shape\t").append(id++).append("\tlatlon\t").append(GeoCorpus.esc(GeoCorpus.spec(gs)));
      Component2D c;
      try {
        c = LatLonGeometry.create(gs);
      } catch (IllegalArgumentException e) {
        sb.append('\t').append(err(e)).append('\n');
        continue;
      }
      sb.append('\n');
      queries(sb, c, gs.length == 1 ? vertices(gs[0]) : new double[0][], true);
    }
    // cartesian
    List<XYGeometry[]> xy = new ArrayList<>();
    for (int i = 0; i < 16; i++) {
      Polygon p = polys.get(R.nextInt(polys.size()));
      try {
        xy.add(new XYGeometry[] {GeoCorpus.xyPolygon(p, GeoCorpus.randomScale(R), (R.nextDouble() - 0.5) * 1e4)});
      } catch (IllegalArgumentException e) {
        // float rounding collapsed the ring
      }
    }
    for (int i = 0; i < 4; i++) {
      double s = GeoCorpus.randomScale(R);
      float[] x = new float[2 + R.nextInt(10)];
      float[] y = new float[x.length];
      for (int k = 0; k < x.length; k++) {
        x[k] = (float) ((R.nextDouble() - 0.5) * s);
        y[k] = (float) ((R.nextDouble() - 0.5) * s);
      }
      xy.add(new XYGeometry[] {new XYLine(x, y)});
    }
    for (int i = 0; i < 4; i++) {
      double s = GeoCorpus.randomScale(R);
      xy.add(new XYGeometry[] {new XYCircle((float) ((R.nextDouble() - 0.5) * s), (float) ((R.nextDouble() - 0.5) * s), (float) (R.nextDouble() * s))});
    }
    for (int i = 0; i < 3; i++) {
      double s = GeoCorpus.randomScale(R);
      float a = (float) ((R.nextDouble() - 0.5) * s);
      float b = (float) ((R.nextDouble() - 0.5) * s);
      xy.add(new XYGeometry[] {new XYRectangle(a, a + (float) (R.nextDouble() * s), b, b + (float) (R.nextDouble() * s))});
    }
    xy.add(new XYGeometry[] {new XYPoint(1.5f, -2.25f)});
    xy.add(new XYGeometry[] {new XYPoint(1.5f, -2.25f), new XYCircle(0, 0, 3), new XYRectangle(-10, -5, 2, 4)});
    for (XYGeometry[] gs : xy) {
      sb.append("shape\t").append(id++).append("\txy\t").append(GeoCorpus.esc(GeoCorpus.spec(gs)));
      Component2D c;
      try {
        c = XYGeometry.create(gs);
      } catch (IllegalArgumentException e) {
        sb.append('\t').append(err(e)).append('\n');
        continue;
      }
      sb.append('\n');
      queries(sb, c, gs.length == 1 ? vertices(gs[0]) : new double[0][], false);
    }
    return sb.toString();
  }

  /** (x, y) vertices of a geometry: the coordinates where relations are most fragile. */
  static double[][] vertices(Object g) {
    List<double[]> v = new ArrayList<>();
    if (g instanceof Polygon p) {
      for (int i = 0; i < p.numPoints(); i++) v.add(new double[] {p.getPolyLon(i), p.getPolyLat(i)});
      for (Polygon hole : p.getHoles())
        for (int i = 0; i < hole.numPoints(); i++) v.add(new double[] {hole.getPolyLon(i), hole.getPolyLat(i)});
    } else if (g instanceof Line l) {
      for (int i = 0; i < l.numPoints(); i++) v.add(new double[] {l.getLon(i), l.getLat(i)});
    } else if (g instanceof XYPolygon p) {
      for (int i = 0; i < p.numPoints(); i++) v.add(new double[] {p.getPolyX(i), p.getPolyY(i)});
    } else if (g instanceof XYLine l) {
      for (int i = 0; i < l.numPoints(); i++) v.add(new double[] {l.getX(i), l.getY(i)});
    }
    return v.toArray(new double[0][]);
  }

  static double[] pick(Component2D c, double[][] verts, boolean geo) {
    double minX = c.getMinX(), maxX = c.getMaxX(), minY = c.getMinY(), maxY = c.getMaxY();
    double w = Math.max(maxX - minX, 1e-9), hgt = Math.max(maxY - minY, 1e-9);
    int k = R.nextInt(10);
    double x, y;
    if (verts.length > 0 && k < 3) {
      double[] v = verts[R.nextInt(verts.length)];
      x = v[0];
      y = v[1];
    } else if (verts.length > 1 && k < 5) {
      // a point on an edge
      int i = R.nextInt(verts.length - 1);
      double t = R.nextDouble();
      x = verts[i][0] + t * (verts[i + 1][0] - verts[i][0]);
      y = verts[i][1] + t * (verts[i + 1][1] - verts[i][1]);
    } else if (k == 5) {
      x = R.nextBoolean() ? minX : maxX;
      y = R.nextBoolean() ? minY : maxY;
    } else {
      x = minX - 0.2 * w + R.nextDouble() * 1.4 * w;
      y = minY - 0.2 * hgt + R.nextDouble() * 1.4 * hgt;
    }
    if (geo) {
      x = GeoCorpus.clampLon(x);
      y = GeoCorpus.clampLat(y);
    }
    return new double[] {x, y};
  }

  static String d(double v) {
    return Double.toString(v);
  }

  static void queries(StringBuilder sb, Component2D c, double[][] verts, boolean geo) {
    sb.append("q\tbounds\t\t").append(d(c.getMinX())).append(',').append(d(c.getMaxX())).append(',').append(d(c.getMinY())).append(',').append(d(c.getMaxY())).append('\n');
    for (int i = 0; i < 24; i++) {
      double[] p = pick(c, verts, geo);
      sb.append("q\tcontains\t").append(d(p[0])).append(',').append(d(p[1])).append('\t').append(c.contains(p[0], p[1]) ? 1 : 0).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      double[] a = pick(c, verts, geo);
      double[] b = i % 5 == 0 ? new double[] {a[0] + 1e-7, a[1] + 1e-7} : pick(c, verts, geo);
      double minX = Math.min(a[0], b[0]), maxX = Math.max(a[0], b[0]);
      double minY = Math.min(a[1], b[1]), maxY = Math.max(a[1], b[1]);
      sb.append("q\trelate\t").append(d(minX)).append(',').append(d(maxX)).append(',').append(d(minY)).append(',').append(d(maxY)).append('\t').append(c.relate(minX, maxX, minY, maxY).ordinal()).append('\n');
    }
    for (int i = 0; i < 8; i++) {
      double[] a = pick(c, verts, geo), b = pick(c, verts, geo);
      String args = d(a[0]) + "," + d(a[1]) + "," + d(b[0]) + "," + d(b[1]);
      sb.append("q\tiline\t").append(args).append('\t').append(c.intersectsLine(a[0], a[1], b[0], b[1]) ? 1 : 0).append('\n');
      sb.append("q\tcline\t").append(args).append('\t').append(c.containsLine(a[0], a[1], b[0], b[1]) ? 1 : 0).append('\n');
      boolean ab = R.nextBoolean();
      sb.append("q\twline\t").append(args).append(',').append(ab ? 1 : 0).append('\t').append(safe(() -> rel(c.withinLine(a[0], a[1], ab, b[0], b[1])))).append('\n');
    }
    for (int i = 0; i < 8; i++) {
      double[] a = pick(c, verts, geo), b = pick(c, verts, geo), e = pick(c, verts, geo);
      String args = d(a[0]) + "," + d(a[1]) + "," + d(b[0]) + "," + d(b[1]) + "," + d(e[0]) + "," + d(e[1]);
      sb.append("q\titri\t").append(args).append('\t').append(c.intersectsTriangle(a[0], a[1], b[0], b[1], e[0], e[1]) ? 1 : 0).append('\n');
      sb.append("q\tctri\t").append(args).append('\t').append(c.containsTriangle(a[0], a[1], b[0], b[1], e[0], e[1]) ? 1 : 0).append('\n');
      boolean ab = R.nextBoolean(), bc = R.nextBoolean(), ca = R.nextBoolean();
      sb.append("q\twtri\t").append(args).append(',').append(ab ? 1 : 0).append(',').append(bc ? 1 : 0).append(',').append(ca ? 1 : 0).append('\t')
          .append(safe(() -> rel(c.withinTriangle(a[0], a[1], ab, b[0], b[1], bc, e[0], e[1], ca)))).append('\n');
    }
    for (int i = 0; i < 5; i++) {
      double[] p = pick(c, verts, geo);
      sb.append("q\twpoint\t").append(d(p[0])).append(',').append(d(p[1])).append('\t').append(safe(() -> rel(c.withinPoint(p[0], p[1])))).append('\n');
    }
    if (geo) {
      GeoEncodingUtils.Component2DPredicate pred;
      try {
        pred = GeoEncodingUtils.createComponentPredicate(c);
      } catch (IllegalArgumentException e) {
        sb.append("q\tpred_err\t\t").append(err(e)).append('\n');
        return;
      }
      for (int i = 0; i < 20; i++) {
        double[] p = pick(c, verts, geo);
        int elat = GeoEncodingUtils.encodeLatitude(p[1]);
        int elon = GeoEncodingUtils.encodeLongitude(p[0]);
        sb.append("q\tpred\t").append(elat).append(',').append(elon).append('\t').append(pred.test(elat, elon) ? 1 : 0).append('\n');
      }
    }
  }
}
