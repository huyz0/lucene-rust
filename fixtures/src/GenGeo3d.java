import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.spatial3d.geom.DistanceStyle;
import org.apache.lucene.spatial3d.geom.GeoArea;
import org.apache.lucene.spatial3d.geom.GeoAreaFactory;
import org.apache.lucene.spatial3d.geom.GeoAreaShape;
import org.apache.lucene.spatial3d.geom.GeoBBox;
import org.apache.lucene.spatial3d.geom.GeoBBoxFactory;
import org.apache.lucene.spatial3d.geom.GeoCircleFactory;
import org.apache.lucene.spatial3d.geom.GeoDistance;
import org.apache.lucene.spatial3d.geom.GeoDistanceShape;
import org.apache.lucene.spatial3d.geom.GeoMembershipShape;
import org.apache.lucene.spatial3d.geom.GeoPath;
import org.apache.lucene.spatial3d.geom.GeoPathFactory;
import org.apache.lucene.spatial3d.geom.GeoPoint;
import org.apache.lucene.spatial3d.geom.GeoPointShapeFactory;
import org.apache.lucene.spatial3d.geom.GeoPolygon;
import org.apache.lucene.spatial3d.geom.GeoPolygonFactory;
import org.apache.lucene.spatial3d.geom.GeoS2ShapeFactory;
import org.apache.lucene.spatial3d.geom.GeoShape;
import org.apache.lucene.spatial3d.geom.GeoSizeable;
import org.apache.lucene.spatial3d.geom.LatLonBounds;
import org.apache.lucene.spatial3d.geom.PlanetModel;
import org.apache.lucene.spatial3d.geom.PlanetObject;
import org.apache.lucene.spatial3d.geom.SerializableObject;
import org.apache.lucene.spatial3d.geom.XYZBounds;
import org.apache.lucene.spatial3d.geom.XYZSolidFactory;

/**
 * Cross-engine ground truth for {@code lucene-util}'s {@code spatial3d} module: {@code
 * geo3d/shapes.tsv}, replayed by {@code crates/lucene-util/tests/geo3d_fixtures.rs}.
 *
 * <p>A seeded corpus over five planet models (SPHERE, WGS84, CLARKE_1866, a random oblate and a
 * prolate ellipsoid):
 * every shape kind the factories build -- boxes (every {@code GeoBBoxFactory} branch: world, zones,
 * slices, degenerate points and lines, wide/north/south rectangles), circles (standard and exact),
 * point shapes, paths (standard and degenerate), polygons (convex, concave, complex via {@code
 * makeLargeGeoPolygon}, with holes, crossing the antimeridian, around the poles, degenerate), S2
 * cells, x/y/z solids and {@code GeoAreaFactory} areas -- each recorded with:
 *
 * <ul>
 *   <li>{@code SH id pm kind args => OK class bytes | ERR class message | NULL}: the factory call
 *       and the result's {@code writePlanetObject} bytes (or its exception);
 *   <li>{@code Q id x y z within style [O outside] [D distance delta] [P nearest center]}: one
 *       probe point -- {@code isWithin}, then, in one distance style, every distance the shape
 *       answers ({@code computeOutsideDistance}; {@code computeDistance}/{@code
 *       computeDeltaDistance}; a path's {@code computeNearestDistance}/{@code
 *       computePathCenterDistance}), each a double or {@code ERR class message};
 *   <li>{@code XB}/{@code LB} the x/y/z and lat/lon bounds, {@code EP} edge points, {@code DB}
 *       distance bounds per distance style, {@code RS} radius and center, {@code EX} bbox
 *       expansion;
 *   <li>{@code REL a b rel}: {@code getRelationship} of every area against shapes and against x/y/z
 *       solids around them -- the call {@code PointInGeo3DShapeQuery} makes per BKD cell.
 * </ul>
 *
 * <p>Every double is its raw bits in hex. Generated with {@code -XX:DisableIntrinsic=_dsin,_dcos,
 * _dtan} (see {@code scripts/gen-fixtures.sh}), so {@code Math.sin/cos/tan} are {@code StrictMath}'s
 * fdlibm and the output is the same on every platform.
 */
public class GenGeo3d {
  /**
   * The corpus seed; {@code -Dgeo3d.seed=N} draws another corpus (with the test's {@code
   * GEO3D_FIXTURES} pointing at it) for a wider one-off sweep.
   */
  static final Random R = new Random(Long.getLong("geo3d.seed", 0x3D_6E0_3DL));
  static final StringBuilder OUT = new StringBuilder();
  static final String[] STYLE_NAMES = {"ARC", "LINEAR", "LINEAR_SQUARED", "NORMAL", "NORMAL_SQUARED"};
  static final DistanceStyle[] STYLES = {
    DistanceStyle.ARC,
    DistanceStyle.LINEAR,
    DistanceStyle.LINEAR_SQUARED,
    DistanceStyle.NORMAL,
    DistanceStyle.NORMAL_SQUARED
  };
  static int nextId = 0;
  /** Probes a shape gets beyond {@link #probes}, by shape id. */
  static final java.util.Map<Integer, List<GeoPoint>> EXTRA_PROBES = new java.util.HashMap<>();

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) {
      sb.append(Character.forDigit((x >> 4) & 0xf, 16)).append(Character.forDigit(x & 0xf, 16));
    }
    return sb.toString();
  }

  static String esc(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static String err(Throwable t) {
    return "ERR\t" + t.getClass().getName() + "\t" + esc(t.getMessage());
  }

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append('\t');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String opt(Double d) {
    return d == null ? "null" : h(d);
  }

  static double rad(double deg) {
    return deg * Math.PI / 180.0;
  }

  /** A random latitude in radians, biased toward the poles and the equator. */
  static double lat() {
    switch (R.nextInt(8)) {
      case 0:
        return Math.PI * 0.5 - R.nextDouble() * 0.05;
      case 1:
        return -Math.PI * 0.5 + R.nextDouble() * 0.05;
      case 2:
        return (R.nextDouble() - 0.5) * 0.02;
      default:
        return (R.nextDouble() - 0.5) * Math.PI;
    }
  }

  /** A random longitude in radians, biased toward the antimeridian. */
  static double lon() {
    switch (R.nextInt(6)) {
      case 0:
        return Math.PI - R.nextDouble() * 0.05;
      case 1:
        return -Math.PI + R.nextDouble() * 0.05;
      default:
        return (R.nextDouble() * 2.0 - 1.0) * Math.PI;
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo3d");
    Files.createDirectories(out);
    PlanetModel custom = new PlanetModel(1.0 + R.nextDouble() * 0.1, 1.0 - R.nextDouble() * 0.1);
    // A prolate model too: some shapes branch on which axis is longer.
    PlanetModel prolate = new PlanetModel(0.95, 1.05);
    PlanetModel[] pms = {
      PlanetModel.SPHERE, PlanetModel.WGS84, PlanetModel.CLARKE_1866, custom, prolate
    };
    for (int p = 0; p < pms.length; p++) {
      PlanetModel pm = pms[p];
      line("PM", p, h(pm.a), h(pm.b));
      planetModelChecks(p, pm);
      List<Shape> shapes = new ArrayList<>();
      makeShapes(p, pm, shapes);
      for (Shape s : shapes) {
        checks(s, pm);
      }
      relations(pm, shapes);
    }
    Files.writeString(out.resolve("shapes.tsv"), OUT.toString());
  }

  // ------------------------------------------------------------- planet models

  static void planetModelChecks(int p, PlanetModel pm) {
    line(
        "PMV", p, h(pm.MAX_VALUE), h(pm.DECODE), pm.MIN_ENCODED_VALUE, pm.MAX_ENCODED_VALUE,
        h(pm.minimumPoleDistance), h(pm.getMeanRadius()), h(pm.scaledFlattening));
    for (int i = 0; i < 40; i++) {
      double v;
      switch (i) {
        case 0:
          v = pm.MAX_VALUE;
          break;
        case 1:
          v = -pm.MAX_VALUE;
          break;
        case 2:
          v = Math.nextUp(pm.MAX_VALUE);
          break;
        case 3:
          v = 0.0;
          break;
        default:
          v = (R.nextDouble() * 2 - 1) * pm.MAX_VALUE;
      }
      String enc;
      try {
        int e = pm.encodeValue(v);
        enc = "OK\t" + e + "\t" + h(pm.decodeValue(e));
      } catch (RuntimeException e) {
        enc = err(e);
      }
      line("ENC", p, h(v), enc);
    }
    for (int i = 0; i < 30; i++) {
      GeoPoint gp = new GeoPoint(pm, lat(), lon());
      double x = gp.x, y = gp.y, z = gp.z;
      if (i == 0) {
        x = pm.getMaximumXValue();
      }
      if (i == 1) {
        x = Math.nextUp(pm.getMaximumXValue());
      }
      String enc;
      try {
        long dv = pm.getDocValueEncoder().encodePoint(x, y, z);
        GeoPoint back = pm.getDocValueEncoder().decodePoint(dv);
        enc = "OK\t" + dv + "\t" + h(back.x) + "\t" + h(back.y) + "\t" + h(back.z);
      } catch (RuntimeException e) {
        enc = err(e);
      }
      line("DVE", p, h(x), h(y), h(z), enc);
    }
    for (int i = 0; i < 30; i++) {
      GeoPoint a = new GeoPoint(pm, lat(), lon());
      GeoPoint b = new GeoPoint(pm, lat(), lon());
      double dist = pm.surfaceDistance(a, b);
      double bearing = (R.nextDouble() * 2 - 1) * Math.PI;
      double len = R.nextDouble() * 2.0;
      String bear;
      try {
        GeoPoint c = pm.surfacePointOnBearing(a, len, bearing);
        bear = "OK\t" + h(c.x) + "\t" + h(c.y) + "\t" + h(c.z);
      } catch (RuntimeException e) {
        bear = err(e);
      }
      GeoPoint bis = pm.bisection(a, b);
      line(
          "SD", p, h(a.getLatitude()), h(a.getLongitude()), h(b.getLatitude()),
          h(b.getLongitude()), h(dist), h(len), h(bearing), bear,
          bis == null ? "null" : h(bis.x) + "," + h(bis.y) + "," + h(bis.z));
    }
  }

  // ------------------------------------------------------------- shapes

  static class Shape {
    final int id;
    final Object shape;

    Shape(int id, Object shape) {
      this.id = id;
      this.shape = shape;
    }
  }

  interface Maker {
    Object make() throws Exception;
  }

  static void record(PlanetModel pm, int p, List<Shape> shapes, String spec, Maker maker) {
    int id = nextId++;
    Object o;
    try {
      o = maker.make();
    } catch (Exception | AssertionError e) {
      line("SH", id, p, spec, "=>", err(e));
      return;
    }
    if (o == null) {
      line("SH", id, p, spec, "=>", "NULL");
      return;
    }
    String bytes;
    try {
      ByteArrayOutputStream bos = new ByteArrayOutputStream();
      SerializableObject.writePlanetObject(bos, (PlanetObject) o);
      byte[] b = bos.toByteArray();
      // Java's own round trip: what it reads back writes the same bytes.
      PlanetObject back = SerializableObject.readPlanetObject(new ByteArrayInputStream(b));
      ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
      SerializableObject.writePlanetObject(bos2, back);
      if (!Arrays.equals(b, bos2.toByteArray())) {
        throw new AssertionError("round trip differs for " + o);
      }
      bytes = hex(b);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    line("SH", id, p, spec, "=>", "OK", o.getClass().getSimpleName(), bytes);
    shapes.add(new Shape(id, o));
  }

  static String pts(double[][] ll) {
    StringBuilder sb = new StringBuilder();
    sb.append(ll.length);
    for (double[] p : ll) {
      sb.append(' ').append(h(p[0])).append(' ').append(h(p[1]));
    }
    return sb.toString();
  }

  static List<GeoPoint> geo(PlanetModel pm, double[][] ll) {
    List<GeoPoint> out = new ArrayList<>();
    for (double[] p : ll) {
      out.add(new GeoPoint(pm, p[0], p[1]));
    }
    return out;
  }

  /** A star-shaped ring of n points around (clat, clon), radius in radians. */
  static double[][] ring(double clat, double clon, int n, double radius, boolean clockwise) {
    return ring(clat, clon, n, radius, clockwise, false);
  }

  /** As above; a spiky ring alternates long and short spokes (deeply concave). */
  static double[][] ring(
      double clat, double clon, int n, double radius, boolean clockwise, boolean spiky) {
    double[][] out = new double[n][];
    double start = R.nextDouble() * Math.PI * 2;
    for (int i = 0; i < n; i++) {
      double a = start + (clockwise ? -1 : 1) * (Math.PI * 2 * i / n + R.nextDouble() * 0.3 / n);
      double r =
          spiky && i % 2 == 1
              ? radius * (0.15 + R.nextDouble() * 0.2)
              : radius * (0.4 + R.nextDouble() * 0.6);
      double la = clat + r * Math.sin(a);
      double lo = clon + r * Math.cos(a) / Math.max(0.05, Math.cos(clat));
      la = Math.max(-Math.PI * 0.5, Math.min(Math.PI * 0.5, la));
      while (lo > Math.PI) lo -= 2 * Math.PI;
      while (lo < -Math.PI) lo += 2 * Math.PI;
      out[i] = new double[] {la, lo};
    }
    return out;
  }

  static String desc(double[][] outer, List<double[][]> holes) {
    StringBuilder sb = new StringBuilder(pts(outer));
    sb.append(' ').append(holes.size());
    for (double[][] hole : holes) {
      sb.append(' ').append(pts(hole)).append(" 0");
    }
    return sb.toString();
  }

  static GeoPolygonFactory.PolygonDescription description(
      PlanetModel pm, double[][] outer, List<double[][]> holes) {
    List<GeoPolygonFactory.PolygonDescription> hs = new ArrayList<>();
    for (double[][] hole : holes) {
      hs.add(new GeoPolygonFactory.PolygonDescription(geo(pm, hole)));
    }
    return new GeoPolygonFactory.PolygonDescription(geo(pm, outer), hs);
  }

  static void makeShapes(int p, PlanetModel pm, List<Shape> shapes) {
    // Boxes: every factory branch, then random ones.
    double[][] boxes = {
      {90, -90, -180, 180}, {30, 30, -180, 180}, {90, 90, -180, 180}, {-90, -90, -180, 180},
      {90, 10, -180, 180}, {-10, -90, -180, 180}, {40, -20, -180, 180}, {90, -90, 30, 30},
      {90, -90, -170, 50}, {90, -90, 10, 40}, {90, -90, 170, -170}, {20, 20, 30, 30},
      {50, -10, 30, 30}, {20, 20, -170, 50}, {90, 90, -170, 50}, {-90, -90, -170, 50},
      {90, 10, -170, 50}, {-10, -90, -170, 50}, {40, -20, -170, 50}, {20, 20, 10, 40},
      {90, 90, 10, 40}, {-90, -90, 10, 40}, {90, 10, 10, 40}, {-10, -90, 10, 40},
      {40, -20, 10, 40}, {40, -20, 170, -170}, {100, -100, -200, 200}, {40, 50, 10, 40},
      {40, -20, 0, 180}, {40, -20, -180, 0}, {89.9999999999, -89.9999999999, -179.99999, 179.99999}
    };
    for (double[] b : boxes) {
      final double t = rad(b[0]), bo = rad(b[1]), l = rad(b[2]), r = rad(b[3]);
      record(pm, p, shapes, "bbox\t" + h(t) + " " + h(bo) + " " + h(l) + " " + h(r),
          () -> GeoBBoxFactory.makeGeoBBox(pm, t, bo, l, r));
    }
    for (int i = 0; i < 40; i++) {
      double a = lat(), b = lat();
      final double t = Math.max(a, b), bo = Math.min(a, b), l = lon(), r = lon();
      record(pm, p, shapes, "bbox\t" + h(t) + " " + h(bo) + " " + h(l) + " " + h(r),
          () -> GeoBBoxFactory.makeGeoBBox(pm, t, bo, l, r));
    }
    // Areas from lat/lon and from x/y/z.
    for (int i = 0; i < 6; i++) {
      double a = lat(), b = lat();
      final double t = Math.max(a, b), bo = Math.min(a, b), l = lon(), r = lon();
      record(pm, p, shapes, "areall\t" + h(t) + " " + h(bo) + " " + h(l) + " " + h(r),
          () -> GeoAreaFactory.makeGeoArea(pm, t, bo, l, r));
    }
    for (int i = 0; i < 40; i++) {
      double[] v = new double[6];
      double[] mins = {pm.getMinimumXValue(), pm.getMinimumYValue(), pm.getMinimumZValue()};
      double[] maxs = {pm.getMaximumXValue(), pm.getMaximumYValue(), pm.getMaximumZValue()};
      for (int d = 0; d < 3; d++) {
        double a = mins[d] + R.nextDouble() * (maxs[d] - mins[d]);
        double c = mins[d] + R.nextDouble() * (maxs[d] - mins[d]);
        int mode = R.nextInt(6);
        if (mode == 0) {
          c = a; // degenerate dimension
        } else if (mode == 1) {
          a = mins[d];
          c = maxs[d];
        } else if (mode == 2) {
          c = a + (R.nextDouble() - 0.5) * 1e-11;
        }
        v[d * 2] = Math.min(a, c);
        v[d * 2 + 1] = Math.max(a, c);
      }
      final double[] s = v;
      String spec = h(s[0]) + " " + h(s[1]) + " " + h(s[2]) + " " + h(s[3]) + " " + h(s[4]) + " " + h(s[5]);
      if (i % 4 == 3) {
        record(pm, p, shapes, "areaxyz\t" + spec,
            () -> GeoAreaFactory.makeGeoArea(pm, s[0], s[1], s[2], s[3], s[4], s[5]));
      } else {
        record(pm, p, shapes, "solid\t" + spec,
            () -> XYZSolidFactory.makeXYZSolid(pm, s[0], s[1], s[2], s[3], s[4], s[5]));
      }
    }
    // Points.
    for (int i = 0; i < 6; i++) {
      final double la = i == 0 ? Math.PI * 0.5 : lat(), lo = lon();
      record(pm, p, shapes, "point\t" + h(la) + " " + h(lo),
          () -> GeoPointShapeFactory.makeGeoPointShape(pm, la, lo));
    }
    // Circles.
    double[] radii = {0.0, 1e-9, 1e-4, 0.01, 0.3, 1.0, Math.PI * 0.5, 2.5, Math.PI - 1e-6, Math.PI, 4.0};
    for (int i = 0; i < 30; i++) {
      final double la = i == 0 ? Math.PI * 0.5 : lat(), lo = lon();
      final double rr = i < radii.length ? radii[i] : R.nextDouble() * (R.nextBoolean() ? 0.1 : 2.0);
      record(pm, p, shapes, "circle\t" + h(la) + " " + h(lo) + " " + h(rr),
          () -> GeoCircleFactory.makeGeoCircle(pm, la, lo, rr));
    }
    for (int i = 0; i < 16; i++) {
      final double la = lat(), lo = lon();
      final double rr =
          i < 5 ? new double[] {1e-12, Math.PI * 0.5, 3.0, 1e-9, 1e-7}[i] : R.nextDouble() * 1.5;
      final double acc = new double[] {1e-12, 1e-6, 1e-3, 0.01}[R.nextInt(4)];
      record(pm, p, shapes, "exactcircle\t" + h(la) + " " + h(lo) + " " + h(rr) + " " + h(acc),
          () -> GeoCircleFactory.makeExactGeoCircle(pm, la, lo, rr, acc));
    }
    // Paths.
    for (int i = 0; i < 30; i++) {
      int n = 1 + R.nextInt(5);
      double[][] ll = new double[n][];
      double la = lat(), lo = lon();
      for (int k = 0; k < n; k++) {
        ll[k] = new double[] {la, lo};
        if (R.nextInt(8) != 0) { // sometimes a repeated point
          la = Math.max(-Math.PI * 0.5, Math.min(Math.PI * 0.5, la + (R.nextDouble() - 0.5) * 0.6));
          lo = lo + (R.nextDouble() - 0.5) * 0.8;
          if (lo > Math.PI) lo -= 2 * Math.PI;
          if (lo < -Math.PI) lo += 2 * Math.PI;
        }
      }
      final double w = i < 4 ? new double[] {0.0, 1e-10, Math.PI * 0.5, 2.0}[i] : R.nextDouble() * 0.2;
      final double[][] fl = ll;
      record(pm, p, shapes, "path\t" + h(w) + " " + pts(ll),
          () -> GeoPathFactory.makeGeoPath(pm, w, geo(pm, fl).toArray(new GeoPoint[0])));
    }
    // Polygons with colinear runs, backtracks and nearly parallel edges: a
    // square whose sides carry extra points on (or a hair off) their great
    // circles, optionally with a hole -- the factory's edge filtering and
    // its tiling failures.
    for (int i = 0; i < 24; i++) {
      double clat = lat() * 0.8, clon = lon();
      double size = new double[] {0.01, 0.3, 1.0}[i % 3];
      double eps = new double[] {0.0, 1e-13, 1e-11, 1e-9, 1e-6}[(i / 3) % 5];
      double[][] corners = ring(clat, clon, 4, size, false);
      List<GeoPoint> cg = geo(pm, corners);
      List<double[]> pts = new ArrayList<>();
      for (int k = 0; k < 4; k++) {
        GeoPoint a = cg.get(k), b = cg.get((k + 1) % 4);
        pts.add(corners[k]);
        int extra = (k + i) % 3;
        for (int e = 1; e <= extra; e++) {
          double t = e / (extra + 1.0);
          GeoPoint m =
              pm.createSurfacePoint(
                  a.x * (1 - t) + b.x * t + eps * R.nextGaussian(),
                  a.y * (1 - t) + b.y * t + eps * R.nextGaussian(),
                  a.z * (1 - t) + b.z * t + eps * R.nextGaussian());
          pts.add(new double[] {m.getLatitude(), m.getLongitude()});
        }
        if (i % 8 == 7 && k == 1) {
          // A backtrack: out to a point and straight back.
          pts.add(corners[k + 1]);
          pts.add(corners[k]);
        }
      }
      double[][] outer = pts.toArray(new double[0][]);
      List<double[][]> holes = new ArrayList<>();
      if (i % 4 == 3) {
        holes.add(ring(clat, clon, 4, size * 0.2, true));
      }
      final double[][] fo = outer;
      final List<double[][]> fh = holes;
      record(pm, p, shapes, "polygon\t" + desc(outer, holes),
          () -> GeoPolygonFactory.makeGeoPolygon(pm, description(pm, fo, fh)));
    }
    // Polygons.
    for (int i = 0; i < 90; i++) {
      double clat = lat(), clon = lon();
      // Rings over SMALL_POLYGON_CUTOFF_EDGES go straight to a complex polygon.
      int n = i % 13 == 12 ? 101 + R.nextInt(40) : 3 + R.nextInt(i < 20 ? 6 : 14);
      double radius = new double[] {1e-7, 0.01, 0.2, 0.8, 1.5}[R.nextInt(5)];
      double[][] outer = ring(clat, clon, n, radius, R.nextBoolean(), i % 3 == 1);
      List<double[][]> holes = new ArrayList<>();
      if (R.nextInt(3) == 0) {
        holes.add(ring(clat, clon, 3 + R.nextInt(4), radius * 0.2, R.nextBoolean()));
      }
      if (i % 10 == 9) { // a degenerate one: a repeated and a colinear point
        outer = Arrays.copyOf(outer, outer.length + 2);
        outer[outer.length - 2] = outer[0];
        outer[outer.length - 1] = new double[] {outer[0][0], outer[1][1]};
      }
      final double[][] fo = outer;
      final List<double[][]> fh = holes;
      record(pm, p, shapes, "polygon\t" + desc(outer, holes),
          () -> GeoPolygonFactory.makeGeoPolygon(pm, description(pm, fo, fh)));
      if (i % 5 == 0) {
        final List<GeoPoint> pl = geo(pm, fo);
        record(pm, p, shapes, "convex\t" + pts(fo),
            () -> GeoPolygonFactory.makeGeoConvexPolygon(pm, pl));
        record(pm, p, shapes, "concave\t" + pts(fo),
            () -> GeoPolygonFactory.makeGeoConcavePolygon(pm, pl));
      }
    }
    for (int i = 0; i < 24; i++) {
      int count = 1 + R.nextInt(3);
      List<double[][]> outers = new ArrayList<>();
      List<List<double[][]>> holes = new ArrayList<>();
      StringBuilder spec = new StringBuilder("largepolygon\t" + count);
      for (int k = 0; k < count; k++) {
        double clat = lat(), clon = lon();
        double radius = new double[] {0.01, 0.3, 1.2}[R.nextInt(3)];
        double[][] outer = ring(clat, clon, 4 + R.nextInt(30), radius, R.nextBoolean());
        List<double[][]> hs = new ArrayList<>();
        if (R.nextBoolean()) {
          hs.add(ring(clat, clon, 3 + R.nextInt(5), radius * 0.2, R.nextBoolean()));
        }
        outers.add(outer);
        holes.add(hs);
        spec.append(' ').append(desc(outer, hs));
      }
      record(pm, p, shapes, spec.toString(), () -> {
        List<GeoPolygonFactory.PolygonDescription> ds = new ArrayList<>();
        for (int k = 0; k < outers.size(); k++) {
          ds.add(description(pm, outers.get(k), holes.get(k)));
        }
        return GeoPolygonFactory.makeLargeGeoPolygon(pm, ds);
      });
    }
    // Complex polygons with a vertex on one of the test point's fixed
    // planes, probed from points that share the vertex's other coordinates:
    // a traversal's intersection point then lands on the polygon's edge,
    // which only the dual crossing iterator can handle. The test point is
    // the center of mass of the smallest ring (the triangle).
    for (int i = 0; i < 9; i++) {
      int axis = i % 3;
      double[][] tri = ring(lat() * 0.5, lon(), 3, 0.05, false);
      double sx = 0, sy = 0, sz = 0;
      for (GeoPoint g : geo(pm, tri)) {
        sx += g.x;
        sy += g.y;
        sz += g.z;
      }
      GeoPoint c = pm.createSurfacePoint(sx, sy, sz);
      GeoPoint v = sharing(pm, c, axis);
      if (v == null) continue;
      double vlat = v.getLatitude(), vlon = v.getLongitude();
      double[][] other = ring(vlat, vlon + 0.1, 5, 0.1, false);
      other[0] = new double[] {vlat, vlon};
      List<GeoPoint> extra = new ArrayList<>();
      for (int b = 0; b < 3; b++) {
        if (b == axis) continue;
        for (int k = 0; k < 3; k++) {
          GeoPoint q = sharing(pm, v, b);
          if (q != null) extra.add(q);
        }
      }
      EXTRA_PROBES.put(nextId, extra);
      final double[][] ft = tri, fo = other;
      String spec = "largepolygon\t2 " + desc(tri, List.of()) + " " + desc(other, List.of());
      record(pm, p, shapes, spec, () -> GeoPolygonFactory.makeLargeGeoPolygon(pm, List.of(
          description(pm, ft, List.of()), description(pm, fo, List.of()))));
    }
    // S2 cells (quadrilaterals).
    for (int i = 0; i < 8; i++) {
      double[][] q = ring(lat(), lon(), 4, 0.05 + R.nextDouble() * 0.3, false);
      final double[][] fq = q;
      record(pm, p, shapes, "s2\t" + pts(q), () -> {
        List<GeoPoint> g = geo(pm, fq);
        return GeoS2ShapeFactory.makeGeoS2Shape(pm, g.get(0), g.get(1), g.get(2), g.get(3));
      });
    }
  }

  // ------------------------------------------------------------- checks

  /** A random surface point with {@code anchor}'s coordinate on {@code axis}, or null. */
  static GeoPoint sharing(PlanetModel pm, GeoPoint anchor, int axis) {
    double t = R.nextDouble() * Math.PI * 2;
    double a = pm.xyScaling, b = pm.zScaling;
    if (axis == 2) {
      double r = 1 - (anchor.z / b) * (anchor.z / b);
      if (r < 0) return null;
      return new GeoPoint(a * Math.sqrt(r) * Math.cos(t), a * Math.sqrt(r) * Math.sin(t), anchor.z);
    }
    double fixed = axis == 0 ? anchor.x : anchor.y;
    double r = 1 - (fixed / a) * (fixed / a);
    if (r < 0) return null;
    double u = a * Math.sqrt(r) * Math.cos(t), z = b * Math.sqrt(r) * Math.sin(t);
    return axis == 0 ? new GeoPoint(fixed, u, z) : new GeoPoint(u, fixed, z);
  }

  static List<GeoPoint> probes(PlanetModel pm, Object o) {
    List<GeoPoint> out = new ArrayList<>();
    for (int i = 0; i < 4; i++) {
      out.add(new GeoPoint(pm, lat(), lon()));
    }
    if (o instanceof GeoShape) {
      for (GeoPoint e : ((GeoShape) o).getEdgePoints()) {
        out.add(e);
        for (double scale : new double[] {1e-13, 1e-3}) {
          out.add(
              new GeoPoint(
                  e.x + (R.nextDouble() - 0.5) * scale,
                  e.y + (R.nextDouble() - 0.5) * scale,
                  e.z + (R.nextDouble() - 0.5) * scale));
        }
        if (out.size() > 16) break;
      }
    }
    if (o.getClass().getSimpleName().equals("GeoComplexPolygon")) {
      // Points sharing a coordinate with the polygon's test point (or with
      // a vertex) put an edge or the point itself on a travel plane: the
      // traversal strategies' fallbacks.
      GeoPoint tp;
      try {
        java.lang.reflect.Field f = o.getClass().getDeclaredField("testPoint1");
        f.setAccessible(true);
        tp = (GeoPoint) f.get(o);
      } catch (ReflectiveOperationException e) {
        throw new RuntimeException(e);
      }
      out.add(tp);
      out.add(new GeoPoint(-tp.x, -tp.y, -tp.z));
      List<GeoPoint> anchors = new ArrayList<>();
      anchors.add(tp);
      GeoPoint[] edges = ((GeoShape) o).getEdgePoints();
      anchors.add(edges[R.nextInt(edges.length)]);
      for (GeoPoint anchor : anchors) {
        for (int axis = 0; axis < 3; axis++) {
          for (int k = 0; k < 2; k++) {
            GeoPoint g = sharing(pm, anchor, axis);
            if (g != null) out.add(g);
          }
        }
      }
    }
    if (o instanceof GeoSizeable) {
      GeoPoint c = ((GeoSizeable) o).getCenter();
      out.add(c);
      out.add(pm.createSurfacePoint(c.x + 0.01, c.y - 0.01, c.z));
    }
    if (o instanceof GeoShape) {
      XYZBounds b = new XYZBounds();
      try {
        ((GeoShape) o).getBounds(b);
      } catch (RuntimeException e) {
        // recorded by XB
      }
      if (b.getMinimumX() != null && b.getMaximumZ() != null && b.getMinimumY() != null) {
        out.add(pm.createSurfacePoint(b.getMinimumX(), b.getMinimumY(), b.getMaximumZ()));
        out.add(
            pm.createSurfacePoint(
                (b.getMinimumX() + b.getMaximumX()) * 0.5,
                (b.getMinimumY() + b.getMaximumY()) * 0.5,
                (b.getMinimumZ() + b.getMaximumZ()) * 0.5));
      }
    }
    return out;
  }

  static String xyzBounds(XYZBounds b) {
    return opt(b.getMinimumX()) + "\t" + opt(b.getMaximumX()) + "\t" + opt(b.getMinimumY()) + "\t"
        + opt(b.getMaximumY()) + "\t" + opt(b.getMinimumZ()) + "\t" + opt(b.getMaximumZ());
  }

  static String latLonBounds(LatLonBounds b) {
    return (b.checkNoTopLatitudeBound() ? "1" : "0") + (b.checkNoBottomLatitudeBound() ? "1" : "0")
        + (b.checkNoLongitudeBound() ? "1" : "0") + "\t" + opt(b.getMaxLatitude()) + "\t"
        + opt(b.getMinLatitude()) + "\t" + opt(b.getLeftLongitude()) + "\t"
        + opt(b.getRightLongitude());
  }

  static String num(DoubleCall c) {
    try {
      return h(c.call());
    } catch (RuntimeException e) {
      return err(e);
    }
  }

  interface DoubleCall {
    double call();
  }

  static void checks(Shape s, PlanetModel pm) {
    Object o = s.shape;
    int id = s.id;
    if (o instanceof GeoShape) {
      GeoShape gs = (GeoShape) o;
      try {
        XYZBounds xb = new XYZBounds();
        gs.getBounds(xb);
        line("XB", id, "OK", xyzBounds(xb));
      } catch (RuntimeException e) {
        line("XB", id, err(e));
      }
      try {
        LatLonBounds lb = new LatLonBounds();
        gs.getBounds(lb);
        line("LB", id, "OK", latLonBounds(lb));
      } catch (RuntimeException e) {
        line("LB", id, err(e));
      }
      StringBuilder ep = new StringBuilder();
      GeoPoint[] eps = gs.getEdgePoints();
      ep.append(eps.length);
      for (GeoPoint e : eps) {
        ep.append(' ').append(h(e.x)).append(' ').append(h(e.y)).append(' ').append(h(e.z));
      }
      line("EP", id, ep);
    }
    if (o instanceof GeoSizeable) {
      GeoSizeable sz = (GeoSizeable) o;
      GeoPoint c = sz.getCenter();
      line("RS", id, h(sz.getRadius()), h(c.x), h(c.y), h(c.z));
    }
    if (o instanceof GeoBBox) {
      for (double angle : new double[] {0.0, 0.01, 0.5}) {
        String res;
        try {
          GeoBBox e = ((GeoBBox) o).expand(angle);
          ByteArrayOutputStream bos = new ByteArrayOutputStream();
          SerializableObject.writePlanetObject(bos, e);
          res = "OK\t" + e.getClass().getSimpleName() + "\t" + hex(bos.toByteArray());
        } catch (Exception e) {
          res = err(e);
        }
        line("EX", id, h(angle), res);
      }
    }
    if (o instanceof GeoDistanceShape) {
      for (int k = 0; k < STYLES.length; k++) {
        final DistanceStyle st = STYLES[k];
        double v = new double[] {0.0, 0.05, 0.5, Double.POSITIVE_INFINITY}[R.nextInt(4)];
        String res;
        try {
          XYZBounds db = new XYZBounds();
          ((GeoDistanceShape) o).getDistanceBounds(db, st, v);
          res = "OK\t" + xyzBounds(db);
        } catch (RuntimeException e) {
          res = err(e);
        }
        line("DB", id, STYLE_NAMES[k], h(v), res);
      }
    }
    List<GeoPoint> allProbes = probes(pm, o);
    allProbes.addAll(EXTRA_PROBES.getOrDefault(id, List.of()));
    for (GeoPoint q : allProbes) {
      final double x = q.x, y = q.y, z = q.z;
      String w;
      try {
        w = ((org.apache.lucene.spatial3d.geom.Membership) o).isWithin(x, y, z) ? "1" : "0";
      } catch (RuntimeException e) {
        w = err(e);
      }
      // One record per probe: its membership, then each distance the shape
      // answers, tagged (O outside, D distance + delta, P nearest + center).
      StringBuilder rec = new StringBuilder();
      // ARC half the time, one of the others otherwise.
      final int k = R.nextBoolean() ? 0 : 1 + R.nextInt(STYLES.length - 1);
      final DistanceStyle st = STYLES[k];
      rec.append(STYLE_NAMES[k]);
      if (o instanceof GeoMembershipShape) {
        rec.append("\tO\t")
            .append(num(() -> ((GeoMembershipShape) o).computeOutsideDistance(st, x, y, z)));
      }
      if (o instanceof GeoDistance) {
        rec.append("\tD\t").append(num(() -> ((GeoDistance) o).computeDistance(st, x, y, z)));
        rec.append('\t').append(num(() -> ((GeoDistance) o).computeDeltaDistance(st, x, y, z)));
      }
      if (o instanceof GeoPath) {
        rec.append("\tP\t").append(num(() -> ((GeoPath) o).computeNearestDistance(st, x, y, z)));
        rec.append('\t').append(num(() -> ((GeoPath) o).computePathCenterDistance(st, x, y, z)));
      }
      line("Q", id, h(x), h(y), h(z), w, rec);
    }
  }

  static String rel(int r) {
    return new String[] {"CONTAINS", "WITHIN", "OVERLAPS", "DISJOINT"}[r];
  }

  static void relations(PlanetModel pm, List<Shape> shapes) {
    for (Shape a : shapes) {
      if (!(a.shape instanceof GeoArea)) continue;
      GeoArea area = (GeoArea) a.shape;
      for (int k = 0; k < 6; k++) {
        Shape b = shapes.get(R.nextInt(shapes.size()));
        if (!(b.shape instanceof GeoShape)) continue;
        String res;
        try {
          res = rel(area.getRelationship((GeoShape) b.shape));
        } catch (RuntimeException e) {
          res = err(e);
        }
        line("REL", a.id, b.id, res);
      }
    }
    // Solids around each shape, as PointInGeo3DShapeQuery's BKD cells are.
    for (Shape s : shapes) {
      if (!(s.shape instanceof GeoShape)) continue;
      GeoShape gs = (GeoShape) s.shape;
      XYZBounds b = new XYZBounds();
      try {
        gs.getBounds(b);
      } catch (RuntimeException e) {
        // recorded by XB
      }
      for (int k = 0; k < 5; k++) {
        double[] v = new double[6];
        Double[] lo = {b.getMinimumX(), b.getMinimumY(), b.getMinimumZ()};
        Double[] hi = {b.getMaximumX(), b.getMaximumY(), b.getMaximumZ()};
        double[] pmin = {pm.getMinimumXValue(), pm.getMinimumYValue(), pm.getMinimumZValue()};
        double[] pmax = {pm.getMaximumXValue(), pm.getMaximumYValue(), pm.getMaximumZValue()};
        for (int d = 0; d < 3; d++) {
          double l = lo[d] == null ? pmin[d] : lo[d];
          double hh = hi[d] == null ? pmax[d] : hi[d];
          double span = Math.max(hh - l, 1e-6);
          double c1 = l - span * 0.5 + R.nextDouble() * span * 2;
          double c2 = c1 + R.nextDouble() * span * (k < 3 ? 0.3 : 1.5);
          v[d * 2] = Math.max(pmin[d], Math.min(c1, c2));
          v[d * 2 + 1] = Math.min(pmax[d], Math.max(c1, c2));
          if (v[d * 2] > v[d * 2 + 1]) {
            v[d * 2 + 1] = v[d * 2];
          }
        }
        String spec = h(v[0]) + " " + h(v[1]) + " " + h(v[2]) + " " + h(v[3]) + " " + h(v[4]) + " " + h(v[5]);
        String res;
        try {
          GeoArea area = GeoAreaFactory.makeGeoArea(pm, v[0], v[1], v[2], v[3], v[4], v[5]);
          res = area.getClass().getSimpleName() + "\t" + rel(area.getRelationship(gs));
        } catch (RuntimeException e) {
          res = err(e);
        }
        line("XREL", s.id, spec, res);
      }
    }
  }
}
