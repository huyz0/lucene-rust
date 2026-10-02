import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.spatial3d.geom.DistanceStyle;
import org.apache.lucene.spatial3d.geom.GeoArea;
import org.apache.lucene.spatial3d.geom.GeoAreaFactory;
import org.apache.lucene.spatial3d.geom.GeoBBoxFactory;
import org.apache.lucene.spatial3d.geom.GeoCircle;
import org.apache.lucene.spatial3d.geom.GeoCircleFactory;
import org.apache.lucene.spatial3d.geom.GeoDistanceShape;
import org.apache.lucene.spatial3d.geom.GeoPath;
import org.apache.lucene.spatial3d.geom.GeoPathFactory;
import org.apache.lucene.spatial3d.geom.GeoPoint;
import org.apache.lucene.spatial3d.geom.GeoPolygon;
import org.apache.lucene.spatial3d.geom.GeoPolygonFactory;
import org.apache.lucene.spatial3d.geom.GeoShape;
import org.apache.lucene.spatial3d.geom.PlanetModel;

/**
 * Java side of the geo3d benchmark pair; the Rust side is {@code
 * benchmarks/rust-runner/src/micro_geo3d.rs}, with the same case names. Both draw the same inputs
 * from a SplitMix64 stream (no {@code java.util.Random}, so the Rust side need not port it): star
 * polygons, circles, paths and boxes on WGS84, x/y/z cells around them (the cells {@code
 * PointInGeo3DShapeQuery} relates per BKD node) and surface points. Each case prints a {@code
 * #check} digest of integer results (relationships, memberships, distances rounded to 1e-6) that the
 * report compares before it shows a ratio: HotSpot's {@code Math.sin/cos} intrinsics can move a
 * coordinate by an ulp, which these digests are coarse enough to absorb.
 */
public final class Geo3dMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final PlanetModel PM = PlanetModel.WGS84;

  interface Op {
    long run();
  }

  static void measure(String name, Op op) {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) System.err.print("");
    return units;
  }

  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }
  }

  static void check(String c, Fnv d, long n) {
    System.out.printf("#check\t%s\t%016x\t%d%n", c, d.h, n);
  }

  /** SplitMix64, identical to the Rust side's. */
  static long state = 0x3D_B3_4C_11L;

  static long next() {
    long z = (state += 0x9E3779B97F4A7C15L);
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
    return z ^ (z >>> 31);
  }

  static double unit() {
    return (next() >>> 11) * 0x1.0p-53;
  }

  static double lat() {
    return (unit() - 0.5) * Math.PI * 0.9;
  }

  static double lon() {
    return (unit() * 2 - 1) * Math.PI;
  }

  /** A star ring of n points around (clat, clon), counter-clockwise. */
  static List<GeoPoint> star(double clat, double clon, int n, double radius) {
    List<GeoPoint> out = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      double a = Math.PI * 2 * i / n;
      double r = radius * (0.5 + unit() * 0.5);
      double la = Math.max(-1.5, Math.min(1.5, clat + r * StrictMath.sin(a)));
      double lo = clon + r * StrictMath.cos(a);
      if (lo > Math.PI) lo -= 2 * Math.PI;
      if (lo < -Math.PI) lo += 2 * Math.PI;
      out.add(new GeoPoint(PM, la, lo));
    }
    return out;
  }

  static long round(double d) {
    return Double.isInfinite(d) ? Long.MAX_VALUE : Math.round(d * 1e6);
  }

  public static void main(String[] args) {
    // Inputs, drawn in a fixed order the Rust side repeats.
    List<double[]> rings = new ArrayList<>();
    List<List<GeoPoint>> ringPoints = new ArrayList<>();
    for (int i = 0; i < 200; i++) {
      double clat = lat(), clon = lon(), radius = 0.02 + unit() * 0.3;
      int n = 6 + (int) (unit() * 14);
      rings.add(new double[] {clat, clon, radius, n});
      ringPoints.add(star(clat, clon, n, radius));
    }
    double[][] circles = new double[2000][];
    for (int i = 0; i < circles.length; i++) {
      circles[i] = new double[] {lat(), lon(), 0.001 + unit() * 0.5};
    }
    List<GeoPoint[]> paths = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      double la = lat(), lo = lon();
      GeoPoint[] p = new GeoPoint[2 + (int) (unit() * 6)];
      for (int k = 0; k < p.length; k++) {
        la = Math.max(-1.5, Math.min(1.5, la + (unit() - 0.5) * 0.2));
        lo = Math.max(-3.1, Math.min(3.1, lo + (unit() - 0.5) * 0.2));
        p[k] = new GeoPoint(PM, la, lo);
      }
      paths.add(p);
    }
    GeoPoint[] points = new GeoPoint[4096];
    for (int i = 0; i < points.length; i++) {
      points[i] = new GeoPoint(PM, lat(), lon());
    }

    // Shapes the relation/membership/distance cases share.
    List<GeoShape> shapes = new ArrayList<>();
    List<GeoDistanceShape> distanceShapes = new ArrayList<>();
    for (int i = 0; i < 40; i++) {
      shapes.add(GeoPolygonFactory.makeGeoPolygon(PM, ringPoints.get(i)));
      GeoCircle c = GeoCircleFactory.makeGeoCircle(PM, circles[i][0], circles[i][1], circles[i][2]);
      shapes.add(c);
      distanceShapes.add(c);
      GeoPath p = GeoPathFactory.makeGeoPath(PM, 0.01, paths.get(i));
      shapes.add(p);
      distanceShapes.add(p);
      double t = circles[i][0], l = circles[i][1];
      shapes.add(GeoBBoxFactory.makeGeoBBox(PM, Math.min(1.5, t + 0.2), t - 0.1, l - 0.2, l));
    }
    // Cells around each shape's center: as small as a BKD leaf and as big as an inner node.
    List<GeoArea> cells = new ArrayList<>();
    for (int i = 0; i < 400; i++) {
      GeoPoint c = points[i];
      double s = new double[] {1e-3, 1e-2, 1e-1, 1.0}[(int) (unit() * 4)] * (0.5 + unit() * 0.5);
      cells.add(
          GeoAreaFactory.makeGeoArea(
              PM, c.x - s, c.x + s * unit(), c.y - s, c.y + s * unit(), c.z - s, c.z + s * unit()));
    }

    Fnv d = new Fnv();
    for (int i = 0; i < 200; i++) {
      GeoPolygon p = GeoPolygonFactory.makeGeoPolygon(PM, ringPoints.get(i));
      for (int k = 0; k < 8; k++) d.add(p.isWithin(points[(i * 8 + k) % points.length]) ? 1 : 0);
    }
    check("geo3d_polygon_build", d, 200);
    measure(
        "geo3d_polygon_build",
        () -> {
          for (List<GeoPoint> r : ringPoints) {
            sink += GeoPolygonFactory.makeGeoPolygon(PM, r).getEdgePoints().length;
          }
          return ringPoints.size();
        });

    d = new Fnv();
    for (double[] c : circles) {
      GeoCircle g = GeoCircleFactory.makeGeoCircle(PM, c[0], c[1], c[2]);
      d.add(g.isWithin(points[(int) (Math.abs(c[0] * 1000)) % points.length]) ? 1 : 0);
    }
    check("geo3d_circle_build", d, circles.length);
    measure(
        "geo3d_circle_build",
        () -> {
          for (double[] c : circles) {
            sink += GeoCircleFactory.makeGeoCircle(PM, c[0], c[1], c[2]).getEdgePoints().length;
          }
          return circles.length;
        });

    d = new Fnv();
    long q = 0;
    for (GeoShape s : shapes) {
      for (GeoArea a : cells) {
        d.add(a.getRelationship(s));
        q++;
      }
    }
    check("geo3d_relate", d, q);
    final long fq = q;
    measure(
        "geo3d_relate",
        () -> {
          long n = 0;
          for (GeoShape s : shapes) {
            for (GeoArea a : cells) {
              n += a.getRelationship(s);
            }
          }
          sink += n;
          return fq;
        });

    d = new Fnv();
    q = 0;
    for (GeoShape s : shapes) {
      for (GeoPoint p : points) {
        d.add(s.isWithin(p) ? 1 : 0);
        q++;
      }
    }
    check("geo3d_within", d, q);
    final long wq = q;
    measure(
        "geo3d_within",
        () -> {
          long n = 0;
          for (GeoShape s : shapes) {
            for (GeoPoint p : points) {
              if (s.isWithin(p.x, p.y, p.z)) n++;
            }
          }
          sink += n;
          return wq;
        });

    d = new Fnv();
    q = 0;
    for (GeoDistanceShape s : distanceShapes) {
      for (GeoPoint p : points) {
        d.add(round(s.computeDistance(DistanceStyle.ARC, p.x, p.y, p.z)));
        d.add(round(s.computeOutsideDistance(DistanceStyle.ARC, p.x, p.y, p.z)));
        q++;
      }
    }
    check("geo3d_distance", d, q);
    final long dq = q;
    measure(
        "geo3d_distance",
        () -> {
          double t = 0;
          for (GeoDistanceShape s : distanceShapes) {
            for (GeoPoint p : points) {
              t += s.computeOutsideDistance(DistanceStyle.ARC, p.x, p.y, p.z);
              double v = s.computeDistance(DistanceStyle.ARC, p.x, p.y, p.z);
              if (v != Double.POSITIVE_INFINITY) t += v;
            }
          }
          sink += (long) t;
          return dq;
        });
  }
}
