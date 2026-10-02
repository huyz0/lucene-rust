import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.spatial3d.geom.DistanceStyle;
import org.apache.lucene.spatial3d.geom.GeoPoint;
import org.apache.lucene.spatial3d.geom.LatLonBounds;
import org.apache.lucene.spatial3d.geom.Membership;
import org.apache.lucene.spatial3d.geom.Plane;
import org.apache.lucene.spatial3d.geom.PlanetModel;
import org.apache.lucene.spatial3d.geom.SidedPlane;
import org.apache.lucene.spatial3d.geom.Vector;
import org.apache.lucene.spatial3d.geom.XYZBounds;

/**
 * geo3d's geometric primitives -- {@code Vector}, {@code GeoPoint}, {@code Plane}, {@code
 * SidedPlane}, {@code PlanetModel} (and its {@code DocValueEncoder}), {@code XYZBounds}, {@code
 * LatLonBounds}, {@code DistanceStyle} -- called directly through their public API on seeded random
 * inputs, near-degenerate ones included, for {@code crates/lucene-util/tests/geo3d_fixtures.rs}.
 *
 * <p>Each line is {@code M op inputs => result}: the inputs are hex double bits (a vector is three,
 * a sided plane is the nine of the {@code SidedPlane(p, A, B)} it is built from, a planet model an
 * index into {@code SPHERE, WGS84, CLARKE_1866, (1.1, 0.9), (0.95, 1.05)}), the result is formatted per op (see
 * {@code fmt*}) or {@code ERR class message}. Generated with the trig intrinsics off, like {@code
 * GenGeo3d}.
 */
public class GenGeo3dMath {
  /**
   * The corpus seed; {@code -Dgeo3d.seed=N} draws another corpus (with the test's {@code
   * GEO3D_FIXTURES} pointing at it) for a wider one-off sweep.
   */
  static final Random R = new Random(Long.getLong("geo3d.seed", 0x3D_3A7_1L));
  static final StringBuilder OUT = new StringBuilder();
  static final PlanetModel[] PMS = {
    PlanetModel.SPHERE,
    PlanetModel.WGS84,
    PlanetModel.CLARKE_1866,
    new PlanetModel(1.1, 0.9),
    new PlanetModel(0.95, 1.05)
  };
  static final DistanceStyle[] STYLES = {
    DistanceStyle.ARC,
    DistanceStyle.LINEAR,
    DistanceStyle.LINEAR_SQUARED,
    DistanceStyle.NORMAL,
    DistanceStyle.NORMAL_SQUARED
  };

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String esc(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static String fmtV(Vector v) {
    return v == null ? "null" : h(v.x) + "," + h(v.y) + "," + h(v.z);
  }

  static String fmtP(Plane p) {
    if (p == null) return "null";
    String s = h(p.x) + "," + h(p.y) + "," + h(p.z) + "," + h(p.D);
    if (p instanceof SidedPlane) s += "," + h(((SidedPlane) p).sigNum);
    return s;
  }

  static String fmtPts(GeoPoint[] pts) {
    if (pts == null) return "null";
    StringBuilder sb = new StringBuilder("[");
    for (int i = 0; i < pts.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(fmtV(pts[i]));
    }
    return sb.append(']').toString();
  }

  static String opt(Double d) {
    return d == null ? "null" : h(d);
  }

  static String fmtXB(XYZBounds b) {
    return opt(b.getMinimumX()) + "," + opt(b.getMaximumX()) + "," + opt(b.getMinimumY()) + ","
        + opt(b.getMaximumY()) + "," + opt(b.getMinimumZ()) + "," + opt(b.getMaximumZ());
  }

  static String fmtLB(LatLonBounds b) {
    return b.checkNoLongitudeBound() + "," + b.checkNoTopLatitudeBound() + ","
        + b.checkNoBottomLatitudeBound() + "," + opt(b.getMaxLatitude()) + ","
        + opt(b.getMinLatitude()) + "," + opt(b.getLeftLongitude()) + ","
        + opt(b.getRightLongitude());
  }

  interface Op {
    Object run() throws Exception;
  }

  /** Records one call: its inputs (already formatted) and its result or exception. */
  static void rec(String op, String inputs, Op f) {
    String result;
    try {
      Object o = f.run();
      result = String.valueOf(o);
    } catch (Exception | AssertionError e) {
      result = "ERR " + e.getClass().getName() + " " + esc(e.getMessage());
    }
    OUT.append("M\t").append(op).append('\t').append(inputs).append("\t=>\t").append(result).append('\n');
  }

  static String in(Object... parts) {
    StringBuilder sb = new StringBuilder();
    for (Object p : parts) {
      if (sb.length() > 0) sb.append(' ');
      if (p instanceof Double) sb.append(h((Double) p));
      else if (p instanceof double[]) {
        double[] a = (double[]) p;
        for (int i = 0; i < a.length; i++) {
          if (i > 0) sb.append(' ');
          sb.append(h(a[i]));
        }
      } else if (p instanceof Vector) {
        Vector v = (Vector) p;
        sb.append(h(v.x)).append(' ').append(h(v.y)).append(' ').append(h(v.z));
      } else sb.append(p);
    }
    return sb.toString();
  }

  // ------------------------------------------------------------- inputs

  static double lat() {
    switch (R.nextInt(6)) {
      case 0:
        return Math.PI * 0.5 - R.nextDouble() * 1e-3;
      case 1:
        return -Math.PI * 0.5 + R.nextDouble() * 1e-3;
      case 2:
        return (R.nextDouble() - 0.5) * 1e-3;
      default:
        return (R.nextDouble() - 0.5) * Math.PI;
    }
  }

  static double lon() {
    return (R.nextDouble() * 2.0 - 1.0) * Math.PI;
  }

  /**
   * A surface point, rebuilt from its coordinates: a point made from a latitude and longitude
   * caches an ellipsoid magnitude that can differ in the last bit from the one its coordinates
   * give, and only the coordinates are recorded.
   */
  static GeoPoint surface(PlanetModel pm) {
    return fresh(new GeoPoint(pm, lat(), lon()));
  }

  static GeoPoint fresh(GeoPoint p) {
    return new GeoPoint(p.x, p.y, p.z);
  }

  /** A point on the surface, or a random or degenerate vector. */
  static Vector vec(PlanetModel pm) {
    switch (R.nextInt(10)) {
      case 0:
        return new Vector(0, 0, 0);
      case 1:
        return new Vector(R.nextGaussian() * 1e-13, R.nextGaussian() * 1e-13, R.nextGaussian());
      case 2:
        return new Vector(R.nextGaussian(), R.nextGaussian(), R.nextGaussian());
      default:
        GeoPoint p = surface(pm);
        return new Vector(p.x, p.y, p.z);
    }
  }

  /** A surface point close to {@code p}, or {@code p} itself, or its antipode. */
  static GeoPoint near(PlanetModel pm, GeoPoint p) {
    switch (R.nextInt(5)) {
      case 0:
        return new GeoPoint(p.x, p.y, p.z);
      case 1:
        return fresh(pm.createSurfacePoint(-p.x, -p.y, -p.z));
      default:
        double e = Math.pow(10, -2 - R.nextInt(12));
        return fresh(pm.createSurfacePoint(
            p.x + R.nextGaussian() * e, p.y + R.nextGaussian() * e, p.z + R.nextGaussian() * e));
    }
  }

  static GeoPoint pointFor(PlanetModel pm, GeoPoint base) {
    return R.nextBoolean() ? near(pm, base) : surface(pm);
  }

  /** A random sided plane through two surface points, as its nine inputs. */
  static double[] sidedSpec(PlanetModel pm) {
    GeoPoint a = surface(pm);
    GeoPoint b = R.nextInt(4) == 0 ? near(pm, a) : surface(pm);
    GeoPoint p = surface(pm);
    return new double[] {p.x, p.y, p.z, a.x, a.y, a.z, b.x, b.y, b.z};
  }

  static SidedPlane sided(double[] s) {
    return new SidedPlane(
        new Vector(s[0], s[1], s[2]), new Vector(s[3], s[4], s[5]), new Vector(s[6], s[7], s[8]));
  }

  static double[] boundsSpec(PlanetModel pm) {
    int k = R.nextInt(3);
    double[] out = new double[1 + 9 * k];
    out[0] = k;
    for (int i = 0; i < k; i++) {
      System.arraycopy(sidedSpec(pm), 0, out, 1 + 9 * i, 9);
    }
    return out;
  }

  static Membership[] bounds(double[] spec) {
    int k = (int) spec[0];
    Membership[] out = new Membership[k];
    for (int i = 0; i < k; i++) {
      double[] s = new double[9];
      System.arraycopy(spec, 1 + 9 * i, s, 0, 9);
      out[i] = sided(s);
    }
    return out;
  }

  /** A plane: through the origin and two points, or horizontal, or vertical, or arbitrary. */
  static Plane plane(PlanetModel pm) {
    switch (R.nextInt(13)) {
      case 11: {
        // Tangent to the planet at a surface point: every degenerate
        // (double-root) branch of the bounds and intersection math.
        GeoPoint t = surface(pm);
        double nx = t.x / (pm.xyScaling * pm.xyScaling);
        double ny = t.y / (pm.xyScaling * pm.xyScaling);
        double nz = t.z / (pm.zScaling * pm.zScaling);
        double m = Math.sqrt(nx * nx + ny * ny + nz * nz);
        nx /= m;
        ny /= m;
        nz /= m;
        return new Plane(nx, ny, nz, -(nx * t.x + ny * t.y + nz * t.z));
      }
      case 12: {
        // Tangent at a pole or an axis point.
        int axis = R.nextInt(3);
        double sign = R.nextBoolean() ? 1.0 : -1.0;
        double r = axis == 2 ? pm.zScaling : pm.xyScaling;
        double[] n = new double[3];
        n[axis] = 1.0;
        return new Plane(n[0], n[1], n[2], -sign * r);
      }
      case 8: {
        double[] n = new double[3];
        n[R.nextInt(3)] = R.nextBoolean() ? 1.0 : -1.0;
        return new Plane(n[0], n[1], n[2], (R.nextDouble() - 0.5) * 1.5);
      }
      case 9:
        return new Plane(0.0, R.nextGaussian(), R.nextGaussian(), (R.nextDouble() - 0.5));
      case 10:
        return new Plane(R.nextGaussian(), 0.0, R.nextGaussian(), (R.nextDouble() - 0.5));
      case 5:
        double b = lon();
        return new Plane(Math.cos(b), Math.sin(b), 0.0, (R.nextDouble() - 0.5) * 1.5);
      case 6:
        return new Plane(0.0, 0.0, 1.0, (R.nextDouble() - 0.5) * 1.5);
      case 7:
        return new Plane(R.nextGaussian(), R.nextGaussian(), R.nextGaussian() * 1e-13, R.nextGaussian());
      case 0:
        return new Plane(pm, Math.sin(lat()));
      case 1:
        double a = lon();
        return new Plane(Math.cos(a), Math.sin(a));
      case 2:
        return new Plane(
            R.nextGaussian(), R.nextGaussian(), R.nextGaussian(), R.nextGaussian() * 0.5);
      default:
        GeoPoint p = surface(pm);
        try {
          return new Plane(p, R.nextBoolean() ? near(pm, p) : surface(pm));
        } catch (IllegalArgumentException e) {
          return Plane.normalZPlane;
        }
    }
  }

  static double[] planeSpec(Plane p) {
    return new double[] {p.x, p.y, p.z, p.D};
  }

  // ------------------------------------------------------------- the ops

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo3d");
    Files.createDirectories(out);
    // StrictMath's tan/atan/atan2 (fdlibm), which geo3d reaches through
    // Math with the intrinsics off: random, tiny, huge and special arguments.
    double[] specials = {0.0, -0.0, 1e-300, 1e-9, 0.5, 1.0, Math.PI / 4, Math.PI / 2, Math.PI, 1e9,
      1e300, Double.MAX_VALUE, Double.MIN_VALUE, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
      Double.NaN};
    for (int i = 0; i < 600; i++) {
      double x =
          i < specials.length
              ? specials[i]
              : switch (i % 4) {
                case 0 -> (R.nextDouble() - 0.5) * 20;
                case 1 -> (R.nextDouble() - 0.5) * 1e-6;
                case 2 -> (R.nextDouble() - 0.5) * 1e6;
                default -> Math.PI / 2 * (R.nextInt(2000) - 1000) + R.nextGaussian() * 1e-12;
              };
      double y =
          i < specials.length ? specials[specials.length - 1 - i] : (R.nextDouble() - 0.5) * 10;
      rec("SM.trig", in(x, y), () -> h(StrictMath.tan(x)) + "," + h(StrictMath.atan(x)) + ","
          + h(StrictMath.atan2(y, x)) + "," + h(StrictMath.atan2(x, y)));
    }
    for (int pmi = 0; pmi < PMS.length; pmi++) {
      final int pi = pmi;
      final PlanetModel pm = PMS[pmi];
      rec("PM.str", in(pi), () -> pm.toString() + "|" + pm.hashCode() + "|" + pm.isSphere());
      for (int i = 0; i < 45; i++) {
        vectorOps(pi, pm);
        pointOps(pi, pm);
        planeOps(pi, pm);
        sidedOps(pi, pm);
        modelOps(pi, pm);
        boundsOps(pi, pm);
      }
    }
    Files.write(out.resolve("math.tsv"), OUT.toString().getBytes(StandardCharsets.UTF_8));
  }

  static void vectorOps(int pi, PlanetModel pm) {
    GeoPoint n = near(pm, surface(pm));
    Vector a = vec(pm), b = R.nextBoolean() ? vec(pm) : new Vector(n.x, n.y, n.z);
    if (R.nextInt(4) == 0) b = new Vector(a.x * 2, a.y * 2, a.z * 2);
    final Vector fb = b;
    Vector c = vec(pm);
    double ang = lon();
    String ab = in(a, b);
    rec("V.normalize", in(a), () -> fmtV(a.normalize()));
    rec("V.perp", ab, () -> fmtV(new Vector(a, fb)));
    rec("V.perpxyz", ab, () -> fmtV(new Vector(a, fb.x, fb.y, fb.z)));
    rec("V.cpez", in(a, b, c), () -> Vector.crossProductEvaluateIsZero(a, fb, c));
    rec("V.dot", ab, () -> h(a.dotProduct(fb)) + "," + h(a.dotProduct(fb.x, fb.y, fb.z)));
    rec("V.translate", ab, () -> fmtV(a.translate(fb.x, fb.y, fb.z)));
    rec("V.rot", in(a, ang), () -> fmtV(a.rotateXY(ang)) + "|" + fmtV(a.rotateXZ(ang)) + "|"
        + fmtV(a.rotateZY(ang)) + "|" + fmtV(a.rotateXY(Math.sin(ang), Math.cos(ang))) + "|"
        + fmtV(a.rotateXZ(Math.sin(ang), Math.cos(ang))) + "|"
        + fmtV(a.rotateZY(Math.sin(ang), Math.cos(ang))));
    rec("V.dist", ab, () -> h(a.linearDistanceSquared(fb)) + "," + h(a.linearDistance(fb)) + ","
        + h(a.normalDistanceSquared(fb)) + "," + h(a.normalDistance(fb)) + ","
        + h(a.linearDistanceSquared(fb.x, fb.y, fb.z)) + "," + h(a.linearDistance(fb.x, fb.y, fb.z))
        + "," + h(a.normalDistanceSquared(fb.x, fb.y, fb.z)) + ","
        + h(a.normalDistance(fb.x, fb.y, fb.z)) + "," + h(a.magnitude()) + ","
        + h(Vector.magnitude(a.x, a.y, a.z)));
    rec("V.same", ab, () -> a.isNumericallyIdentical(fb) + "," + a.isNumericallyIdentical(fb.x, fb.y, fb.z)
        + "," + a.isParallel(fb) + "," + a.isParallel(fb.x, fb.y, fb.z) + "," + a.equals(fb) + ","
        + a.hashCode() + "," + a.toString());
  }

  static void pointOps(int pi, PlanetModel pm) {
    double la = lat(), lo = lon();
    if (R.nextInt(10) == 0) la = (R.nextBoolean() ? 1 : -1) * (Math.PI * 0.5 + 1e-3);
    if (R.nextInt(10) == 0) lo = (R.nextBoolean() ? 1 : -1) * (Math.PI + 1e-3);
    final double fla = la, flo = lo;
    rec("G.ctor", in(pi, la, lo), () -> {
      GeoPoint p = new GeoPoint(pm, fla, flo);
      return fmtV(p) + "|" + h(p.getLatitude()) + "|" + h(p.getLongitude()) + "|" + h(p.magnitude());
    });
    GeoPoint p = surface(pm);
    Vector v = vec(pm);
    double m = R.nextBoolean() ? 1.0 : 0.5 + R.nextDouble();
    rec("G.mag", in(m, v), () -> {
      GeoPoint q = new GeoPoint(m, v.x, v.y, v.z);
      return fmtV(q) + "|" + h(q.getLatitude()) + "|" + h(q.getLongitude()) + "|" + h(q.magnitude());
    });
    rec("G.xyz", in(v), () -> {
      GeoPoint q = new GeoPoint(v.x, v.y, v.z);
      return h(q.getLatitude()) + "|" + h(q.getLongitude()) + "|" + h(q.magnitude()) + "|"
          + q.hashCode() + "|" + q.toString();
    });
    rec("G.arc", in(p, v), () -> h(p.arcDistance(v)) + "," + h(p.arcDistance(v.x, v.y, v.z)) + ","
        + p.isIdentical(new GeoPoint(v.x, v.y, v.z)) + "," + p.isIdentical(v.x, v.y, v.z) + ","
        + p.isIdentical(p.x, p.y, p.z));
    rec("G.trig", in(pi, m, la, lo, v), () -> {
      GeoPoint q = new GeoPoint(pm, Math.sin(fla), Math.sin(flo), Math.cos(fla), Math.cos(flo), fla, flo);
      GeoPoint r = new GeoPoint(pm, Math.sin(fla), Math.sin(flo), Math.cos(fla), Math.cos(flo));
      GeoPoint t = new GeoPoint(fla, flo, v.x, v.y, v.z);
      String s = fmtV(q) + "|" + fmtV(r) + "|" + h(r.getLatitude()) + "|" + h(t.getLatitude()) + "|"
          + h(t.magnitude());
      GeoPoint u = new GeoPoint(m, v.x, v.y, v.z, fla, flo);
      return s + "|" + fmtV(u) + "|" + h(u.getLongitude()) + "|" + h(u.magnitude());
    });
  }

  static void planeOps(int pi, PlanetModel pm) {
    Plane pl = plane(pm);
    Plane q;
    switch (R.nextInt(6)) {
      case 0:
        q = new Plane(pl.x, pl.y, pl.z, pl.D + R.nextGaussian() * 1e-13);
        break;
      case 1:
        // Parallel, apart.
        q = new Plane(pl.x, pl.y, pl.z, pl.D + 0.05 + R.nextDouble() * 0.5);
        break;
      default:
        q = plane(pm);
    }
    String ps = in(planeSpec(pl));
    String pq = in(planeSpec(pl), planeSpec(q));
    GeoPoint a = surface(pm);
    GeoPoint b = pointFor(pm, a);
    GeoPoint c = pointFor(pm, a);
    rec("P.ctor", in(pi, a, b), () -> fmtP(new Plane(a, b)) + "|" + fmtP(new Plane(a, b.x, b.y, b.z)));
    double sl = Math.sin(lat()), vx = R.nextGaussian(), vy = R.nextGaussian();
    rec("P.simple", in(pi, sl, vx, vy, a), () -> fmtP(new Plane(pm, sl)) + "|" + fmtP(new Plane(vx, vy))
        + "|" + fmtP(new Plane(a, sl)) + "|" + fmtP(new Plane(new Plane(a, sl), true)) + "|"
        + fmtP(new Plane(new Plane(a, sl), false)));
    rec("P.center1", in(planeSpec(pl), a), () -> fmtP(Plane.constructPerpendicularCenterPlaneOnePoint(pl, a)));
    rec("P.center2", in(a, b), () -> fmtP(Plane.constructPerpendicularCenterPlaneTwoPoints(a, b)));
    double D = R.nextGaussian() * 0.3;
    rec("P.norm", in(a, b, c, D), () -> fmtP(Plane.constructNormalizedZPlane(a, b, c)) + "|"
        + fmtP(Plane.constructNormalizedYPlane(a, b, c)) + "|" + fmtP(Plane.constructNormalizedXPlane(a, b, c))
        + "|" + fmtP(Plane.constructNormalizedZPlane(a.x, a.y)) + "|"
        + fmtP(Plane.constructNormalizedYPlane(a.x, a.z, D)) + "|"
        + fmtP(Plane.constructNormalizedXPlane(a.y, a.z, D)) + "|"
        + fmtP(Plane.constructNormalizedZPlane(a.x * 1e-13, a.y * 1e-13)) + "|"
        + fmtP(Plane.constructNormalizedYPlane(a.x * 1e-13, a.z * 1e-13, D)) + "|"
        + fmtP(Plane.constructNormalizedXPlane(a.y * 1e-13, a.z * 1e-13, D)));
    rec("P.eval", in(planeSpec(pl), a), () -> h(pl.evaluate(a)) + "," + pl.evaluateIsZero(a) + ","
        + pl.evaluateIsZero(a.x, a.y, a.z) + "," + fmtP(pl.normalize()) + "," + pl.hashCode());
    double[] bs = boundsSpec(pm);
    GeoPoint t = R.nextBoolean() ? surface(pm) : near(pm, a);
    rec("P.dist", in(pi, planeSpec(pl), t, bs), () -> {
      Membership[] bd = bounds(bs);
      return h(pl.arcDistance(pm, t, bd)) + "," + h(pl.arcDistance(pm, t.x, t.y, t.z, bd)) + ","
          + h(pl.normalDistance(t, bd)) + "," + h(pl.normalDistance(t.x, t.y, t.z, bd)) + ","
          + h(pl.normalDistanceSquared(t, bd)) + "," + h(pl.normalDistanceSquared(t.x, t.y, t.z, bd))
          + "," + h(pl.linearDistance(pm, t, bd)) + "," + h(pl.linearDistance(pm, t.x, t.y, t.z, bd))
          + "," + h(pl.linearDistanceSquared(pm, t, bd)) + ","
          + h(pl.linearDistanceSquared(pm, t.x, t.y, t.z, bd));
    });
    rec("P.inter", in(pi, pq, bs), () -> {
      Membership[] bd = bounds(bs);
      return fmtPts(pl.findIntersections(pm, q, bd)) + "|" + fmtPts(pl.findCrossings(pm, q, bd)) + "|"
          + pl.isFunctionallyIdentical(q) + "|" + pl.isNumericallyIdentical(q) + "|"
          + fmtV(pl.getSampleIntersectionPoint(pm, q));
    });
    // A plane through two surface points, cut by other planes.
    Plane edge;
    try {
      edge = new Plane(a, b);
    } catch (IllegalArgumentException e) {
      edge = pl;
    }
    final Plane fe = edge;
    double[] bs2 = boundsSpec(pm);
    GeoPoint n1 = surface(pm), n2 = surface(pm);
    rec("P.isect", in(pi, planeSpec(fe), planeSpec(q), a, b, n1, n2, bs, bs2), () -> {
      Membership[] bd = bounds(bs), bd2 = bounds(bs2);
      GeoPoint[] notable = {a, b};
      GeoPoint[] notable2 = {n1, n2};
      return fe.intersects(pm, q, notable, notable2, bd, bd2) + ","
          + fe.crosses(pm, q, notable, notable2, bd, bd2) + ","
          + q.intersects(pm, fe, new GeoPoint[0], notable, bd2) + ","
          + q.crosses(pm, fe, new GeoPoint[0], notable, bd2);
    });
    final Plane bp = R.nextBoolean() ? fe : pl;
    rec("P.bounds", in(pi, planeSpec(bp), planeSpec(q), bs), () -> {
      Membership[] bd = bounds(bs);
      XYZBounds xb = new XYZBounds();
      bp.recordBounds(pm, xb, bd);
      XYZBounds xb2 = new XYZBounds();
      bp.recordBounds(pm, xb2, q, bd);
      LatLonBounds lb = new LatLonBounds();
      bp.recordBounds(pm, lb, bd);
      LatLonBounds lb2 = new LatLonBounds();
      bp.recordBounds(pm, lb2, q, bd);
      return fmtXB(xb) + "|" + fmtXB(xb2) + "|" + fmtLB(lb) + "|" + fmtLB(lb2);
    });
    double dist = R.nextInt(4) == 0 ? 0.0 : R.nextDouble() * 1.5;
    rec("P.arcpts", in(pi, planeSpec(fe), dist, a, b, c, bs), () -> {
      Membership[] bd = bounds(bs);
      String s = fmtPts(fe.findArcDistancePoints(pm, dist, a, bd));
      for (DistanceStyle st : STYLES) {
        try {
          s += "|" + fmtPts(st.findDistancePoints(pm, dist, a, fe, bd));
        } catch (Exception e) {
          s += "|ERR " + e.getClass().getName();
        }
        try {
          s += "," + h(st.findMinimumArcDistance(pm, dist)) + "," + h(st.findMaximumArcDistance(pm, dist));
        } catch (Exception e) {
          s += ",ERR " + e.getClass().getName();
        }
        s += "," + h(st.computeDistance(a, b)) + "," + h(st.computeDistance(pm, fe, c, bd)) + ","
            + h(st.toAggregationForm(dist)) + "," + h(st.fromAggregationForm(dist)) + ","
            + h(st.aggregateDistances(dist, dist * 0.5, 1.0));
      }
      return s;
    });
    int k = 1 + R.nextInt(4);
    double[] props = new double[k];
    for (int i = 0; i < k; i++) props[i] = R.nextDouble() * (R.nextInt(5) == 0 ? 2 : 1);
    final GeoPoint is = R.nextInt(6) == 0 ? c : a, ie = R.nextInt(6) == 0 ? c : b;
    rec("P.interp", in(pi, planeSpec(fe), is, ie, k, props), () -> fmtPts(fe.interpolate(pm, is, ie, props)));
    rec("P.coplanar", in(a, b, c), () -> Plane.arePointsCoplanar(a, b, c));
  }

  static void sidedOps(int pi, PlanetModel pm) {
    GeoPoint p = surface(pm), a = surface(pm), b = pointFor(pm, a), c = pointFor(pm, a);
    double sl = Math.sin(lat()), vx = R.nextGaussian(), vy = R.nextGaussian(), vz = R.nextGaussian();
    double D = R.nextGaussian() * 0.3;
    String s = in(pi, p, a, b, c, sl, vx, vy, vz, D);
    rec("S.ctor", s, () -> {
      StringBuilder sb = new StringBuilder();
      Object[] tries = new Object[11];
      for (int i = 0; i < tries.length; i++) {
        try {
          switch (i) {
            case 0: tries[i] = fmtP(new SidedPlane(p, a, b)); break;
            case 1: tries[i] = fmtP(new SidedPlane(a, b)); break;
            case 2: tries[i] = fmtP(new SidedPlane(p, a, b.x, b.y, b.z)); break;
            case 3: tries[i] = fmtP(new SidedPlane(p, true, a, b)); break;
            case 4: tries[i] = fmtP(new SidedPlane(p, false, a, b)); break;
            case 5: tries[i] = fmtP(new SidedPlane(p, pm, sl)); break;
            case 6: tries[i] = fmtP(new SidedPlane(p, vx, vy)); break;
            case 7: tries[i] = fmtP(new SidedPlane(p, vx, vy, vz, D)); break;
            case 8: tries[i] = fmtP(new SidedPlane(p, new Vector(vx, vy, vz), D)); break;
            case 9: tries[i] = fmtP(new SidedPlane(p.x, p.y, p.z, new Vector(vx, vy, vz), D)); break;
            default: tries[i] = fmtP(new SidedPlane(new SidedPlane(p, a, b))); break;
          }
        } catch (Exception e) {
          tries[i] = "ERR " + e.getClass().getName() + " " + esc(e.getMessage());
        }
        if (i > 0) sb.append('|');
        sb.append(tries[i]);
      }
      return sb.toString();
    });
    rec("S.static", s, () -> {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < 5; i++) {
        String r;
        try {
          switch (i) {
            case 0: r = fmtP(SidedPlane.constructNormalizedPerpendicularSidedPlane(p, new Vector(vx, vy, vz), a, b)); break;
            case 1: r = fmtP(SidedPlane.constructSidedPlaneFromTwoPoints(p, a, b)); break;
            case 2: r = fmtP(SidedPlane.constructSidedPlaneFromOnePoint(p, new Plane(a, b), c)); break;
            case 3: r = fmtP(SidedPlane.constructNormalizedThreePointSidedPlane(p, a, b, c)); break;
            default: {
              SidedPlane sp = new SidedPlane(p, a, b);
              r = sp.isWithin(c) + "," + sp.isWithin(c.x, c.y, c.z) + "," + sp.strictlyWithin(c) + ","
                  + sp.strictlyWithin(c.x, c.y, c.z) + "," + sp.strictlyWithin(a) + "," + sp.hashCode()
                  + "," + sp.equals(new SidedPlane(p, a, b)) + "," + sp.isWithin(a);
            }
          }
        } catch (Exception e) {
          r = "ERR " + e.getClass().getName() + " " + esc(e.getMessage());
        }
        if (i > 0) sb.append('|');
        sb.append(r);
      }
      return sb.toString();
    });
  }

  static void modelOps(int pi, PlanetModel pm) {
    Vector v = vec(pm);
    double s = 0.9 + R.nextDouble() * 0.2;
    final Vector w = new Vector(v.x * s, v.y * s, v.z * s);
    rec("PM.pt", in(pi, w), () -> pm.pointOnSurface(w) + "," + pm.pointOnSurface(w.x, w.y, w.z) + ","
        + pm.pointOutside(w) + "," + pm.pointOutside(w.x, w.y, w.z) + "," + fmtV(pm.createSurfacePoint(w))
        + "," + h(pm.getMinimumMagnitude()) + "," + h(pm.getMaximumMagnitude()) + ","
        + h(pm.getMinimumXValue()) + "," + h(pm.getMaximumZValue()));
    PlanetModel.DocValueEncoder enc = pm.getDocValueEncoder();
    double x = (R.nextDouble() * 2 - 1) * pm.getMaximumMagnitude() * 1.1;
    rec("PM.round", in(pi, x), () -> h(enc.roundDownX(x)) + "," + h(enc.roundUpX(x)) + ","
        + h(enc.roundDownY(x)) + "," + h(enc.roundUpY(x)) + "," + h(enc.roundDownZ(x)) + ","
        + h(enc.roundUpZ(x)));
    GeoPoint p = surface(pm);
    rec("PM.dv", in(pi, p), () -> {
      long dv = enc.encodePoint(p);
      return dv + "," + h(enc.decodeXValue(dv)) + "," + h(enc.decodeYValue(dv)) + ","
          + h(enc.decodeZValue(dv)) + "," + fmtV(enc.decodePoint(dv));
    });
  }

  static void boundsOps(int pi, PlanetModel pm) {
    int n = R.nextInt(5);
    double[] pts = new double[1 + 3 * n];
    pts[0] = n;
    XYZBounds b = new XYZBounds();
    LatLonBounds lb = new LatLonBounds();
    for (int i = 0; i < n; i++) {
      GeoPoint p = surface(pm);
      pts[1 + 3 * i] = p.x;
      pts[2 + 3 * i] = p.y;
      pts[3 + 3 * i] = p.z;
    }
    int flags = R.nextInt(64);
    Vector probe = vec(pm);
    rec("B.ops", in(pi, flags, pts, probe), () -> {
      for (int i = 0; i < n; i++) {
        GeoPoint p = new GeoPoint(pts[1 + 3 * i], pts[2 + 3 * i], pts[3 + 3 * i]);
        if ((flags & 1) != 0 && i == 0) {
          b.addXValue(p).addYValue(p).addZValue(p);
          lb.addXValue(p).addYValue(p).addZValue(p);
        } else {
          b.addPoint(p);
          lb.addPoint(p);
        }
      }
      if ((flags & 2) != 0) {
        b.isWide().noLongitudeBound();
        lb.isWide();
      }
      if ((flags & 4) != 0) {
        lb.noLongitudeBound();
      }
      if ((flags & 8) != 0) {
        lb.noTopLatitudeBound();
        b.noTopLatitudeBound();
      }
      if ((flags & 16) != 0) {
        lb.noBottomLatitudeBound();
        b.noBottomLatitudeBound();
      }
      if ((flags & 32) != 0) {
        lb.noBound(pm);
        b.noBound(pm);
      }
      XYZBounds other = new XYZBounds();
      other.addPoint(new GeoPoint(probe.x, probe.y, probe.z));
      XYZBounds sum = new XYZBounds();
      b.addBounds(sum);
      other.addBounds(sum);
      return fmtXB(b) + "|" + fmtLB(lb) + "|" + b.isWithin(probe) + "," + b.isWithin(probe.x, probe.y, probe.z)
          + "," + b.overlaps(other) + "," + other.overlaps(b) + "," + b.overlaps(sum) + ","
          + b.isSmallestMinX(pm) + b.isLargestMaxX(pm) + b.isSmallestMinY(pm) + b.isLargestMaxY(pm)
          + b.isSmallestMinZ(pm) + b.isLargestMaxZ(pm) + "|" + fmtXB(sum) + "|" + b + "|" + lb;
    });
  }
}
