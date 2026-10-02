import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import com.google.common.geometry.S2Cell;
import com.google.common.geometry.S2CellId;
import com.google.common.geometry.S2LatLng;
import com.google.common.geometry.S2Point;
import com.google.common.geometry.S2Projections;
import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.context.SpatialContextFactory;
import org.locationtech.spatial4j.distance.DistanceUtils;
import org.locationtech.spatial4j.io.GeohashUtils;
import org.locationtech.spatial4j.shape.Point;
import org.locationtech.spatial4j.shape.Rectangle;
import org.locationtech.spatial4j.shape.Shape;

/**
 * The Spatial4j 0.8 subset spatial-extras uses, Lucene's Geo3D bridge to it
 * ({@code org.apache.lucene.spatial.spatial4j}) and the S2 cell ids {@code S2PrefixTree} uses,
 * called directly on seeded random inputs, for {@code crates/lucene-util/tests/spatial4j_fixtures.rs}.
 *
 * <p>Each line is {@code op TAB inputs TAB => TAB result}. Shapes are written as specs the test
 * rebuilds through the same factory calls: {@code P x y}, {@code R minX maxX minY maxY}, {@code C
 * x y d}, {@code L buf n x1 y1 ...} (a line string), {@code M n spec...} (a shape collection), with
 * doubles as hex bits. A context is an index into {@link #CTX_ARGS}. Results are formatted per op,
 * or {@code ERR class message}. Generated with the trig intrinsics off, like {@code GenGeo3d}.
 */
public class GenSpatial4j {
  static final Random R = new Random(Long.getLong("spatial4j.seed", 0x5E47_1A_4L));
  static final StringBuilder OUT = new StringBuilder();

  /** The contexts, as the args {@code SpatialContextFactory.makeSpatialContext} reads. */
  static final String[][] CTX_ARGS = {
    {},
    {"distCalculator", "lawOfCosines"},
    {"distCalculator", "vincentySphere"},
    {"normWrapLongitude", "true"},
    {"geo", "false"},
    {"geo", "false", "worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)"},
    {"spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory"},
    {
      "spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
      "planetModel", "wgs84"
    },
  };

  static SpatialContext[] CTX = new SpatialContext[CTX_ARGS.length];

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static double d(String s) {
    return Double.longBitsToDouble(Long.parseUnsignedLong(s, 16));
  }

  static String esc(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  interface Op {
    Object run() throws Exception;
  }

  static String err(Throwable e) {
    String s = "ERR " + e.getClass().getName() + " " + esc(e.getMessage());
    if (e instanceof ParseException) s += " @" + ((ParseException) e).getErrorOffset();
    return s;
  }

  static void rec(String op, String inputs, Op f) {
    String result;
    try {
      result = String.valueOf(f.run());
    } catch (Exception e) {
      result = err(e);
    }
    OUT.append(op).append('\t').append(inputs).append("\t=>\t").append(result).append('\n');
  }

  static boolean geo3d(int c) {
    return c >= 6;
  }

  // ------------------------------------------------------------- shapes from specs

  static int pos;

  static Shape parse(SpatialContext ctx, String[] t) {
    String k = t[pos++];
    switch (k) {
      case "P":
        return ctx.getShapeFactory().pointXY(d(t[pos++]), d(t[pos++]));
      case "R":
        return ctx.getShapeFactory().rect(d(t[pos++]), d(t[pos++]), d(t[pos++]), d(t[pos++]));
      case "C":
        return ctx.getShapeFactory().circle(d(t[pos++]), d(t[pos++]), d(t[pos++]));
      case "L":
        {
          double buf = d(t[pos++]);
          int n = Integer.parseInt(t[pos++]);
          List<Point> pts = new ArrayList<>();
          for (int i = 0; i < n; i++) {
            pts.add(ctx.getShapeFactory().pointXY(d(t[pos++]), d(t[pos++])));
          }
          return ctx.getShapeFactory().lineString(pts, buf);
        }
      case "M":
        {
          int n = Integer.parseInt(t[pos++]);
          List<Shape> shapes = new ArrayList<>();
          for (int i = 0; i < n; i++) shapes.add(parse(ctx, t));
          return ctx.getShapeFactory().multiShape(shapes);
        }
      default:
        throw new IllegalStateException(k);
    }
  }

  static Shape shape(int c, String spec) {
    pos = 0;
    return parse(CTX[c], spec.split(" "));
  }

  // ------------------------------------------------------------- random inputs

  static double lon() {
    switch (R.nextInt(8)) {
      case 0:
        return 180;
      case 1:
        return -180;
      case 2:
        return 180 - R.nextDouble() * 1e-6;
      case 3:
        return Math.round(R.nextDouble() * 36) * 10 - 180.0;
      default:
        return R.nextDouble() * 360 - 180;
    }
  }

  static double lat() {
    switch (R.nextInt(8)) {
      case 0:
        return 90;
      case 1:
        return -90;
      case 2:
        return 90 - R.nextDouble() * 1e-3;
      case 3:
        return Math.round(R.nextDouble() * 18) * 10 - 90.0;
      default:
        return R.nextDouble() * 180 - 90;
    }
  }

  static double cart() {
    switch (R.nextInt(6)) {
      case 0:
        return 1000;
      case 1:
        return -1000;
      case 2:
        return Math.round(R.nextDouble() * 20) * 100 - 1000.0;
      default:
        return R.nextDouble() * 2000 - 1000;
    }
  }

  static double dist(boolean geo) {
    switch (R.nextInt(8)) {
      case 0:
        return 0;
      case 1:
        return geo ? 90 + R.nextDouble() * 90 : R.nextDouble() * 1000;
      case 2:
        return geo ? 180 : 500;
      case 3:
        return R.nextDouble() * 1e-3;
      default:
        return R.nextDouble() * (geo ? 40 : 300);
    }
  }

  static String spec(int c, int depth) {
    boolean geo = CTX[c].isGeo();
    int kinds = geo3d(c) ? 4 : (depth > 0 ? 4 : 5);
    switch (R.nextInt(kinds)) {
      case 0:
        return "P " + h(geo ? lon() : cart()) + " " + h(geo ? lat() : cart());
      case 1:
        {
          double a = geo ? lon() : cart(), b = geo ? lon() : cart();
          double y1 = geo ? lat() : cart(), y2 = geo ? lat() : cart();
          if (!geo && a > b) {
            double tmp = a;
            a = b;
            b = tmp;
          }
          if (R.nextInt(6) == 0) b = a; // a vertical line
          return "R " + h(a) + " " + h(b) + " " + h(Math.min(y1, y2)) + " " + h(Math.max(y1, y2));
        }
      case 2:
        {
          double x = geo ? lon() : cart(), y = geo ? lat() : cart();
          if (!geo) x = x / 2;
          if (!geo) y = y / 2;
          return "C " + h(x) + " " + h(y) + " " + h(dist(geo) / (geo ? 1 : 4));
        }
      case 3:
        {
          int n = 1 + R.nextInt(4);
          StringBuilder sb = new StringBuilder("L " + h(dist(geo) / (geo ? 8 : 20)) + " " + n);
          double x = geo ? R.nextDouble() * 300 - 150 : cart() / 2;
          double y = geo ? R.nextDouble() * 120 - 60 : cart() / 2;
          for (int i = 0; i < n; i++) {
            sb.append(' ').append(h(x)).append(' ').append(h(y));
            if (R.nextInt(5) != 0) {
              x += (R.nextDouble() - 0.5) * (geo ? 40 : 300);
              y += (R.nextDouble() - 0.5) * (geo ? 30 : 300);
              x = Math.max(geo ? -180 : -1000, Math.min(geo ? 180 : 1000, x));
              y = Math.max(geo ? -90 : -1000, Math.min(geo ? 90 : 1000, y));
            }
          }
          return sb.toString();
        }
      default:
        {
          int n = R.nextInt(4);
          StringBuilder sb = new StringBuilder("M " + n);
          for (int i = 0; i < n; i++) sb.append(' ').append(spec(c, depth + 1));
          return sb.toString();
        }
    }
  }

  static String bbox(Shape s) {
    Rectangle r = s.getBoundingBox();
    return h(r.getMinX()) + "," + h(r.getMaxX()) + "," + h(r.getMinY()) + "," + h(r.getMaxY());
  }

  static String pt(Point p) {
    return h(p.getX()) + "," + h(p.getY());
  }

  static String safe(Op f) {
    try {
      return String.valueOf(f.run());
    } catch (Exception e) {
      return err(e);
    }
  }

  // ------------------------------------------------------------- the records

  static void shapes() {
    for (int c = 0; c < CTX.length; c++) {
      final int cc = c;
      SpatialContext ctx = CTX[c];
      List<String> specs = new ArrayList<>();
      for (int i = 0; i < 160; i++) specs.add(spec(c, 0));
      for (String s : specs) {
        double bd = R.nextDouble() * (ctx.isGeo() ? 10 : 100);
        rec("shape", c + "\t" + s + "\t" + h(bd), () -> {
          Shape sh = shape(cc, s);
          StringBuilder sb = new StringBuilder();
          sb.append(geo3d(cc) ? "-" : esc(sh.toString()));
          sb.append(" | ").append(safe(() -> bbox(sh)));
          sb.append(" | ").append(safe(() -> pt(sh.getCenter())));
          sb.append(" | ").append(sh.hasArea()).append(' ').append(sh.isEmpty());
          sb.append(" | ").append(safe(() -> h(sh.getArea(ctx))));
          sb.append(" | ").append(safe(() -> h(sh.getArea(null))));
          sb.append(" | ").append(safe(() -> {
            Shape b = sh.getBuffered(bd, ctx);
            return (geo3d(cc) ? "-" : esc(b.toString())) + " " + bbox(b);
          }));
          return sb.toString();
        });
      }
      // relations and equality between random pairs, and each with itself
      for (int i = 0; i < 700; i++) {
        String a = specs.get(R.nextInt(specs.size()));
        String b = R.nextInt(10) == 0 ? a : specs.get(R.nextInt(specs.size()));
        rec("rel", c + "\t" + a + "\t" + b, () -> {
          Shape sa = shape(cc, a);
          Shape sb = shape(cc, b);
          return safe(() -> sa.relate(sb)) + " " + safe(() -> sa.equals(sb));
        });
      }
      // relations against small rectangles and points (what a prefix tree asks)
      for (int i = 0; i < 300; i++) {
        String a = specs.get(R.nextInt(specs.size()));
        String b;
        if (R.nextBoolean()) {
          b = "P " + h(ctx.isGeo() ? lon() : cart()) + " " + h(ctx.isGeo() ? lat() : cart());
        } else {
          double w = Math.pow(2, -R.nextInt(12)) * (ctx.isGeo() ? 90 : 500);
          double x = ctx.isGeo() ? R.nextDouble() * (360 - w) - 180 : R.nextDouble() * (2000 - w) - 1000;
          double y = ctx.isGeo() ? R.nextDouble() * (180 - w / 2) - 90 : R.nextDouble() * (2000 - w) - 1000;
          if (R.nextBoolean()) {
            // near the shape: inside, across its edge, around its center
            try {
              Rectangle bb = shape(cc, a).getBoundingBox();
              double bx = R.nextBoolean() ? bb.getMinX() : bb.getCenter().getX();
              double by = R.nextBoolean() ? bb.getMinY() : bb.getCenter().getY();
              w = Math.max(1e-9, bb.getWidth() * Math.pow(2, -R.nextInt(6)));
              x = Math.max(ctx.isGeo() ? -180 : -1000, Math.min((ctx.isGeo() ? 180 : 1000) - w, bx - w * R.nextDouble() * 0.5));
              y = Math.max(ctx.isGeo() ? -90 : -1000, Math.min((ctx.isGeo() ? 90 : 1000) - w / 2, by - w * R.nextDouble() * 0.25));
            } catch (Exception e) {
              // keep the random box
            }
          }
          b = "R " + h(x) + " " + h(x + w) + " " + h(y) + " " + h(y + w / 2);
        }
        rec("rel", c + "\t" + a + "\t" + b, () -> {
          Shape sa = shape(cc, a);
          Shape sb = shape(cc, b);
          return safe(() -> sa.relate(sb)) + " " + safe(() -> sa.equals(sb));
        });
      }
      // distances
      for (int i = 0; i < 120; i++) {
        double x1 = ctx.isGeo() ? lon() : cart(), y1 = ctx.isGeo() ? lat() : cart();
        double x2 = ctx.isGeo() ? lon() : cart(), y2 = ctx.isGeo() ? lat() : cart();
        double dd = dist(ctx.isGeo());
        double bearing = R.nextDouble() * 720 - 360;
        rec("dist", c + "\t" + h(x1) + " " + h(y1) + " " + h(x2) + " " + h(y2) + " " + h(dd) + " " + h(bearing), () -> {
          Point p = ctx.getShapeFactory().pointXY(x1, y1);
          Point q = ctx.getShapeFactory().pointXY(x2, y2);
          return safe(() -> h(ctx.getDistCalc().distance(p, q)))
              + " " + safe(() -> h(ctx.getDistCalc().distance(p, x2, y2)))
              + " " + safe(() -> ctx.getDistCalc().within(p, x2, y2, dd))
              + " " + safe(() -> pt(ctx.getDistCalc().pointOnBearing(p, dd, bearing, ctx, null)))
              + " " + safe(() -> bbox(ctx.getDistCalc().calcBoxByDistFromPt(p, dd, ctx, null)))
              + " " + safe(() -> h(ctx.getDistCalc().calcBoxByDistFromPt_yHorizAxisDEG(p, dd, ctx)));
        });
      }
      // the binary codec, written and read back
      for (int i = 0; i < 60; i++) {
        String s = specs.get(R.nextInt(specs.size()));
        rec("codec", c + "\t" + s, () -> {
          Shape sh = shape(cc, s);
          ByteArrayOutputStream bytes = new ByteArrayOutputStream();
          DataOutputStream out = new DataOutputStream(bytes);
          ctx.getBinaryCodec().writeShape(out, sh);
          out.flush();
          StringBuilder hex = new StringBuilder();
          for (byte b : bytes.toByteArray()) hex.append(String.format("%02x", b));
          Shape back =
              ctx.getBinaryCodec()
                  .readShape(new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())));
          return hex + " " + bbox(back) + " " + back.equals(sh);
        });
      }
    }
  }

  static final String[] WKT = {
    "POINT(1 2)",
    " point ( -180 90 ) ",
    "POINT EMPTY",
    "POINT Z (1 2 3)",
    "POINT ZM EMPTY",
    "POINT Z X (1 2)",
    "POINT(1)",
    "POINT(1 2",
    "POINT(1 2) x",
    "POINT(181 0)",
    "POINT(1-2 3)",
    "POINT(1e2 -3.5E-1)",
    "POINT(.5 +2.)",
    "ENVELOPE(-10, 20, 15, 10)",
    "ENVELOPE(170, -170, 10, -10)",
    "ENVELOPE(10, 20, 10, 15)",
    "ENVELOPE(-180, 180, 90, -90)",
    "MULTIPOINT(1 2, 3 4)",
    "MULTIPOINT((1 2), (3 4))",
    "MULTIPOINT EMPTY",
    "LINESTRING(0 0, 10 10, 20 0)",
    "LINESTRING EMPTY",
    "MULTILINESTRING((0 0, 1 1), (5 5, 6 7))",
    "MULTILINESTRING EMPTY",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 2))",
    "POLYGON EMPTY",
    "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 0)), ((20 20, 30 20, 30 30, 20 20)))",
    "GEOMETRYCOLLECTION(POINT(1 2), ENVELOPE(0, 5, 5, 0))",
    "GEOMETRYCOLLECTION EMPTY",
    "GEOMETRYCOLLECTION(POINT(1 2), FOO(1))",
    "BUFFER(POINT(1 2), 3)",
    "BUFFER(LINESTRING(0 0, 10 10), 1.5)",
    "BUFFER(ENVELOPE(0, 5, 5, 0), 2)",
    "FOO(1 2)",
    "",
    "   ",
    "1 2",
    "ENVELOPE(1, 2, 3)",
    "ENVELOPE(1 2 3 4)",
    "POINT(NaN 1)",
    "POINT(1 2)) ",
    "LINESTRING(0 0, 1 a)",
    "BUFFER(POINT(1 2) 3)",
    "BUFFER(POINT(0 0), -1)",
    "POLYGON((0 0, 1 0, 2 0, 0 0))",
    "MULTIPOLYGON EMPTY",
    "ENVELOPE(-10, 10, 95, 0)",
    "POINT(-1500 1)",
  };

  static void wkt() {
    for (int c = 0; c < CTX.length; c++) {
      final int cc = c;
      SpatialContext ctx = CTX[c];
      for (String w : WKT) {
        rec("wkt", c + "\t" + esc(w), () -> {
          Shape s = ctx.getFormats().getWktReader().read(w);
          return (geo3d(cc) ? "-" : esc(s.toString()))
              + " | " + safe(() -> bbox(s));
        });
      }
    }
  }

  static void distanceUtils() {
    for (int i = 0; i < 400; i++) {
      double a = R.nextInt(5) == 0 ? (R.nextDouble() - 0.5) * 4000 : lon();
      double b = R.nextInt(5) == 0 ? (R.nextDouble() - 0.5) * 4000 : lat();
      double dd = dist(true);
      double la1 = Math.toRadians(lat()), lo1 = Math.toRadians(lon());
      double la2 = Math.toRadians(lat()), lo2 = Math.toRadians(lon());
      if (R.nextInt(10) == 0) {
        la2 = la1;
        lo2 = lo1;
      }
      final double fla2 = la2, flo2 = lo2;
      rec("du", h(a) + " " + h(b) + " " + h(dd) + " " + h(la1) + " " + h(lo1) + " " + h(la2) + " " + h(lo2), () ->
          h(DistanceUtils.normLonDEG(a)) + " " + h(DistanceUtils.normLatDEG(b)) + " "
              + h(DistanceUtils.calcBoxByDistFromPt_deltaLonDEG(b, a, dd)) + " "
              + h(DistanceUtils.calcBoxByDistFromPt_latHorizAxisDEG(b, a, dd)) + " "
              + h(DistanceUtils.calcLonDegreesAtLat(b, dd)) + " "
              + h(DistanceUtils.distHaversineRAD(la1, lo1, fla2, flo2)) + " "
              + h(DistanceUtils.distLawOfCosinesRAD(la1, lo1, fla2, flo2)) + " "
              + h(DistanceUtils.distVincentyRAD(la1, lo1, fla2, flo2)) + " "
              + h(DistanceUtils.dist2Degrees(dd, DistanceUtils.EARTH_MEAN_RADIUS_KM)) + " "
              + h(DistanceUtils.degrees2Dist(dd, DistanceUtils.EARTH_MEAN_RADIUS_KM)) + " "
              + h(DistanceUtils.toRadians(a)) + " " + h(DistanceUtils.toDegrees(la1)));
    }
  }

  static void geohash() {
    for (int i = 0; i < 300; i++) {
      double la = lat(), lo = lon();
      int prec = 1 + R.nextInt(GeohashUtils.MAX_PRECISION);
      rec("gh", h(la) + " " + h(lo) + " " + prec, () -> {
        String hash = GeohashUtils.encodeLatLon(la, lo, prec);
        Rectangle r = GeohashUtils.decodeBoundary(hash, CTX[0]);
        Point p = GeohashUtils.decode(hash.toUpperCase(), CTX[0]);
        return hash + " " + bbox(r) + " " + pt(p) + " " + String.join(",", GeohashUtils.getSubGeohashes(hash.substring(0, prec - 1)));
      });
    }
    for (int i = 0; i < 200; i++) {
      double lonErr = Math.pow(10, R.nextDouble() * 8 - 6), latErr = Math.pow(10, R.nextDouble() * 8 - 6);
      rec("ghlen", h(lonErr) + " " + h(latErr), () -> GeohashUtils.lookupHashLenForWidthHeight(lonErr, latErr));
    }
    for (int len = 0; len <= GeohashUtils.MAX_PRECISION; len++) {
      final int l = len;
      rec("ghsize", String.valueOf(len), () -> {
        double[] s = GeohashUtils.lookupDegreesSizeForHashLen(l);
        return h(s[0]) + " " + h(s[1]);
      });
    }
  }

  static void s2() {
    for (int i = 0; i < 400; i++) {
      double la = lat(), lo = lon();
      int level = R.nextInt(S2CellId.MAX_LEVEL + 1);
      rec("s2", h(la) + " " + h(lo) + " " + level, () -> {
        S2CellId leaf = S2CellId.fromLatLng(S2LatLng.fromDegrees(la, lo));
        S2CellId id = leaf.parent(level);
        StringBuilder sb = new StringBuilder();
        sb.append(Long.toHexString(leaf.id())).append(' ').append(Long.toHexString(id.id()));
        sb.append(' ').append(id.level()).append(' ').append(id.face()).append(' ').append(id.toToken());
        sb.append(' ').append(id.toString().replace(' ', '_'));
        sb.append(' ').append(id.isLeaf()).append(' ').append(id.isFace()).append(' ').append(id.isValid());
        for (int l = 1; l <= level; l++) sb.append(l == 1 ? " " : ",").append(id.childPosition(l));
        if (level < S2CellId.MAX_LEVEL) {
          S2CellId child = id.childBegin(level + 1);
          sb.append(' ').append(Long.toHexString(child.id())).append(' ').append(Long.toHexString(child.next().id()));
          sb.append(' ').append(id.contains(child)).append(' ').append(child.contains(id));
          sb.append(' ').append(id.compareTo(child)).append(' ').append(child.compareTo(id.next()));
        }
        S2Cell cell = new S2Cell(id);
        for (int k = 0; k < 4; k++) {
          S2Point v = cell.getVertexRaw(k);
          sb.append(' ').append(h(v.get(0))).append(',').append(h(v.get(1))).append(',').append(h(v.get(2)));
        }
        return sb.toString();
      });
    }
    for (int face = 0; face < 6; face++) {
      final int f = face;
      rec("s2face", String.valueOf(face), () -> Long.toHexString(S2CellId.fromFacePosLevel(f, 0, 0).id()));
    }
    for (int i = 0; i < 100; i++) {
      double v = Math.pow(10, R.nextDouble() * 10 - 9) * (R.nextInt(20) == 0 ? -1 : 1);
      rec("s2min", h(v), () -> S2Projections.MAX_WIDTH.getMinLevel(v));
    }
    for (int level = -2; level <= 31; level++) {
      final int l = level;
      rec("s2val", String.valueOf(level), () -> h(S2Projections.MAX_WIDTH.getValue(l)));
    }
  }

  public static void main(String[] args) throws Exception {
    for (int c = 0; c < CTX_ARGS.length; c++) {
      Map<String, String> m = new HashMap<>();
      for (int i = 0; i < CTX_ARGS[c].length; i += 2) m.put(CTX_ARGS[c][i], CTX_ARGS[c][i + 1]);
      CTX[c] = SpatialContextFactory.makeSpatialContext(m, GenSpatial4j.class.getClassLoader());
    }
    shapes();
    wkt();
    distanceUtils();
    geohash();
    s2();
    Path dir = Path.of(args[0], "spatial4j");
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("spatial4j.tsv"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
