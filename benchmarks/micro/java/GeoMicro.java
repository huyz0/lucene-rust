import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.Component2D;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Point;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.geo.Tessellator;
import org.apache.lucene.geo.XYCircle;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.geo.XYLine;
import org.apache.lucene.geo.XYPoint;
import org.apache.lucene.geo.XYPolygon;
import org.apache.lucene.geo.XYRectangle;
import org.apache.lucene.util.SloppyMath;

/**
 * Java side of the geo benchmark pair; the Rust side is {@code
 * benchmarks/rust-runner/src/micro_geo.rs}, with the same case names. Both read their inputs from
 * the differential fixtures under {@code fixtures/data/geo/} (the polygons the tessellator test
 * replays, the shapes and queries the {@code Component2D} test replays), so they time exactly the
 * work the tests verify, and each case prints a {@code #check} digest of its results that the report
 * compares before it shows a ratio.
 */
public final class GeoMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final Path DIR = Path.of("fixtures/data/geo");

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

  // ---------------------------------------------------------------- spec parsing

  static String unesc(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '\\' && i + 1 < s.length()) {
        char n = s.charAt(++i);
        b.append(n == 't' ? '\t' : n == 'n' ? '\n' : n == 'r' ? '\r' : n);
      } else {
        b.append(c);
      }
    }
    return b.toString();
  }

  static double[][] ringD(String s) {
    String[] pts = s.split(";");
    double[] a = new double[pts.length], b = new double[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] xy = pts[i].split(" ");
      a[i] = Double.parseDouble(xy[0]);
      b[i] = Double.parseDouble(xy[1]);
    }
    return new double[][] {a, b};
  }

  static float[][] ringF(String s) {
    String[] pts = s.split(";");
    float[] a = new float[pts.length], b = new float[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] xy = pts[i].split(" ");
      a[i] = Float.parseFloat(xy[0]);
      b[i] = Float.parseFloat(xy[1]);
    }
    return new float[][] {a, b};
  }

  static Polygon polygon(String body) {
    String[] rings = body.split("\\|");
    Polygon[] holes = new Polygon[rings.length - 1];
    for (int i = 1; i < rings.length; i++) {
      double[][] r = ringD(rings[i]);
      holes[i - 1] = new Polygon(r[0], r[1]);
    }
    double[][] s = ringD(rings[0]);
    return new Polygon(s[0], s[1], holes);
  }

  static XYPolygon xyPolygon(String body) {
    String[] rings = body.split("\\|");
    XYPolygon[] holes = new XYPolygon[rings.length - 1];
    for (int i = 1; i < rings.length; i++) {
      float[][] r = ringF(rings[i]);
      holes[i - 1] = new XYPolygon(r[0], r[1]);
    }
    float[][] s = ringF(rings[0]);
    return new XYPolygon(s[0], s[1], holes);
  }

  static double[] nums(String s) {
    String[] p = s.split(",");
    double[] d = new double[p.length];
    for (int i = 0; i < p.length; i++) d[i] = Double.parseDouble(p[i]);
    return d;
  }

  static LatLonGeometry[] latlon(String spec) {
    String[] gs = spec.split(" \\+ ");
    LatLonGeometry[] out = new LatLonGeometry[gs.length];
    for (int i = 0; i < gs.length; i++) {
      String kind = gs[i].substring(0, 2), body = gs[i].substring(2);
      switch (kind) {
        case "P:" -> {
          double[] v = nums(body);
          out[i] = new Point(v[0], v[1]);
        }
        case "L:" -> {
          double[][] r = ringD(body);
          out[i] = new Line(r[0], r[1]);
        }
        case "G:" -> out[i] = polygon(body);
        case "C:" -> {
          double[] v = nums(body);
          out[i] = new Circle(v[0], v[1], v[2]);
        }
        default -> {
          double[] v = nums(body);
          out[i] = new Rectangle(v[0], v[1], v[2], v[3]);
        }
      }
    }
    return out;
  }

  static XYGeometry[] xy(String spec) {
    String[] gs = spec.split(" \\+ ");
    XYGeometry[] out = new XYGeometry[gs.length];
    for (int i = 0; i < gs.length; i++) {
      String kind = gs[i].substring(0, 2), body = gs[i].substring(2);
      float[] v = null;
      if (!kind.equals("L:") && !kind.equals("G:")) {
        String[] p = body.split(",");
        v = new float[p.length];
        for (int k = 0; k < p.length; k++) v[k] = Float.parseFloat(p[k]);
      }
      switch (kind) {
        case "P:" -> out[i] = new XYPoint(v[0], v[1]);
        case "L:" -> {
          float[][] r = ringF(body);
          out[i] = new XYLine(r[0], r[1]);
        }
        case "G:" -> out[i] = xyPolygon(body);
        case "C:" -> out[i] = new XYCircle(v[0], v[1], v[2]);
        default -> out[i] = new XYRectangle(v[0], v[1], v[2], v[3]);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- cases

  public static void main(String[] args) throws IOException {
    tessellate();
    components();
    haversin();
  }

  static void tessellate() throws IOException {
    List<Polygon> latlon = new ArrayList<>();
    List<XYPolygon> xyPolys = new ArrayList<>();
    String[] lines = Files.readString(DIR.resolve("tessellator.tsv")).split("\n");
    for (int i = 0; i < lines.length; i++) {
      String[] f = lines[i].split("\t");
      // checked polygons that Lucene tessellates
      if (!f[0].equals("poly") || !f[3].equals("1") || i + 1 >= lines.length || lines[i + 1].startsWith("ERR")) continue;
      String body = unesc(f[4]).substring(2);
      if (f[2].equals("latlon")) latlon.add(polygon(body));
      else xyPolys.add(xyPolygon(body));
    }
    for (boolean check : new boolean[] {true, false}) {
      String name = "tessellate_latlon" + (check ? "_checked" : "");
      Fnv d = new Fnv();
      long tris = 0;
      for (Polygon p : latlon) {
        for (Tessellator.Triangle t : Tessellator.tessellate(p, check)) {
          tris++;
          for (int v = 0; v < 3; v++) {
            d.add(t.getEncodedX(v));
            d.add(t.getEncodedY(v));
            d.add(t.isEdgefromPolygon(v) ? 1 : 0);
          }
        }
      }
      check(name, d, tris);
      measure(name, () -> {
        long n = 0;
        for (Polygon p : latlon) n += Tessellator.tessellate(p, check).size();
        sink += n;
        return latlon.size();
      });
    }
    Fnv d = new Fnv();
    long tris = 0;
    for (XYPolygon p : xyPolys) {
      for (Tessellator.Triangle t : Tessellator.tessellate(p, true)) {
        tris++;
        d.add(t.getEncodedX(0));
        d.add(t.getEncodedY(2));
      }
    }
    check("tessellate_xy_checked", d, tris);
    measure("tessellate_xy_checked", () -> {
      long n = 0;
      for (XYPolygon p : xyPolys) n += Tessellator.tessellate(p, true).size();
      sink += n;
      return xyPolys.size();
    });
  }

  record Shape(boolean geo, String spec, List<double[]> relate, List<double[]> contains, List<double[]> tris) {}

  static void components() throws IOException {
    List<Shape> shapes = new ArrayList<>();
    Shape cur = null;
    for (String line : Files.readString(DIR.resolve("component2d.tsv")).split("\n")) {
      String[] f = line.split("\t");
      if (f[0].equals("shape")) {
        cur = f.length > 4 ? null : new Shape(f[2].equals("latlon"), unesc(f[3]), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        if (cur != null) shapes.add(cur);
        continue;
      }
      if (cur == null) continue;
      switch (f[1]) {
        case "relate" -> cur.relate.add(nums(f[2]));
        case "contains" -> cur.contains.add(nums(f[2]));
        case "itri" -> cur.tris.add(nums(f[2]));
        default -> {}
      }
    }
    List<Component2D> comps = new ArrayList<>();
    for (Shape s : shapes) comps.add(s.geo ? LatLonGeometry.create(latlon(s.spec)) : XYGeometry.create(xy(s.spec)));

    measure("component_build", () -> {
      long n = 0;
      for (Shape s : shapes) {
        Component2D c = s.geo ? LatLonGeometry.create(latlon(s.spec)) : XYGeometry.create(xy(s.spec));
        n += Double.doubleToRawLongBits(c.getMinX());
      }
      sink += n;
      return shapes.size();
    });

    // each query repeated so one batch is long enough to time
    final int reps = 20;
    Fnv d = new Fnv();
    long q = 0;
    for (int i = 0; i < shapes.size(); i++) {
      for (double[] b : shapes.get(i).relate) {
        d.add(comps.get(i).relate(b[0], b[1], b[2], b[3]).ordinal());
        q++;
      }
    }
    check("component_relate", d, q);
    final long relateQueries = q;
    measure("component_relate", () -> {
      long n = 0;
      for (int r = 0; r < reps; r++) {
        for (int i = 0; i < shapes.size(); i++) {
          Component2D c = comps.get(i);
          for (double[] b : shapes.get(i).relate) n += c.relate(b[0], b[1], b[2], b[3]).ordinal();
        }
      }
      sink += n;
      return relateQueries * reps;
    });

    d = new Fnv();
    q = 0;
    for (int i = 0; i < shapes.size(); i++) {
      for (double[] p : shapes.get(i).contains) {
        d.add(comps.get(i).contains(p[0], p[1]) ? 1 : 0);
        q++;
      }
    }
    check("component_contains", d, q);
    final long containsQueries = q;
    measure("component_contains", () -> {
      long n = 0;
      for (int r = 0; r < reps; r++) {
        for (int i = 0; i < shapes.size(); i++) {
          Component2D c = comps.get(i);
          for (double[] p : shapes.get(i).contains) n += c.contains(p[0], p[1]) ? 1 : 0;
        }
      }
      sink += n;
      return containsQueries * reps;
    });

    d = new Fnv();
    q = 0;
    for (int i = 0; i < shapes.size(); i++) {
      for (double[] t : shapes.get(i).tris) {
        d.add(comps.get(i).intersectsTriangle(t[0], t[1], t[2], t[3], t[4], t[5]) ? 1 : 0);
        q++;
      }
    }
    check("component_intersects_triangle", d, q);
    final long triQueries = q;
    measure("component_intersects_triangle", () -> {
      long n = 0;
      for (int r = 0; r < reps; r++) {
        for (int i = 0; i < shapes.size(); i++) {
          Component2D c = comps.get(i);
          for (double[] t : shapes.get(i).tris) n += c.intersectsTriangle(t[0], t[1], t[2], t[3], t[4], t[5]) ? 1 : 0;
        }
      }
      sink += n;
      return triQueries * reps;
    });
  }

  static void haversin() throws IOException {
    List<double[]> pts = new ArrayList<>();
    for (String line : Files.readString(DIR.resolve("sloppy_math.tsv")).split("\n")) {
      String[] f = line.split("\t");
      if (!f[0].equals("hav")) continue;
      String[] h = f[1].split(",");
      double[] p = new double[4];
      for (int k = 0; k < 4; k++) p[k] = Double.longBitsToDouble(Long.parseUnsignedLong(h[k], 16));
      pts.add(p);
    }
    Fnv d = new Fnv();
    for (double[] p : pts) d.add(Double.doubleToRawLongBits(SloppyMath.haversinMeters(p[0], p[1], p[2], p[3])));
    check("haversin_meters", d, pts.size());
    measure("haversin_meters", () -> {
      double s = 0;
      for (double[] p : pts) s += SloppyMath.haversinMeters(p[0], p[1], p[2], p[3]);
      sink += Double.doubleToRawLongBits(s);
      return pts.size();
    });
  }
}
