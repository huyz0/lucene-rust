import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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

/**
 * The seeded random geometry corpus shared by the geo fixture generators ({@link GenGeo}, {@link
 * GenGeoTessellator}), and its text serialization, which {@code
 * crates/lucene-util/tests/geo_fixtures.rs} parses back into the same geometries.
 *
 * <p>Spec grammar (numbers are {@code Double.toString} for lat/lon, {@code Float.toString} for
 * cartesian, both shortest-round-trip, so the Rust side re-reads the exact bits):
 *
 * <pre>
 *   P:a,b              point (lat,lon | x,y)
 *   L:a b;a b;...      line
 *   G:ring|hole|...    polygon; a ring is "a b;a b;..." closed
 *   C:a,b,r            circle
 *   R:a,b,c,d          rectangle (minLat,maxLat,minLon,maxLon | minX,maxX,minY,maxY)
 * </pre>
 *
 * Several geometries are joined by {@code " + "}. Lat/lon pairs are written lat first, cartesian x
 * first.
 *
 * <p>Not a generator: no {@code Gen} prefix, so {@code gen-fixtures.sh} compiles it and never runs
 * it.
 */
public final class GeoCorpus {
  private GeoCorpus() {}

  // ---------------------------------------------------------------- random lat/lon polygons

  static double clampLat(double v) {
    return Math.max(-90, Math.min(90, v));
  }

  static double clampLon(double v) {
    return Math.max(-180, Math.min(180, v));
  }

  /** A star-shaped ring around (clat, clon): simple by construction, then clamped to the globe. */
  static double[][] starRing(
      Random r, double clat, double clon, double radius, int n, double minFrac, boolean round) {
    double[] angles = new double[n];
    for (int i = 0; i < n; i++) angles[i] = r.nextDouble() * 2 * Math.PI;
    java.util.Arrays.sort(angles);
    double[] lats = new double[n + 1];
    double[] lons = new double[n + 1];
    for (int i = 0; i < n; i++) {
      double rad = radius * (minFrac + (1 - minFrac) * r.nextDouble());
      double lat = clampLat(clat + rad * Math.sin(angles[i]));
      double lon = clampLon(clon + rad * Math.cos(angles[i]));
      if (round) {
        // coarse coordinates make collinear and duplicate points likely
        lat = Math.rint(lat * 4) / 4;
        lon = Math.rint(lon * 4) / 4;
      }
      lats[i] = lat;
      lons[i] = lon;
    }
    lats[n] = lats[0];
    lons[n] = lons[0];
    return new double[][] {lats, lons};
  }

  static double[][] reverse(double[][] ring) {
    int n = ring[0].length;
    double[] lats = new double[n];
    double[] lons = new double[n];
    for (int i = 0; i < n; i++) {
      lats[i] = ring[0][n - 1 - i];
      lons[i] = ring[1][n - 1 - i];
    }
    return new double[][] {lats, lons};
  }

  /** Same vertices, random order: almost always self-intersecting. */
  static double[][] shuffled(Random r, double[][] ring) {
    int n = ring[0].length - 1;
    double[] lats = new double[n + 1];
    double[] lons = new double[n + 1];
    int[] perm = new int[n];
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
      int j = r.nextInt(i + 1);
      int t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
    }
    for (int i = 0; i < n; i++) {
      lats[i] = ring[0][perm[i]];
      lons[i] = ring[1][perm[i]];
    }
    lats[n] = lats[0];
    lons[n] = lons[0];
    return new double[][] {lats, lons};
  }

  /** A random point on the globe, biased toward the poles and the dateline one time in three. */
  static double[] center(Random r) {
    double lat = r.nextDouble() * 160 - 80;
    double lon = r.nextDouble() * 340 - 170;
    switch (r.nextInt(6)) {
      case 0:
        lat = r.nextBoolean() ? 89 + r.nextDouble() : -89 - r.nextDouble();
        break;
      case 1:
        lon = r.nextBoolean() ? 179 + r.nextDouble() : -179 - r.nextDouble();
        break;
      default:
        break;
    }
    return new double[] {lat, lon};
  }

  /** Every kind of polygon the corpus holds; each entry is a spec of one or more polygons. */
  static List<Polygon> polygons(Random r, int count) {
    List<Polygon> out = new ArrayList<>();
    int attempts = 0;
    while (out.size() < count && attempts++ < count * 20) {
      try {
        out.add(polygon(r, out.size()));
      } catch (IllegalArgumentException e) {
        // an invalid construction (e.g. collapsed by clamping): try another
      }
    }
    // hand-built cases from Lucene's own TestTessellator / TestPolygon2D scenarios
    out.addAll(fixedPolygons());
    return out;
  }

  static Polygon polygon(Random r, int i) {
    double[] c = center(r);
    int kind = i % 10;
    double radius = Math.pow(10, -3 + r.nextDouble() * 4.5); // 0.001 .. ~30 degrees
    switch (kind) {
      case 0:
      case 1:
        {
          double[][] ring = starRing(r, c[0], c[1], radius, 3 + r.nextInt(20), 0.4, false);
          if (r.nextBoolean()) ring = reverse(ring);
          return new Polygon(ring[0], ring[1]);
        }
      case 2:
        {
          // morton path: more than 80 vertices
          double[][] ring = starRing(r, c[0], c[1], radius, 81 + r.nextInt(90), 0.6, false);
          if (r.nextBoolean()) ring = reverse(ring);
          return new Polygon(ring[0], ring[1]);
        }
      case 3:
      case 4:
        {
          // holes: small stars around points well inside the outer ring
          double[][] ring = starRing(r, c[0], c[1], radius, 6 + r.nextInt(30), 0.85, false);
          int numHoles = 1 + r.nextInt(kind == 4 ? 6 : 2);
          List<Polygon> holes = new ArrayList<>();
          for (int h = 0; h < numHoles; h++) {
            double a = 2 * Math.PI * h / numHoles;
            double off = numHoles == 1 ? 0 : radius * 0.45;
            double[][] hole =
                starRing(
                    r,
                    clampLat(c[0] + off * Math.sin(a)),
                    clampLon(c[1] + off * Math.cos(a)),
                    radius * 0.18,
                    3 + r.nextInt(8),
                    0.5,
                    false);
            if (r.nextBoolean()) hole = reverse(hole);
            holes.add(new Polygon(hole[0], hole[1]));
          }
          return new Polygon(ring[0], ring[1], holes.toArray(new Polygon[0]));
        }
      case 5:
        {
          // self-intersecting
          double[][] ring = shuffled(r, starRing(r, c[0], c[1], radius, 5 + r.nextInt(10), 0.3, false));
          return new Polygon(ring[0], ring[1]);
        }
      case 6:
        {
          // coarse grid coordinates: duplicates and collinear runs
          double[][] ring = starRing(r, c[0], c[1], 1 + r.nextDouble() * 3, 5 + r.nextInt(25), 0.2, true);
          return new Polygon(ring[0], ring[1]);
        }
      case 7:
        {
          // touching a pole or the dateline exactly
          double lat0 = r.nextBoolean() ? 90 : -90;
          double lon0 = r.nextBoolean() ? 180 : -180;
          double w = 0.5 + r.nextDouble() * 10;
          double sLat = Math.signum(lat0);
          double sLon = Math.signum(lon0);
          if (r.nextBoolean()) {
            double[] lats = {lat0, lat0 - sLat * w, lat0 - sLat * w, lat0, lat0};
            double[] lons = {c[1], c[1], c[1] + w, c[1] + w, c[1]};
            for (int k = 0; k < 5; k++) lons[k] = clampLon(lons[k]);
            return new Polygon(lats, lons);
          } else {
            double[] lats = {c[0], c[0], c[0] + w, c[0] + w, c[0]};
            double[] lons = {lon0, lon0 - sLon * w, lon0 - sLon * w, lon0, lon0};
            for (int k = 0; k < 5; k++) lats[k] = clampLat(lats[k]);
            return new Polygon(lats, lons);
          }
        }
      case 8:
        if (i % 20 == 8) {
          // one adjacent pair swapped: a small local self-intersection, which
          // the tessellator's CURE pass removes when not checking
          int n = (i % 40 == 8) ? 6 + r.nextInt(30) : 85 + r.nextInt(40);
          double[][] ring = starRing(r, c[0], c[1], radius, n, 0.9, false);
          int k = 1 + r.nextInt(n - 3);
          double t = ring[0][k];
          ring[0][k] = ring[0][k + 1];
          ring[0][k + 1] = t;
          t = ring[1][k];
          ring[1][k] = ring[1][k + 1];
          ring[1][k + 1] = t;
          return new Polygon(ring[0], ring[1]);
        }
        {
          // nearly degenerate: a sliver
          double d = radius * 1e-6;
          double[] lats = {c[0], c[0] + d, c[0] + 2 * d, c[0] + radius, c[0]};
          double[] lons = {c[1], c[1] + radius, c[1] + 2 * radius, c[1] + d, c[1]};
          for (int k = 0; k < 5; k++) {
            lats[k] = clampLat(lats[k]);
            lons[k] = clampLon(lons[k]);
          }
          return new Polygon(lats, lons);
        }
      default:
        {
          // zero area: every vertex on one line
          double[] lats = {c[0], c[0] + radius, c[0] + 2 * radius, c[0] + radius, c[0]};
          double[] lons = {c[1], c[1] + radius, c[1] + 2 * radius, c[1] + radius, c[1]};
          for (int k = 0; k < 5; k++) {
            lats[k] = clampLat(lats[k]);
            lons[k] = clampLon(lons[k]);
          }
          return new Polygon(lats, lons);
        }
    }
  }

  static List<Polygon> fixedPolygons() {
    List<Polygon> out = new ArrayList<>();
    // a square with a hole touching the shell at one vertex
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {0, 5, 5, 0}, new double[] {0, 2, 5, 0})));
    // two holes sharing a vertex
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {2, 5, 2, 2}, new double[] {2, 5, 5, 2}),
            new Polygon(new double[] {5, 8, 8, 5}, new double[] {5, 5, 8, 5})));
    // a hole outside the shell
    out.add(
        new Polygon(
            new double[] {0, 0, 1, 1, 0},
            new double[] {0, 1, 1, 0, 0},
            new Polygon(new double[] {5, 6, 6, 5}, new double[] {5, 5, 6, 5})));
    // a bow tie
    out.add(new Polygon(new double[] {0, 1, 0, 1, 0}, new double[] {0, 1, 1, 0, 0}));
    // the dateline-spanning world box and the whole world
    out.add(new Polygon(new double[] {-90, -90, 90, 90, -90}, new double[] {-180, 180, 180, -180, -180}));
    // a comb: many reflex vertices
    {
      int teeth = 30;
      double[] lats = new double[teeth * 2 + 3];
      double[] lons = new double[teeth * 2 + 3];
      int k = 0;
      for (int t = 0; t < teeth; t++) {
        lats[k] = 0;
        lons[k++] = t;
        lats[k] = 5;
        lons[k++] = t + 0.5;
      }
      lats[k] = 0;
      lons[k++] = teeth;
      lats[k] = -1;
      lons[k++] = teeth / 2.0;
      lats[k] = lats[0];
      lons[k] = lons[0];
      out.add(new Polygon(lats, lons));
    }
    // holes touching the shell at its corner, sharing that vertex
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {0, 1, 2, 0}, new double[] {0, 4, 4, 0}),
            new Polygon(new double[] {0, 4, 4, 0}, new double[] {0, 2, 1, 0}),
            new Polygon(new double[] {0, 3, 2, 0}, new double[] {0, 3, 2, 0})));
    // three holes sharing one interior vertex (chained shared-vertex merges)
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {5, 5, 6, 5}, new double[] {5, 7, 7, 5}),
            new Polygon(new double[] {5, 7, 7, 5}, new double[] {5, 5, 4, 5}),
            new Polygon(new double[] {5, 5, 4, 5}, new double[] {5, 3, 3, 5}),
            new Polygon(new double[] {5, 3, 3, 5}, new double[] {5, 5, 6, 5})));
    // holes whose leftmost vertices coincide (the hole sort's last tie-break)
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {2, 1, 2.5, 2}, new double[] {2, 4, 4, 2}),
            new Polygon(new double[] {2, 3, 4, 2}, new double[] {2, 4, 3, 2})));
    // a hole touching the shell's edge (not a vertex): the bridge ray hits it exactly
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {5, 4, 6, 5}, new double[] {0, 3, 3, 0})));
    // a hole whose leftmost vertex is level with a shell vertex to its left
    out.add(
        new Polygon(
            new double[] {0, 5, 10, 10, 0, 0},
            new double[] {1, 0, 1, 10, 10, 1},
            new Polygon(new double[] {5, 4, 6, 5}, new double[] {3, 5, 5, 3})));
    // a hole of collinear points, and a hole far outside with a long ring
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 0},
            new double[] {0, 10, 10, 0, 0},
            new Polygon(new double[] {1, 2, 3, 1}, new double[] {1, 2, 3, 1})));
    {
      int n = 40;
      double[] lats = new double[n + 1];
      double[] lons = new double[n + 1];
      for (int k = 0; k < n; k++) {
        lats[k] = 50 + 3 * Math.sin(2 * Math.PI * k / n);
        lons[k] = 50 + 3 * Math.cos(2 * Math.PI * k / n);
      }
      lats[n] = lats[0];
      lons[n] = lons[0];
      out.add(
          new Polygon(
              new double[] {0, 0, 10, 10, 0},
              new double[] {0, 10, 10, 0, 0},
              new Polygon(lats, lons)));
    }
    // a spike doubling back along itself (ring self-overlap)
    out.add(
        new Polygon(
            new double[] {0, 0, 5, 10, 5, 10, 10, 0},
            new double[] {0, 10, 10, 10, 10, 10, 0, 0}));
    out.add(
        new Polygon(
            new double[] {0, 0, 10, 10, 5, 5, 5, 0},
            new double[] {0, 10, 10, 0, 0, 5, 2, 0}));
    // combs past the morton threshold: the SPLIT pass on z-ordered nodes
    for (int teeth : new int[] {45, 70}) {
      double[] lats = new double[teeth * 2 + 3];
      double[] lons = new double[teeth * 2 + 3];
      int k = 0;
      for (int t = 0; t < teeth; t++) {
        lats[k] = 0;
        lons[k++] = t * 0.1;
        lats[k] = 5 + (t % 3);
        lons[k++] = t * 0.1 + 0.05;
      }
      lats[k] = 0;
      lons[k++] = teeth * 0.1;
      lats[k] = -1;
      lons[k++] = teeth * 0.05;
      lats[k] = lats[0];
      lons[k] = lons[0];
      out.add(new Polygon(lats, lons));
    }
    // a spiral: deep SPLIT recursion
    {
      int n = 120;
      double[] lats = new double[2 * n + 1];
      double[] lons = new double[2 * n + 1];
      for (int k = 0; k < n; k++) {
        double a = k * 0.25;
        double rad = 1 + k * 0.08;
        lats[k] = rad * Math.sin(a);
        lons[k] = rad * Math.cos(a);
        double rad2 = rad + 0.5;
        lats[2 * n - 1 - k] = rad2 * Math.sin(a);
        lons[2 * n - 1 - k] = rad2 * Math.cos(a);
      }
      lats[2 * n] = lats[0];
      lons[2 * n] = lons[0];
      out.add(new Polygon(lats, lons));
    }
    // past the morton threshold, polygons that need the SPLIT pass: an
    // earlier corpus polygon (holes on the pole) and the four holes sharing
    // one vertex, each with an extra 80-point hole elsewhere
    out.add(withCircleHole(fromSpec("G:-86.86737826154896 26.51454402962736;-84.4712724352173 26.09376343767987;-76.25355817827216 9.181299289049523;-76.60563758407736 7.214210164740096;-79.57433616804926 3.4897899309380236;-84.40820021900568 0.6857091029569009;-85.92640268234479 0.8782562408888506;-90.0 2.8263168243244827;-90.0 4.597102435947102;-90.0 5.6391846906056;-90.0 14.105011898137173;-90.0 19.778369747749295;-90.0 22.566639414502674;-86.86737826154896 26.51454402962736|-87.49341134820244 19.979778286956353;-90.0 18.46440296320393;-90.0 21.493435169262472;-87.49341134820244 19.979778286956353|-87.21810549623036 8.141296141436479;-89.65496009215225 8.142924166776647;-90.0 7.103439437222652;-90.0 6.5683028475352625;-90.0 6.1713462579135125;-88.16473787815741 4.731716690706579;-87.21810549623036 8.141296141436479"), -80, 15, 0.3));
    out.add(withCircleHole(fixedFourHoles(), 8, 2, 0.5));
    // a ring with a duplicated vertex and a spike
    out.add(
        new Polygon(
            new double[] {0, 0, 0, 3, 3, 3, 0},
            new double[] {0, 1, 1, 1, 5, 0, 0}));
    return out;
  }

  static Polygon fixedFourHoles() {
    return new Polygon(
        new double[] {0, 0, 10, 10, 0},
        new double[] {0, 10, 10, 0, 0},
        new Polygon(new double[] {5, 5, 6, 5}, new double[] {5, 7, 7, 5}),
        new Polygon(new double[] {5, 7, 7, 5}, new double[] {5, 5, 4, 5}),
        new Polygon(new double[] {5, 5, 4, 5}, new double[] {5, 3, 3, 5}),
        new Polygon(new double[] {5, 3, 3, 5}, new double[] {5, 5, 6, 5}));
  }

  /** A polygon from a {@code G:} spec body (or a whole spec). */
  static Polygon fromSpec(String spec) {
    String body = spec.startsWith("G:") ? spec.substring(2) : spec;
    String[] rings = body.split("\\|");
    Polygon[] holes = new Polygon[rings.length - 1];
    for (int i = 1; i < rings.length; i++) {
      double[][] r = parseRing(rings[i]);
      holes[i - 1] = new Polygon(r[0], r[1]);
    }
    double[][] shell = parseRing(rings[0]);
    return new Polygon(shell[0], shell[1], holes);
  }

  static double[][] parseRing(String ring) {
    String[] pts = ring.split(";");
    double[] a = new double[pts.length];
    double[] b = new double[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] xy = pts[i].split(" ");
      a[i] = Double.parseDouble(xy[0]);
      b[i] = Double.parseDouble(xy[1]);
    }
    return new double[][] {a, b};
  }

  /** {@code p} plus an 80-point circular hole: pushes it past the morton threshold. */
  static Polygon withCircleHole(Polygon p, double lat, double lon, double radius) {
    int n = 80;
    double[] lats = new double[n + 1];
    double[] lons = new double[n + 1];
    for (int k = 0; k < n; k++) {
      lats[k] = lat + radius * Math.sin(2 * Math.PI * k / n);
      lons[k] = lon + radius * Math.cos(2 * Math.PI * k / n);
    }
    lats[n] = lats[0];
    lons[n] = lons[0];
    Polygon[] holes = new Polygon[p.numHoles() + 1];
    for (int i = 0; i < p.numHoles(); i++) holes[i] = p.getHoles()[i];
    holes[p.numHoles()] = new Polygon(lats, lons);
    return new Polygon(p.getPolyLats(), p.getPolyLons(), holes);
  }

  static List<Line> lines(Random r, int count) {
    List<Line> out = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      double[] c = center(r);
      int n = 2 + r.nextInt(i % 3 == 0 ? 60 : 8);
      double radius = Math.pow(10, -3 + r.nextDouble() * 4);
      double[] lats = new double[n];
      double[] lons = new double[n];
      for (int k = 0; k < n; k++) {
        lats[k] = clampLat(c[0] + (r.nextDouble() * 2 - 1) * radius);
        lons[k] = clampLon(c[1] + (r.nextDouble() * 2 - 1) * radius);
      }
      out.add(new Line(lats, lons));
    }
    return out;
  }

  // ---------------------------------------------------------------- cartesian

  static float[] toFloats(double[] d, double scale, double shift) {
    float[] f = new float[d.length];
    for (int i = 0; i < d.length; i++) f[i] = (float) (d[i] * scale + shift);
    return f;
  }

  static XYPolygon xyPolygon(Polygon p, double scale, double shift) {
    XYPolygon[] holes = new XYPolygon[p.numHoles()];
    for (int h = 0; h < holes.length; h++) {
      Polygon hole = p.getHoles()[h];
      holes[h] =
          new XYPolygon(
              toFloats(hole.getPolyLons(), scale, shift), toFloats(hole.getPolyLats(), scale, shift));
    }
    return new XYPolygon(
        toFloats(p.getPolyLons(), scale, shift), toFloats(p.getPolyLats(), scale, shift), holes);
  }

  static double randomScale(Random r) {
    return Math.pow(10, -3 + r.nextDouble() * 9);
  }

  // ---------------------------------------------------------------- serialization

  static String d(double v) {
    return Double.toString(v);
  }

  static String f(float v) {
    return Float.toString(v);
  }

  static String ring(double[] a, double[] b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < a.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(d(a[i])).append(' ').append(d(b[i]));
    }
    return sb.toString();
  }

  static String ring(float[] a, float[] b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < a.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(f(a[i])).append(' ').append(f(b[i]));
    }
    return sb.toString();
  }

  static String spec(Polygon p) {
    StringBuilder sb = new StringBuilder("G:").append(ring(p.getPolyLats(), p.getPolyLons()));
    for (Polygon h : p.getHoles()) sb.append('|').append(ring(h.getPolyLats(), h.getPolyLons()));
    return sb.toString();
  }

  static String spec(XYPolygon p) {
    StringBuilder sb = new StringBuilder("G:").append(ring(p.getPolyX(), p.getPolyY()));
    for (XYPolygon h : p.getHoles()) sb.append('|').append(ring(h.getPolyX(), h.getPolyY()));
    return sb.toString();
  }

  static String spec(LatLonGeometry g) {
    if (g instanceof Polygon p) return spec(p);
    if (g instanceof Point p) return "P:" + d(p.getLat()) + "," + d(p.getLon());
    if (g instanceof Line l) return "L:" + ring(l.getLats(), l.getLons());
    if (g instanceof Circle c) return "C:" + d(c.getLat()) + "," + d(c.getLon()) + "," + d(c.getRadius());
    if (g instanceof Rectangle r)
      return "R:" + d(r.minLat) + "," + d(r.maxLat) + "," + d(r.minLon) + "," + d(r.maxLon);
    throw new IllegalArgumentException(g.toString());
  }

  static String spec(XYGeometry g) {
    if (g instanceof XYPolygon p) return spec(p);
    if (g instanceof XYPoint p) return "P:" + f(p.getX()) + "," + f(p.getY());
    if (g instanceof XYLine l) return "L:" + ring(l.getX(), l.getY());
    if (g instanceof XYCircle c) return "C:" + f(c.getX()) + "," + f(c.getY()) + "," + f(c.getRadius());
    if (g instanceof XYRectangle r)
      return "R:" + f(r.minX) + "," + f(r.maxX) + "," + f(r.minY) + "," + f(r.maxY);
    throw new IllegalArgumentException(g.toString());
  }

  static String spec(LatLonGeometry[] gs) {
    StringBuilder sb = new StringBuilder();
    for (LatLonGeometry g : gs) {
      if (sb.length() > 0) sb.append(" + ");
      sb.append(spec(g));
    }
    return sb.toString();
  }

  static String spec(XYGeometry[] gs) {
    StringBuilder sb = new StringBuilder();
    for (XYGeometry g : gs) {
      if (sb.length() > 0) sb.append(" + ");
      sb.append(spec(g));
    }
    return sb.toString();
  }

  /** One line of text: tabs, newlines and backslashes escaped. */
  static String esc(String s) {
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r");
  }
}
