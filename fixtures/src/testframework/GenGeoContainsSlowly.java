import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.tests.geo.GeoTestUtil;

/** Actual GeoTestUtil.containsSlowly, with raw-double inputs and exact exception messages. */
public class GenGeoContainsSlowly {
  private static final StringBuilder OUT = new StringBuilder();
  private static final List<Polygon> POLYGONS = new ArrayList<>();
  private static int queries;

  private static String bits(double value) {
    return Long.toUnsignedString(Double.doubleToRawLongBits(value));
  }

  private static String bits(double[] values) {
    StringJoiner result = new StringJoiner(",");
    for (double value : values) result.add(bits(value));
    return result.toString();
  }

  private static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i != 0) OUT.append('\t');
      OUT.append(fields[i]);
    }
    OUT.append('\n');
  }

  private static int polygon(String label, double[] lats, double[] lons, int... holes) {
    Polygon[] inner = new Polygon[holes.length];
    StringJoiner holeIds = new StringJoiner(",");
    for (int i = 0; i < holes.length; i++) {
      inner[i] = POLYGONS.get(holes[i]);
      holeIds.add(Integer.toString(holes[i]));
    }
    Polygon polygon = new Polygon(lats, lons, inner);
    int id = POLYGONS.size();
    POLYGONS.add(polygon);
    row("P", id, label, bits(lats), bits(lons), holes.length == 0 ? "-" : holeIds);
    return id;
  }

  private static int reversed(String label, int source) {
    double[] lats = POLYGONS.get(source).getPolyLats();
    double[] lons = POLYGONS.get(source).getPolyLons();
    for (int i = 0, j = lats.length - 1; i < j; i++, j--) {
      double lat = lats[i]; lats[i] = lats[j]; lats[j] = lat;
      double lon = lons[i]; lons[i] = lons[j]; lons[j] = lon;
    }
    return polygon(label, lats, lons);
  }

  private static void query(int id, String label, double lat, double lon) {
    String expected;
    try {
      expected = Boolean.toString(GeoTestUtil.containsSlowly(POLYGONS.get(id), lat, lon));
    } catch (UnsupportedOperationException error) {
      expected = "error:" + error.getMessage();
    }
    row("Q", id, label, bits(lat), bits(lon), expected);
    queries++;
  }

  private static void grid(int id, String label, double[] lats, double[] lons) {
    for (int i = 0; i < lats.length; i++) {
      for (int j = 0; j < lons.length; j++) query(id, label + "-" + i + "-" + j, lats[i], lons[j]);
    }
  }

  private static void edgeNeighbors(int id) {
    Polygon polygon = POLYGONS.get(id);
    double[] lats = polygon.getPolyLats();
    double[] lons = polygon.getPolyLons();
    for (int i = 0; i < lats.length - 1; i++) {
      // Include vertices, edge midpoints and each coordinate's one-ulp neighbours.
      for (int part = 0; part < 2; part++) {
        double lat = part == 0 ? lats[i] : lats[i] + (lats[i + 1] - lats[i]) * 0.5;
        double lon = part == 0 ? lons[i] : lons[i] + (lons[i + 1] - lons[i]) * 0.5;
        grid(id, "edge-" + i + "-" + part,
            new double[] {Math.nextDown(lat), lat, Math.nextUp(lat)},
            new double[] {Math.nextDown(lon), lon, Math.nextUp(lon)});
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    row("# P: id, label, latitude raw bits, longitude raw bits, preceding hole IDs or -");
    row("# Q: polygon ID, label, latitude raw bits, longitude raw bits, actual Java result");
    int square = polygon("square", new double[] {0, 4, 4, 0, 0}, new double[] {0, 0, 4, 4, 0});
    reversed("square-reversed", square);
    int concave = polygon("concave", new double[] {0, 6, 6, 2, 2, 0, 0},
        new double[] {0, 0, 2, 2, 6, 6, 0});
    reversed("concave-reversed", concave);
    int triangle = polygon("triangle", new double[] {-2, 3, 0, -2}, new double[] {0, 0, 5, 0});
    reversed("triangle-reversed", triangle);
    polygon("repeated-collinear", new double[] {0, 0, 2, 4, 4, 4, 0, 0},
        new double[] {0, 0, 0, 0, 2, 4, 4, 0});
    polygon("signed-zero", new double[] {-0.0, 1, 1, 0.0, -0.0},
        new double[] {0.0, -0.0, 1, 1, 0.0});
    polygon("dateline-wide-bbox", new double[] {-5, 5, 5, -5, -5},
        new double[] {-179, -179, 179, 179, -179});
    polygon("geographic-extrema", new double[] {-90, 90, 90, -90, -90},
        new double[] {-180, -180, 180, 180, -180});
    polygon("ulp-thin", new double[] {1, Math.nextUp(1.0), Math.nextUp(1.0), 1, 1},
        new double[] {2, 2, Math.nextUp(2.0), Math.nextUp(2.0), 2});
    polygon("precision-diagonal", new double[] {45, Math.nextUp(45.0), 46, 45},
        new double[] {90, Math.nextUp(90.0), 89, 90});
    polygon("subnormal", new double[] {0, Double.MIN_VALUE * 8, Double.MIN_VALUE * 8, 0, 0},
        new double[] {0, 0, Double.MIN_VALUE * 8, Double.MIN_VALUE * 8, 0});
    polygon("degenerate-line", new double[] {0, 1, 2, 0}, new double[] {0, 1, 2, 0});
    int hole = polygon("hole", new double[] {1, 2, 2, 1, 1}, new double[] {1, 1, 2, 2, 1});
    int holed = polygon("square-with-hole", POLYGONS.get(square).getPolyLats(),
        POLYGONS.get(square).getPolyLons(), hole);

    // Random points are serialized: replay never depends on a Rust RNG implementation.
    Random random = new Random(0x67454f434f4e5441L);
    for (int id = 0; id < POLYGONS.size(); id++) {
      edgeNeighbors(id);
      grid(id, "special", new double[] {-91, -90, -2, -0.0, 0.0, 1, 2, 4, 90, 91,
          Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY},
          new double[] {-181, -180, -2, -0.0, 0.0, 1, 2, 4, 180, 181,
              Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY});
      Polygon p = POLYGONS.get(id);
      for (int i = 0; i < 128; i++) {
        double lat = p.minLat + (p.maxLat - p.minLat) * (random.nextDouble() * 1.5 - 0.25);
        double lon = p.minLon + (p.maxLon - p.minLon) * (random.nextDouble() * 1.5 - 0.25);
        query(id, "seeded-bbox-" + i, lat, lon);
        query(id, "seeded-world-" + i, random.nextDouble() * 180 - 90,
            random.nextDouble() * 360 - 180);
      }
    }
    // Explicitly name exception precedence, including inputs that ordinary bbox checks reject.
    query(holed, "holes-before-outside-bbox", 89, 179);
    query(holed, "holes-before-nan", Double.NaN, Double.NaN);
    query(holed, "holes-before-infinity", Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
    Files.writeString(out.resolve("geo_contains_slowly.tsv"), OUT);
    System.out.println("GenGeoContainsSlowly: " + POLYGONS.size() + " polygons, " + queries + " queries");
  }
}
