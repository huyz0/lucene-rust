import java.util.Locale;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.tests.geo.GeoTestUtil;
import org.apache.lucene.util.RamUsageEstimator;

/** Actual GeoTestUtil helper; polygons and query arrays are built before timing. */
public final class GeoContainsSlowlyMicro {
  static volatile long sink;
  static final int QUERY_COUNT = 128;
  // Longitude, latitude. The concave L has perimeter 1024 on the integer grid.
  static final int[][] CORNERS = {
    {-176, -80}, {176, -80}, {176, -16}, {-48, -16}, {-48, 80}, {-176, 80}, {-176, -80}
  };
  record Input(String name, Polygon polygon, double[] latitudes, double[] longitudes, long expected) {}

  static Polygon polygon(boolean subdivided) {
    int count = subdivided ? 1025 : CORNERS.length;
    double[] latitudes = new double[count], longitudes = new double[count];
    int at = 0;
    for (int edge = 0; edge < CORNERS.length - 1; edge++) {
      int[] from = CORNERS[edge], to = CORNERS[edge + 1];
      int dx = Integer.signum(to[0] - from[0]), dy = Integer.signum(to[1] - from[1]);
      int length = Math.abs(to[0] - from[0]) + Math.abs(to[1] - from[1]);
      int points = subdivided ? length : 1;
      for (int step = 0; step < points; step++) {
        longitudes[at] = from[0] + step * dx;
        latitudes[at++] = from[1] + step * dy;
      }
    }
    longitudes[at] = CORNERS[0][0]; latitudes[at++] = CORNERS[0][1];
    if (at != count) throw new AssertionError("ring size");
    return new Polygon(latitudes, longitudes);
  }

  // Independent analytic membership for this axis-aligned L, including its boundary.
  static boolean expected(double latitude, double longitude) {
    return latitude >= -80 && latitude <= 80 && longitude >= -176 && longitude <= 176
        && (latitude <= -16 || longitude <= -48);
  }

  static Input input(String name, Polygon polygon, int kind) {
    double[] latitudes = new double[QUERY_COUNT], longitudes = new double[QUERY_COUNT];
    long digest = 0;
    for (int at = 0; at < QUERY_COUNT; at++) {
      double lat, lon;
      double jitter = (at % 8) * 0.25;
      if (kind == 0) {
        switch (at % 4) {
          case 0 -> { lat = -40 + jitter; lon = -20 + jitter; }
          case 1 -> { lat = 20 + jitter; lon = -100 + jitter; }
          case 2 -> { lat = 20 + jitter; lon = 20 + jitter; }
          default -> { lat = 60 + jitter; lon = 100 + jitter; }
        }
      } else if (kind == 1) {
        lat = switch (at % 4) { case 0 -> -81; case 1 -> 81; default -> 0; };
        lon = switch (at % 4) { case 2 -> -177; case 3 -> 177; default -> 0; };
      } else {
        // All points are strictly inside the first unit-long horizontal edge.
        lat = -80; lon = -176 + (at % 3 + 1) * 0.25;
      }
      latitudes[at] = lat; longitudes[at] = lon;
      digest = digest * 31 + (expected(lat, lon) ? 1 : 0);
    }
    return new Input(name, polygon, latitudes, longitudes, digest);
  }

  static long batch(Input input) {
    long digest = 0;
    for (int at = 0; at < input.latitudes.length; at++) {
      digest = digest * 31 + (GeoTestUtil.containsSlowly(input.polygon,
          input.latitudes[at], input.longitudes[at]) ? 1 : 0);
    }
    return digest;
  }

  static long[] run(Input input, long millis) {
    long started = System.nanoTime(), count = 0;
    do {
      long actual = batch(input);
      if (actual != input.expected) throw new AssertionError(input.name + " analytic digest differs");
      sink = actual;
      count++;
    } while (System.nanoTime() - started < millis * 1_000_000L);
    return new long[] {System.nanoTime() - started, count};
  }

  static void memory(String size, Polygon polygon) {
    // Getter clones have the same shallow array sizes as the retained originals.
    // No holes: shared winding enum, query arrays, allocator/GC costs are excluded.
    double[] lats = polygon.getPolyLats(), lons = polygon.getPolyLons();
    Polygon[] holes = polygon.getHoles();
    long coordinates = RamUsageEstimator.sizeOf(lats) + RamUsageEstimator.sizeOf(lons);
    long payload = 2L * polygon.numPoints() * Double.BYTES;
    System.out.printf("geo_contains_%s_polygon_coordinate_payload_bytes\t%d%n", size, payload);
    System.out.printf("geo_contains_%s_polygon_retained_estimated_bytes\t%d%n", size,
        RamUsageEstimator.shallowSizeOf(polygon) + coordinates + RamUsageEstimator.shallowSizeOf(holes));
    System.out.printf("geo_contains_%s_per_call_temporary_coordinate_payload_bytes\t%d%n", size, payload);
    System.out.printf("geo_contains_%s_per_call_temporary_arrays_estimated_bytes\t%d%n", size,
        coordinates + RamUsageEstimator.shallowSizeOf(holes));
  }

  public static void main(String[] args) {
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid durations");
    Polygon small = polygon(false), large = polygon(true);
    System.out.println("#memory_scope\tretained polygon plus owned arrays; shared enum/query arrays excluded");
    System.out.println("#memory_scope\ttemporary arrays are source-level clone estimates before JIT elimination; not measured allocation/peak RSS");
    memory("small", small); memory("large", large);
    Input[] inputs = {
      input("geo_contains_small_full_scan", small, 0),
      input("geo_contains_large_full_scan", large, 0),
      input("geo_contains_large_bbox_rejection", large, 1),
      input("geo_contains_large_boundary_early_exit", large, 2)
    };
    for (Input input : inputs) {
      System.out.printf("#check\t%s\t%s\t%d\t%d%n", input.name,
          Long.toUnsignedString(input.expected), QUERY_COUNT, input.polygon.numPoints());
      run(input, warmup);
      long[] timing = run(input, measure);
      System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", input.name,
          (double) timing[0] / timing[1], timing[1]);
    }
  }
}
