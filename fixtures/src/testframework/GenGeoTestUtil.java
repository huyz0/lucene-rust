import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.carrotsearch.randomizedtesting.Randomness;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.StringJoiner;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.tests.geo.GeoTestUtil;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;

/**
 * Actual GeoTestUtil generators (polygons, lines, points and boxes near polygons and boxes,
 * regular polygons), each seeded with a java.util.Random installed as the test's random(); every
 * row records the result's raw double bits and the stream's next int.
 *
 * <pre>
 * geo_test_util_generators.tsv: seed  op  at  result  nextInt
 * geo_test_util_regular.tsv:    case  gons  result
 * </pre>
 *
 * Results: {@code poly:lats;lons[|hole lats;lons...]}, {@code line:lats;lons}, {@code
 * point:lat,lon}, {@code box:minLat,maxLat,minLon,maxLon}, {@code circle:lat,lon,radius}, or
 * {@code error:message} for an IllegalArgumentException Java lets escape.
 */
public class GenGeoTestUtil {
  static final StringBuilder OUT = new StringBuilder();
  static final int OPS = 8;

  static String bits(double... values) {
    StringJoiner out = new StringJoiner(",");
    for (double v : values) out.add(Long.toUnsignedString(Double.doubleToRawLongBits(v)));
    return out.toString();
  }

  static String ring(Polygon p) {
    return bits(p.getPolyLats()) + ";" + bits(p.getPolyLons());
  }

  static String poly(Polygon p) {
    StringBuilder sb = new StringBuilder("poly:").append(ring(p));
    for (Polygon h : p.getHoles()) sb.append('|').append(ring(h));
    return sb.toString();
  }

  static String box(Rectangle r) {
    return "box:" + bits(r.minLat, r.maxLat, r.minLon, r.maxLon);
  }

  /** A polygon with a hole, so the hole-targeting branches run. */
  static final Polygon HOLED =
      new Polygon(
          new double[] {-10, -10, 10, 10, -10},
          new double[] {-20, 20, 20, -20, -20},
          new Polygon(new double[] {-2, 2, 2, -2, -2}, new double[] {-3, -3, 3, 3, -3}));

  static String result(int op) {
    try {
      return switch (op) {
        case 0 -> poly(GeoTestUtil.nextPolygon());
        case 1 -> {
          Line l = GeoTestUtil.nextLine();
          yield "line:" + bits(l.getLats()) + ";" + bits(l.getLons());
        }
        case 2 -> "point:" + bits(GeoTestUtil.nextPointNear(GeoTestUtil.nextPolygon()));
        case 3 -> box(GeoTestUtil.nextBoxNear(GeoTestUtil.nextPolygon()));
        case 4 -> "point:" + bits(GeoTestUtil.nextPointNear(GeoTestUtil.nextBox()));
        case 5 -> "point:" + bits(GeoTestUtil.nextPointNear(HOLED));
        case 6 -> box(GeoTestUtil.nextBoxNear(HOLED));
        case 7 -> {
          var p = GeoTestUtil.nextPoint();
          var c = GeoTestUtil.nextCircle();
          yield "circle:" + bits(p.getLat(), p.getLon(), c.getLat(), c.getLon(), c.getRadius());
        }
        default -> throw new AssertionError();
      };
    } catch (IllegalArgumentException e) {
      return "error:" + e.getMessage();
    }
  }

  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test
    public void dump() throws Exception {
      for (long seed = 0; seed < 100; seed++) {
        for (int op = 0; op < OPS; op++) {
          final long s = seed;
          final int operation = op;
          RandomizedContext.current()
              .runWithPrivateRandomness(
                  new Randomness(seed, Random::new),
                  () -> {
                    Random r = RandomizedContext.current().getRandom();
                    for (int at = 0; at < 4; at++) {
                      OUT.append(s)
                          .append('\t')
                          .append(operation)
                          .append('\t')
                          .append(at)
                          .append('\t')
                          .append(result(operation))
                          .append('\t')
                          .append(r.nextInt())
                          .append('\n');
                    }
                    return null;
                  });
        }
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Result result = JUnitCore.runClasses(Dump.class);
    for (Failure f : result.getFailures()) System.err.println(f.getTrace());
    if (!result.wasSuccessful() || result.getRunCount() != 1) {
      throw new AssertionError("GeoTestUtil fixture failed");
    }
    Files.writeString(out.resolve("geo_test_util_generators.tsv"), OUT);
    StringBuilder text = new StringBuilder();
    double[][] centers = {
      {0, 0, 1000}, {45, 90, 50_000}, {-60, -120, 2_000_000}, {89, 0, 500_000}, {0, 179.9, 50_000},
      {10, 10, 0.05}, {33.3, -44.4, 7_654_321}
    };
    for (int c = 0; c < centers.length; c++) {
      for (int gons : new int[] {3, 4, 7, 64}) {
        String value;
        try {
          value =
              poly(GeoTestUtil.createRegularPolygon(centers[c][0], centers[c][1], centers[c][2], gons));
        } catch (IllegalArgumentException e) {
          value = "error:" + e.getMessage();
        }
        text.append(c).append('\t').append(gons).append('\t').append(value).append('\n');
      }
    }
    Files.writeString(out.resolve("geo_test_util_regular.tsv"), text);
  }
}
