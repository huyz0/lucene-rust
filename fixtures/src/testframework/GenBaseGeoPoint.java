import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.carrotsearch.randomizedtesting.Randomness;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.StringJoiner;
import org.apache.lucene.document.Document;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.search.Query;
import org.apache.lucene.tests.geo.BaseGeoPointTestCase;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;

/**
 * The points BaseGeoPointTestCase's random test methods generate, and its nextGeometry hook, each
 * seeded with a java.util.Random installed as the test's random(). A capture subclass overrides
 * the four protected verify methods, so nothing is indexed and no further draw is made; quantizing
 * is the default identity.
 *
 * <pre>
 * base_geo_point_points.tsv:   case  seed  length  verifies  Arrays.hashCode(lats)  (lons)  nextInt
 * base_geo_point_geometry.tsv: seed  at  geometries  nextInt
 * </pre>
 *
 * Geometries: {@code box:minLat,maxLat,minLon,maxLon}, {@code circle:lat,lon,radius} or {@code
 * poly:lats;lons}, as raw double bits, joined by {@code |}.
 */
public class GenBaseGeoPoint {
  static final StringBuilder POINTS = new StringBuilder();
  static final StringBuilder GEOMETRY = new StringBuilder();
  static final String[] CASES = {
    "random-tiny", "random-medium", "same-point", "low-cardinality", "all-lat", "all-lon"
  };

  static String bits(double... values) {
    StringJoiner out = new StringJoiner(",");
    for (double v : values) out.add(Long.toUnsignedString(Double.doubleToRawLongBits(v)));
    return out.toString();
  }

  static String geometry(LatLonGeometry g) {
    if (g instanceof Rectangle r) return "box:" + bits(r.minLat, r.maxLat, r.minLon, r.maxLon);
    if (g instanceof Circle c) return "circle:" + bits(c.getLat(), c.getLon(), c.getRadius());
    Polygon p = (Polygon) g;
    if (p.numHoles() != 0) throw new AssertionError("unexpected generated holes");
    return "poly:" + bits(p.getPolyLats()) + ";" + bits(p.getPolyLons());
  }

  public static class Capture extends BaseGeoPointTestCase {
    double[] lats;
    double[] lons;
    final StringJoiner visits = new StringJoiner(",");

    void capture(String name, double[] la, double[] lo) {
      if (lats == null) {
        lats = la.clone();
        lons = lo.clone();
      } else if (!Arrays.equals(lats, la) || !Arrays.equals(lons, lo)) {
        throw new AssertionError("verify hooks received different arrays");
      }
      visits.add(name);
    }

    @Override
    protected void verifyRandomRectangles(double[] la, double[] lo) {
      capture("rectangles", la, lo);
    }

    @Override
    protected void verifyRandomDistances(double[] la, double[] lo) {
      capture("distances", la, lo);
    }

    @Override
    protected void verifyRandomPolygons(double[] la, double[] lo) {
      capture("polygons", la, lo);
    }

    @Override
    protected void verifyRandomGeometries(double[] la, double[] lo) {
      capture("geometries", la, lo);
    }

    void generate(String name) throws Exception {
      switch (name) {
        case "random-tiny" -> testRandomTiny();
        case "random-medium" -> testRandomMedium();
        case "same-point" -> testSamePointManyTimes();
        case "low-cardinality" -> testLowCardinality();
        case "all-lat" -> testAllLatEqual();
        case "all-lon" -> testAllLonEqual();
        default -> throw new AssertionError(name);
      }
    }

    String nextGeometries() {
      StringJoiner out = new StringJoiner("|");
      for (LatLonGeometry g : nextGeometry()) out.add(geometry(g));
      return out.toString();
    }

    @Override
    protected void addPointToDoc(String field, Document doc, double lat, double lon) {
      throw new AssertionError("capture fixture must not index");
    }

    @Override
    protected Query newRectQuery(
        String field, double minLat, double maxLat, double minLon, double maxLon) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newDistanceQuery(
        String field, double centerLat, double centerLon, double radiusMeters) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newPolygonQuery(String field, Polygon... polygon) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newGeometryQuery(String field, LatLonGeometry... geometry) {
      throw new AssertionError("capture fixture must not query");
    }
  }

  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test
    public void dump() throws Exception {
      if (LuceneTestCase.RANDOM_MULTIPLIER != 1) throw new AssertionError("multiplier must be one");
      for (long seed = 0; seed < 20; seed++) {
        final long s = seed;
        for (String name : CASES) {
          RandomizedContext.current()
              .runWithPrivateRandomness(
                  new Randomness(seed, Random::new),
                  () -> {
                    Capture c = new Capture();
                    c.generate(name);
                    POINTS.append(name).append('\t').append(s).append('\t')
                        .append(c.lats.length).append('\t').append(c.visits).append('\t')
                        .append(Arrays.hashCode(c.lats)).append('\t')
                        .append(Arrays.hashCode(c.lons)).append('\t')
                        .append(RandomizedContext.current().getRandom().nextInt()).append('\n');
                    return null;
                  });
        }
        RandomizedContext.current()
            .runWithPrivateRandomness(
                new Randomness(seed, Random::new),
                () -> {
                  Capture c = new Capture();
                  for (int at = 0; at < 8; at++) {
                    GEOMETRY.append(s).append('\t').append(at).append('\t')
                        .append(c.nextGeometries()).append('\t')
                        .append(RandomizedContext.current().getRandom().nextInt()).append('\n');
                  }
                  return null;
                });
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Result result = JUnitCore.runClasses(Dump.class);
    for (Failure f : result.getFailures()) System.err.println(f.getTrace());
    if (!result.wasSuccessful() || result.getRunCount() != 1) {
      throw new AssertionError("BaseGeoPoint fixture failed");
    }
    Files.writeString(out.resolve("base_geo_point_points.tsv"), POINTS);
    Files.writeString(out.resolve("base_geo_point_geometry.tsv"), GEOMETRY);
  }
}
