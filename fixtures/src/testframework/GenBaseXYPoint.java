import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.carrotsearch.randomizedtesting.Randomness;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.StringJoiner;
import org.apache.lucene.document.Document;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.geo.XYPoint;
import org.apache.lucene.geo.XYPolygon;
import org.apache.lucene.geo.XYRectangle;
import org.apache.lucene.search.Query;
import org.apache.lucene.tests.geo.BaseXYPointTestCase;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;

/** Actual inherited BaseXY generators, default hooks and package-private oracle helpers. */
public class GenBaseXYPoint {
  static final StringBuilder GENERATORS = new StringBuilder();
  static final StringBuilder HOOKS = new StringBuilder();
  static final StringBuilder ORACLES = new StringBuilder();
  static final String[] CASES = {
    "random-tiny", "random-medium", "same-point", "low-cardinality", "all-x", "all-y"
  };
  static final long[] SEEDS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    42, -1, Long.MIN_VALUE, Long.MAX_VALUE
  };

  static String bits(float... values) {
    StringJoiner text = new StringJoiner(",");
    for (float value : values) {
      text.add(Integer.toUnsignedString(Float.floatToRawIntBits(value)));
    }
    return text.toString();
  }

  static String doubleBits(double value) {
    return Long.toUnsignedString(Double.doubleToRawLongBits(value));
  }

  static String polygon(XYPolygon polygon) {
    if (polygon.numHoles() != 0) throw new AssertionError("unexpected generated holes");
    return "polygon:" + bits(polygon.getPolyX()) + ";" + bits(polygon.getPolyY());
  }

  static String rectangle(XYRectangle rectangle) {
    return "rectangle:" + bits(rectangle.minX, rectangle.maxX, rectangle.minY, rectangle.maxY);
  }

  static String geometries(XYGeometry[] geometries) {
    StringJoiner text = new StringJoiner("|");
    for (XYGeometry geometry : geometries) {
      if (geometry instanceof XYPoint point) {
        text.add("point:" + bits(point.getX(), point.getY()));
      } else if (geometry instanceof XYRectangle rectangle) {
        text.add(rectangle(rectangle));
      } else if (geometry instanceof XYPolygon polygon) {
        text.add(polygon(polygon));
      } else {
        throw new AssertionError("unexpected default nextGeometry type: " + geometry.getClass());
      }
    }
    return text.toString();
  }

  /**
   * The actual private verify method dispatches to these four protected hooks.
   * Only capture them: no writer, query or searcher is constructed, and no RNG
   * draws are added. Inherited generator quirks, including allYEqual's copy
   * branch leaving X at its default zero, therefore remain Java's own behavior.
   */
  public static class Capture extends BaseXYPointTestCase {
    float[] xs;
    float[] ys;
    final StringJoiner visits = new StringJoiner(",");

    void capture(String name, float[] x, float[] y) {
      if (x.length != y.length) throw new AssertionError("coordinate length mismatch");
      if (xs == null) {
        xs = x.clone();
        ys = y.clone();
      } else if (!Arrays.equals(xs, x) || !Arrays.equals(ys, y)) {
        throw new AssertionError("verify hooks received different arrays");
      }
      visits.add(name);
    }

    @Override
    protected void verifyRandomRectangles(float[] xs, float[] ys) {
      capture("rectangles", xs, ys);
    }

    @Override
    protected void verifyRandomDistances(float[] xs, float[] ys) {
      capture("distances", xs, ys);
    }

    @Override
    protected void verifyRandomPolygons(float[] xs, float[] ys) {
      capture("polygons", xs, ys);
    }

    @Override
    protected void verifyRandomGeometries(float[] xs, float[] ys) {
      capture("geometries", xs, ys);
    }

    void generate(String name) throws Exception {
      switch (name) {
        case "random-tiny" -> testRandomTiny();
        case "random-medium" -> testRandomMedium();
        case "same-point" -> testSamePointManyTimes();
        case "low-cardinality" -> testLowCardinality();
        case "all-x" -> testAllXEqual();
        case "all-y" -> testAllYEqual();
        default -> throw new AssertionError(name);
      }
      if (!visits.toString().equals("rectangles,distances,polygons,geometries")) {
        throw new AssertionError("unexpected verify dispatch: " + visits);
      }
    }

    String hook(String name) {
      return switch (name) {
        case "x" -> bits(nextX());
        case "y" -> bits(nextY());
        case "box" -> rectangle(nextBox());
        case "polygon" -> polygon(nextPolygon());
        case "geometry" -> geometries(nextGeometry());
        default -> throw new AssertionError(name);
      };
    }

    @Override
    protected void addPointToDoc(String field, Document document, float x, float y) {
      throw new AssertionError("capture fixture must not index");
    }

    @Override
    protected Query newRectQuery(String field, float minX, float maxX, float minY, float maxY) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newDistanceQuery(String field, float centerX, float centerY, float radius) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newPolygonQuery(String field, XYPolygon... polygons) {
      throw new AssertionError("capture fixture must not query");
    }

    @Override
    protected Query newGeometryQuery(String field, XYGeometry... geometries) {
      throw new AssertionError("capture fixture must not query");
    }
  }

  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test
    public void dump() throws Exception {
      if (LuceneTestCase.RANDOM_MULTIPLIER != 1) throw new AssertionError("multiplier must be one");
      for (long seed : SEEDS) {
        for (String name : CASES) {
          RandomizedContext.current().runWithPrivateRandomness(
              new Randomness(seed, Random::new),
              () -> {
                Capture capture = new Capture();
                capture.generate(name);
                GENERATORS.append("case\t").append(name).append('\t').append(seed)
                    .append('\t').append(capture.xs.length).append('\t').append(capture.visits)
                    .append('\t').append(RandomizedContext.current().getRandom().nextInt())
                    .append('\n');
                GENERATORS.append("x\t").append(bits(capture.xs)).append('\n');
                GENERATORS.append("y\t").append(bits(capture.ys)).append('\n');
                return null;
              });
        }
        for (String name : new String[] {"x", "y", "box", "polygon", "geometry"}) {
          RandomizedContext.current().runWithPrivateRandomness(
              new Randomness(seed, Random::new),
              () -> {
                Capture capture = new Capture();
                for (int at = 0; at < 16; at++) {
                  HOOKS.append(name).append('\t').append(seed).append('\t').append(at)
                      .append('\t').append(capture.hook(name)).append('\t')
                      .append(RandomizedContext.current().getRandom().nextInt()).append('\n');
                }
                return null;
              });
        }
      }
      oracleCases();
    }
  }

  static void oracleCases() throws Exception {
    // The helpers are package-private. Invoke the actual source methods rather
    // than restating the oracle arithmetic in this default-package generator.
    Method rectangle = BaseXYPointTestCase.class.getDeclaredMethod(
        "rectContainsPoint", XYRectangle.class, double.class, double.class);
    Method distance = BaseXYPointTestCase.class.getDeclaredMethod(
        "cartesianDistance", double.class, double.class, double.class, double.class);
    rectangle.setAccessible(true);
    distance.setAccessible(true);
    XYRectangle[] boxes = {
      new XYRectangle(-2, 3, -1, 4),
      new XYRectangle(-Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, Float.MAX_VALUE),
      new XYRectangle(-0.0f, 0.0f, -0.0f, 0.0f)
    };
    double[] coordinates = {
      Double.NEGATIVE_INFINITY, -Float.MAX_VALUE, -3, -2, -1, -Float.MIN_VALUE,
      -0.0, 0.0, Float.MIN_VALUE, 1, 3, 4, 5, Float.MAX_VALUE,
      Double.POSITIVE_INFINITY, Double.NaN
    };
    for (int box = 0; box < boxes.length; box++) {
      for (double x : coordinates) {
        for (double y : coordinates) {
          ORACLES.append("rectangle\t").append(box).append('\t').append(doubleBits(x))
              .append('\t').append(doubleBits(y)).append('\t')
              .append(rectangle.invoke(null, boxes[box], x, y)).append('\n');
        }
      }
    }
    double[][] points = {
      {0.0, 0.0}, {-0.0, -0.0}, {3, 4}, {-2, 7},
      {Float.MIN_VALUE, -Float.MIN_VALUE}, {Float.MAX_VALUE, -Float.MAX_VALUE},
      {-Float.MAX_VALUE, Float.MAX_VALUE}, {Double.MAX_VALUE, -Double.MAX_VALUE},
      {Double.POSITIVE_INFINITY, 0}, {Double.NEGATIVE_INFINITY, 0},
      {0, Double.POSITIVE_INFINITY}, {Double.NaN, 1}, {1, Double.NaN}
    };
    for (double[] a : points) {
      for (double[] b : points) {
        double value = (double) distance.invoke(null, a[0], a[1], b[0], b[1]);
        ORACLES.append("distance\t").append(doubleBits(a[0])).append('\t')
            .append(doubleBits(a[1])).append('\t').append(doubleBits(b[0])).append('\t')
            .append(doubleBits(b[1])).append('\t').append(doubleBits(value)).append('\n');
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    if (args.length == 3 && args[1].equals("--worker")) {
      boolean nightly = Boolean.parseBoolean(args[2]);
      Result result = JUnitCore.runClasses(Dump.class);
      for (Failure failure : result.getFailures()) System.err.println(failure.getTrace());
      if (!result.wasSuccessful() || result.getRunCount() != 1) {
        throw new AssertionError("BaseXY context fixture failed");
      }
      if (LuceneTestCase.TEST_NIGHTLY != nightly) throw new AssertionError("wrong nightly JVM");
      Files.writeString(out.resolve("base_xy_point_generators_" + nightly + ".tsv"), GENERATORS);
      Files.writeString(out.resolve("base_xy_point_hooks_" + nightly + ".tsv"), HOOKS);
      if (!nightly) Files.writeString(out.resolve("base_xy_point_oracles.tsv"), ORACLES);
      return;
    }
    for (boolean nightly : new boolean[] {false, true}) {
      Process process = new ProcessBuilder(
          Path.of(System.getProperty("java.home"), "bin", "java").toString(),
          "-ea", "-Dtests.seed=0", "-Dtests.multiplier=1", "-Dtests.nightly=" + nightly,
          "-cp", System.getProperty("java.class.path"), "GenBaseXYPoint", out.toString(),
          "--worker", Boolean.toString(nightly)).inheritIO().start();
      if (process.waitFor() != 0) throw new AssertionError("BaseXY worker failed");
    }
  }
}
