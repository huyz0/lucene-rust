package org.apache.lucene.tests.geo;

import com.carrotsearch.randomizedtesting.*;
import java.util.Locale;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.geo.*;
import org.apache.lucene.search.Query;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.RamUsageEstimator;
import org.junit.Test;
import org.junit.runner.*;
import org.junit.runner.notification.Failure;

/** Actual inherited generators; randomized context construction is outside timing. */
public class BaseXYPointMicro {
  static volatile long sink;
  static long warmup, measure;
  static long step(long d, long value) { return d * 31 + value; }
  static long values(long d, float[] values) {
    d = step(d, values.length);
    for (float value : values) d = step(d, Integer.toUnsignedLong(Float.floatToRawIntBits(value)));
    return d;
  }
  static long polygon(long d, XYPolygon value) {
    if (value.numHoles() != 0) throw new AssertionError("unexpected holes");
    return step(values(values(d, value.getPolyX()), value.getPolyY()), 0);
  }
  static long rectangle(long d, XYRectangle value) {
    return values(d, new float[] {value.minX, value.maxX, value.minY, value.maxY});
  }
  static long geometries(long d, XYGeometry[] values) {
    d = step(d, values.length);
    for (XYGeometry value : values) {
      if (value instanceof XYPoint point) d = values(step(d, 0), new float[] {point.getX(), point.getY()});
      else if (value instanceof XYRectangle box) d = rectangle(step(d, 1), box);
      else if (value instanceof XYPolygon p) d = polygon(step(d, 2), p);
      else throw new AssertionError("unexpected default geometry");
    }
    return d;
  }
  static class Capture extends BaseXYPointTestCase {
    long digest;
    void capture(int callback, float[] xs, float[] ys) {
      if (xs.length != ys.length) throw new AssertionError("coordinate length mismatch");
      digest = values(values(step(digest, callback), xs), ys);
    }
    protected void verifyRandomRectangles(float[] xs, float[] ys) { capture(1, xs, ys); }
    protected void verifyRandomDistances(float[] xs, float[] ys) { capture(2, xs, ys); }
    protected void verifyRandomPolygons(float[] xs, float[] ys) { capture(3, xs, ys); }
    protected void verifyRandomGeometries(float[] xs, float[] ys) { capture(4, xs, ys); }
    void generate(int which) throws Exception {
      switch (which) {
        case 0 -> testRandomTiny(); case 1 -> testRandomMedium();
        case 2 -> testSamePointManyTimes(); case 3 -> testLowCardinality();
        case 4 -> testAllXEqual(); case 5 -> testAllYEqual();
        default -> throw new AssertionError("unknown generator");
      }
    }
    long hooks(long d) {
      for (int at = 0; at < 16; at++) {
        d = step(d, Integer.toUnsignedLong(Float.floatToRawIntBits(nextX())));
        d = step(d, Integer.toUnsignedLong(Float.floatToRawIntBits(nextY())));
        d = rectangle(d, nextBox());
        d = polygon(d, nextPolygon());
        d = geometries(d, nextGeometry());
      }
      return d;
    }
    protected void addPointToDoc(String field, Document doc, float x, float y) { throw new AssertionError("capture must not index"); }
    protected Query newRectQuery(String field, float minX, float maxX, float minY, float maxY) { throw new AssertionError("capture must not query"); }
    protected Query newDistanceQuery(String field, float x, float y, float radius) { throw new AssertionError("capture must not query"); }
    protected Query newPolygonQuery(String field, XYPolygon... polygons) { throw new AssertionError("capture must not query"); }
    protected Query newGeometryQuery(String field, XYGeometry... geometries) { throw new AssertionError("capture must not query"); }
  }
  static long batch(int which, Random random, Capture capture, XYRectangle oracle) throws Exception {
    random.setSeed(42);
    capture.digest = 0;
    long d = 0;
    if (which < 6) { capture.generate(which); d = capture.digest; }
    else if (which == 6) d = capture.hooks(d);
    else if (which == 7) {
      for (int at = 0; at < 1024; at++) {
        double x = (at % 17) / 4.0 - 2, y = (at % 19) / 4.0 - 1;
        d = step(d, BaseXYPointTestCase.rectContainsPoint(oracle, x, y) ? 1 : 0);
        d = step(d, Double.doubleToRawLongBits(BaseXYPointTestCase.cartesianDistance(x, y, (at % 13) / 8.0 - 1, (at % 11) / 8.0 - 1)));
      }
    } else throw new AssertionError("unknown case");
    return step(d, Integer.toUnsignedLong(random.nextInt()));
  }
  static long[] run(int which, Random random, Capture capture, XYRectangle oracle, long millis, long expected) throws Exception {
    long start = System.nanoTime(), count = 0;
    do {
      long actual = batch(which, random, capture, oracle);
      if (actual != expected) throw new AssertionError("complete batch digest changed");
      sink = actual;
      count++;
    } while (System.nanoTime() - start < millis * 1_000_000);
    return new long[] {System.nanoTime() - start, count};
  }
  @RunWith(RandomizedRunner.class) public static class Work {
    @Test public void benchmark() throws Exception {
      if (!BaseXYPointMicro.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
      if (LuceneTestCase.TEST_NIGHTLY || LuceneTestCase.RANDOM_MULTIPLIER != 1) throw new AssertionError("nightly=false, multiplier=1 required");
      Random random = new Random(42);
      Capture capture = new Capture();
      XYRectangle oracle = new XYRectangle(-2, 3, -1, 4);
      XYPolygon example = new XYPolygon(new float[] {0, 3, 3, 1, 0}, new float[] {0, 0, 2, 4, 0});
      System.out.printf("base_xy_capture_owned_bytes\t%d%n", RamUsageEstimator.shallowSizeOf(capture));
      System.out.printf("base_xy_1000_point_arrays_owned_bytes\t%d%n", 2 * RamUsageEstimator.sizeOf(new float[1000]));
      System.out.printf("base_xy_polygon_owned_bytes\t%d%n", RamUsageEstimator.shallowSizeOf(example) + RamUsageEstimator.sizeOf(example.getPolyX()) + RamUsageEstimator.sizeOf(example.getPolyY()) + RamUsageEstimator.shallowSizeOf(example.getHoles()));
      RandomizedContext.current().runWithPrivateRandomness(new Randomness(42, seed -> random), () -> {
        // RandomizedContext may wrap the factory result in an asserting Random.
        // Prove stream sharing rather than object identity, outside timing.
        random.setSeed(42);
        Random control = new Random(42);
        if (RandomizedContext.current().getRandom().nextInt() != control.nextInt()
            || random.nextInt() != control.nextInt()) {
          throw new AssertionError("randomized context does not share the supplied RNG stream");
        }
        random.setSeed(42);
        String[] names = {"base_xy_random_tiny", "base_xy_random_medium", "base_xy_same_point", "base_xy_low_cardinality", "base_xy_all_x", "base_xy_all_y", "base_xy_shape_hooks", "base_xy_oracles"};
        for (int which = 0; which < names.length; which++) {
          long expected = batch(which, random, capture, oracle);
          System.out.printf("#check\t%s\t%s\t1%n", names[which], Long.toUnsignedString(expected));
          run(which, random, capture, oracle, warmup, expected);
          long[] result = run(which, random, capture, oracle, measure, expected);
          System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", names[which], (double) result[0] / result[1], result[1]);
        }
        return null;
      });
    }
  }
  public static void main(String[] args) {
    warmup = Long.parseLong(args[0]); measure = Long.parseLong(args[1]);
    Result result = JUnitCore.runClasses(Work.class);
    for (Failure failure : result.getFailures()) System.err.println(failure.getTrace());
    if (!result.wasSuccessful() || result.getRunCount() != 1) throw new AssertionError("BaseXY benchmark context failed");
  }
}
