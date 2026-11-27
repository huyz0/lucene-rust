import java.util.Arrays;
import java.util.Random;
import org.opensearch.search.aggregations.pipeline.MovingFunctions;

/**
 * Prints what OpenSearch 3.10.0's {@code MovingFunctions} returns (or throws) for fixed inputs, one line per
 * call: {@code name args -> result}. Run with the OpenSearch jars on the class path:
 * {@code java -cp "$CP" GenMovingFunctions.java > fixtures/data/painless-moving/java-moving-functions.txt}.
 */
public final class GenMovingFunctions {
    static String d(double v) { return Double.toString(v); }

    static String arr(double[] v) { return Arrays.toString(v); }

    static void line(String name, String args, java.util.function.Supplier<Object> f) {
        String result;
        try {
            Object o = f.get();
            result = o instanceof Double x ? d(x) : String.valueOf(o);
        } catch (Exception e) {
            result = "!" + e.getClass().getName() + ": " + e.getMessage();
        }
        System.out.println(name + " " + args + " -> " + result);
    }

    public static void main(String[] args) {
        double nan = Double.NaN, inf = Double.POSITIVE_INFINITY;
        double[][] inputs = {
            {}, {nan}, {1}, {1, 2, 3}, {3, 1, 2}, {1, nan, 3}, {nan, nan}, {-0.0, 0.0}, {0.0, -0.0},
            {inf, 1}, {-inf, 1}, {inf, -inf}, {1e308, 1e308}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {5, 0, 5, 0, 5, 0, 5, 0},
            {1.5, 2.5, nan, 4.5, 5.5, 6.5, nan, 8.5}, {0, 0, 0, 0}, {10, 20, 30, 40, 10, 20, 30, 40},
        };
        Random r = new Random(7);
        double[][] more = new double[12][];
        for (int i = 0; i < more.length; i++) {
            more[i] = new double[1 + r.nextInt(14)];
            for (int j = 0; j < more[i].length; j++) { more[i][j] = r.nextInt(8) == 0 ? nan : Math.round(r.nextDouble() * 1000) / 10.0; }
        }
        double[][] all = new double[inputs.length + more.length][];
        System.arraycopy(inputs, 0, all, 0, inputs.length);
        System.arraycopy(more, 0, all, inputs.length, more.length);
        for (double[] v : all) {
            String a = arr(v);
            line("max", a, () -> MovingFunctions.max(v));
            line("min", a, () -> MovingFunctions.min(v));
            line("sum", a, () -> MovingFunctions.sum(v));
            line("unweightedAvg", a, () -> MovingFunctions.unweightedAvg(v));
            line("linearWeightedAvg", a, () -> MovingFunctions.linearWeightedAvg(v));
            for (double avg : new double[] {0, 2.5, nan, 10}) {
                line("stdDev", a + " " + d(avg), () -> MovingFunctions.stdDev(v, avg));
            }
            for (double alpha : new double[] {0.0, 0.3, 0.5, 1.0}) {
                line("ewma", a + " " + d(alpha), () -> MovingFunctions.ewma(v, alpha));
            }
            for (double alpha : new double[] {0.3, 0.8}) {
                for (double beta : new double[] {0.1, 0.5}) {
                    line("holt", a + " " + d(alpha) + " " + d(beta), () -> MovingFunctions.holt(v, alpha, beta));
                }
            }
            for (int period : new int[] {1, 2, 4}) {
                for (boolean mult : new boolean[] {false, true}) {
                    line("holtWinters", a + " 0.3 0.1 0.2 " + period + " " + mult,
                        () -> MovingFunctions.holtWinters(v, 0.3, 0.1, 0.2, period, mult));
                    line("holtWinters", a + " 0.8 0.5 0.9 " + period + " " + mult,
                        () -> MovingFunctions.holtWinters(v, 0.8, 0.5, 0.9, period, mult));
                }
            }
        }
    }
}
