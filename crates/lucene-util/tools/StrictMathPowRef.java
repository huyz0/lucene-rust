/** The expected values of `strict_math::tests::pow_is_the_jdks_bit_for_bit`: `java StrictMathPowRef.java`. */
public class StrictMathPowRef {
    static long state = 0x9E3779B97F4A7C15L;

    static double unit() {
        state = state * 6364136223846793005L + 1;
        return (state >>> 11) / (double) (1L << 53);
    }

    static long bitsOf() {
        state = state * 6364136223846793005L + 1;
        return state;
    }

    public static void main(String[] a) {
        double[][] fixed = {
            {2, 10}, {2, 0.5}, {-8, 1.0 / 3}, {-2, 3}, {-2, 4}, {-2, 0.5}, {0.0, -1}, {-0.0, -3}, {-0.0, -2},
            {Double.NEGATIVE_INFINITY, 0.5}, {Double.NEGATIVE_INFINITY, 3}, {Double.NEGATIVE_INFINITY, -3},
            {-1, Double.POSITIVE_INFINITY}, {0.5, Double.NEGATIVE_INFINITY}, {3, Double.NEGATIVE_INFINITY},
            {Double.NaN, 0}, {1, Double.NaN}, {-1, 1e300}, {-1, 2.5}, {1.0000001, 1e10}, {0.9999999, 1e10},
            {1.0000001, -1e10}, {0.5, 3e9}, {1.5, 3e9}, {1.0000000001, 3e9}, {0.9999999999, -3e9},
            {10, 308}, {10, 309}, {10, -323}, {10, -324}, {2, 1023.9999}, {2, -1074}, {2, -1075}, {2, -1074.5},
            {4.9e-324, 0.3}, {1e-310, 1.5}, {7, -0.5}, {1.7, 1.3}, {1.7320508075688772, 7.1}, {Math.E, 1023.0 * Math.log(2)},
            {-3, -9007199254740993.0}, {-3, 1e16},
        };
        StringBuilder sb = new StringBuilder();
        for (double[] p : fixed) {
            sb.append(String.format("(0x%016x, 0x%016x, 0x%016x),%n",
                Double.doubleToRawLongBits(p[0]), Double.doubleToRawLongBits(p[1]),
                Double.doubleToRawLongBits(StrictMath.pow(p[0], p[1]))));
        }
        System.out.print(sb);
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < 300_000; i++) {
            double x, y;
            switch (i % 3) {
                case 0 -> { x = unit() * 20 - 5; y = unit() * 60 - 30; }
                case 1 -> { x = Double.longBitsToDouble(bitsOf() >>> 1 & 0x7fefffffffffffffL); y = unit() * 4 - 2; }
                default -> { x = 1 + (unit() - 0.5) * 1e-6; y = (unit() - 0.5) * 1e10; }
            }
            h = (h ^ Double.doubleToRawLongBits(StrictMath.pow(x, y))) * 0x100000001b3L;
        }
        System.out.printf("digest 0x%016x%n", h);
    }
}
