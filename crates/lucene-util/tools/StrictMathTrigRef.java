/** The expected values of `strict_math::tests::large_arguments_are_the_jdks_bit_for_bit`: `java StrictMathTrigRef.java`. */
public class StrictMathTrigRef {
    public static void main(String[] a) {
        long state = 0x2545F4914F6CDD1DL;
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < 100_000; i++) {
            state = state * 6364136223846793005L + 1;
            double x;
            if (i % 4 == 3) {
                // a power of two times a small odd integer: long zero runs
                x = Math.scalb((double) ((state >>> 40) | 1), (int) ((state >>> 8) % 1000));
            } else {
                // exponents 20..1023, both signs
                long exp = 1043 + ((state >>> 1) % 1004);
                x = Double.longBitsToDouble((state & 0x800fffffffffffffL) | (exp << 52));
            }
            h = (h ^ Double.doubleToRawLongBits(StrictMath.sin(x))) * 0x100000001b3L;
            h = (h ^ Double.doubleToRawLongBits(StrictMath.cos(x))) * 0x100000001b3L;
            h = (h ^ Double.doubleToRawLongBits(StrictMath.tan(x))) * 0x100000001b3L;
        }
        System.out.printf("0x%016x%n", h);
        for (double x : new double[] {1e22, 1e9, -1e9, 1e300, Double.MAX_VALUE, Math.pow(2, 60), Math.pow(2, 100)}) {
            System.out.printf("%s 0x%016x 0x%016x 0x%016x%n", x, Double.doubleToRawLongBits(StrictMath.sin(x)),
                Double.doubleToRawLongBits(StrictMath.cos(x)), Double.doubleToRawLongBits(StrictMath.tan(x)));
        }
    }
}
