/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */

/**
 * The vectors in {@code crates/lucene-search/src/tdigest.rs}'s tests, from t-digest 3.3's own
 * {@code MergingDigest} (what OpenSearch's {@code TDigestState} extends): digests of a few value
 * sequences at several compressions, across merges in both directions, each state's every field
 * (counters, {@code min}/{@code max}, the merged and the buffered centroids) folded into one
 * FNV-1a hash of their bits. Run against the t-digest jar:
 *
 * <pre>
 * javac -proc:none -cp t-digest-3.3.jar GenTDigest.java && java -cp t-digest-3.3.jar:. GenTDigest
 * </pre>
 */
import com.tdunning.math.stats.MergingDigest;
import java.lang.reflect.Field;
import java.util.Random;

public class GenTDigest {
    static long h;

    static void mix(long v) {
        for (int i = 0; i < 8; i++) {
            h ^= (v >>> (8 * i)) & 0xff;
            h *= 0x100000001b3L;
        }
    }

    static Object get(Object o, Class<?> c, String name) throws Exception {
        Field f = c.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(o);
    }

    public static void main(String[] a) throws Exception {
        Class<?> md = MergingDigest.class;
        Class<?> abs = com.tdunning.math.stats.TDigest.class;
        for (double compression : new double[] { 100, 20, 5, 300 }) {
            for (int kind = 0; kind < 4; kind++) {
                for (int n : new int[] { 0, 1, 2, 3, 50, 1500, 20000 }) {
                    Random r = new Random(31 * kind + n);
                    MergingDigest d = new MergingDigest(compression);
                    for (int i = 0; i < n; i++) {
                        double x = switch (kind) {
                            case 0 -> r.nextGaussian() * 100;
                            case 1 -> r.nextInt(20);
                            case 2 -> i % 3 == 0 ? -0.0 : (i % 3 == 1 ? 0.0 : i);
                            default -> (r.nextDouble() - 0.5) * Math.scalb(1.0, r.nextInt(80) - 40);
                        };
                        d.add(x);
                    }
                    h = 0xcbf29ce484222325L;
                    int last = (int) get(d, md, "lastUsedCell");
                    int temp = (int) get(d, md, "tempUsed");
                    mix(Double.doubleToRawLongBits((double) get(d, md, "publicCompression")));
                    mix(Double.doubleToRawLongBits((double) get(d, md, "compression")));
                    mix((int) get(d, md, "mergeCount"));
                    mix(last);
                    mix(Double.doubleToRawLongBits((double) get(d, md, "totalWeight")));
                    mix(Double.doubleToRawLongBits((double) get(d, md, "unmergedWeight")));
                    mix(temp);
                    mix(Double.doubleToRawLongBits((double) get(d, abs, "min")));
                    mix(Double.doubleToRawLongBits((double) get(d, abs, "max")));
                    double[] mean = (double[]) get(d, md, "mean");
                    double[] weight = (double[]) get(d, md, "weight");
                    double[] tempMean = (double[]) get(d, md, "tempMean");
                    double[] tempWeight = (double[]) get(d, md, "tempWeight");
                    mix(mean.length);
                    mix(tempMean.length);
                    for (int i = 0; i < last; i++) {
                        mix(Double.doubleToRawLongBits(mean[i]));
                        mix(Double.doubleToRawLongBits(weight[i]));
                    }
                    for (int i = 0; i < temp; i++) {
                        mix(Double.doubleToRawLongBits(tempMean[i]));
                        mix(Double.doubleToRawLongBits(tempWeight[i]));
                    }
                    System.out.println("(" + compression + ", " + kind + ", " + n + ", " + h + "),");
                }
            }
        }
    }
}
