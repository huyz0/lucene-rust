/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */

/**
 * The vectors in {@code crates/lucene-search/src/cardinality_sketch.rs}'s tests, from OpenSearch
 * 3.8.0's own {@code MurmurHash3}, {@code BitMixer} and {@code HyperLogLogPlusPlus}: hashes of a few
 * terms and longs, then sketches of {@code n} longs at several precisions, across the
 * linear-counting threshold, as {@code writeTo} writes them (length, CRC-32, and the bytes when
 * short). Run against the distribution's jars:
 *
 * <pre>
 * CP=$(ls $OPENSEARCH_HOME/lib/*.jar | tr '\n' ':')
 * javac -proc:none -cp "$CP" GenHll.java && java -cp "$CP:." GenHll
 * </pre>
 */
import org.opensearch.common.hash.MurmurHash3;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.util.BigArrays;
import org.opensearch.common.util.BitMixer;
import org.opensearch.search.aggregations.metrics.HyperLogLogPlusPlus;
import java.nio.charset.StandardCharsets;
import java.util.HexFormat;
import java.util.Random;

public class GenHll {
    public static void main(String[] a) throws Exception {
        HexFormat hex = HexFormat.of();
        // Hashes.
        for (String s : new String[] { "", "a", "hello", "0123456789abcdef", "0123456789abcdefXYZ", "héllo wörld, a longer term" }) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            MurmurHash3.Hash128 h = MurmurHash3.hash128(b, 0, b.length, 0, new MurmurHash3.Hash128());
            System.out.println("murmur\t" + hex.formatHex(b) + "\t" + h.h1);
        }
        for (long v : new long[] { 0, 1, -1, 42, Long.MIN_VALUE, Long.MAX_VALUE, Double.doubleToLongBits(2.5) }) {
            System.out.println("mix64\t" + v + "\t" + BitMixer.mix64(v));
        }
        // Sketches: longs 0..n (mixed) at several precisions and sizes across the linear-counting threshold.
        Random r = new Random(7);
        for (int p : new int[] { 4, 5, 10, 14 }) {
            for (int n : new int[] { 1, 3, 4, 20, 700, 5000 }) {
                HyperLogLogPlusPlus counts = new HyperLogLogPlusPlus(p, BigArrays.NON_RECYCLING_INSTANCE, 1);
                StringBuilder vals = new StringBuilder();
                long seed = r.nextLong();
                for (int i = 0; i < n; i++) {
                    long v = seed + i * 7919L;
                    counts.collect(0, BitMixer.mix64(v));
                }
                BytesStreamOutput out = new BytesStreamOutput();
                counts.writeTo(0, out);
                byte[] bytes = java.util.Arrays.copyOf(out.bytes().toBytesRef().bytes, out.bytes().length());
                java.util.zip.CRC32 crc = new java.util.zip.CRC32();
                crc.update(bytes);
                System.out.println("sketch\t" + p + "\t" + seed + "\t" + n + "\t" + bytes.length + "\t" + crc.getValue() + "\t" + (bytes.length <= 400 ? hex.formatHex(bytes) : ""));
            }
        }
    }
}
