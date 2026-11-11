/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

/**
 * The bytes of one {@code multi_vector} value, as its binary doc value holds them: the layouts of
 * {@code crates/lucene-tensor/src/encoding.rs}, which scores them natively.
 *
 * <ul>
 *   <li>{@code float}: {@code int dim}, then every token's floats, little-endian (Lucene's {@code
 *       LateInteractionField});
 *   <li>{@code float16}: {@code int -1}, {@code int dim}, {@code int tokens}, then the tokens in
 *       blocks of 16, component-major, as IEEE half floats;
 *   <li>{@code int8}: {@code int -2}, {@code int dim}, {@code int tokens}, a {@code float} scale per
 *       token, then blocks as above, signed bytes {@code round(x / scale)}.
 * </ul>
 *
 * A block's entry {@code k * 16 + l} is component {@code k} of token {@code 16 b + l}, and lanes past
 * the last token are zero.
 */
public enum MultiVectorEncoding {
    FLOAT("float"),
    FLOAT16("float16"),
    INT8("int8");

    static final int LANES = 16;
    private static final int FLOAT16_MARKER = -1;
    private static final int INT8_MARKER = -2;
    private static final int HEADER = 12;
    private static final float HALF_MAX = 65504f;

    private final String name;

    MultiVectorEncoding(String name) {
        this.name = name;
    }

    /** The mapping's name for this encoding. */
    public String getName() {
        return name;
    }

    /** The encoding called {@code name}. */
    public static MultiVectorEncoding of(String name) {
        for (MultiVectorEncoding e : values()) {
            if (e.name.equals(name)) {
                return e;
            }
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "unknown multi_vector encoding [%s], expected one of [float, float16, int8]", name)
        );
    }

    private static int blocked(int tokens, int dim) {
        return Math.multiplyExact(Math.multiplyExact((int) (((long) tokens + LANES - 1) / LANES), dim), LANES);
    }

    /** {@code tokens} (each {@code dim} long, at least one) encoded as this encoding. */
    public byte[] encode(float[][] tokens, int dim) {
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Value should not be null or empty");
        }
        for (float[] t : tokens) {
            if (t.length != dim) {
                throw new IllegalArgumentException("all composing token vectors should have the same length, the field's dimension " + dim);
            }
            if (this != FLOAT) {
                for (float x : t) {
                    if (Float.isFinite(x) == false) {
                        throw new IllegalArgumentException(name + " stores finite components only, got " + x);
                    }
                    if (this == FLOAT16 && Math.abs(x) > HALF_MAX) {
                        throw new IllegalArgumentException("float16 stores components up to 65504.0 in magnitude, got " + x);
                    }
                }
            }
        }
        int n = tokens.length;
        if (this == FLOAT) {
            ByteBuffer b = ByteBuffer.allocate(Math.addExact(4, Math.multiplyExact(Math.multiplyExact(n, dim), 4))).order(ByteOrder.LITTLE_ENDIAN);
            b.putInt(dim);
            for (float[] t : tokens) {
                for (float x : t) {
                    b.putFloat(x);
                }
            }
            return b.array();
        }
        int len = blocked(n, dim);
        if (this == FLOAT16) {
            ByteBuffer b = ByteBuffer.allocate(Math.addExact(HEADER, Math.multiplyExact(len, 2))).order(ByteOrder.LITTLE_ENDIAN);
            b.putInt(FLOAT16_MARKER).putInt(dim).putInt(n);
            for (int i = 0; i < n; i++) {
                int base = HEADER + 2 * ((i / LANES) * dim * LANES + i % LANES);
                for (int k = 0; k < dim; k++) {
                    b.putShort(base + 2 * k * LANES, Float.floatToFloat16(tokens[i][k]));
                }
            }
            return b.array();
        }
        int body = Math.addExact(HEADER, Math.multiplyExact(n, 4));
        ByteBuffer b = ByteBuffer.allocate(Math.addExact(body, len)).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(INT8_MARKER).putInt(dim).putInt(n);
        for (int i = 0; i < n; i++) {
            float max = 0f;
            for (float x : tokens[i]) {
                max = Math.max(max, Math.abs(x));
            }
            // A subnormal max would underflow to a zero scale: the smallest positive float keeps
            // every scaled component within the clamp.
            float scale = max > 0f ? Math.max(max / 127f, Float.MIN_VALUE) : 1f;
            b.putFloat(scale);
            int base = body + (i / LANES) * dim * LANES + i % LANES;
            for (int k = 0; k < dim; k++) {
                float v = tokens[i][k] / scale;
                // Rust's f32::round: half away from zero, exact in double for these magnitudes.
                double r = Math.copySign(Math.floor(Math.abs((double) v) + 0.5), v);
                b.put(base + k * LANES, (byte) Math.max(-127, Math.min(127, (int) r)));
            }
        }
        return b.array();
    }

    /**
     * The tokens' mean, scaled to unit length ({@code lucene_tensor::pooled_mean}: summed and scaled
     * in double), or null when it is zero: a document's one vector for finding candidates.
     */
    public static float[] pooledMean(float[][] tokens) {
        int dim = tokens[0].length;
        double[] mean = new double[dim];
        for (float[] t : tokens) {
            for (int k = 0; k < dim; k++) {
                mean[k] += t[k];
            }
        }
        double norm = 0;
        for (double m : mean) {
            norm += m * m;
        }
        norm = Math.sqrt(norm);
        if (norm == 0 || Double.isFinite(norm) == false) {
            return null;
        }
        float[] out = new float[dim];
        for (int k = 0; k < dim; k++) {
            out[k] = (float) (mean[k] / norm);
        }
        return out;
    }

    /** A decoded value: {@code tokens} tokens of {@code dim} floats in the block layout. */
    public record Blocks(int dim, int tokens, float[] values) {}

    private static IllegalArgumentException notLucene() {
        return new IllegalArgumentException("Provided payload does not appear to have been encoded via LateInteractionField.encode");
    }

    private static IllegalArgumentException bad() {
        return new IllegalArgumentException("Provided payload is not a multi-vector this field encodes");
    }

    /** {@code value} (any encoding) widened to floats in the block layout. */
    public static Blocks decode(byte[] bytes, int offset, int length) {
        if (length < 4) {
            throw bad();
        }
        ByteBuffer b = ByteBuffer.wrap(bytes, offset, length).slice().order(ByteOrder.LITTLE_ENDIAN);
        int first = b.getInt(0);
        if (first > 0) {
            int dim = first;
            long width = 4L * dim;
            if ((length - 4) % width != 0) {
                throw notLucene();
            }
            int n = (int) ((length - 4) / width);
            float[] out;
            try {
                out = new float[blocked(n, dim)];
            } catch (ArithmeticException e) {
                throw notLucene();
            }
            for (int i = 0; i < n; i++) {
                int base = (i / LANES) * dim * LANES + i % LANES;
                for (int k = 0; k < dim; k++) {
                    out[base + k * LANES] = b.getFloat(4 + 4 * (i * dim + k));
                }
            }
            return new Blocks(dim, n, out);
        }
        if (length < HEADER) {
            throw bad();
        }
        int dim = b.getInt(4);
        int n = b.getInt(8);
        if (dim <= 0 || n < 0) {
            throw bad();
        }
        int len;
        try {
            len = blocked(n, dim);
        } catch (ArithmeticException e) {
            throw bad();
        }
        // The lengths are checked before allocating: a corrupt header must not size the buffer.
        if (first == FLOAT16_MARKER ? (long) len * 2 != length - HEADER : first != INT8_MARKER || HEADER + 4L * n + len != length) {
            throw bad();
        }
        float[] out = new float[len];
        if (first == FLOAT16_MARKER) {
            for (int i = 0; i < len; i++) {
                out[i] = Float.float16ToFloat(b.getShort(HEADER + 2 * i));
            }
            return new Blocks(dim, n, out);
        }
        if (first == INT8_MARKER) {
            long body = HEADER + 4L * n;
            for (int i = 0; i < len; i++) {
                int token = (i / (dim * LANES)) * LANES + i % LANES;
                float scale = token < n ? b.getFloat(HEADER + 4 * token) : 0f;
                out[i] = b.get((int) body + i) * scale;
            }
            return new Blocks(dim, n, out);
        }
        throw bad();
    }

    /**
     * The sum, over {@code query}'s tokens, of each one's best dot product with a token of {@code
     * doc}: {@code lucene_util::max_sim}'s scalar specification, which every native kernel equals
     * bit for bit (a fused multiply-add per component, in component order). A value without tokens
     * scores {@code Float.MIN_VALUE}, as Lucene's {@code SUM_MAX_SIM} does.
     */
    public static float maxSim(float[][] query, Blocks doc) {
        if (doc.tokens() == 0) {
            return Float.MIN_VALUE;
        }
        int dim = doc.dim();
        if (query[0].length != dim) {
            throw new IllegalArgumentException(
                "Provided multi-vectors are incompatible. Their composing token vectors should have the same dimension, got "
                    + query[0].length
                    + " != "
                    + dim
            );
        }
        float[] v = doc.values();
        float total = 0f;
        for (float[] q : query) {
            float best = Float.NEGATIVE_INFINITY;
            for (int blk = 0; blk * LANES < doc.tokens(); blk++) {
                int base = blk * dim * LANES;
                int valid = Math.min(LANES, doc.tokens() - blk * LANES);
                for (int l = 0; l < valid; l++) {
                    float acc = 0f;
                    for (int k = 0; k < dim; k++) {
                        acc = Math.fma(q[k], v[base + k * LANES + l], acc);
                    }
                    best = best > acc ? best : acc;
                }
            }
            total += best;
        }
        return total;
    }
}
