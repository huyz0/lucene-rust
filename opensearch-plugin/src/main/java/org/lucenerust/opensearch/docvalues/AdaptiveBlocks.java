/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.docvalues;

import java.io.IOException;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.NumericUtils;

/**
 * One adaptive value array, read a block at a time: {@code adaptive_values.rs}'s {@code
 * decode_block}, with the block last decoded kept, as a doc-values iterator reads its values in
 * order.
 *
 * <p>A block is a token byte ({@code CONSTANT}, {@code DELTA}, {@code GCD}, {@code DECIMAL} bits);
 * a constant block then holds one zig-zag varint, any other the width, the decimal exponent, the
 * first value (delta), the minimum, the GCD and the values bit-packed least significant bit first.
 */
final class AdaptiveBlocks {
    static final int BLOCK_SHIFT = 7;
    static final int BLOCK_SIZE = 1 << BLOCK_SHIFT;
    /** Zero bytes after the last block: every block can be read with 16 more bytes. */
    static final int PADDING = 16;

    private static final int DELTA = 0x01;
    private static final int GCD = 0x02;
    private static final int DECIMAL = 0x04;
    private static final int CONSTANT = 0x08;
    private static final int MAX_EXPONENT = 18;
    private static final double[] POW10 = new double[MAX_EXPONENT + 1];

    static {
        double p = 1;
        for (int i = 0; i <= MAX_EXPONENT; i++) {
            POW10[i] = p;
            p *= 10;
        }
    }

    private final RandomAccessInput values;
    private final LongValues addresses;
    private final long numValues;
    private final long[] block = new long[BLOCK_SIZE];
    private long current = -1;
    private byte[] buffer = new byte[64];

    AdaptiveBlocks(RandomAccessInput values, LongValues addresses, long numValues) {
        this.values = values;
        this.addresses = addresses;
        this.numValues = numValues;
    }

    /** The value at {@code index} of the array. */
    long get(long index) throws IOException {
        long b = index >>> BLOCK_SHIFT;
        if (b != current) {
            load(b);
        }
        return block[(int) (index & (BLOCK_SIZE - 1))];
    }

    private void load(long b) throws IOException {
        current = -1;
        long start = addresses.get(b);
        long end = addresses.get(b + 1);
        long n = Math.min(BLOCK_SIZE, numValues - (b << BLOCK_SHIFT));
        if (start < 0 || end < start || n <= 0 || end - start > Integer.MAX_VALUE - PADDING) {
            throw new CorruptIndexException("adaptive block " + b + " at [" + start + ", " + end + ")", values.toString());
        }
        int length = (int) (end - start) + PADDING;
        if (buffer.length < length) {
            buffer = new byte[length];
        }
        values.readBytes(start, buffer, 0, length);
        decode(buffer, (int) n, block);
        current = b;
    }

    /** Fills {@code out[0..n)} from the block at the start of {@code buf}. */
    static void decode(byte[] buf, int n, long[] out) throws CorruptIndexException {
        int[] pos = {0};
        int token = buf[pos[0]++] & 0xff;
        if ((token & CONSTANT) != 0) {
            java.util.Arrays.fill(out, 0, n, zlong(buf, pos));
            return;
        }
        int bpv = buf[pos[0]++] & 0xff;
        if (bpv > 64) {
            throw new CorruptIndexException("adaptive block width " + bpv, "block");
        }
        int exponent = 0;
        if ((token & DECIMAL) != 0) {
            exponent = buf[pos[0]++] & 0xff;
            if (exponent > MAX_EXPONENT) {
                throw new CorruptIndexException("adaptive block exponent " + exponent, "block");
            }
        }
        long first = (token & DELTA) != 0 ? zlong(buf, pos) : 0;
        long min = zlong(buf, pos);
        long gcd = (token & GCD) != 0 ? zlong(buf, pos) : 1;
        int from = (token & DELTA) != 0 ? 1 : 0;
        int packed = pos[0];
        if (bpv == 0) {
            java.util.Arrays.fill(out, from, n, 0L);
        } else {
            long mask = bpv == 64 ? -1L : (1L << bpv) - 1;
            for (int i = from; i < n; i++) {
                long bit = (long) (i - from) * bpv;
                int at = packed + (int) (bit >>> 3);
                int shift = (int) (bit & 7);
                long v = (long) BitUtil.VH_LE_LONG.get(buf, at) >>> shift;
                if (shift + bpv > 64) {
                    v |= (buf[at + 8] & 0xffL) << (64 - shift);
                }
                out[i] = v & mask;
            }
        }
        if (from == 1) {
            long acc = first;
            out[0] = first;
            for (int i = 1; i < n; i++) {
                acc += min + out[i] * gcd;
                out[i] = acc;
            }
        } else {
            for (int i = 0; i < n; i++) {
                out[i] = min + out[i] * gcd;
            }
        }
        if ((token & DECIMAL) != 0) {
            double pow = POW10[exponent];
            for (int i = 0; i < n; i++) {
                out[i] = NumericUtils.doubleToSortableLong((double) out[i] / pow);
            }
        }
    }

    private static long zlong(byte[] buf, int[] pos) throws CorruptIndexException {
        long z = 0;
        for (int shift = 0;; shift += 7) {
            if (shift > 63) {
                throw new CorruptIndexException("adaptive block varint too long", "block");
            }
            int b = buf[pos[0]++] & 0xff;
            z |= (long) (b & 0x7f) << shift;
            if ((b & 0x80) == 0) {
                return (z >>> 1) ^ -(z & 1);
            }
        }
    }
}
