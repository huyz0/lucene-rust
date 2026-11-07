/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import java.util.List;

/** The stock plugin's vector value checks ({@code KNNVectorFieldMapperUtil}, {@code KNNQueryBuilder}). */
final class KnnVectors {
    private KnnVectors() {}

    static float[] toFloats(List<Float> values, SpaceType space) {
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = values.get(i);
        }
        checkFloats(v, space);
        return v;
    }

    static void checkFloats(float[] v, SpaceType space) {
        boolean zero = true;
        for (float f : v) {
            if (Float.isNaN(f)) {
                throw new IllegalArgumentException("KNN vector values cannot be NaN");
            }
            if (Float.isInfinite(f)) {
                throw new IllegalArgumentException("KNN vector values cannot be infinity");
            }
            zero &= f == 0;
        }
        if (zero && space == SpaceType.COSINESIMIL) {
            throw new IllegalArgumentException("zero vector is not supported when space type is [cosinesimil]");
        }
    }

    /** {@code KNNValidationUtil.validateHalfFloatVectorValue}'s range check (finite values were checked already). */
    static void checkHalfFloats(float[] v) {
        for (float f : v) {
            if (f < -65504f || f > 65504f) {
                throw new IllegalArgumentException(
                    "[data_type] field was set as HALF_FLOAT in index mapping. But, KNN vector values are not within in the half_float range [-65504.000000, 65504.000000]"
                );
            }
        }
    }

    /** {@code KNNVectorAsCollectionOfHalfFloatsSerializer.floatToByteArray}: two little-endian bytes a component. */
    static byte[] halfFloatBytes(float[] v) {
        byte[] out = new byte[v.length * 2];
        for (int i = 0; i < v.length; i++) {
            short h = Float.floatToFloat16(v[i]);
            out[2 * i] = (byte) h;
            out[2 * i + 1] = (byte) (h >>> 8);
        }
        return out;
    }

    static byte[] toBytes(List<Float> values) {
        byte[] v = new byte[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = toByte(values.get(i));
        }
        return v;
    }

    static byte toByte(float f) {
        if (Float.isNaN(f) || Float.isInfinite(f)) {
            throw new IllegalArgumentException("KNN vector values cannot be NaN or infinity");
        }
        if (f != (float) Math.rint(f)) {
            throw new IllegalArgumentException(
                "[data_type] field was set as [byte] in index mapping. But, KNN vector values are floats instead of byte integers"
            );
        }
        if (f < Byte.MIN_VALUE || f > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                "[data_type] field was set as [byte] in index mapping. But, KNN vector values are not within in the byte range [-128, 127]"
            );
        }
        return (byte) f;
    }
}
