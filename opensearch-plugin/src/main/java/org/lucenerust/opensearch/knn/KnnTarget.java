/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import java.util.Arrays;

/** A KNN query vector: {@code float[]}, or {@code byte[]} for a {@code byte} field. Exactly one is non-null. */
public record KnnTarget(float[] floats, byte[] bytes) {
    public static KnnTarget of(float[] v) {
        return new KnnTarget(v, null);
    }

    public static KnnTarget of(byte[] v) {
        return new KnnTarget(null, v);
    }

    public int dimension() {
        return floats != null ? floats.length : bytes.length;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof KnnTarget t && Arrays.equals(floats, t.floats) && Arrays.equals(bytes, t.bytes);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(floats) + Arrays.hashCode(bytes);
    }

    @Override
    public String toString() {
        return floats != null ? Arrays.toString(floats) : Arrays.toString(bytes);
    }
}
