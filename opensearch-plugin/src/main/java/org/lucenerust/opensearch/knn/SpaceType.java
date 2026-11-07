/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.Locale;

/**
 * k-NN's {@code SpaceType}s this node serves, each with the Lucene similarity whose score equals
 * k-NN's score translation for it: {@code l2} {@code 1 / (1 + d²)}, {@code cosinesimil} {@code (1 +
 * cos) / 2}, {@code innerproduct} the maximum-inner-product scaling, {@code hamming} {@code 1 / (1 +
 * d)} by its format's scorer.
 */
public enum SpaceType {
    L2("l2", VectorSimilarityFunction.EUCLIDEAN),
    COSINESIMIL("cosinesimil", VectorSimilarityFunction.COSINE),
    INNER_PRODUCT("innerproduct", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT),
    /** Packed bit vectors: their format scores {@code 1 / (1 + hamming distance)}; the field records euclidean, as k-NN's does. */
    HAMMING("hamming", VectorSimilarityFunction.EUCLIDEAN);

    public static final String DEFAULT = "l2";
    /** {@code SpaceType.UNDEFINED}: the mapping named none. */
    public static final String UNDEFINED = "undefined";

    private final String value;
    private final VectorSimilarityFunction similarity;

    SpaceType(String value, VectorSimilarityFunction similarity) {
        this.value = value;
        this.similarity = similarity;
    }

    public String value() {
        return value;
    }

    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    /** The space type {@code name} names: one of k-NN's, refused when this node does not serve it. */
    public static SpaceType of(String name) {
        for (SpaceType s : values()) {
            if (s.value.equals(name)) {
                return s;
            }
        }
        switch (name) {
            case "l1", "linf", "hammingbit" -> throw new IllegalArgumentException(
                String.format(Locale.ROOT, "space_type [%s] is not supported by this node's knn_vector fields", name)
            );
            default -> throw new IllegalArgumentException(String.format(Locale.ROOT, "Unable to find space: %s", name));
        }
    }

    /**
     * The least score a radial search keeps for {@code max_distance} ({@code KNNQueryBuilder}'s
     * distance-to-score translation for the {@code lucene} engine).
     */
    public float scoreForDistance(float distance) {
        return switch (this) {
            case L2, HAMMING -> 1 / (1 + distance);
            case COSINESIMIL -> Math.max((2 - distance) / 2, 0);
            case INNER_PRODUCT -> distance < 0 ? 1 / (1 - distance) : distance + 1;
        };
    }

    @Override
    public String toString() {
        return value;
    }
}
