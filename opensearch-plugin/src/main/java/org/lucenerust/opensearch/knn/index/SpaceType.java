/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.index;

import java.util.Arrays;
import java.util.Locale;

/**
 * k-NN's {@code SpaceType} as its search processors see it: the names, their parsing and errors, and
 * each space's similarity function ({@code l1} and {@code linf} have none). The mapping side is
 * {@link org.lucenerust.opensearch.knn.SpaceType}.
 */
public enum SpaceType {
    UNDEFINED("undefined", null),
    L2("l2", KNNVectorSimilarityFunction.EUCLIDEAN),
    COSINESIMIL("cosinesimil", KNNVectorSimilarityFunction.COSINE),
    L1("l1", null),
    LINF("linf", null),
    INNER_PRODUCT("innerproduct", KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT),
    HAMMING("hamming", KNNVectorSimilarityFunction.HAMMING);

    public static final SpaceType DEFAULT = L2;
    public static final SpaceType DEFAULT_BINARY = HAMMING;

    public static final String[] VALID_VALUES = Arrays.stream(values()).filter(s -> s != UNDEFINED).map(SpaceType::getValue).toArray(String[]::new);

    private final String value;
    private final KNNVectorSimilarityFunction similarity;

    SpaceType(String value, KNNVectorSimilarityFunction similarity) {
        this.value = value;
        this.similarity = similarity;
    }

    public String getValue() {
        return value;
    }

    public KNNVectorSimilarityFunction getKnnVectorSimilarityFunction() {
        return similarity;
    }

    public static SpaceType getSpace(String spaceTypeName) {
        for (SpaceType s : values()) {
            if (s.getValue().equalsIgnoreCase(spaceTypeName)) {
                return s;
            }
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "Unable to find space: %s . Valid values are: %s", spaceTypeName, Arrays.toString(VALID_VALUES))
        );
    }
}
