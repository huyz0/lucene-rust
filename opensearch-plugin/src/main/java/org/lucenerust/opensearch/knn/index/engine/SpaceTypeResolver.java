/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.index.engine;

import org.lucenerust.opensearch.knn.index.SpaceType;
import org.lucenerust.opensearch.knn.index.VectorDataType;

/** k-NN's {@code SpaceTypeResolver.getDefaultSpaceType}: hamming for binary vectors, l2 otherwise. */
public final class SpaceTypeResolver {
    private SpaceTypeResolver() {}

    public static SpaceType getDefaultSpaceType(final VectorDataType vectorDataType) {
        return vectorDataType == VectorDataType.BINARY ? SpaceType.DEFAULT_BINARY : SpaceType.DEFAULT;
    }
}
