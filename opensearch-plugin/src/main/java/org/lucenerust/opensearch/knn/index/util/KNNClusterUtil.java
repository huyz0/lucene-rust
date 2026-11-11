/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.index.util;

import org.lucenerust.opensearch.neural.util.NeuralSearchClusterUtil;
import org.opensearch.action.IndicesRequest;
import org.opensearch.cluster.metadata.IndexMetadata;

import java.util.List;

/** k-NN's {@code KNNClusterUtil}, over the node's cluster state that neural search already holds. */
public final class KNNClusterUtil {
    private static final KNNClusterUtil INSTANCE = new KNNClusterUtil();

    private KNNClusterUtil() {}

    public static KNNClusterUtil instance() {
        return INSTANCE;
    }

    public List<IndexMetadata> getIndexMetadataList(final IndicesRequest searchRequest) {
        return NeuralSearchClusterUtil.instance().getIndexMetadataList(searchRequest);
    }

    public boolean isSystemGeneratedSearchFactoryEnabled(String factoryName) {
        return NeuralSearchClusterUtil.instance().isSystemGeneratedFactoryEnabled(factoryName);
    }
}
