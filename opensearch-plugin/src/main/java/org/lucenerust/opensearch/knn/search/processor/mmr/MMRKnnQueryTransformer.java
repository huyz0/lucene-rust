/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.search.processor.mmr;

import org.opensearch.core.action.ActionListener;
import org.lucenerust.opensearch.knn.KnnQueryBuilder;
import static org.lucenerust.opensearch.knn.search.processor.mmr.MMRUtil.resolveKnnVectorFieldInfo;

/**
 * A transformer to transform the knn query for MMR
 */
public class MMRKnnQueryTransformer implements MMRQueryTransformer<KnnQueryBuilder> {
    @Override
    public void transform(KnnQueryBuilder queryBuilder, ActionListener<Void> listener, MMRTransformContext mmrTransformContext) {
        try {
            if (queryBuilder.getMaxDistance() == null && queryBuilder.getMinScore() == null) {
                queryBuilder.setK(mmrTransformContext.getCandidates());
            }
            if (mmrTransformContext.isVectorFieldInfoResolved()) {
                listener.onResponse(null);
                return;
            }
            MMRRerankContext mmrRerankContext = mmrTransformContext.getMmrRerankContext();
            String knnVectorFieldPath = queryBuilder.fieldName();
            if (knnVectorFieldPath == null) {
                throw new IllegalArgumentException("Failed to transform the knn query for MMR. Field name of the knn query should not be null.");
            }
            mmrRerankContext.setVectorFieldPath(knnVectorFieldPath);
            resolveKnnVectorFieldInfo(knnVectorFieldPath, mmrTransformContext.getUserProvidedSpaceType(), mmrTransformContext.getUserProvidedVectorDataType(), mmrTransformContext.getLocalIndexMetadataList(), mmrTransformContext.getClient(), ActionListener.wrap(vectorFieldInfo -> {
                mmrRerankContext.setVectorDataType(vectorFieldInfo.getVectorDataType());
                mmrRerankContext.setSpaceType(vectorFieldInfo.getSpaceType());
                listener.onResponse(null);
            }, listener::onFailure));
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    @Override
    public String getQueryName() {
        return KnnQueryBuilder.NAME;
    }

    
    
    public MMRKnnQueryTransformer() {
    }
}
