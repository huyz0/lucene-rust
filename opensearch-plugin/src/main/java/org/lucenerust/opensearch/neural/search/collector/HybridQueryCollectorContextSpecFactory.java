/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search.collector;

import org.apache.lucene.search.Query;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorArguments;
import org.opensearch.search.query.QueryCollectorContextSpec;
import org.opensearch.search.query.QueryCollectorContextSpecFactory;
import java.util.Optional;
import static org.lucenerust.opensearch.neural.util.HybridQueryUtil.isHybridQuery;

/**
 * Factory class for HybridQueryCollectorContextSpec. In case of hybrid query, it will create the spec which will retrieved in the QueryPhase.
 */
public class HybridQueryCollectorContextSpecFactory implements QueryCollectorContextSpecFactory {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(HybridQueryCollectorContextSpecFactory.class);

    @Override
    public Optional<QueryCollectorContextSpec> createQueryCollectorContextSpec(SearchContext searchContext, Query query, QueryCollectorArguments queryCollectorArguments) {
        if (isHybridQuery(query, searchContext)) {
            return Optional.of(new HybridQueryCollectorContextSpec(searchContext, query));
        }
        return Optional.empty();
    }
}
