/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search.collector;

import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.Query;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorManager;
import org.lucenerust.opensearch.neural.search.query.HybridCollectorResultsUtilParams;
import org.lucenerust.opensearch.neural.search.query.util.HybridSearchCollectorResultUtil;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContextSpec;
import org.opensearch.search.query.QuerySearchResult;
import org.opensearch.search.query.ReduceableSearchResult;
import java.io.IOException;
import static org.opensearch.search.profile.query.CollectorResult.REASON_SEARCH_TOP_HITS;

/**
 * Create hybrid collector manager and injects the hybrid collector into the query phase to allow custom scoring and top doc collection
 */
public class HybridQueryCollectorContextSpec implements QueryCollectorContextSpec {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(HybridQueryCollectorContextSpec.class);
    private final HybridCollectorManager collectorManager;
    private final HybridSearchCollector collector;
    private final SearchContext searchContext;

    public HybridQueryCollectorContextSpec(final SearchContext searchContext, final Query query) {
        this.searchContext = searchContext;
        this.collectorManager = (HybridCollectorManager) HybridCollectorManager.createHybridCollectorManager(searchContext, query);
        this.collector = (HybridSearchCollector) collectorManager.newCollector();
    }

    @Override
    public String getContextName() {
        return REASON_SEARCH_TOP_HITS;
    }

    @Override
    public Collector create(Collector in) throws IOException {
        return collector;
    }

    @Override
    public CollectorManager<?, ReduceableSearchResult> createManager(CollectorManager<?, ReduceableSearchResult> in) {
        return collectorManager;
    }

    @Override
    public void postProcess(QuerySearchResult result) throws IOException {
        HybridSearchCollectorResultUtil hybridSearchCollectorResultUtil = new HybridSearchCollectorResultUtil(new HybridCollectorResultsUtilParams.Builder().searchContext(searchContext).build(), collector);
        TopDocsAndMaxScore topDocsAndMaxScore = hybridSearchCollectorResultUtil.getTopDocsAndMaxScore();
        hybridSearchCollectorResultUtil.reduceCollectorResults(result, topDocsAndMaxScore);
    }
}
