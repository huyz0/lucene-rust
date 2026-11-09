/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.stats;

/** The neural-search event counters the hybrid query and its processors bump, under their stock names and paths. */
public enum EventStatName {
    NORMALIZATION_PROCESSOR_EXECUTIONS("normalization_processor_executions", "processors.search.hybrid"),
    NORM_TECHNIQUE_L2_EXECUTIONS("norm_l2_executions", "processors.search.hybrid"),
    NORM_TECHNIQUE_MINMAX_EXECUTIONS("norm_minmax_executions", "processors.search.hybrid"),
    NORM_TECHNIQUE_NORM_ZSCORE_EXECUTIONS("norm_zscore_executions", "processors.search.hybrid"),
    COMB_TECHNIQUE_ARITHMETIC_EXECUTIONS("comb_arithmetic_executions", "processors.search.hybrid"),
    COMB_TECHNIQUE_GEOMETRIC_EXECUTIONS("comb_geometric_executions", "processors.search.hybrid"),
    COMB_TECHNIQUE_HARMONIC_EXECUTIONS("comb_harmonic_executions", "processors.search.hybrid"),
    RRF_PROCESSOR_EXECUTIONS("rank_based_normalization_processor_executions", "processors.search.hybrid"),
    COMB_TECHNIQUE_RRF_EXECUTIONS("comb_rrf_executions", "processors.search.hybrid"),
    HYBRID_QUERY_REQUESTS("hybrid_query_requests", "query.hybrid"),
    HYBRID_QUERY_INNER_HITS_REQUESTS("hybrid_query_with_inner_hits_requests", "query.hybrid"),
    HYBRID_QUERY_FILTER_REQUESTS("hybrid_query_with_filter_requests", "query.hybrid"),
    HYBRID_QUERY_PAGINATION_REQUESTS("hybrid_query_with_pagination_requests", "query.hybrid");

    private final String nameString;
    private final String path;

    EventStatName(String nameString, String path) {
        this.nameString = nameString;
        this.path = path;
    }

    public String getNameString() {
        return nameString;
    }

    public String getPath() {
        return path;
    }
}
