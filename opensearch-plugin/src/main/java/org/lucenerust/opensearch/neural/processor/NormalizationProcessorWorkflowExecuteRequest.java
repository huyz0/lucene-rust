/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import org.opensearch.action.search.SearchPhaseContext;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationTechnique;
import org.opensearch.search.fetch.FetchSearchResult;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.query.QuerySearchResult;
import java.util.List;
import java.util.Optional;

/**
 * DTO class to hold request parameters for normalization and combination
 */
public class NormalizationProcessorWorkflowExecuteRequest {
    final List<QuerySearchResult> querySearchResults;
    final Optional<FetchSearchResult> fetchSearchResultOptional;
    final ScoreNormalizationTechnique normalizationTechnique;
    final ScoreCombinationTechnique combinationTechnique;
    boolean explain;
    final PipelineProcessingContext pipelineProcessingContext;
    final SearchPhaseContext searchPhaseContext;


    @java.lang.SuppressWarnings("all")
    public static class NormalizationProcessorWorkflowExecuteRequestBuilder {
        @java.lang.SuppressWarnings("all")
        private List<QuerySearchResult> querySearchResults;
        @java.lang.SuppressWarnings("all")
        private Optional<FetchSearchResult> fetchSearchResultOptional;
        @java.lang.SuppressWarnings("all")
        private ScoreNormalizationTechnique normalizationTechnique;
        @java.lang.SuppressWarnings("all")
        private ScoreCombinationTechnique combinationTechnique;
        @java.lang.SuppressWarnings("all")
        private boolean explain;
        @java.lang.SuppressWarnings("all")
        private PipelineProcessingContext pipelineProcessingContext;
        @java.lang.SuppressWarnings("all")
        private SearchPhaseContext searchPhaseContext;

        @java.lang.SuppressWarnings("all")
        NormalizationProcessorWorkflowExecuteRequestBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder querySearchResults(final List<QuerySearchResult> querySearchResults) {
            this.querySearchResults = querySearchResults;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder fetchSearchResultOptional(final Optional<FetchSearchResult> fetchSearchResultOptional) {
            this.fetchSearchResultOptional = fetchSearchResultOptional;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder normalizationTechnique(final ScoreNormalizationTechnique normalizationTechnique) {
            this.normalizationTechnique = normalizationTechnique;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder combinationTechnique(final ScoreCombinationTechnique combinationTechnique) {
            this.combinationTechnique = combinationTechnique;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder explain(final boolean explain) {
            this.explain = explain;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder pipelineProcessingContext(final PipelineProcessingContext pipelineProcessingContext) {
            this.pipelineProcessingContext = pipelineProcessingContext;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder searchPhaseContext(final SearchPhaseContext searchPhaseContext) {
            this.searchPhaseContext = searchPhaseContext;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public NormalizationProcessorWorkflowExecuteRequest build() {
            return new NormalizationProcessorWorkflowExecuteRequest(this.querySearchResults, this.fetchSearchResultOptional, this.normalizationTechnique, this.combinationTechnique, this.explain, this.pipelineProcessingContext, this.searchPhaseContext);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder(querySearchResults=" + this.querySearchResults + ", fetchSearchResultOptional=" + this.fetchSearchResultOptional + ", normalizationTechnique=" + this.normalizationTechnique + ", combinationTechnique=" + this.combinationTechnique + ", explain=" + this.explain + ", pipelineProcessingContext=" + this.pipelineProcessingContext + ", searchPhaseContext=" + this.searchPhaseContext + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder builder() {
        return new NormalizationProcessorWorkflowExecuteRequest.NormalizationProcessorWorkflowExecuteRequestBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public NormalizationProcessorWorkflowExecuteRequest(final List<QuerySearchResult> querySearchResults, final Optional<FetchSearchResult> fetchSearchResultOptional, final ScoreNormalizationTechnique normalizationTechnique, final ScoreCombinationTechnique combinationTechnique, final boolean explain, final PipelineProcessingContext pipelineProcessingContext, final SearchPhaseContext searchPhaseContext) {
        this.querySearchResults = querySearchResults;
        this.fetchSearchResultOptional = fetchSearchResultOptional;
        this.normalizationTechnique = normalizationTechnique;
        this.combinationTechnique = combinationTechnique;
        this.explain = explain;
        this.pipelineProcessingContext = pipelineProcessingContext;
        this.searchPhaseContext = searchPhaseContext;
    }

    @java.lang.SuppressWarnings("all")
    public List<QuerySearchResult> getQuerySearchResults() {
        return this.querySearchResults;
    }

    @java.lang.SuppressWarnings("all")
    public Optional<FetchSearchResult> getFetchSearchResultOptional() {
        return this.fetchSearchResultOptional;
    }

    @java.lang.SuppressWarnings("all")
    public ScoreNormalizationTechnique getNormalizationTechnique() {
        return this.normalizationTechnique;
    }

    @java.lang.SuppressWarnings("all")
    public ScoreCombinationTechnique getCombinationTechnique() {
        return this.combinationTechnique;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isExplain() {
        return this.explain;
    }

    @java.lang.SuppressWarnings("all")
    public PipelineProcessingContext getPipelineProcessingContext() {
        return this.pipelineProcessingContext;
    }

    @java.lang.SuppressWarnings("all")
    public SearchPhaseContext getSearchPhaseContext() {
        return this.searchPhaseContext;
    }
}
