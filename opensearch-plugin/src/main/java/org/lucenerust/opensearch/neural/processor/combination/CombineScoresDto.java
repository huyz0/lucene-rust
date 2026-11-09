/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.combination;

import java.util.List;
import org.apache.lucene.search.Sort;
import org.opensearch.common.Nullable;
import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.opensearch.search.query.QuerySearchResult;

/**
 * DTO object to hold data required for Score Combination.
 */
public class CombineScoresDto {
    
    private List<CompoundTopDocs> queryTopDocs;
    
    private ScoreCombinationTechnique scoreCombinationTechnique;
    
    private List<QuerySearchResult> querySearchResults;
    @Nullable
    private Sort sort;
    private int fromValueForSingleShard;
    private boolean isSingleShard;
    private Float minScore;
    private boolean isCollapseEnabled;


    @java.lang.SuppressWarnings("all")
    public static class CombineScoresDtoBuilder {
        @java.lang.SuppressWarnings("all")
        private List<CompoundTopDocs> queryTopDocs;
        @java.lang.SuppressWarnings("all")
        private ScoreCombinationTechnique scoreCombinationTechnique;
        @java.lang.SuppressWarnings("all")
        private List<QuerySearchResult> querySearchResults;
        @java.lang.SuppressWarnings("all")
        private Sort sort;
        @java.lang.SuppressWarnings("all")
        private int fromValueForSingleShard;
        @java.lang.SuppressWarnings("all")
        private boolean isSingleShard;
        @java.lang.SuppressWarnings("all")
        private Float minScore;
        @java.lang.SuppressWarnings("all")
        private boolean isCollapseEnabled;

        @java.lang.SuppressWarnings("all")
        CombineScoresDtoBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder queryTopDocs(final List<CompoundTopDocs> queryTopDocs) {
            if (queryTopDocs == null) {
                throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
            }
            this.queryTopDocs = queryTopDocs;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder scoreCombinationTechnique(final ScoreCombinationTechnique scoreCombinationTechnique) {
            if (scoreCombinationTechnique == null) {
                throw new java.lang.NullPointerException("scoreCombinationTechnique is marked non-null but is null");
            }
            this.scoreCombinationTechnique = scoreCombinationTechnique;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder querySearchResults(final List<QuerySearchResult> querySearchResults) {
            if (querySearchResults == null) {
                throw new java.lang.NullPointerException("querySearchResults is marked non-null but is null");
            }
            this.querySearchResults = querySearchResults;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder sort(final Sort sort) {
            this.sort = sort;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder fromValueForSingleShard(final int fromValueForSingleShard) {
            this.fromValueForSingleShard = fromValueForSingleShard;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder isSingleShard(final boolean isSingleShard) {
            this.isSingleShard = isSingleShard;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder minScore(final Float minScore) {
            this.minScore = minScore;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombineScoresDto.CombineScoresDtoBuilder isCollapseEnabled(final boolean isCollapseEnabled) {
            this.isCollapseEnabled = isCollapseEnabled;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public CombineScoresDto build() {
            return new CombineScoresDto(this.queryTopDocs, this.scoreCombinationTechnique, this.querySearchResults, this.sort, this.fromValueForSingleShard, this.isSingleShard, this.minScore, this.isCollapseEnabled);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "CombineScoresDto.CombineScoresDtoBuilder(queryTopDocs=" + this.queryTopDocs + ", scoreCombinationTechnique=" + this.scoreCombinationTechnique + ", querySearchResults=" + this.querySearchResults + ", sort=" + this.sort + ", fromValueForSingleShard=" + this.fromValueForSingleShard + ", isSingleShard=" + this.isSingleShard + ", minScore=" + this.minScore + ", isCollapseEnabled=" + this.isCollapseEnabled + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static CombineScoresDto.CombineScoresDtoBuilder builder() {
        return new CombineScoresDto.CombineScoresDtoBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public CombineScoresDto(final List<CompoundTopDocs> queryTopDocs, final ScoreCombinationTechnique scoreCombinationTechnique, final List<QuerySearchResult> querySearchResults, final Sort sort, final int fromValueForSingleShard, final boolean isSingleShard, final Float minScore, final boolean isCollapseEnabled) {
        if (queryTopDocs == null) {
            throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
        }
        if (scoreCombinationTechnique == null) {
            throw new java.lang.NullPointerException("scoreCombinationTechnique is marked non-null but is null");
        }
        if (querySearchResults == null) {
            throw new java.lang.NullPointerException("querySearchResults is marked non-null but is null");
        }
        this.queryTopDocs = queryTopDocs;
        this.scoreCombinationTechnique = scoreCombinationTechnique;
        this.querySearchResults = querySearchResults;
        this.sort = sort;
        this.fromValueForSingleShard = fromValueForSingleShard;
        this.isSingleShard = isSingleShard;
        this.minScore = minScore;
        this.isCollapseEnabled = isCollapseEnabled;
    }

    
    @java.lang.SuppressWarnings("all")
    public List<CompoundTopDocs> getQueryTopDocs() {
        return this.queryTopDocs;
    }

    
    @java.lang.SuppressWarnings("all")
    public ScoreCombinationTechnique getScoreCombinationTechnique() {
        return this.scoreCombinationTechnique;
    }

    
    @java.lang.SuppressWarnings("all")
    public List<QuerySearchResult> getQuerySearchResults() {
        return this.querySearchResults;
    }

    @java.lang.SuppressWarnings("all")
    public Sort getSort() {
        return this.sort;
    }

    @java.lang.SuppressWarnings("all")
    public int getFromValueForSingleShard() {
        return this.fromValueForSingleShard;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isSingleShard() {
        return this.isSingleShard;
    }

    @java.lang.SuppressWarnings("all")
    public Float getMinScore() {
        return this.minScore;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isCollapseEnabled() {
        return this.isCollapseEnabled;
    }
}
