/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.dto;

import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationTechnique;
import java.util.List;

/**
 * DTO object to hold data required for score normalization.
 */
public class NormalizeScoresDTO {
    
    private List<CompoundTopDocs> queryTopDocs;
    
    private ScoreNormalizationTechnique normalizationTechnique;
    private boolean singleShard;


    @java.lang.SuppressWarnings("all")
    public static class NormalizeScoresDTOBuilder {
        @java.lang.SuppressWarnings("all")
        private List<CompoundTopDocs> queryTopDocs;
        @java.lang.SuppressWarnings("all")
        private ScoreNormalizationTechnique normalizationTechnique;
        @java.lang.SuppressWarnings("all")
        private boolean singleShard;

        @java.lang.SuppressWarnings("all")
        NormalizeScoresDTOBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizeScoresDTO.NormalizeScoresDTOBuilder queryTopDocs(final List<CompoundTopDocs> queryTopDocs) {
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
        public NormalizeScoresDTO.NormalizeScoresDTOBuilder normalizationTechnique(final ScoreNormalizationTechnique normalizationTechnique) {
            if (normalizationTechnique == null) {
                throw new java.lang.NullPointerException("normalizationTechnique is marked non-null but is null");
            }
            this.normalizationTechnique = normalizationTechnique;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public NormalizeScoresDTO.NormalizeScoresDTOBuilder singleShard(final boolean singleShard) {
            this.singleShard = singleShard;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public NormalizeScoresDTO build() {
            return new NormalizeScoresDTO(this.queryTopDocs, this.normalizationTechnique, this.singleShard);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "NormalizeScoresDTO.NormalizeScoresDTOBuilder(queryTopDocs=" + this.queryTopDocs + ", normalizationTechnique=" + this.normalizationTechnique + ", singleShard=" + this.singleShard + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static NormalizeScoresDTO.NormalizeScoresDTOBuilder builder() {
        return new NormalizeScoresDTO.NormalizeScoresDTOBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public NormalizeScoresDTO(final List<CompoundTopDocs> queryTopDocs, final ScoreNormalizationTechnique normalizationTechnique, final boolean singleShard) {
        if (queryTopDocs == null) {
            throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
        }
        if (normalizationTechnique == null) {
            throw new java.lang.NullPointerException("normalizationTechnique is marked non-null but is null");
        }
        this.queryTopDocs = queryTopDocs;
        this.normalizationTechnique = normalizationTechnique;
        this.singleShard = singleShard;
    }

    
    @java.lang.SuppressWarnings("all")
    public List<CompoundTopDocs> getQueryTopDocs() {
        return this.queryTopDocs;
    }

    
    @java.lang.SuppressWarnings("all")
    public ScoreNormalizationTechnique getNormalizationTechnique() {
        return this.normalizationTechnique;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isSingleShard() {
        return this.singleShard;
    }
}
