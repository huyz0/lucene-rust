/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.dto;

import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationTechnique;

/**
 * DTO object to hold data required for validation.
 */
public class TechniqueCompatibilityCheckDTO {
    
    private ScoreCombinationTechnique scoreCombinationTechnique;
    
    private ScoreNormalizationTechnique scoreNormalizationTechnique;


    @java.lang.SuppressWarnings("all")
    public static class TechniqueCompatibilityCheckDTOBuilder {
        @java.lang.SuppressWarnings("all")
        private ScoreCombinationTechnique scoreCombinationTechnique;
        @java.lang.SuppressWarnings("all")
        private ScoreNormalizationTechnique scoreNormalizationTechnique;

        @java.lang.SuppressWarnings("all")
        TechniqueCompatibilityCheckDTOBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public TechniqueCompatibilityCheckDTO.TechniqueCompatibilityCheckDTOBuilder scoreCombinationTechnique(final ScoreCombinationTechnique scoreCombinationTechnique) {
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
        public TechniqueCompatibilityCheckDTO.TechniqueCompatibilityCheckDTOBuilder scoreNormalizationTechnique(final ScoreNormalizationTechnique scoreNormalizationTechnique) {
            if (scoreNormalizationTechnique == null) {
                throw new java.lang.NullPointerException("scoreNormalizationTechnique is marked non-null but is null");
            }
            this.scoreNormalizationTechnique = scoreNormalizationTechnique;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public TechniqueCompatibilityCheckDTO build() {
            return new TechniqueCompatibilityCheckDTO(this.scoreCombinationTechnique, this.scoreNormalizationTechnique);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "TechniqueCompatibilityCheckDTO.TechniqueCompatibilityCheckDTOBuilder(scoreCombinationTechnique=" + this.scoreCombinationTechnique + ", scoreNormalizationTechnique=" + this.scoreNormalizationTechnique + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static TechniqueCompatibilityCheckDTO.TechniqueCompatibilityCheckDTOBuilder builder() {
        return new TechniqueCompatibilityCheckDTO.TechniqueCompatibilityCheckDTOBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public TechniqueCompatibilityCheckDTO(final ScoreCombinationTechnique scoreCombinationTechnique, final ScoreNormalizationTechnique scoreNormalizationTechnique) {
        if (scoreCombinationTechnique == null) {
            throw new java.lang.NullPointerException("scoreCombinationTechnique is marked non-null but is null");
        }
        if (scoreNormalizationTechnique == null) {
            throw new java.lang.NullPointerException("scoreNormalizationTechnique is marked non-null but is null");
        }
        this.scoreCombinationTechnique = scoreCombinationTechnique;
        this.scoreNormalizationTechnique = scoreNormalizationTechnique;
    }

    
    @java.lang.SuppressWarnings("all")
    public ScoreCombinationTechnique getScoreCombinationTechnique() {
        return this.scoreCombinationTechnique;
    }

    
    @java.lang.SuppressWarnings("all")
    public ScoreNormalizationTechnique getScoreNormalizationTechnique() {
        return this.scoreNormalizationTechnique;
    }
}
