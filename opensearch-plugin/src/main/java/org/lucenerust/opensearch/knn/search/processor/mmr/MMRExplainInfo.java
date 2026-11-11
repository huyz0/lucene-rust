/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.search.processor.mmr;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * A DTO to hold the explain info for a single hit selected by the MMR algorithm.
 * This captures the scoring details at the moment the document was chosen.
 * Note: the selection order and previously-selected documents can be inferred from
 * the position of each hit in the result list.
 */
public final class MMRExplainInfo {
    public static final String ORIGINAL_SCORE_FIELD = "original_score";
    public static final String MAX_SIMILARITY_TO_SELECTED_FIELD = "max_similarity_to_selected";
    public static final String MMR_SCORE_FIELD = "mmr_score";
    public static final String MMR_FORMULA_FIELD = "mmr_formula";
    /**
     * The original relevance score from the KNN/neural search.
     */
    private final float originalScore;
    /**
     * The maximum vector similarity between this document and any already-selected document.
     * For the first selected document, this is 0.0.
     */
    private final float maxSimilarityToSelected;
    /**
     * The computed MMR score at selection time: (1 - diversity) * originalScore - diversity * maxSimilarityToSelected.
     */
    private final double mmrScore;
    /**
     * The diversity parameter (lambda) used in the MMR formula.
     */
    private final float diversity;

    /**
     * Converts this explain info to a map suitable for injection into a search hit's _source.
     *
     * @return a map representing the mmr_explain payload
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put(ORIGINAL_SCORE_FIELD, originalScore);
        map.put(MAX_SIMILARITY_TO_SELECTED_FIELD, maxSimilarityToSelected);
        map.put(MMR_SCORE_FIELD, mmrScore);
        map.put(MMR_FORMULA_FIELD, String.format(Locale.ROOT, "(1 - %.4f) * %.4f - %.4f * %.4f = %.4f", diversity, originalScore, diversity, maxSimilarityToSelected, mmrScore));
        return map;
    }

    /**
     * Creates a new {@code MMRExplainInfo} instance.
     *
     * @param originalScore The original relevance score from the KNN/neural search.
     * @param maxSimilarityToSelected The maximum vector similarity between this document and any already-selected document.
     * For the first selected document, this is 0.0.
     * @param mmrScore The computed MMR score at selection time: (1 - diversity) * originalScore - diversity * maxSimilarityToSelected.
     * @param diversity The diversity parameter (lambda) used in the MMR formula.
     */
    
    
    MMRExplainInfo(final float originalScore, final float maxSimilarityToSelected, final double mmrScore, final float diversity) {
        this.originalScore = originalScore;
        this.maxSimilarityToSelected = maxSimilarityToSelected;
        this.mmrScore = mmrScore;
        this.diversity = diversity;
    }


    
    
    public static class MMRExplainInfoBuilder {
        
        
        private float originalScore;
        
        
        private float maxSimilarityToSelected;
        
        
        private double mmrScore;
        
        
        private float diversity;

        
        
        MMRExplainInfoBuilder() {
        }

        /**
         * The original relevance score from the KNN/neural search.
         * @return {@code this}.
         */
        
        
        public MMRExplainInfo.MMRExplainInfoBuilder originalScore(final float originalScore) {
            this.originalScore = originalScore;
            return this;
        }

        /**
         * The maximum vector similarity between this document and any already-selected document.
         * For the first selected document, this is 0.0.
         * @return {@code this}.
         */
        
        
        public MMRExplainInfo.MMRExplainInfoBuilder maxSimilarityToSelected(final float maxSimilarityToSelected) {
            this.maxSimilarityToSelected = maxSimilarityToSelected;
            return this;
        }

        /**
         * The computed MMR score at selection time: (1 - diversity) * originalScore - diversity * maxSimilarityToSelected.
         * @return {@code this}.
         */
        
        
        public MMRExplainInfo.MMRExplainInfoBuilder mmrScore(final double mmrScore) {
            this.mmrScore = mmrScore;
            return this;
        }

        /**
         * The diversity parameter (lambda) used in the MMR formula.
         * @return {@code this}.
         */
        
        
        public MMRExplainInfo.MMRExplainInfoBuilder diversity(final float diversity) {
            this.diversity = diversity;
            return this;
        }

        
        
        public MMRExplainInfo build() {
            return new MMRExplainInfo(this.originalScore, this.maxSimilarityToSelected, this.mmrScore, this.diversity);
        }

        @java.lang.Override
        
        
        public java.lang.String toString() {
            return "MMRExplainInfo.MMRExplainInfoBuilder(originalScore=" + this.originalScore + ", maxSimilarityToSelected=" + this.maxSimilarityToSelected + ", mmrScore=" + this.mmrScore + ", diversity=" + this.diversity + ")";
        }
    }

    
    
    public static MMRExplainInfo.MMRExplainInfoBuilder builder() {
        return new MMRExplainInfo.MMRExplainInfoBuilder();
    }

    /**
     * The original relevance score from the KNN/neural search.
     */
    
    
    public float getOriginalScore() {
        return this.originalScore;
    }

    /**
     * The maximum vector similarity between this document and any already-selected document.
     * For the first selected document, this is 0.0.
     */
    
    
    public float getMaxSimilarityToSelected() {
        return this.maxSimilarityToSelected;
    }

    /**
     * The computed MMR score at selection time: (1 - diversity) * originalScore - diversity * maxSimilarityToSelected.
     */
    
    
    public double getMmrScore() {
        return this.mmrScore;
    }

    /**
     * The diversity parameter (lambda) used in the MMR formula.
     */
    
    
    public float getDiversity() {
        return this.diversity;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof MMRExplainInfo)) return false;
        final MMRExplainInfo other = (MMRExplainInfo) o;
        if (java.lang.Float.compare(this.getOriginalScore(), other.getOriginalScore()) != 0) return false;
        if (java.lang.Float.compare(this.getMaxSimilarityToSelected(), other.getMaxSimilarityToSelected()) != 0) return false;
        if (java.lang.Double.compare(this.getMmrScore(), other.getMmrScore()) != 0) return false;
        if (java.lang.Float.compare(this.getDiversity(), other.getDiversity()) != 0) return false;
        return true;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + java.lang.Float.floatToIntBits(this.getOriginalScore());
        result = result * PRIME + java.lang.Float.floatToIntBits(this.getMaxSimilarityToSelected());
        final long $mmrScore = java.lang.Double.doubleToLongBits(this.getMmrScore());
        result = result * PRIME + (int) ($mmrScore >>> 32 ^ $mmrScore);
        result = result * PRIME + java.lang.Float.floatToIntBits(this.getDiversity());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "MMRExplainInfo(originalScore=" + this.getOriginalScore() + ", maxSimilarityToSelected=" + this.getMaxSimilarityToSelected() + ", mmrScore=" + this.getMmrScore() + ", diversity=" + this.getDiversity() + ")";
    }
}
