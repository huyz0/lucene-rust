/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.explain;

/**
 * DTO class to hold explain details for normalization and combination
 */
public class CombinedExplanationDetails {
    private ExplanationDetails normalizationExplanations;
    private ExplanationDetails combinationExplanations;


    @java.lang.SuppressWarnings("all")
    public static class CombinedExplanationDetailsBuilder {
        @java.lang.SuppressWarnings("all")
        private ExplanationDetails normalizationExplanations;
        @java.lang.SuppressWarnings("all")
        private ExplanationDetails combinationExplanations;

        @java.lang.SuppressWarnings("all")
        CombinedExplanationDetailsBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombinedExplanationDetails.CombinedExplanationDetailsBuilder normalizationExplanations(final ExplanationDetails normalizationExplanations) {
            this.normalizationExplanations = normalizationExplanations;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public CombinedExplanationDetails.CombinedExplanationDetailsBuilder combinationExplanations(final ExplanationDetails combinationExplanations) {
            this.combinationExplanations = combinationExplanations;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public CombinedExplanationDetails build() {
            return new CombinedExplanationDetails(this.normalizationExplanations, this.combinationExplanations);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "CombinedExplanationDetails.CombinedExplanationDetailsBuilder(normalizationExplanations=" + this.normalizationExplanations + ", combinationExplanations=" + this.combinationExplanations + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static CombinedExplanationDetails.CombinedExplanationDetailsBuilder builder() {
        return new CombinedExplanationDetails.CombinedExplanationDetailsBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public CombinedExplanationDetails(final ExplanationDetails normalizationExplanations, final ExplanationDetails combinationExplanations) {
        this.normalizationExplanations = normalizationExplanations;
        this.combinationExplanations = combinationExplanations;
    }

    @java.lang.SuppressWarnings("all")
    public ExplanationDetails getNormalizationExplanations() {
        return this.normalizationExplanations;
    }

    @java.lang.SuppressWarnings("all")
    public ExplanationDetails getCombinationExplanations() {
        return this.combinationExplanations;
    }
}
