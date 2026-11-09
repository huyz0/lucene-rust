/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.query;

import org.apache.lucene.search.Scorable;
import java.io.IOException;
import java.util.Arrays;

/**
 * Scorer implementation for Hybrid Query. This object is light and expected to be re-used between different doc ids
 */
public class HybridSubQueryScorer extends Scorable {
    // array of scores from all sub-queries for a single doc id
    private final float[] subQueryScores;
    // array of min competitive scores, score is shard level
    private final float[] minScores;

    public HybridSubQueryScorer(int numOfSubQueries) {
        this.minScores = new float[numOfSubQueries];
        this.subQueryScores = new float[numOfSubQueries];
    }

    @Override
    public float score() throws IOException {
        // for scenarios when scorer is needed (like in aggregations) for one doc id return sum of sub-query scores
        float totalScore = 0.0F;
        for (float score : subQueryScores) {
            totalScore += score;
        }
        return totalScore;
    }

    /**
     * Reset sub-query scores to 0.0f so this scorer can be reused for next doc id
     */
    public void resetScores() {
        Arrays.fill(subQueryScores, 0.0F);
    }

    public int getNumOfSubQueries() {
        return subQueryScores.length;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof HybridSubQueryScorer)) return false;
        final HybridSubQueryScorer other = (HybridSubQueryScorer) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        if (!super.equals(o)) return false;
        if (!java.util.Arrays.equals(this.getSubQueryScores(), other.getSubQueryScores())) return false;
        if (!java.util.Arrays.equals(this.getMinScores(), other.getMinScores())) return false;
        return true;
    }

    @java.lang.SuppressWarnings("all")
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof HybridSubQueryScorer;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public int hashCode() {
        final int PRIME = 59;
        int result = super.hashCode();
        result = result * PRIME + java.util.Arrays.hashCode(this.getSubQueryScores());
        result = result * PRIME + java.util.Arrays.hashCode(this.getMinScores());
        return result;
    }

    @java.lang.SuppressWarnings("all")
    public float[] getSubQueryScores() {
        return this.subQueryScores;
    }

    @java.lang.SuppressWarnings("all")
    public float[] getMinScores() {
        return this.minScores;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "HybridSubQueryScorer(subQueryScores=" + java.util.Arrays.toString(this.getSubQueryScores()) + ", minScores=" + java.util.Arrays.toString(this.getMinScores()) + ")";
    }
}
