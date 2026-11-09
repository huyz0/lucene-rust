/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.combination;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.lucenerust.opensearch.neural.processor.explain.ExplainableTechnique;
import static org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationUtil.PARAM_NAME_WEIGHTS;
import static org.lucenerust.opensearch.neural.processor.explain.ExplanationUtils.describeCombinationTechnique;

/**
 * Abstracts combination of scores based on arithmetic mean method
 */
public class ArithmeticMeanScoreCombinationTechnique implements ScoreCombinationTechnique, ExplainableTechnique {
    public static final String TECHNIQUE_NAME = "arithmetic_mean";
    private static final Set<String> SUPPORTED_PARAMS = Set.of(PARAM_NAME_WEIGHTS);
    private static final Float ZERO_SCORE = 0.0F;
    private final List<Float> weights;
    private final ScoreCombinationUtil scoreCombinationUtil;

    public ArithmeticMeanScoreCombinationTechnique(final Map<String, Object> params, final ScoreCombinationUtil combinationUtil) {
        scoreCombinationUtil = combinationUtil;
        scoreCombinationUtil.validateParams(params, SUPPORTED_PARAMS);
        weights = scoreCombinationUtil.getWeights(params);
    }

    /**
     * Arithmetic mean method for combining scores.
     * score = (weight1*score1 + weight2*score2 +...+ weightN*scoreN)/(weight1 + weight2 + ... + weightN)
     *
     * Zero (0.0) scores are excluded from number of scores N
     */
    @Override
    public float combine(final float[] scores) {
        scoreCombinationUtil.validateIfWeightsMatchScores(scores, weights);
        float combinedScore = 0.0F;
        float sumOfWeights = 0;
        for (int indexOfSubQuery = 0; indexOfSubQuery < scores.length; indexOfSubQuery++) {
            float score = scores[indexOfSubQuery];
            if (score >= 0.0) {
                float weight = scoreCombinationUtil.getWeightForSubQuery(weights, indexOfSubQuery);
                score = score * weight;
                combinedScore += score;
                sumOfWeights += weight;
            }
        }
        if (sumOfWeights == 0.0F) {
            return ZERO_SCORE;
        }
        return combinedScore / sumOfWeights;
    }

    @Override
    public String techniqueName() {
        return TECHNIQUE_NAME;
    }

    @Override
    public String describe() {
        return describeCombinationTechnique(TECHNIQUE_NAME, weights);
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "ArithmeticMeanScoreCombinationTechnique(TECHNIQUE_NAME=" + ArithmeticMeanScoreCombinationTechnique.TECHNIQUE_NAME + ", weights=" + this.weights + ")";
    }
}
