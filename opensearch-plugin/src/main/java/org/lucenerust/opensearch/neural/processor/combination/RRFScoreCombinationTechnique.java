/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.combination;

import org.lucenerust.opensearch.neural.processor.explain.ExplainableTechnique;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import static org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationUtil.PARAM_NAME_WEIGHTS;
import static org.lucenerust.opensearch.neural.processor.explain.ExplanationUtils.describeCombinationTechnique;

/**
 * Abstracts combination of scores based on reciprocal rank fusion algorithm
 */
public class RRFScoreCombinationTechnique implements ScoreCombinationTechnique, ExplainableTechnique {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(RRFScoreCombinationTechnique.class);
    public static final String TECHNIQUE_NAME = "rrf";
    private static final Set<String> SUPPORTED_PARAMS = Set.of(PARAM_NAME_WEIGHTS);
    private final List<Float> weights;
    private static final Float ZERO_SCORE = 0.0F;
    private final ScoreCombinationUtil scoreCombinationUtil;

    public RRFScoreCombinationTechnique(final Map<String, Object> params, final ScoreCombinationUtil combinationUtil) {
        scoreCombinationUtil = combinationUtil;
        scoreCombinationUtil.validateParams(params, SUPPORTED_PARAMS);
        weights = scoreCombinationUtil.getWeights(params);
    }

    @Override
    public float combine(final float[] scores) {
        if (Objects.isNull(scores)) {
            throw new IllegalArgumentException("scores array cannot be null");
        }
        scoreCombinationUtil.validateIfWeightsMatchScores(scores, weights);
        float sumScores = 0.0F;
        float sumOfWeights = 0;
        for (int indexOfSubQuery = 0; indexOfSubQuery < scores.length; indexOfSubQuery++) {
            float score = scores[indexOfSubQuery];
            if (score >= 0.0) {
                float weight = scoreCombinationUtil.getWeightForSubQuery(weights, indexOfSubQuery);
                score = score * weight;
                sumScores += score;
                sumOfWeights += weight;
            }
        }
        if (sumOfWeights == 0.0F) {
            return ZERO_SCORE;
        }
        return sumScores;
    }

    @Override
    public String techniqueName() {
        return TECHNIQUE_NAME;
    }

    @Override
    public String describe() {
        return describeCombinationTechnique(TECHNIQUE_NAME, List.of());
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "RRFScoreCombinationTechnique(TECHNIQUE_NAME=" + RRFScoreCombinationTechnique.TECHNIQUE_NAME + ", weights=" + this.weights + ")";
    }
}
