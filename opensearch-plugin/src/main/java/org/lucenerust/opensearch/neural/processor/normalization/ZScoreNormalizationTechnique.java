/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.normalization;

import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.lucenerust.opensearch.neural.processor.dto.ExplainDTO;
import org.lucenerust.opensearch.neural.processor.dto.NormalizeScoresDTO;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Map;
import java.util.HashMap;
import java.util.Locale;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.lucenerust.opensearch.neural.processor.explain.DocIdAtSearchShard;
import org.lucenerust.opensearch.neural.processor.explain.ExplainableTechnique;
import org.lucenerust.opensearch.neural.processor.explain.ExplanationDetails;
import static org.lucenerust.opensearch.neural.processor.explain.ExplanationUtils.getDocIdAtQueryForNormalization;
import static org.lucenerust.opensearch.neural.processor.util.ProcessorUtils.getNumOfSubqueries;

/**
 * Abstracts normalization of scores based on z score method
 */
public class ZScoreNormalizationTechnique implements ScoreNormalizationTechnique, ExplainableTechnique {
    public static final String TECHNIQUE_NAME = "z_score";
    private static final float SINGLE_RESULT_SCORE = 1.0F;
    private static final float MIN_SCORE = 0.001F;

    /**
     * Z-score normalization transforms the data based on its mean and standard deviation, making it more robust to outliers.
     * This technique preserves the shape of the original distribution while ensuring that all features contribute equally to the analysis,
     * regardless of their original scales.
     * Z-score calculation: score - mean // standard_deviation
     * mean = sum of all scores / number of scores
     * standard_deviation = square root of (sum of all scores - mean) / number of scores
     * Main algorithm steps:
     * - calculate mean and standard deviation for each sub query
     * - iterate over each result and update score as per formula above where "score" is raw score returned by Hybrid query
     *
     * @param normalizeScoresDTO is a data transfer object that contains queryTopDocs
     * original query results from multiple shards and multiple sub-queries, ScoreNormalizationTechnique,
     * and nullable rankConstant that is only used in RRF technique
     */
    @Override
    public void normalize(NormalizeScoresDTO normalizeScoresDTO) {
        List<CompoundTopDocs> queryTopDocs = normalizeScoresDTO.getQueryTopDocs();
        ZScores zscores = getZScoreResults(queryTopDocs);
        // do normalization using actual score and z-scores for corresponding sub query
        for (CompoundTopDocs compoundQueryTopDocs : queryTopDocs) {
            if (Objects.isNull(compoundQueryTopDocs)) {
                continue;
            }
            List<TopDocs> topDocsPerSubQuery = compoundQueryTopDocs.getTopDocs();
            for (int j = 0; j < topDocsPerSubQuery.size(); j++) {
                TopDocs subQueryTopDoc = topDocsPerSubQuery.get(j);
                for (ScoreDoc scoreDoc : subQueryTopDoc.scoreDocs) {
                    scoreDoc.score = normalizeSingleScore(scoreDoc.score, zscores.stdPerSubquery[j], zscores.meanPerSubQuery[j], zscores.maxPerSubQuery[j], zscores.minPerSubQuery[j]);
                }
            }
        }
    }

    @Override
    public String techniqueName() {
        return TECHNIQUE_NAME;
    }

    @Override
    public String describe() {
        return String.format(Locale.ROOT, "%s", TECHNIQUE_NAME);
    }

    @Override
    public Map<DocIdAtSearchShard, ExplanationDetails> explain(final ExplainDTO explainDTO) {
        List<CompoundTopDocs> queryTopDocs = explainDTO.getQueryTopDocs();
        ZScoreNormalizationTechnique.ZScores zScores = getZScoreResults(queryTopDocs);
        Map<DocIdAtSearchShard, List<Float>> normalizedScores = new HashMap<>();
        for (CompoundTopDocs compoundQueryTopDocs : queryTopDocs) {
            if (Objects.isNull(compoundQueryTopDocs)) {
                continue;
            }
            List<TopDocs> topDocsPerSubQuery = compoundQueryTopDocs.getTopDocs();
            int numberOfSubQueries = topDocsPerSubQuery.size();
            for (int subQueryIndex = 0; subQueryIndex < numberOfSubQueries; subQueryIndex++) {
                TopDocs subQueryTopDoc = topDocsPerSubQuery.get(subQueryIndex);
                for (ScoreDoc scoreDoc : subQueryTopDoc.scoreDocs) {
                    DocIdAtSearchShard docIdAtSearchShard = new DocIdAtSearchShard(scoreDoc.doc, compoundQueryTopDocs.getSearchShard());
                    float normalizedScore = normalizeSingleScore(scoreDoc.score, zScores.stdPerSubquery[subQueryIndex], zScores.meanPerSubQuery[subQueryIndex], zScores.maxPerSubQuery[subQueryIndex], zScores.minPerSubQuery[subQueryIndex]);
                    ScoreNormalizationUtil.setNormalizedScore(normalizedScores, docIdAtSearchShard, subQueryIndex, numberOfSubQueries, normalizedScore);
                    scoreDoc.score = normalizedScore;
                }
            }
        }
        return getDocIdAtQueryForNormalization(normalizedScores, this);
    }

    private static float[] calculateMaxScorePerSubquery(final List<CompoundTopDocs> queryTopDocs, final int numOfSubqueries) {
        DescriptiveStatistics[] statsPerSubquery = calculateStatsPerSubquery(queryTopDocs, numOfSubqueries);
        float[] maxPerSubQuery = new float[numOfSubqueries];
        for (int i = 0; i < numOfSubqueries; i++) {
            maxPerSubQuery[i] = (float) statsPerSubquery[i].getMax();
        }
        return maxPerSubQuery;
    }

    private static float[] calculateMinScorePerSubquery(final List<CompoundTopDocs> queryTopDocs, final int numOfSubqueries) {
        DescriptiveStatistics[] statsPerSubquery = calculateStatsPerSubquery(queryTopDocs, numOfSubqueries);
        float[] minPerSubQuery = new float[numOfSubqueries];
        for (int i = 0; i < numOfSubqueries; i++) {
            minPerSubQuery[i] = (float) statsPerSubquery[i].getMin();
        }
        return minPerSubQuery;
    }

    private static DescriptiveStatistics[] calculateStatsPerSubquery(final List<CompoundTopDocs> queryTopDocs, final int numOfSubqueries) {
        DescriptiveStatistics[] statsPerSubquery = new DescriptiveStatistics[numOfSubqueries];
        for (int i = 0; i < numOfSubqueries; i++) {
            statsPerSubquery[i] = new DescriptiveStatistics();
        }
        for (CompoundTopDocs compoundQueryTopDocs : queryTopDocs) {
            if (Objects.isNull(compoundQueryTopDocs)) {
                continue;
            }
            List<TopDocs> topDocsPerSubQuery = compoundQueryTopDocs.getTopDocs();
            for (int subQueryIndex = 0; subQueryIndex < topDocsPerSubQuery.size(); subQueryIndex++) {
                TopDocs topDocs = topDocsPerSubQuery.get(subQueryIndex);
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    statsPerSubquery[subQueryIndex].addValue(scoreDoc.score);
                }
            }
        }
        return statsPerSubquery;
    }

    private static float[] calculateMeanPerSubquery(final List<CompoundTopDocs> queryTopDocs, final int numOfSubqueries) {
        DescriptiveStatistics[] statsPerSubquery = calculateStatsPerSubquery(queryTopDocs, numOfSubqueries);
        float[] meanPerSubQuery = new float[numOfSubqueries];
        for (int i = 0; i < numOfSubqueries; i++) {
            meanPerSubQuery[i] = (float) statsPerSubquery[i].getMean();
        }
        return meanPerSubQuery;
    }

    private static float[] calculateStandardDeviationPerSubquery(final List<CompoundTopDocs> queryTopDocs, final int numOfSubqueries) {
        DescriptiveStatistics[] statsPerSubquery = calculateStatsPerSubquery(queryTopDocs, numOfSubqueries);
        float[] stdPerSubQuery = new float[numOfSubqueries];
        for (int i = 0; i < numOfSubqueries; i++) {
            stdPerSubQuery[i] = (float) statsPerSubquery[i].getStandardDeviation();
        }
        return stdPerSubQuery;
    }

    private ZScores getZScoreResults(final List<CompoundTopDocs> queryTopDocs) {
        int numOfSubqueries = getNumOfSubqueries(queryTopDocs);
        // to be done for each subquery
        float[] maxPerSubquery = calculateMaxScorePerSubquery(queryTopDocs, numOfSubqueries);
        float[] minPerSubquery = calculateMinScorePerSubquery(queryTopDocs, numOfSubqueries);
        float[] meanPerSubQuery = calculateMeanPerSubquery(queryTopDocs, numOfSubqueries);
        float[] stdPerSubquery = calculateStandardDeviationPerSubquery(queryTopDocs, numOfSubqueries);
        return new ZScores(meanPerSubQuery, stdPerSubquery, maxPerSubquery, minPerSubquery);
    }

    private static float normalizeSingleScore(final float score, final float standardDeviation, final float mean, final float maxScore, final float minScore) {
        // edge case when there is only one score and z scores are same
        if (Float.compare(mean, score) == 0) {
            return maxScore;
        }
        // Case when sd is 0
        if (Float.compare(standardDeviation, 0.0F) == 0) {
            return minScore;
        }
        float normalizedScore = (score - mean) / standardDeviation;
        return normalizedScore <= 0.0F ? MIN_SCORE : normalizedScore;
    }


    /**
     * Record to hold mean, std dev, max and min scores for each sub query
     */
    private record ZScores(float[] meanPerSubQuery, float[] stdPerSubquery, float[] maxPerSubQuery, float[] minPerSubQuery) {
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "ZScoreNormalizationTechnique(TECHNIQUE_NAME=" + ZScoreNormalizationTechnique.TECHNIQUE_NAME + ")";
    }

    /** The scores of one sub-query, with commons-math's {@code DescriptiveStatistics} mean and standard deviation. */
    private static final class DescriptiveStatistics {
        private double[] values = new double[8];
        private int n;

        void addValue(double v) {
            if (n == values.length) {
                values = Arrays.copyOf(values, n * 2);
            }
            values[n++] = v;
        }

        /** Commons-math {@code Max}: NaN when empty. */
        double getMax() {
            double max = Double.NaN;
            for (int i = 0; i < n; i++) {
                if (values[i] > max || Double.isNaN(max)) {
                    max = values[i];
                }
            }
            return max;
        }

        /** Commons-math {@code Min}: NaN when empty. */
        double getMin() {
            double min = Double.NaN;
            for (int i = 0; i < n; i++) {
                if (values[i] < min || Double.isNaN(min)) {
                    min = values[i];
                }
            }
            return min;
        }

        /** Commons-math {@code Mean}: the sum over n, then its rounding error corrected. */
        double getMean() {
            if (n == 0) {
                return Double.NaN;
            }
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += values[i];
            }
            double xbar = sum / n;
            double correction = 0;
            for (int i = 0; i < n; i++) {
                correction += values[i] - xbar;
            }
            return xbar + correction / n;
        }

        /** Commons-math's bias-corrected {@code Variance}, square-rooted; 0 for one value. */
        double getStandardDeviation() {
            if (n == 0) {
                return Double.NaN;
            }
            if (n == 1) {
                return 0.0;
            }
            double mean = getMean();
            double accum = 0;
            double dev = 0;
            for (int i = 0; i < n; i++) {
                double d = values[i] - mean;
                accum += d * d;
                dev += d;
            }
            return Math.sqrt((accum - dev * dev / n) / (n - 1));
        }
    }
}
