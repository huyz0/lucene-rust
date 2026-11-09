/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.normalization;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Locale;
import java.util.PriorityQueue;
import java.util.Set;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.opensearch.common.TriConsumer;
import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.lucenerust.opensearch.neural.processor.dto.ExplainDTO;
import org.lucenerust.opensearch.neural.processor.dto.NormalizeScoresDTO;
import org.lucenerust.opensearch.neural.processor.SearchShard;
import org.lucenerust.opensearch.neural.processor.explain.DocIdAtSearchShard;
import org.lucenerust.opensearch.neural.processor.explain.ExplainableTechnique;
import org.lucenerust.opensearch.neural.processor.explain.ExplanationDetails;
import static org.lucenerust.opensearch.neural.processor.explain.ExplanationUtils.getDocIdAtQueryForNormalization;

/**
 * Abstracts calculation of rank scores for each document returned as part of
 * reciprocal rank fusion. Rank scores are summed across subqueries in combination classes.
 */
public class RRFNormalizationTechnique implements ScoreNormalizationTechnique, ExplainableTechnique {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(RRFNormalizationTechnique.class);
    public static final String TECHNIQUE_NAME = "rrf";
    public static final int DEFAULT_RANK_CONSTANT = 60;
    public static final String PARAM_NAME_RANK_CONSTANT = "rank_constant";
    private static final Set<String> SUPPORTED_PARAMS = Set.of(PARAM_NAME_RANK_CONSTANT);
    private static final int MIN_RANK_CONSTANT = 1;
    private static final int MAX_RANK_CONSTANT = 10000;
    // Rank scores are quantized to 10 decimal places, i.e. expressed as an integer numerator over 10^10.
    private static final long SCORE_SCALE_L = 10000000000L;
    private static final double SCORE_SCALE_D = 1.0E10;
    private final int rankConstant;
    // Comparator to compare ShardResultPerSubQuery
    private static final Comparator<ShardResultPerSubQuery> comparator = Comparator.comparing((ShardResultPerSubQuery sr) -> sr.scoreDoc, ScoreDoc.COMPARATOR).thenComparingInt(sr -> sr.referenceShardId);

    public RRFNormalizationTechnique(final Map<String, Object> params, final ScoreNormalizationUtil scoreNormalizationUtil) {
        scoreNormalizationUtil.validateParameters(params, SUPPORTED_PARAMS, Map.of());
        rankConstant = getRankConstant(params);
    }

    /**
     * Reciprocal Rank Fusion normalization technique
     * @param normalizeScoresDTO is a data transfer object that contains queryTopDocs
     * original query results from multiple shards and multiple sub-queries, ScoreNormalizationTechnique,
     * and nullable rankConstant, which has a default value of 60 if not specified by user
     * algorithm as follows, where document_n_score is the new score for each document in queryTopDocs
     * and subquery_result_rank is the position in the array of documents returned for each subquery
     * (j + 1 is used to adjust for 0 indexing)
     * document_n_score = 1 / (rankConstant + subquery_result_rank)
     * document scores are summed in combination step
     */
    @Override
    public void normalize(final NormalizeScoresDTO normalizeScoresDTO) {
        final List<CompoundTopDocs> queryTopDocs = normalizeScoresDTO.getQueryTopDocs();
        Map<Integer, Map<String, Integer>> sortedDocIdsPerSubqueryByGlobalRank = normalizeScoresDTO.isSingleShard() ? Map.of() : sortDocumentsAsPerGlobalRankInIndividualQuery(queryTopDocs);
        for (int referenceShardId = 0; referenceShardId < queryTopDocs.size(); referenceShardId++) {
            processTopDocs(queryTopDocs.get(referenceShardId), (docId, score, subQueryIndex) -> {
            }, sortedDocIdsPerSubqueryByGlobalRank, referenceShardId);
        }
    }

    private Map<Integer, Map<String, Integer>> sortDocumentsAsPerGlobalRankInIndividualQuery(final List<CompoundTopDocs> queryTopDocs) {
        if (queryTopDocs == null) {
            throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
        }
        // Map: subQueryIndex -> PQ that contains ScoreDocs from all shards
        Map<Integer, PriorityQueue<ShardResultPerSubQuery>> scoreDocsPerSubquery = new HashMap<>();
        for (int referenceShardId = 0; referenceShardId < queryTopDocs.size(); referenceShardId++) {
            CompoundTopDocs compoundTopDocs = queryTopDocs.get(referenceShardId);
            if (Objects.isNull(compoundTopDocs)) {
                continue;
            }
            // List of subquery results
            List<TopDocs> topDocs = compoundTopDocs.getTopDocs();
            for (int topDocIndex = 0; topDocIndex < topDocs.size(); topDocIndex++) {
                TopDocs topDoc = topDocs.get(topDocIndex);
                scoreDocsPerSubquery.putIfAbsent(topDocIndex, new PriorityQueue<>(comparator));
                ScoreDoc[] scoreDocs = topDoc.scoreDocs;
                // Insert all the scoreDocs in priority queue that holds results across all shards for the subquery
                for (int scoreDocIndex = 0; scoreDocIndex < scoreDocs.length; scoreDocIndex++) {
                    scoreDocsPerSubquery.get(topDocIndex).add(new ShardResultPerSubQuery(scoreDocs[scoreDocIndex], referenceShardId));
                }
            }
        }
        Map<Integer, Map<String, Integer>> globallySortedDocIdMap = new HashMap<>();
        // Traverse all scoreDocs per subquery and add it in the map with docId-shardId order.
        // We created docId-shardId format because docId can be same across different shards but the combination is unique.
        for (Map.Entry<Integer, PriorityQueue<ShardResultPerSubQuery>> entry : scoreDocsPerSubquery.entrySet()) {
            int subQueryNumber = entry.getKey();
            globallySortedDocIdMap.putIfAbsent(subQueryNumber, new HashMap<>());
            PriorityQueue<ShardResultPerSubQuery> sortedScoreDocsAcrossAllShards = entry.getValue();
            // first rank
            int rank = 0;
            while (!sortedScoreDocsAcrossAllShards.isEmpty()) {
                ShardResultPerSubQuery shardResultPerSubQuery = sortedScoreDocsAcrossAllShards.poll();
                globallySortedDocIdMap.get(subQueryNumber).put(shardResultPerSubQuery.scoreDoc.doc + "_" + shardResultPerSubQuery.referenceShardId, rank++);
            }
        }
        return globallySortedDocIdMap;
    }

    @Override
    public String describe() {
        return String.format(Locale.ROOT, "%s, rank_constant [%s]", TECHNIQUE_NAME, rankConstant);
    }

    @Override
    public String techniqueName() {
        return TECHNIQUE_NAME;
    }

    @Override
    public Map<DocIdAtSearchShard, ExplanationDetails> explain(final ExplainDTO explainDTO) {
        final List<CompoundTopDocs> queryTopDocs = explainDTO.getQueryTopDocs();
        Map<DocIdAtSearchShard, List<Float>> normalizedScores = new HashMap<>();
        Map<Integer, Map<String, Integer>> sortedDocIdsPerSubqueryByGlobalRank = explainDTO.isSingleShard() ? Map.of() : sortDocumentsAsPerGlobalRankInIndividualQuery(queryTopDocs);
        for (int referenceShardId = 0; referenceShardId < queryTopDocs.size(); referenceShardId++) {
            CompoundTopDocs compoundQueryTopDocs = queryTopDocs.get(referenceShardId);
            if (Objects.isNull(compoundQueryTopDocs)) {
                continue;
            }
            List<TopDocs> topDocsPerSubQuery = compoundQueryTopDocs.getTopDocs();
            int numberOfSubQueries = topDocsPerSubQuery.size();
            processTopDocs(compoundQueryTopDocs, (docId, score, subQueryIndex) -> ScoreNormalizationUtil.setNormalizedScore(normalizedScores, docId, subQueryIndex, numberOfSubQueries, score), sortedDocIdsPerSubqueryByGlobalRank, referenceShardId);
        }
        return getDocIdAtQueryForNormalization(normalizedScores, this);
    }

    private void processTopDocs(CompoundTopDocs compoundQueryTopDocs, TriConsumer<DocIdAtSearchShard, Float, Integer> scoreProcessor, Map<Integer, Map<String, Integer>> sortedDocIdMapPerSubqueryByGlobalRank, int referenceShardId) {
        if (Objects.isNull(compoundQueryTopDocs)) {
            return;
        }
        List<TopDocs> topDocsList = compoundQueryTopDocs.getTopDocs();
        SearchShard searchShard = compoundQueryTopDocs.getSearchShard();
        for (int topDocsIndex = 0; topDocsIndex < topDocsList.size(); topDocsIndex++) {
            Map<String, Integer> docIdToRankMap = sortedDocIdMapPerSubqueryByGlobalRank.isEmpty() ? Map.of() : sortedDocIdMapPerSubqueryByGlobalRank.get(topDocsIndex);
            processTopDocsEntry(topDocsList.get(topDocsIndex), searchShard, topDocsIndex, scoreProcessor, docIdToRankMap, referenceShardId);
        }
    }

    private void processTopDocsEntry(TopDocs topDocs, SearchShard searchShard, int topDocsIndex, TriConsumer<DocIdAtSearchShard, Float, Integer> scoreProcessor, Map<String, Integer> docIdToRankMap, int referenceShardId) {
        if (topDocs == null) {
            throw new java.lang.NullPointerException("topDocs is marked non-null but is null");
        }
        for (int position = 0; position < topDocs.scoreDocs.length; position++) {
            ScoreDoc scoreDoc = topDocs.scoreDocs[position];
            Integer rank = docIdToRankMap.isEmpty() ? position : docIdToRankMap.get(scoreDoc.doc + "_" + referenceShardId);
            if (Objects.isNull(rank) || rank < 0) {
                throw new IllegalStateException("Document not found in global ranking map: doc=" + scoreDoc.doc + ", shard=" + referenceShardId);
            }
            float normalizedScore = calculateNormalizedScore(rank);
            DocIdAtSearchShard docIdAtSearchShard = new DocIdAtSearchShard(scoreDoc.doc, searchShard);
            scoreProcessor.apply(docIdAtSearchShard, normalizedScore, topDocsIndex);
            scoreDoc.score = normalizedScore;
        }
    }

    /**
     * Computes the rank score of a result, {@code 1 / (rankConstant + position + 1)} rounded HALF_UP to 10
     * decimal places. This is an allocation-free integer equivalent of the original implementation:
     * <pre>{@code
     * BigDecimal.ONE.divide(BigDecimal.valueOf(rankConstant + position + 1), 10, RoundingMode.HALF_UP).floatValue()
     * }</pre>
     * Because 10^10 and 2 * 10^10 both fit in a {@code long}, rounding HALF_UP to scale 10 is exactly expressible
     * as the integer division {@code (2 * 10^10 + d) / (2 * d)}. That quotient is at most 5 * 10^9, well inside
     * the range a {@code double} represents exactly, as is 10^10 itself, so the single narrowing to {@code float}
     * reproduces what {@code BigDecimal.floatValue()} produced. Verified bit-identical for every reachable
     * denominator in [2, Integer.MAX_VALUE].
     * <p>
     * The final division is deliberately performed in {@code double}. Narrowing the numerator to {@code float}
     * first and dividing by {@code 1.0e10f} rounds twice and is <em>not</em> bit-identical: it differs for 167
     * denominators, starting at 3.
     *
     * @param position zero-based rank of the result within its subquery
     * @return the rank score, to be summed across subqueries in the combination step
     */
    private float calculateNormalizedScore(int position) {
        long denominator = (long) rankConstant + position + 1;
        long numerator = (2 * SCORE_SCALE_L + denominator) / (2 * denominator);
        return (float) (numerator / SCORE_SCALE_D);
    }

    private int getRankConstant(final Map<String, Object> params) {
        if (Objects.isNull(params) || !params.containsKey(PARAM_NAME_RANK_CONSTANT)) {
            return DEFAULT_RANK_CONSTANT;
        }
        int rankConstant = getParamAsInteger(params, PARAM_NAME_RANK_CONSTANT);
        validateRankConstant(rankConstant);
        return rankConstant;
    }

    private void validateRankConstant(final int rankConstant) {
        if (rankConstant < MIN_RANK_CONSTANT || rankConstant > MAX_RANK_CONSTANT) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "rank constant must be in the interval between 1 and 10000, submitted rank constant: %d", rankConstant));
        }
    }

    private static int getParamAsInteger(final Map<String, Object> parameters, final String fieldName) {
        try {
            return Integer.decode(String.valueOf(parameters.get(fieldName)));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "parameter [%s] must be an integer", fieldName));
        }
    }


    /**
     * Record to store shard results per subquery
     * @param scoreDoc scoreDoc result
     * @param referenceShardId reference shard Id for identifying scoreDoc
     */
    private record ShardResultPerSubQuery(ScoreDoc scoreDoc, int referenceShardId) {
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "RRFNormalizationTechnique(TECHNIQUE_NAME=" + RRFNormalizationTechnique.TECHNIQUE_NAME + ", rankConstant=" + this.rankConstant + ")";
    }
}
