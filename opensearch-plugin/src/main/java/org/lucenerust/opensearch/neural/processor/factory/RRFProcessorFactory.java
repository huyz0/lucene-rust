/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.factory;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import org.lucenerust.opensearch.neural.processor.RRFProcessor;
import org.lucenerust.opensearch.neural.processor.NormalizationProcessorWorkflow;
import org.lucenerust.opensearch.neural.processor.combination.RRFScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationFactory;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.RRFNormalizationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationFactory;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationTechnique;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchPhaseResultsProcessor;
import static org.opensearch.ingest.ConfigurationUtils.readOptionalMap;
import static org.opensearch.ingest.ConfigurationUtils.readStringProperty;

/**
 * Factory class to instantiate RRF processor based on user provided input.
 */
public class RRFProcessorFactory implements Processor.Factory<SearchPhaseResultsProcessor> {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(RRFProcessorFactory.class);
    public static final String COMBINATION_CLAUSE = "combination";
    public static final String TECHNIQUE = "technique";
    public static final String PARAMETERS = "parameters";
    private final NormalizationProcessorWorkflow normalizationProcessorWorkflow;
    private ScoreNormalizationFactory scoreNormalizationFactory;
    private ScoreCombinationFactory scoreCombinationFactory;

    @Override
    public SearchPhaseResultsProcessor create(final Map<String, Processor.Factory<SearchPhaseResultsProcessor>> processorFactories, final String tag, final String description, final boolean ignoreFailure, final Map<String, Object> config, final Processor.PipelineContext pipelineContext) throws Exception {
        // assign defaults
        ScoreNormalizationTechnique normalizationTechnique = scoreNormalizationFactory.createNormalization(RRFNormalizationTechnique.TECHNIQUE_NAME);
        ScoreCombinationTechnique scoreCombinationTechnique = scoreCombinationFactory.createCombination(RRFScoreCombinationTechnique.TECHNIQUE_NAME);
        Map<String, Object> combinationClause = readOptionalMap(RRFProcessor.TYPE, tag, config, COMBINATION_CLAUSE);
        if (Objects.nonNull(combinationClause)) {
            String combinationTechnique = readStringProperty(RRFProcessor.TYPE, tag, combinationClause, TECHNIQUE, RRFScoreCombinationTechnique.TECHNIQUE_NAME);
            // This processor normalizes by rank, so rrf is the only combination that means anything here. The other
            // techniques registered in ScoreCombinationFactory (the means) belong to the normalization-processor, where
            // they combine normalized scores. Accepting one here produced a pipeline that threw on every query, so
            // reject it at pipeline creation with a 400 instead of failing at search time.
            if (RRFScoreCombinationTechnique.TECHNIQUE_NAME.equals(combinationTechnique) == false) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "provided combination technique is not supported by [%s], supported technique is [%s], got [%s]", RRFProcessor.TYPE, RRFScoreCombinationTechnique.TECHNIQUE_NAME, combinationTechnique));
            }
            String rankConstantParam = RRFNormalizationTechnique.PARAM_NAME_RANK_CONSTANT;
            if (combinationClause.containsKey(rankConstantParam)) {
                normalizationTechnique = scoreNormalizationFactory.createNormalization(RRFNormalizationTechnique.TECHNIQUE_NAME, Map.of(rankConstantParam, combinationClause.get(rankConstantParam)));
            }
            Map<String, Object> params = readOptionalMap(RRFProcessor.TYPE, tag, combinationClause, PARAMETERS);
            scoreCombinationTechnique = scoreCombinationFactory.createCombination(combinationTechnique, params);
        }
        log.info("Creating search phase results processor of type [{}] with normalization [{}] and combination [{}]", RRFProcessor.TYPE, normalizationTechnique, scoreCombinationTechnique);
        return new RRFProcessor(tag, description, normalizationTechnique, scoreCombinationTechnique, normalizationProcessorWorkflow);
    }

    @java.lang.SuppressWarnings("all")
    public RRFProcessorFactory(final NormalizationProcessorWorkflow normalizationProcessorWorkflow, final ScoreNormalizationFactory scoreNormalizationFactory, final ScoreCombinationFactory scoreCombinationFactory) {
        this.normalizationProcessorWorkflow = normalizationProcessorWorkflow;
        this.scoreNormalizationFactory = scoreNormalizationFactory;
        this.scoreCombinationFactory = scoreCombinationFactory;
    }
}
