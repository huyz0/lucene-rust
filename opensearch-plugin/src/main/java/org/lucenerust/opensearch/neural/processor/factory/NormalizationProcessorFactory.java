/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.factory;

import static org.opensearch.ingest.ConfigurationUtils.readOptionalMap;
import static org.opensearch.ingest.ConfigurationUtils.readStringProperty;
import java.util.Map;
import java.util.Objects;
import org.lucenerust.opensearch.neural.processor.NormalizationProcessor;
import org.lucenerust.opensearch.neural.processor.NormalizationProcessorWorkflow;
import org.lucenerust.opensearch.neural.processor.dto.TechniqueCompatibilityCheckDTO;
import org.lucenerust.opensearch.neural.processor.combination.ArithmeticMeanScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationFactory;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.MinMaxScoreNormalizationTechnique;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationFactory;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationTechnique;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchPhaseResultsProcessor;

/**
 * Factory for query results normalization processor for search pipeline. Instantiates processor based on user provided input.
 */
public class NormalizationProcessorFactory implements Processor.Factory<SearchPhaseResultsProcessor> {
    @java.lang.SuppressWarnings("all")
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(NormalizationProcessorFactory.class);
    public static final String NORMALIZATION_CLAUSE = "normalization";
    public static final String COMBINATION_CLAUSE = "combination";
    public static final String TECHNIQUE = "technique";
    public static final String PARAMETERS = "parameters";
    private final NormalizationProcessorWorkflow normalizationProcessorWorkflow;
    private ScoreNormalizationFactory scoreNormalizationFactory;
    private ScoreCombinationFactory scoreCombinationFactory;

    @Override
    public SearchPhaseResultsProcessor create(final Map<String, Processor.Factory<SearchPhaseResultsProcessor>> processorFactories, final String tag, final String description, final boolean ignoreFailure, final Map<String, Object> config, final Processor.PipelineContext pipelineContext) throws Exception {
        Map<String, Object> normalizationClause = readOptionalMap(NormalizationProcessor.TYPE, tag, config, NORMALIZATION_CLAUSE);
        ScoreNormalizationTechnique normalizationTechnique = ScoreNormalizationFactory.DEFAULT_METHOD;
        if (Objects.nonNull(normalizationClause)) {
            String normalizationTechniqueName = readStringProperty(NormalizationProcessor.TYPE, tag, normalizationClause, TECHNIQUE, MinMaxScoreNormalizationTechnique.TECHNIQUE_NAME);
            Map<String, Object> normalizationParams = readOptionalMap(NormalizationProcessor.TYPE, tag, normalizationClause, PARAMETERS);
            normalizationTechnique = scoreNormalizationFactory.createNormalization(normalizationTechniqueName, normalizationParams);
        }
        Map<String, Object> combinationClause = readOptionalMap(NormalizationProcessor.TYPE, tag, config, COMBINATION_CLAUSE);
        ScoreCombinationTechnique scoreCombinationTechnique = ScoreCombinationFactory.DEFAULT_METHOD;
        if (Objects.nonNull(combinationClause)) {
            String combinationTechnique = readStringProperty(NormalizationProcessor.TYPE, tag, combinationClause, TECHNIQUE, ArithmeticMeanScoreCombinationTechnique.TECHNIQUE_NAME);
            // check for optional combination params
            Map<String, Object> combinationParams = readOptionalMap(NormalizationProcessor.TYPE, tag, combinationClause, PARAMETERS);
            scoreCombinationTechnique = scoreCombinationFactory.createCombination(combinationTechnique, combinationParams);
        }
        TechniqueCompatibilityCheckDTO techniqueCompatibilityCheckDTO = TechniqueCompatibilityCheckDTO.builder().scoreNormalizationTechnique(normalizationTechnique).scoreCombinationTechnique(scoreCombinationTechnique).build();
        scoreNormalizationFactory.isTechniquesCompatible(techniqueCompatibilityCheckDTO);
        log.info("Creating search phase results processor of type [{}] with normalization [{}] and combination [{}]", NormalizationProcessor.TYPE, normalizationTechnique, scoreCombinationTechnique);
        return new NormalizationProcessor(tag, description, normalizationTechnique, scoreCombinationTechnique, normalizationProcessorWorkflow);
    }

    @java.lang.SuppressWarnings("all")
    public NormalizationProcessorFactory(final NormalizationProcessorWorkflow normalizationProcessorWorkflow, final ScoreNormalizationFactory scoreNormalizationFactory, final ScoreCombinationFactory scoreCombinationFactory) {
        this.normalizationProcessorWorkflow = normalizationProcessorWorkflow;
        this.scoreNormalizationFactory = scoreNormalizationFactory;
        this.scoreCombinationFactory = scoreCombinationFactory;
    }
}
