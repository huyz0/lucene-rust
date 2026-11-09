/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural;

import org.lucenerust.opensearch.neural.executors.HybridQueryExecutor;
import org.lucenerust.opensearch.neural.processor.ExplanationResponseProcessor;
import org.lucenerust.opensearch.neural.processor.NormalizationProcessor;
import org.lucenerust.opensearch.neural.processor.NormalizationProcessorWorkflow;
import org.lucenerust.opensearch.neural.processor.RRFProcessor;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombinationFactory;
import org.lucenerust.opensearch.neural.processor.combination.ScoreCombiner;
import org.lucenerust.opensearch.neural.processor.factory.ExplanationResponseProcessorFactory;
import org.lucenerust.opensearch.neural.processor.factory.NormalizationProcessorFactory;
import org.lucenerust.opensearch.neural.processor.factory.RRFProcessorFactory;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizationFactory;
import org.lucenerust.opensearch.neural.processor.normalization.ScoreNormalizer;
import org.lucenerust.opensearch.neural.query.HybridQueryBuilder;
import org.lucenerust.opensearch.neural.search.HybridQuerySearchRequestFilter;
import org.lucenerust.opensearch.neural.search.collector.HybridQueryCollectorContextSpecFactory;
import org.opensearch.action.support.ActionFilter;
import org.opensearch.common.settings.Settings;
import org.opensearch.plugins.SearchPlugin.QuerySpec;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchPhaseResultsProcessor;
import org.opensearch.search.pipeline.SearchResponseProcessor;
import org.opensearch.search.query.QueryCollectorContextSpecFactory;
import org.opensearch.threadpool.ExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;

import java.util.List;
import java.util.Map;

/**
 * neural-search's hybrid search, as its plugin registers it: the {@code hybrid} query, the
 * {@code normalization-processor} and {@code score-ranker-processor} search phase results
 * processors, the {@code hybrid_score_explanation} response processor, the collector that keeps
 * each sub-query's top hits, the request filter, and the thread pool sub-queries rewrite on.
 */
public final class HybridSearch {
    private HybridSearch() {}

    public static QuerySpec<?> query() {
        return new QuerySpec<>(HybridQueryBuilder.NAME, HybridQueryBuilder::new, HybridQueryBuilder::fromXContent);
    }

    public static Map<String, Processor.Factory<SearchPhaseResultsProcessor>> phaseResultsProcessors() {
        NormalizationProcessorWorkflow workflow = new NormalizationProcessorWorkflow(new ScoreNormalizer(), new ScoreCombiner());
        ScoreNormalizationFactory normalizations = new ScoreNormalizationFactory();
        ScoreCombinationFactory combinations = new ScoreCombinationFactory();
        return Map.of(
            NormalizationProcessor.TYPE,
            new NormalizationProcessorFactory(workflow, normalizations, combinations),
            RRFProcessor.TYPE,
            new RRFProcessorFactory(workflow, normalizations, combinations)
        );
    }

    public static Map<String, Processor.Factory<SearchResponseProcessor>> responseProcessors() {
        return Map.of(ExplanationResponseProcessor.TYPE, new ExplanationResponseProcessorFactory());
    }

    public static QueryCollectorContextSpecFactory collectorContextSpecFactory() {
        return new HybridQueryCollectorContextSpecFactory();
    }

    public static ActionFilter requestFilter() {
        return new HybridQuerySearchRequestFilter();
    }

    public static ExecutorBuilder<?> executorBuilder(Settings settings) {
        return HybridQueryExecutor.getExecutorBuilder(settings);
    }

    public static void initialize(ThreadPool threadPool) {
        HybridQueryExecutor.initialize(threadPool);
    }
}
