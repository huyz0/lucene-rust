/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.opensearch.search.aggregations.AggregationInitializationException;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.aggregations.BucketCollectorProcessor;
import org.opensearch.search.aggregations.ConcurrentAggregationProcessor;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.aggregations.GlobalAggCollectorManagerWithSingleCollector;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManagerWithSingleCollector;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.ReduceableSearchResult;

/**
 * {@code delegate}'s processing, with the aggregators of a search over a view of the writer's
 * buffers ({@link LiveDirectoryReader}) built only when their collector is first asked for: such a
 * search's reader is new, so building them builds its keyword fields' global ordinals again, which
 * the native pass that answers most aggregations never reads; and a search Lucene answers gives
 * Lucene the buffered documents first ({@link LiveDirectoryReader#readByLucene}), which aggregators
 * built before then would not see ({@link #rebuild}).
 *
 * <p>OpenSearch builds them up front so that a failure to build them fails the request before it
 * runs; deferred, the same failure fails it as it runs. A request Lucene collects builds its
 * aggregators when the query phase asks for their collector, and reduces them as OpenSearch's own
 * processor would.
 */
final class DeferredAggregationProcessor implements AggregationProcessor {
    private static final List<Class<?>> KEYS = List.of(NonGlobalAggCollectorManager.class, GlobalAggCollectorManager.class);

    private final AggregationProcessor delegate;

    DeferredAggregationProcessor(AggregationProcessor delegate) {
        this.delegate = delegate;
    }

    @Override
    public void preProcess(SearchContext context) {
        if (context.aggregations() == null || LiveDirectoryReader.find(context.searcher().getIndexReader()) == null) {
            delegate.preProcess(context);
            return;
        }
        // What the delegate's preProcess does, but for building the managers.
        context.setBucketCollectorProcessor(new BucketCollectorProcessor());
        boolean concurrent = delegate instanceof ConcurrentAggregationProcessor;
        if (context.aggregations().factories().hasNonGlobalAggregator()) {
            context.queryCollectorManagers()
                .put(
                    NonGlobalAggCollectorManager.class,
                    new Deferred(
                        () -> concurrent
                            ? new NonGlobalAggCollectorManager(context)
                            : new NonGlobalAggCollectorManagerWithSingleCollector(context)
                    )
                );
        }
        if (context.aggregations().factories().hasGlobalAggregator()) {
            context.queryCollectorManagers()
                .put(
                    GlobalAggCollectorManager.class,
                    new Deferred(
                        () -> concurrent ? new GlobalAggCollectorManager(context) : new GlobalAggCollectorManagerWithSingleCollector(context)
                    )
                );
        }
    }

    @Override
    public void postProcess(SearchContext context) {
        for (Class<?> key : KEYS) {
            // The global aggregations are still there only when nothing answered them; the others
            // when Lucene collected them or nothing answered. The delegate reduces (or runs) the
            // managers it knows, as it would have.
            if (context.queryCollectorManagers().get(key) instanceof Deferred d
                && (key == GlobalAggCollectorManager.class || d.isBuilt() || context.queryResult().hasAggs() == false)) {
                try {
                    context.queryCollectorManagers().put(key, d.manager());
                } catch (IOException e) {
                    throw new AggregationInitializationException("Could not initialize aggregators", e);
                }
            }
        }
        delegate.postProcess(context);
    }

    /**
     * Has the aggregators built so far built again when a collector is next asked for: the reader
     * changed under them ({@link LiveDirectoryReader#readByLucene}).
     */
    static void rebuild(SearchContext context) {
        for (Class<?> key : KEYS) {
            if (context.queryCollectorManagers().get(key) instanceof Deferred d) {
                d.reset();
            }
        }
    }

    /** What builds a collector manager. */
    @FunctionalInterface
    interface Builder {
        CollectorManager<Collector, ReduceableSearchResult> build() throws IOException;
    }

    /** A collector manager the delegate's preProcess would have registered, built on its first collector. */
    static final class Deferred implements CollectorManager<Collector, ReduceableSearchResult> {
        private final Builder builder;
        private CollectorManager<Collector, ReduceableSearchResult> built;

        Deferred(Builder builder) {
            this.builder = builder;
        }

        /** Whether the aggregators were built. */
        synchronized boolean isBuilt() {
            return built != null;
        }

        /** Drops the aggregators built: the next call builds them again. */
        synchronized void reset() {
            built = null;
        }

        /** The manager, built on the first call. */
        synchronized CollectorManager<Collector, ReduceableSearchResult> manager() throws IOException {
            if (built == null) {
                built = builder.build();
            }
            return built;
        }

        @Override
        public Collector newCollector() throws IOException {
            return manager().newCollector();
        }

        @Override
        public ReduceableSearchResult reduce(Collection<Collector> collectors) throws IOException {
            return manager().reduce(collectors);
        }
    }
}
