/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.util.Collection;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.opensearch.search.aggregations.AggregationInitializationException;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.aggregations.BucketCollectorProcessor;
import org.opensearch.search.aggregations.ConcurrentAggregationProcessor;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManagerWithSingleCollector;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.ReduceableSearchResult;

/**
 * {@code delegate}'s processing, with the aggregators of a search over a view of the writer's
 * buffers ({@link LiveDirectoryReader}) built only when Lucene collects them: such a search's
 * reader is new, so building them builds its keyword fields' global ordinals again, which the
 * native pass ({@link NativeAggregations}) that answers the aggregations never reads.
 *
 * <p>Only aggregations {@link NativeAggregations#plan} takes are deferred: OpenSearch builds the
 * others up front so that a failure to build them fails the request before it runs, and the native
 * pass does not answer them anyway. A deferred request that falls back to Lucene builds its
 * aggregators when the query phase asks for their collector, as OpenSearch's own processor would
 * have, and reduces them as it would.
 */
final class DeferredAggregationProcessor implements AggregationProcessor {
    private final AggregationProcessor delegate;

    DeferredAggregationProcessor(AggregationProcessor delegate) {
        this.delegate = delegate;
    }

    @Override
    public void preProcess(SearchContext context) {
        if (context.aggregations() == null
            || LiveDirectoryReader.find(context.searcher().getIndexReader()) == null
            || NativeAggregations.plan(context) == null) {
            delegate.preProcess(context);
            return;
        }
        // What the delegate's preProcess does, but for building the collector: a plan has no
        // global aggregator, and every aggregation it takes is a non-global one.
        context.setBucketCollectorProcessor(new BucketCollectorProcessor());
        context.queryCollectorManagers()
            .put(NonGlobalAggCollectorManager.class, new Deferred(context, delegate instanceof ConcurrentAggregationProcessor));
    }

    @Override
    public void postProcess(SearchContext context) {
        if (context.queryCollectorManagers().get(NonGlobalAggCollectorManager.class) instanceof Deferred d
            && (d.isBuilt() || context.queryResult().hasAggs() == false)) {
            // Lucene collected them, or nothing answered: the delegate reduces the manager it
            // knows, as it would have.
            try {
                context.queryCollectorManagers().put(NonGlobalAggCollectorManager.class, d.manager());
            } catch (IOException e) {
                throw new AggregationInitializationException("Could not initialize aggregators", e);
            }
        }
        delegate.postProcess(context);
    }

    /** The non-global aggregations' collector manager, built on its first collector. */
    static final class Deferred implements CollectorManager<Collector, ReduceableSearchResult> {
        private final SearchContext context;
        private final boolean concurrent;
        private CollectorManager<Collector, ReduceableSearchResult> built;

        Deferred(SearchContext context, boolean concurrent) {
            this.context = context;
            this.concurrent = concurrent;
        }

        /** Whether the aggregators were built. */
        boolean isBuilt() {
            return built != null;
        }

        /** The manager the delegate's preProcess would have registered, built on the first call. */
        synchronized CollectorManager<Collector, ReduceableSearchResult> manager() throws IOException {
            if (built == null) {
                built = concurrent ? new NonGlobalAggCollectorManager(context) : new NonGlobalAggCollectorManagerWithSingleCollector(context);
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
