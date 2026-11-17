/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.lucenerust.opensearch.neural.search.query.HybridAggregationProcessor;
import org.lucenerust.opensearch.neural.search.query.HybridQueryPhaseSearcher;
import org.lucenerust.opensearch.painless.PainlessScore;
import org.apache.lucene.search.Query;
import org.opensearch.common.settings.Setting;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryPhaseExecutionException;
import org.opensearch.search.query.QueryPhaseSearcher;
import org.opensearch.search.query.QueryPhaseSearcherWrapper;

import java.io.IOException;
import java.util.LinkedList;

/**
 * Runs the query phase of a shard search natively when the whole request is inside the supported
 * matrix, and on Lucene otherwise -- per query, transparently.
 *
 * <p>This hooks {@link QueryPhaseSearcher}, the extension point OpenSearch gives a plugin for the
 * query phase, rather than an {@code EngineFactory}: the query phase is the only part of a search
 * this milestone moves, and hooking it leaves indexing, refresh, fetch and every unsupported request
 * exactly on OpenSearch's own code. The fallback is OpenSearch's own default ({@link
 * QueryPhaseSearcherWrapper}, which also picks concurrent segment search when the index asks for
 * it), so an unsupported request runs precisely as it would without this plugin.
 *
 * <p>A request runs native when it is a top-hits search -- by score, or by a sort {@link
 * SortEncoder} can encode, with or without {@code search_after}, {@code post_filter}, {@code timeout},
 * scroll, {@code terminate_after} and {@code min_score} -- with no collapse, rescore or profile, no aggregation beyond the
 * metrics {@link NativeAggregations} plans, and a query {@link QueryEncoder} can encode over fields
 * whose similarities {@link SimilarityEncoder} can send. What it produces is what OpenSearch's own {@code
 * SimpleTopDocsCollectorContext} would: the top {@code from + size} hits (with their sort values),
 * the max score, and total hits under the same {@code track_total_hits} rules; and, with
 * aggregations, the shard results their aggregators would have built, stored before {@code
 * DefaultAggregationProcessor.postProcess} runs so that it keeps them rather than reducing its own
 * (unfed) collectors.
 *
 * <p>A native failure is logged, counted, and the query is re-run on Lucene: it never fails a
 * search that Lucene could answer.
 */
public final class AcceleratedQueryPhaseSearcher implements QueryPhaseSearcher {
    private static final Logger logger = LogManager.getLogger(AcceleratedQueryPhaseSearcher.class);

    /** Per-index switch, dynamic: {@code PUT idx/_settings {"index.search.accelerated": false}}. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "index.search.accelerated",
        true,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * Which encodable shapes run native: {@code fast} (default) only those measured at least as fast
     * as Lucene ({@link QueryEncoder#isFast}); {@code all} every shape the engine answers correctly.
     * The two agree today, since every encodable shape is measured faster.
     */
    public static final Setting<String> NATIVE_SHAPES = new Setting<>(
        "index.search.accelerated_shapes",
        "fast",
        v -> {
            if (v.equals("fast") == false && v.equals("all") == false) {
                throw new IllegalArgumentException("index.search.accelerated_shapes must be [fast] or [all], got [" + v + "]");
            }
            return v;
        },
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * Per-index opt-in, dynamic: aggregations keep each committed segment's collected part in the
     * node's aggregation segment cache ({@code indices.aggregation_segment_cache.size}),
     * so a dashboard repeated after a refresh collects only the segments the refresh added. Off by
     * default; results are the same either way.
     */
    public static final Setting<Boolean> AGG_SEGMENT_CACHE = Setting.boolSetting(
        "index.search.aggregation_segment_cache.enabled",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * Whether a match-all or range OpenSearch resolved to its approximation (a sort on the
     * range's field, {@code track_total_hits} not exact) runs native: the exact top hits, by
     * the native bounded newest-N search. OpenSearch's approximation collects whole BKD leaves
     * in value order and keeps the top hits of those, so where the last hit's value is tied
     * across a leaf boundary it may keep other documents of that value than the exact answer
     * does (the same values, other ties): the answer OpenSearch gives without the approximation.
     * On by default, since it is 4-5x faster on a sorted match-all or range; {@code false} keeps
     * OpenSearch's approximate answer.
     */
    public static final Setting<Boolean> NATIVE_APPROXIMATE = Setting.boolSetting(
        "index.search.approximation.exact",
        true,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    private final QueryPhaseSearcher fallback = new QueryPhaseSearcherWrapper();
    /** neural-search's searcher, for a query with a {@code hybrid} query anywhere in it. */
    private final org.lucenerust.opensearch.neural.search.query.HybridQueryPhaseSearcher hybrid =
        new org.lucenerust.opensearch.neural.search.query.HybridQueryPhaseSearcher();
    private final SearchStats stats;
    private final NativeExecution execution;
    private final HybridPhase hybridPhase;

    public AcceleratedQueryPhaseSearcher(NativeReaders readers, SearchStats stats) {
        this.stats = stats;
        this.execution = new NativeExecution(readers, stats);
        this.hybridPhase = new HybridPhase(readers, stats);
    }

    @Override
    public AggregationProcessor aggregationProcessor(SearchContext searchContext) {
        AggregationProcessor processor = fallback.aggregationProcessor(searchContext);
        Query query = searchContext.query();
        DeferredAggregationProcessor deferred = new DeferredAggregationProcessor(processor);
        return query != null && containsHybrid(query) ? new HybridAggregationProcessor(deferred) : deferred;
    }

    @Override
    public boolean searchWith(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        PainlessScore.beginRequest(() -> ctx.request().source() == null ? null : ctx.request().source().toString());
        // QueryPhase took the timeout's start just before calling here.
        NativeMemory.queryStarted(hasTimeout ? ctx.timeout().nanos() : 0);
        try {
            NativeProfile.begin(ctx);
            return containsHybrid(query)
                ? searchHybridOrLucene(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout)
                : searchPlain(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
        } finally {
            NativeProfile.end();
            NativeMemory.queryStarted(0);
            PainlessScore.endRequest();
        }
    }

    private boolean searchHybridOrLucene(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        // As HybridQueryPhaseSearcher: the hybrid query out of its filters and DLS rules; a collapse's
        // inner hits fan out to a plain bool query, searched as any other.
        Query phaseQuery = hybrid.phaseQuery(ctx, query);
        if (phaseQuery instanceof HybridQuery == false) {
            return searchPlain(ctx, searcher, phaseQuery, collectors, hasFilterCollector, hasTimeout);
        }
        long nativeStart = System.nanoTime();
        String why = hybridPhase.search(ctx, searcher, phaseQuery, collectors, hasFilterCollector, hasTimeout);
        if (why == null) {
            stats.nativeQuery();
            stats.nativeTime(System.nanoTime() - nativeStart);
            return false;
        }
        if (FallbackReason.ACCELERATED_ERROR.key().equals(why) && NativeExecution.scriptFailed()) {
            throw execution.scriptFailure(searcher.getIndexReader(), false);
        }
        stats.fallback(why);
        long luceneStart = System.nanoTime();
        LiveDirectoryReader view = LiveDirectoryReader.find(searcher.getIndexReader());
        if (view != null) {
            // The sub-queries' rewrites are neural-search's own, kept as they are.
            execution.readByLucene(ctx, searcher, view);
        }
        boolean rescore = hybrid.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
        stats.luceneTime(System.nanoTime() - luceneStart);
        return rescore;
    }

    private boolean searchPlain(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        long start = System.nanoTime();
        // Aggregations OpenSearch's DefaultAggregationProcessor.preProcess registered (their one
        // collector context is the one in "collectors"), when all of them run natively.
        NativeAggregations.Plan aggs = ctx.queryCollectorManagers().isEmpty() ? null : NativeAggregations.plan(ctx);
        if (aggs != null && execution.termsFromDocFreqs(ctx, searcher, query, aggs, collectors, hasFilterCollector)) {
            stats.nativeQuery();
            stats.nativeTime(System.nanoTime() - start);
            return false;
        }
        // Anything else -- bucket aggregations, sub-aggregations, global -- as a native tree.
        NativeAggregationTree.Tree tree = aggs != null || ctx.queryCollectorManagers().isEmpty() || SearchFlags.of(ctx).enabled() == false
            ? null
            : NativeAggregationTree.plan(ctx, field -> true);
        String reason = SearchEligibility.ineligible(ctx, collectors, hasFilterCollector, aggs != null || tree != null);
        if (reason == null || FallbackReason.SEARCH_AFTER.key().equals(reason)) {
            // A request Lucene's collector refuses: its error, as QueryPhase raises it.
            RuntimeException refused = SortEncoder.searchAfterRefusal(ctx.sort() == null ? null : ctx.sort().sort, ctx.searchAfter(), ctx.size(), searcher.getIndexReader());
            if (refused != null) {
                throw new QueryPhaseExecutionException(ctx.shardTarget(), "Failed to execute main query", refused);
            }
        }
        PlainSearch plan = new PlainSearch(ctx, searcher, query);
        if (reason == null) {
            reason = plan.encodeQuery();
        }
        if (reason == null) {
            reason = plan.encodeRescores();
        }
        if (reason == null) {
            reason = plan.refuseApproximate();
        }
        if (reason == null) {
            reason = plan.encodeSort();
        }
        if (reason == null) {
            reason = execution.executeNative(plan, aggs, tree, hasTimeout);
            if (reason == null) {
                stats.nativeQuery();
                if (plan.memoryTimedOut || hasTimeout && SearchSupport.deadlinePassed(ctx)) {
                    SearchSupport.timedOut(ctx);
                }
                stats.nativeTime(System.nanoTime() - start);
                if (NativeProfile.requested(ctx)) {
                    SearchProfiling.profile(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout, plan.searchedHandle, plan.blob, SearchProfiling.similarities(searcher), System.nanoTime() - start);
                }
                return false;
            }
        }
        // A native search that ran a Painless script fails as the script did: running it again in
        // Java would run the script twice.
        if (FallbackReason.ACCELERATED_ERROR.key().equals(reason) && NativeExecution.scriptFailed()) {
            throw execution.scriptFailure(searcher.getIndexReader(), true);
        }
        stats.fallback(reason);
        // Lucene's own time only: not the plugin's checks, nor a native attempt that fell back.
        long luceneStart = System.nanoTime();
        LiveDirectoryReader view = LiveDirectoryReader.find(searcher.getIndexReader());
        if (view != null && execution.readByLucene(ctx, searcher, view) && query == ctx.query() && ctx.parsedQuery() != null) {
            // Rewritten again as SearchContext.preProcess rewrote it: a rewrite that read the
            // reader's terms (a scoring multi-term query, blended term statistics) read them
            // without the buffered documents'.
            query = searcher.rewrite(ctx.buildFilteredQuery(ctx.parsedQuery().query()));
        }
        boolean rescore = fallback.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
        stats.luceneTime(System.nanoTime() - luceneStart);
        return rescore;
    }

    /** Whether {@code query} is or holds a {@link HybridQuery}. */
    static boolean containsHybrid(Query query) {
        return org.lucenerust.opensearch.neural.util.HybridQueryUtil.containsHybridQuery(query);
    }

}
