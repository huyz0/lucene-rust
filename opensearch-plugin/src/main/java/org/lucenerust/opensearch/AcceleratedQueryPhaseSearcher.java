/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.lucenerust.opensearch.neural.query.HybridQuery;
import org.lucenerust.opensearch.neural.search.query.HybridAggregationProcessor;
import org.lucenerust.opensearch.neural.search.query.HybridQueryPhaseSearcher;
import org.lucenerust.opensearch.painless.PainlessScore;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.common.settings.Setting;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.internal.ScrollContext;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryPhaseExecutionException;
import org.opensearch.search.query.QueryPhaseSearcher;
import org.opensearch.search.query.QueryPhaseSearcherWrapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

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
    private final NativeReaders readers;
    private final SearchStats stats;

    public AcceleratedQueryPhaseSearcher(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
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

    /**
     * Whether the native search failed because a Painless script of this request threw (or
     * scored invalidly): the search then fails as the script does in Java, never re-run there.
     * Any other native failure falls back to Lucene as before.
     */
    private static boolean scriptFailed() {
        return PainlessScore.ran() && PainlessScore.scriptFailed(NativeBridge.lastError());
    }

    /**
     * The exception a script's failure ({@link #scriptFailed}) raises ({@link
     * PainlessScore#failure}); {@code counted} when the failure was already counted as a native
     * error.
     */
    private RuntimeException scriptFailure(IndexReader reader, boolean counted) {
        stats.scriptError(counted);
        return PainlessScore.failure(NativeBridge.lastError(), reader);
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
        String why = searchHybrid(ctx, searcher, phaseQuery, collectors, hasFilterCollector, hasTimeout);
        if (why == null) {
            stats.nativeQuery();
            stats.nativeTime(System.nanoTime() - nativeStart);
            return false;
        }
        if (FallbackReason.NATIVE_ERROR.key().equals(why) && scriptFailed()) {
            throw scriptFailure(searcher.getIndexReader(), false);
        }
        stats.fallback(why);
        long luceneStart = System.nanoTime();
        LiveDirectoryReader view = LiveDirectoryReader.find(searcher.getIndexReader());
        if (view != null) {
            // The sub-queries' rewrites are neural-search's own, kept as they are.
            readByLucene(ctx, searcher, view);
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
        if (aggs != null && termsFromDocFreqs(ctx, searcher, query, aggs, collectors, hasFilterCollector)) {
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
            reason = executeNative(plan, aggs, tree, hasTimeout);
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
        if (FallbackReason.NATIVE_ERROR.key().equals(reason) && scriptFailed()) {
            throw scriptFailure(searcher.getIndexReader(), true);
        }
        stats.fallback(reason);
        // Lucene's own time only: not the plugin's checks, nor a native attempt that fell back.
        long luceneStart = System.nanoTime();
        LiveDirectoryReader view = LiveDirectoryReader.find(searcher.getIndexReader());
        if (view != null && readByLucene(ctx, searcher, view) && query == ctx.query() && ctx.parsedQuery() != null) {
            // Rewritten again as SearchContext.preProcess rewrote it: a rewrite that read the
            // reader's terms (a scoring multi-term query, blended term statistics) read them
            // without the buffered documents'.
            query = searcher.rewrite(ctx.buildFilteredQuery(ctx.parsedQuery().query()));
        }
        boolean rescore = fallback.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
        stats.luceneTime(System.nanoTime() - luceneStart);
        return rescore;
    }

    /**
     * Runs {@code p}'s request on the native engine: null when it did, else the reason it did not.
     * The writer's buffered documents are read too when the searcher has a view of them.
     */
    private String executeNative(PlainSearch p, NativeAggregations.Plan aggs, NativeAggregationTree.Tree tree, boolean hasTimeout)
        throws IOException {
        SearchContext ctx = p.ctx;
        ContextIndexSearcher searcher = p.searcher;
        NativeReaders.Acquired acquired = readers.acquire(searcher.getIndexReader());
        String reason = acquired.fallbackReason();
        if (reason == null) {
            // A timeout already past: Lucene would skip every segment and answer nothing, timed
            // out -- which is what its own path does fastest.
            if (ctx.isCancelled()) {
                reason = FallbackReason.CANCELLED.key();
            } else if (hasTimeout && SearchSupport.deadlinePassed(ctx)) {
                reason = FallbackReason.TIMEOUT.key();
            } else {
                // What the native calls allocate is charged to the request breaker while they run.
                NativeMemory.Scope memory = NativeMemory.open(ctx, "accelerated_search");
                try {
                    reason = searchReaderAndBuffers(p, acquired, aggs, tree);
                } finally {
                    memory.close();
                    stats.nativeMemory(memory.peak());
                }
                p.memoryTimedOut = memory.timedOut();
            }
        }
        return reason;
    }

    /**
     * The native search of {@code p}'s request over the reader, and over the writer's buffered
     * documents too when the searcher has a view of them and the request is one the buffers
     * answer: null when it ran, else the fallback reason.
     */
    private String searchReaderAndBuffers(
        PlainSearch p,
        NativeReaders.Acquired acquired,
        NativeAggregations.Plan aggs,
        NativeAggregationTree.Tree tree
    ) throws IOException {
        SearchContext ctx = p.ctx;
        ContextIndexSearcher searcher = p.searcher;
        Query query = p.query;
        byte[] blob = p.blob;
        byte[] hitsBlob = p.hitsBlob;
        byte[] sortBlob = p.sortBlob;
        byte[] collapseBlob = p.collapseBlob;
        List<NativeRescore> rescores = p.rescores;
        Sort sort = p.sort;
        boolean scoreDocs = p.scoreDocs;
        int terminateAfter = p.terminateAfter;
        String reason = null;
        // The writer's buffered documents too, when the searcher has a view of them and
        // the request is one the buffers answer; otherwise the reader's documents alone.
        LiveDirectoryReader live = LiveDirectoryReader.find(searcher.getIndexReader());
        p.searchedHandle = acquired.handle();
        boolean plain = blob == hitsBlob && aggs == null && tree == null && sortBlob == null && collapseBlob == null
            && rescores == null && sort == null && terminateAfter == 0;
        if (live == null) {
            reason = searchNative(ctx, acquired.handle(), blob, hitsBlob, sortBlob, collapseBlob, rescores, sort, scoreDocs, aggs, tree, false);
        } else if (SearchEligibility.liveIneligible(ctx) != null) {
            // Lucene sees the view's documents as deleted and answers from the reader's.
            reason = FallbackReason.REALTIME_VIEW.key();
        } else if (plain && SearchEligibility.plainIneligible(ctx) == false && query instanceof KnnHitsQuery hits) {
            // A k-NN search's hits, found when the query was rewritten: nothing of the
            // buffers is read again.
            storeKnnHits(ctx, hits);
        } else if (plain == false || SearchEligibility.plainIneligible(ctx) || searchLive(ctx, acquired.handle(), live, blob) == false) {
            // Everything else reads the buffers written as segments, one per leaf of the
            // view: aggregations, sorts, collapse, rescore, and what the plain path's
            // buffers cannot answer.
            if (ctx.mapperService().hasNested() || ctx.explain()) {
                // The fetch phase finds a buffered hit's root document by doc values the
                // view's leaves read only from the buffers given to Lucene as segments,
                // and explains a hit by their postings.
                readByLucene(ctx, searcher, live);
            }
            long[] whole = new long[1];
            long built = System.nanoTime();
            // The request's terms only: the hits query's (the post_filter's terms with the
            // query's), the rescore queries' and the aggregation tree's.
            byte[] request = hitsBlob;
            byte[][] rescoreBlobs = rescores == null ? null : rescores.stream().map(NativeRescore::blob).toArray(byte[][]::new);
            int[][] treeSlices = tree == null ? null : NativeAggregations.slices(ctx);
            byte[] treeBlob = treeSlices == null ? null : tree.blob(treeSlices);
            if (tree != null && treeBlob == null) {
                request = null;
            }
            int rc = NativeLive.liveViewReader(acquired.handle(), live.view().id(), request, treeBlob, rescoreBlobs, whole);
            if (rc == NativeBridge.OK) {
                stats.liveReaderTime(System.nanoTime() - built);
            }
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                logger.debug("accelerated-search: the buffers cannot be read as segments ({}): {}", rc, NativeBridge.lastError());
                reason = FallbackReason.REALTIME_VIEW.key();
            } else {
                p.searchedHandle = whole[0];
                reason = searchNative(ctx, whole[0], blob, hitsBlob, sortBlob, collapseBlob, rescores, sort, scoreDocs, aggs, tree, true);
                if (reason == null) {
                    stats.liveQuery();
                }
            }
        }
        return reason;
    }

    /**
     * A match-all {@code size: 0} search whose only aggregation is a plain keyword {@code terms}
     * ({@link TermsDocFreqShortcut}): its buckets from the segments' term document frequencies and
     * its total from {@code numDocs}, as OpenSearch answers it. False, with nothing stored, for any
     * other request.
     */
    private boolean termsFromDocFreqs(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        NativeAggregations.Plan aggs,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector
    ) {
        if (aggs.entries().size() != 1 || aggs.terms().size() != 1 || ctx.size() != 0 || ctx.from() != 0 || ctx.sort() != null
            || ctx.searchAfter() != null || ctx.scrollContext() != null || ctx.collapse() != null || ctx.parsedPostFilter() != null
            || ctx.minimumScore() != null || ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER || hasFilterCollector
            || ctx.isCancelled() || NativeProfile.requested(ctx) || SearchSupport.unwrapShortcut(query).getClass() != MatchAllDocsQuery.class
            || NativeRescore.applies(ctx) || SearchFlags.of(ctx).enabled() == false
            || LiveDirectoryReader.find(searcher.getIndexReader()) != null
            || SearchEligibility.ineligible(ctx, collectors, hasFilterCollector, true) != null) {
            return false;
        }
        int[][] slices = NativeAggregations.slices(ctx);
        if (slices == null) {
            return false;
        }
        NativeAggregations.Terms terms = aggs.terms().get(0);
        byte[] result;
        try {
            result = TermsDocFreqShortcut.result(searcher.getIndexReader(), terms.field(), terms.thresholds().getShardSize(), slices);
        } catch (IOException | RuntimeException e) {
            return false;
        }
        if (result == null) {
            return false;
        }
        InternalAggregations built = aggs.build(slices, new long[0], new double[0], result, ctx.partialOnShard());
        NativeAggregations.consumeBuckets(ctx, built);
        int shortcut;
        try {
            shortcut = ctx.trackTotalHitsUpTo() == SearchContext.TRACK_TOTAL_HITS_DISABLED
                ? -1
                : SearchSupport.shortcutTotalHitCount(searcher.getIndexReader(), query, 0L);
        } catch (IOException e) {
            shortcut = -1;
        }
        if (searchUnsorted(ctx, 0L, null, 0, 0, shortcut, null, null, null) != null) {
            return false;
        }
        ctx.queryResult().aggregations(built);
        return true;
    }

    /** How long a search Lucene answers waits for a buffer being flushed to be published. */
    private static final long FLUSH_WAIT_MILLIS = 30_000;

    /**
     * Gives Lucene the buffered documents of the searcher's view before it answers, so it finds
     * them as the native engine would: the buffers as segments, and the reader's documents deleted
     * since its commit left out. The aggregators built so far are built again over them. True when
     * Lucene reads them.
     */
    private boolean readByLucene(SearchContext ctx, ContextIndexSearcher searcher, LiveDirectoryReader view) throws IOException {
        if (view.readsLucene()) {
            // Another search gave them to Lucene, perhaps after this one rewrote its query or
            // built its aggregators: they are rebuilt over them all the same.
            DeferredAggregationProcessor.rebuild(ctx);
            return true;
        }
        NativeReaders.Acquired acquired = readers.acquire(searcher.getIndexReader());
        if (acquired.fallbackReason() != null) {
            // No native reader of the commit: nothing reads the buffers, natively or not.
            logger.debug("accelerated-search: the buffers cannot be read without a native reader ({})", acquired.fallbackReason());
            return false;
        }
        boolean read;
        try {
            read = view.readByLucene(acquired.handle(), FLUSH_WAIT_MILLIS, searcher.getIndexReader());
        } catch (IOException | RuntimeException e) {
            // Not the index's own files that could not be read: the search fails, the shard stays
            // (OpenSearch fails a shard whose search reports a corrupt index anywhere in its causes).
            logger.warn("accelerated-search: the buffered documents cannot be read by Lucene", e);
            throw new IllegalStateException("realtime search: the buffered documents cannot be read by Lucene: " + e);
        }
        if (read == false) {
            throw new IllegalStateException("realtime search: the buffered documents cannot be read by Lucene: " + NativeBridge.lastError());
        }
        DeferredAggregationProcessor.rebuild(ctx);
        return true;
    }

    /**
     * A hybrid query's shard phase run natively ({@link NativeHybrid}), or why it cannot be: the
     * stock checks first (search type, sub-queries), which fail the request as they do in
     * Java.
     */
    private String searchHybrid(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout
    ) throws IOException {
        if (SearchFlags.of(ctx).enabled() == false) return FallbackReason.DISABLED.key();
        // The aggregations as the plain path plans them; they run over the documents the hybrid
        // collection counted.
        long searchStart = System.nanoTime();
        NativeAggregations.Plan aggs = ctx.queryCollectorManagers().isEmpty() ? null : NativeAggregations.plan(ctx);
        NativeAggregationTree.Tree tree = aggs != null || ctx.queryCollectorManagers().isEmpty()
            ? null
            : NativeAggregationTree.plan(ctx, field -> true);
        String reason = NativeHybrid.ineligible(ctx, query, collectors, hasFilterCollector, aggs != null || tree != null);
        if (reason != null) return reason;
        HybridQuery hybrid = (HybridQuery) query;
        org.lucenerust.opensearch.neural.util.HybridQueryUtil.validateHybridQuerySearchType(ctx.searchType());
        org.lucenerust.opensearch.neural.util.HybridQueryUtil.validateHybridQuery(hybrid);
        int[][] slices = NativeAggregations.slices(ctx);
        if (slices == null) return FallbackReason.INTRA_SEGMENT.key();
        if (LiveDirectoryReader.find(searcher.getIndexReader()) != null) return FallbackReason.REALTIME_VIEW.key();
        SimilarityEncoder similarities = new SimilarityEncoder(searcher.getSimilarity());
        // The post filter is rewritten as QueryPhase rewrites it; an approximate range in it matches
        // only the first documents it visits, which the native engine would not reproduce.
        Query postFilter = ctx.parsedPostFilter() == null ? null : searcher.rewrite(ctx.parsedPostFilter().query());
        if (postFilter != null && SearchSupport.approximated(postFilter) && SearchSupport.buffered(searcher) == false && SearchFlags.of(ctx).nativeApproximate() == false) return FallbackReason.APPROXIMATE.key();
        NativeHybrid.Encoded enc = NativeHybrid.encode(hybrid, similarities, searcher, SearchFlags.of(ctx).allShapes(), EncodeContext.of(ctx, searcher.getTopReaderContext()));
        if (enc.fallbackReason() != null) return enc.fallbackReason();
        byte[] filterBlob = null;
        if (postFilter != null) {
            Query filter = postFilter;
            QueryEncoder.Encoded encoded = QueryEncoder.encode(filter, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            if (encoded.fallbackReason() != null) return encoded.fallbackReason();
            filterBlob = similarities.prefix(encoded.blob());
        }
        // The rescore queries (HybridSearchCollectorResultUtil rescores each sub-query's hits), encoded
        // before anything runs.
        List<NativeRescore> rescores = null;
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) {
            if (NativeRescore.supported(ctx) == false) return FallbackReason.RESCORE.key();
            rescores = new ArrayList<>();
            reason = NativeRescore.encode(ctx, searcher, new SimilarityEncoder(searcher.getSimilarity()), rescores);
            if (reason != null) return reason;
        }
        NativeReaders.Acquired acquired = readers.acquire(searcher.getIndexReader());
        if (acquired.fallbackReason() != null) return acquired.fallbackReason();
        if (ctx.isCancelled()) return FallbackReason.CANCELLED.key();
        if (hasTimeout && SearchSupport.deadlinePassed(ctx)) return FallbackReason.TIMEOUT.key();
        // HybridCollectorManager's own checks and sizes (pagination_depth, from on one shard).
        int topN = NativeHybrid.numHits(ctx, hybrid);
        // The hits sorted: by the request's sort, after the search_after document when there is one.
        // A collapse (HybridCollectorFactory) is sorted by the request's sort or the score, ignores
        // search_after, and groups by a keyword or number field.
        SortField[] sortFields = null;
        byte[] sortBlob = null;
        byte[] collapseBlob = null;
        String collapseField = null;
        if (ctx.collapse() != null) {
            MappedFieldType type = ctx.collapse().getFieldType();
            byte kind;
            if (type instanceof KeywordFieldMapper.KeywordFieldType) {
                kind = NativeCollapse.KEYWORD;
            } else if (type instanceof NumberFieldMapper.NumberFieldType) {
                kind = NativeCollapse.NUMERIC;
            } else {
                throw new IllegalStateException(
                    "unknown type for collapse field " + ctx.collapse().getFieldName() + ", only keywords and numbers are accepted"
                );
            }
            collapseField = ctx.collapse().getFieldName();
            java.io.ByteArrayOutputStream blob = new java.io.ByteArrayOutputStream();
            byte[] name = type.name().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            NativeAggregations.writeInt(blob, name.length);
            blob.writeBytes(name);
            blob.write(kind);
            collapseBlob = blob.toByteArray();
        }
        if (ctx.sort() != null || collapseBlob != null) {
            Sort keys = ctx.sort() != null ? ctx.sort().sort : new Sort(new SortField(null, SortField.Type.SCORE));
            sortFields = keys.getSort();
            FieldDoc after = collapseBlob != null ? null : ctx.searchAfter();
            if (after != null) {
                if (after.fields == null) {
                    throw new IllegalArgumentException("after.fields wasn't set; you must pass fillFields=true for the previous search");
                }
                if (after.fields.length != sortFields.length) {
                    throw new IllegalArgumentException(
                        "after.fields has " + after.fields.length + " values but sort has " + sortFields.length
                    );
                }
            }
            byte[] indexSorted = SortEncoder.indexSorted(searcher.getIndexReader(), keys);
            SortEncoder.Encoded sorted = SortEncoder.encode(keys, after, false, new int[0][], 0, false, indexSorted, EncodeContext.of(ctx));
            if (sorted.fallbackReason() != null) return sorted.fallbackReason();
            sortBlob = sorted.blob();
        }
        long threshold = Math.max(topN, ctx.trackTotalHitsUpTo());
        boolean aggregating = aggs != null || tree != null;
        byte[][] out = new byte[1][];
        int rc;
        NativeMemory.Scope memory = NativeMemory.open(ctx, "accelerated_hybrid_search");
        try {
            rc = NativeSearch.searchHybrid(acquired.handle(), NativeHybrid.blob(enc.blobs(), topN, slices, aggregating, filterBlob, sortBlob, threshold, collapseBlob, NativeHybrid.terminates(ctx) ? ctx.terminateAfter() : -1), out);
            NativeMemory.check(rc);
            if (rc != NativeBridge.OK) {
                logger.debug("accelerated-search: native hybrid search failed ({}): {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            NativeHybrid.Answer answer = NativeHybrid.read(out[0], enc.blobs().size(), aggregating, sortFields, collapseField, NativeHybrid.terminates(ctx));
            InternalAggregations[] aggOut = new InternalAggregations[1];
            InternalAggregations aggResult = null;
            if (aggregating) {
                String failed = hybridAggregations(ctx, searcher, acquired.handle(), similarities, answer.counted(), aggs, tree, slices, aggOut);
                if (failed != null) return failed;
                aggResult = aggOut[0];
            }
            // A sorted result is never rescored (the result util's sorted branch).
            if (rescores != null && sortFields == null) {
                reason = NativeHybrid.rescore(answer, acquired.handle(), rescores);
                if (reason != null) return reason;
            }
            NativeHybrid.store(ctx, answer, rescores != null && sortFields == null);
            if (NativeProfile.requested(ctx)) {
                SearchProfiling.profileHybrid(ctx, searcher, hybrid, query, collectors, hasFilterCollector, hasTimeout, acquired.handle(), enc.blobs(), similarities, topN, threshold, System.nanoTime() - searchStart);
            }
            if (aggResult != null) {
                ctx.queryResult().aggregations(aggResult);
                if (tree != null && tree.global().isEmpty() == false) {
                    ctx.queryCollectorManagers().remove(GlobalAggCollectorManager.class);
                }
            }
        } finally {
            memory.close();
            stats.nativeMemory(memory.peak());
        }
        if (memory.timedOut() || hasTimeout && SearchSupport.deadlinePassed(ctx)) {
            SearchSupport.timedOut(ctx);
        }
        return null;
    }

    /**
     * The aggregations over the documents the hybrid collection counted (what the aggregators
     * see: the bulk scorer's windows, not the post filter's), each scored with its sub-queries'
     * scores summed. The result in {@code out[0]}, or the reason it cannot be had.
     */
    private String hybridAggregations(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        long handle,
        SimilarityEncoder similarities,
        ScoreDoc[] counted,
        NativeAggregations.Plan aggs,
        NativeAggregationTree.Tree tree,
        int[][] slices,
        InternalAggregations[] out
    ) throws IOException {
        QueryEncoder.Encoded docs = QueryEncoder.encode(KnnHitsQuery.of(searcher.getIndexReader(), counted), similarities, searcher.getTopReaderContext());
        if (docs.fallbackReason() != null) return docs.fallbackReason();
        byte[] blob = docs.blob();
        if (aggs != null) {
            int states = aggs.metrics().size() * Math.max(1, slices.length);
            long[] aggCounts = new long[states];
            double[] aggValues = new double[states * NativeAggregations.VALUES];
            byte[][] terms = new byte[1][];
            int rc = NativeSearch.aggregate(
                handle, blob, aggs.blob(slices), aggCounts, aggValues, terms, -1, new long[] { -1, 0, 0 },
                new long[searcher.getIndexReader().leaves().size()]
            );
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            out[0] = aggs.build(slices, aggCounts, aggValues, terms[0], ctx.partialOnShard());
            NativeAggregations.consumeBuckets(ctx, out[0]);
            return null;
        }
        byte[][] result = new byte[1][];
        int rc = NativeSearch.aggregateTree(handle, blob, tree.blob(slices, NativeAggregations.bucketLimit(ctx)), false, result);
        if (rc != NativeBridge.OK) {
            NativeAggregations.checkBuckets(ctx, rc);
            NativeMemory.check(rc);
            stats.nativeError();
            logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return FallbackReason.NATIVE_ERROR.key();
        }
        try {
            out[0] = tree.build(result[0], slices, ctx.partialOnShard());
        } catch (JavaRefuses e) {
            return FallbackReason.AGGREGATIONS.key();
        } catch (ReflectiveOperationException | RuntimeException e) {
            stats.nativeError();
            logger.warn("accelerated-search: building native aggregation results failed, re-running on Lucene", e);
            return FallbackReason.NATIVE_ERROR.key();
        }
        NativeAggregations.consumeBuckets(ctx, out[0]);
        return null;
    }

    /** Whether {@code query} is or holds a {@link HybridQuery}. */
    static boolean containsHybrid(Query query) {
        return org.lucenerust.opensearch.neural.util.HybridQueryUtil.containsHybridQuery(query);
    }

    /**
     * A plain scored search over the reader's segments and the writer's buffered documents of
     * {@code live}: the hits and total the reader and the buffers give together, a buffered hit
     * numbered after the reader's documents. False when the buffers cannot answer the query, or the
     * native search failed; nothing is stored then.
     */
    private boolean searchLive(SearchContext ctx, long handle, LiveDirectoryReader live, byte[] blob) {
        int size = ctx.size();
        int numDocs = size == 0 ? 0
            : Math.min(ctx.from() + size, Math.max(1, ctx.searcher().getIndexReader().numDocs() + live.view().maxDoc()));
        int trackUpTo = ctx.trackTotalHitsUpTo();
        long countLimit = trackUpTo == SearchContext.TRACK_TOTAL_HITS_DISABLED ? 0
            : trackUpTo == SearchContext.TRACK_TOTAL_HITS_ACCURATE ? Long.MAX_VALUE
            : Math.max(trackUpTo, numDocs);
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[4];
        int rc = NativeLive.searchLive(handle, live.view().id(), blob, numDocs, countLimit, docs, scores, counts);
        if (rc != NativeBridge.OK) {
            NativeMemory.check(rc);
            stats.nativeError();
            if (scriptFailed()) {
                throw scriptFailure(ctx.searcher().getIndexReader(), true);
            }
            logger.warn("accelerated-search: live search failed ({}), searching the reader alone: {}", rc, NativeBridge.lastError());
            return false;
        }
        if (counts[3] == 0) {
            return false;
        }
        int n = (int) counts[0];
        ScoreDoc[] hits = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        TotalHits total = countLimit == 0 ? new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO)
            : new TotalHits(counts[1], counts[2] != 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO);
        store(ctx, new TopDocsAndMaxScore(new TopDocs(total, hits), n == 0 ? Float.NaN : scores[0]), null);
        stats.liveQuery();
        return true;
    }

    /**
     * The top hits and total of {@code hits} alone, as the native plain search gives them: best
     * score first, the lower doc on a tie, {@code from + size} of them, counted up to
     * {@code track_total_hits}.
     */
    private void storeKnnHits(SearchContext ctx, KnnHitsQuery hits) {
        int[] docs = hits.docs();
        float[] scores = hits.scores();
        ScoreDoc[] all = new ScoreDoc[docs.length];
        for (int i = 0; i < docs.length; i++) {
            all[i] = new ScoreDoc(docs[i], scores[i]);
        }
        java.util.Arrays.sort(all, (a, b) -> a.score != b.score ? Float.compare(b.score, a.score) : Integer.compare(a.doc, b.doc));
        int n = ctx.size() == 0 ? 0 : Math.min(all.length, ctx.from() + ctx.size());
        ScoreDoc[] top = java.util.Arrays.copyOf(all, n);
        int trackUpTo = ctx.trackTotalHitsUpTo();
        TotalHits total;
        if (trackUpTo == SearchContext.TRACK_TOTAL_HITS_DISABLED) {
            total = new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO);
        } else if (trackUpTo != SearchContext.TRACK_TOTAL_HITS_ACCURATE && all.length > Math.max(trackUpTo, n)) {
            total = new TotalHits(Math.max(trackUpTo, n), TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO);
        } else {
            total = new TotalHits(all.length, TotalHits.Relation.EQUAL_TO);
        }
        store(ctx, new TopDocsAndMaxScore(new TopDocs(total, top), n == 0 ? Float.NaN : top[0].score), null);
        stats.liveQuery();
    }

    /**
     * Runs the blob and stores the result; returns null, or the fallback reason when it did not.
     *
     * <p>Cancellation is checked before the call, not during it: a native search runs to completion
     * (see docs/accelerated-queries.md, Known limits).
     */
    private String searchNative(
        SearchContext ctx,
        long handle,
        byte[] blob,
        byte[] hitsBlob,
        byte[] sortBlob,
        byte[] collapseBlob,
        List<NativeRescore> rescores,
        Sort sort,
        boolean scoreDocs,
        NativeAggregations.Plan aggs,
        NativeAggregationTree.Tree tree,
        boolean live
    ) {
        int size = ctx.size();
        ScrollContext scroll = ctx.scrollContext();
        // A concurrent size-0 search reports terminated_early per slice: without whole-segment
        // slices there is nothing to replay.
        if (size == 0 && ctx.shouldUseConcurrentSearch() && NativeAggregations.slices(ctx) == null) {
            return FallbackReason.INTRA_SEGMENT.key();
        }
        // A scroll page is "size" hits whatever "from" says, and counts them on its first page only
        // (TopDocsCollectorContext.createTopDocsCollectorContext).
        // With the buffers (live), the searcher's reader counts none of their documents.
        int readerDocs = live ? ctx.searcher().getIndexReader().maxDoc() : ctx.searcher().getIndexReader().numDocs();
        int numDocs = size == 0 ? 0
            : Math.min(scroll != null ? size : ctx.from() + size, Math.max(1, readerDocs));
        // A rescored search collects at least each rescorer's window
        // (createTopDocsCollectorContext).
        if (rescores != null) {
            for (NativeRescore r : rescores) {
                numDocs = Math.max(numDocs, r.window());
            }
        }
        boolean terminating = ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER;
        int trackUpTo = scroll == null ? ctx.trackTotalHitsUpTo()
            : scroll.totalHits != null ? SearchContext.TRACK_TOTAL_HITS_DISABLED
            : SearchContext.TRACK_TOTAL_HITS_ACCURATE;
        // track_total_hits: false -> count nothing; true -> exact; N -> exact below N.
        // TopScoreDocCollectorManager stores max(totalHitsThreshold, numHits).
        long countLimit = trackUpTo == SearchContext.TRACK_TOTAL_HITS_DISABLED ? 0
            : trackUpTo == SearchContext.TRACK_TOTAL_HITS_ACCURATE ? Long.MAX_VALUE
            : Math.max(trackUpTo, numDocs);
        // OpenSearch answers some totals without counting (TopDocsCollectorContext
        // .shortcutTotalHitCount) and then collects with a threshold of 1; so must we, or
        // the total differs ({10000, gte} for its {N, eq}) and the search scores documents
        // Lucene never looks at.
        int shortcut = -1;
        // A term query's shortcut, left to the unsorted search's own native call (searchDocFreq):
        // each crossing into native code costs more than the term lookup it makes.
        Term shortcutTerm = null;
        // TopDocsCollectorContext: "hasFilterCollector ? -1 : SearchSupport.shortcutTotalHitCount(...)".
        // (A collapsed search counts every match itself, whatever track_total_hits says.)
        // The shortcuts read the searcher's reader, to which the buffered documents are deleted:
        // with them (live), every total is counted.
        if (trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED
            && live == false
            && collapseBlob == null
            && ctx.parsedPostFilter() == null
            && terminating == false
            && ctx.minimumScore() == null) {
            if (sortBlob == null && aggs == null && tree == null) {
                shortcutTerm = SearchSupport.shortcutTerm(ctx.searcher().getIndexReader(), ctx.query());
            }
            if (shortcutTerm == null) {
                try {
                    shortcut = SearchSupport.shortcutTotalHitCount(ctx.searcher().getIndexReader(), ctx.query(), handle);
                } catch (IOException e) {
                    return FallbackReason.NATIVE_ERROR.key();
                }
            }
            if (shortcut >= 0 || shortcutTerm != null) {
                countLimit = numDocs;
            }
        }
        if (ctx.isCancelled()) {
            return FallbackReason.CANCELLED.key();
        }
        NativeProfile.Phases phases = NativeProfile.phases();
        if (phases != null) {
            phases.topN = numDocs;
            phases.countLimit = countLimit;
            phases.topScores = numDocs > 0 && (sort == null || SearchSupport.sortByScore(sort));
        }
        long aggStart = System.nanoTime();
        // The aggregations first, kept aside until the hits are in too: a failure in either
        // re-runs the whole request on Lucene.
        InternalAggregations aggResult = null;
        // The size-0 count the aggregations' pass already made: {total, lower bound}, or null.
        long[] counted = null;
        // And each segment's live match count it saw, when it saw every segment's: what replays
        // a concurrent count's early termination without searching again.
        long[] segCounts = null;
        if (aggs != null) {
            int[][] slices = NativeAggregations.slices(ctx);
            if (slices == null) {
                return FallbackReason.INTRA_SEGMENT.key();
            }
            int states = aggs.metrics().size() * Math.max(1, slices.length);
            long[] aggCounts = new long[states];
            double[] aggValues = new double[states * NativeAggregations.VALUES];
            // Every aggregation in one native pass: each segment's matches collected once -- and
            // counted there too, for a size-0 search of the same query (Lucene's MultiCollector
            // counts in the same pass), instead of a second search for the count.
            byte[][] terms = new byte[1][];
            // The hits' query is the aggregations' only without a post_filter (min_score, when
            // there is one, sits in front of both, and behind it both count the passing documents).
            boolean countHere = numDocs == 0 && sortBlob == null && ctx.parsedPostFilter() == null
                && shortcut < 0 && countLimit > 0;
            long[] total = new long[] { -1, 0, 0 };
            long[] perSegment = new long[ctx.searcher().getIndexReader().leaves().size()];
            int rc = NativeSearch.aggregate(
                handle, blob, aggs.blob(slices), aggCounts, aggValues, terms, countHere ? countLimit : -1, total, perSegment
            );
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            aggResult = aggs.build(slices, aggCounts, aggValues, terms[0], ctx.partialOnShard());
            NativeAggregations.consumeBuckets(ctx, aggResult);
            if (countHere && total[0] >= 0) {
                counted = total;
                if (total[2] == 1) {
                    segCounts = perSegment;
                }
            }
        } else if (tree != null) {
            int[][] slices = NativeAggregations.slices(ctx);
            if (slices == null) {
                return FallbackReason.INTRA_SEGMENT.key();
            }
            byte[][] result = new byte[1][];
            int rc = NativeSearch.aggregateTree(handle, blob, tree.blob(slices, NativeAggregations.bucketLimit(ctx)), SearchFlags.of(ctx).aggCache(), result);
            if (rc != NativeBridge.OK) {
                NativeAggregations.checkBuckets(ctx, rc);
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            try {
                aggResult = tree.build(result[0], slices, ctx.partialOnShard());
            } catch (JavaRefuses e) {
                // Values the Java aggregator would refuse (a NaN for a t-digest, a negative value
                // for an HDR histogram): Lucene's pass raises its own error.
                return FallbackReason.AGGREGATIONS.key();
            } catch (ReflectiveOperationException | RuntimeException e) {
                stats.nativeError();
                logger.warn("accelerated-search: building native aggregation results failed, re-running on Lucene", e);
                return FallbackReason.NATIVE_ERROR.key();
            }
            NativeAggregations.consumeBuckets(ctx, aggResult);
        }
        if (phases != null) {
            phases.aggNanos = aggs == null && tree == null ? 0 : System.nanoTime() - aggStart;
        }
        long hitsStart = System.nanoTime();
        String reason = collapseBlob != null ? searchCollapsed(ctx, handle, hitsBlob, sortBlob, collapseBlob, Math.max(1, numDocs))
            : sortBlob != null
            ? searchSorted(ctx, handle, hitsBlob, sortBlob, sort, scoreDocs, Math.max(1, numDocs), countLimit, shortcut)
            : searchUnsorted(ctx, handle, hitsBlob, numDocs, countLimit, shortcut, shortcutTerm, counted, segCounts);
        if (phases != null) {
            phases.hitsNanos = System.nanoTime() - hitsStart;
        }
        if (reason == null && rescores != null) {
            // RescoreProcessor, after the query phase: each rescorer over the hits so far.
            reason = rescore(ctx, handle, rescores);
        }
        if (reason == null && aggResult != null) {
            // DefaultAggregationProcessor.postProcess keeps a result already there (hasAggs).
            ctx.queryResult().aggregations(aggResult);
            // ConcurrentAggregationProcessor.postProcess runs the global aggregations whatever is
            // there, and adds them to it: the native pass answered them, so it has none to run.
            if (tree != null && tree.global().isEmpty() == false) {
                ctx.queryCollectorManagers().remove(GlobalAggCollectorManager.class);
            }
        }
        return reason;
    }

    /** The unsorted half of {@link #searchNative}. */
    private String searchUnsorted(
        SearchContext ctx,
        long handle,
        byte[] blob,
        int numDocs,
        long countLimit,
        int shortcut,
        Term shortcutTerm,
        long[] counted,
        long[] segCounts
    ) {
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[4];
        if (counted != null) {
            // Counted in the aggregations' pass: no hits to find, nothing left to search.
            counts[1] = counted[0];
            counts[2] = counted[1];
        } else if (shortcutTerm != null) {
            // The total from the term's document frequency, the hits (if any are asked for) from
            // the same native call.
            BytesRef bytes = shortcutTerm.bytes();
            int rc = NativeSearch.searchDocFreq(
                handle,
                blob,
                numDocs,
                countLimit,
                shortcutTerm.field().getBytes(java.nio.charset.StandardCharsets.UTF_8),
                java.util.Arrays.copyOfRange(bytes.bytes, bytes.offset, bytes.offset + bytes.length),
                docs,
                scores,
                counts
            );
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            shortcut = Math.toIntExact(counts[3]);
        } else if (numDocs == 0 && (shortcut >= 0 || countLimit == 0)) {
            // No hits, and a total already known (the shortcut) or not asked for: nothing the
            // native search would answer is read below.
        } else {
            int rc = NativeSearch.search(handle, blob, numDocs, countLimit, docs, scores, counts);
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
        }
        int n = (int) counts[0];
        ScoreDoc[] hits = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        // Lucene's own shape: with counting off, 0 hits "or more".
        TotalHits total = shortcut >= 0 ? new TotalHits(shortcut, TotalHits.Relation.EQUAL_TO)
            : countLimit == 0 ? new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO)
            : new TotalHits(counts[1], counts[2] != 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO);
        float maxScore = n == 0 ? Float.NaN : scores[0];
        if (numDocs == 0 && ctx.shouldUseConcurrentSearch()) {
            Boolean terminated = countTerminatedEarly(ctx, handle, blob, shortcut, total, segCounts);
            if (terminated == null) {
                return FallbackReason.NATIVE_ERROR.key();
            }
            if (terminated) {
                ctx.queryResult().terminatedEarly(true);
            }
        }
        if (ctx.sort() != null) {
            // A sorted search with no hits asked for (size 0): what EmptyTopDocsCollectorContext
            // stores, a TopFieldDocs carrying the sort -- the coordinator reads the class.
            ctx.queryResult().topDocs(
                new TopDocsAndMaxScore(new TopFieldDocs(total, new ScoreDoc[0], ctx.sort().sort.getSort()), Float.NaN),
                null
            );
            return null;
        }
        store(ctx, new TopDocsAndMaxScore(new TopDocs(total, hits), maxScore), null);
        return null;
    }

    /**
     * Stores a search's hits, as ScrollingTopDocsCollectorContext does in a scroll: its first page
     * keeps the total and max score for the later ones, which report those; on a single shard it
     * also remembers the last hit, which the next page searches after (on more, the fetch phase
     * does, from the coordinator's merge).
     */
    static void store(SearchContext ctx, TopDocsAndMaxScore td, DocValueFormat[] formats) {
        ScrollContext scroll = ctx.scrollContext();
        if (scroll != null) {
            if (scroll.totalHits == null) {
                scroll.totalHits = td.topDocs.totalHits;
                scroll.maxScore = td.maxScore;
            } else {
                td.topDocs.totalHits = scroll.totalHits;
                td.maxScore = scroll.maxScore;
            }
            if (ctx.numberOfShards() == 1 && td.topDocs.scoreDocs.length > 0) {
                scroll.lastEmittedDoc = td.topDocs.scoreDocs[td.topDocs.scoreDocs.length - 1];
            }
        }
        ctx.queryResult().topDocs(td, formats);
    }

    /**
     * Whether a concurrent {@code size: 0} search reports {@code terminated_early}: its count runs
     * in an EarlyTerminatingCollectorManager (EmptyTopDocsCollectorContext.createManager) whose
     * reduce says so when any slice's collector stopped -- at once when nothing is counted (a
     * disabled total, or one answered from index statistics: a limit of 0), past the limit
     * otherwise, which is when the total is a lower bound.
     */
    private Boolean countTerminatedEarly(
        SearchContext ctx,
        long handle,
        byte[] blob,
        int shortcut,
        TotalHits total,
        long[] segCounts
    ) {
        boolean[] used = new boolean[1];
        Boolean terminated = countTerminatedEarly(ctx, handle, blob, shortcut, total, segCounts, used);
        if (terminated != null && used[0] == false) {
            try {
                SegmentQueryCache.useQueryCache(ctx);
            } catch (IOException e) {
                return null;
            }
        }
        return terminated;
    }

    private Boolean countTerminatedEarly(
        SearchContext ctx,
        long handle,
        byte[] blob,
        int shortcut,
        TotalHits total,
        long[] segCounts,
        boolean[] used
    ) {
        int trackUpTo = ctx.trackTotalHitsUpTo();
        List<LeafReaderContext> leaves = ctx.searcher().getIndexReader().leaves();
        if (trackUpTo == SearchContext.TRACK_TOTAL_HITS_DISABLED || shortcut >= 0) {
            return leaves.isEmpty() == false;
        }
        if (trackUpTo == SearchContext.TRACK_TOTAL_HITS_ACCURATE
            || (total.relation() == TotalHits.Relation.EQUAL_TO && total.value() < trackUpTo)) {
            // A slice's collector stops only past trackUpTo documents it iterated.
            return false;
        }
        // Past the limit, it depends on which segments the count collector
        // answers from Weight.count (nothing reaches the terminating collector)
        // and which it iterates: Lucene's own weight, the query cache included,
        // says; the native side replays each slice.
        int[][] slices = NativeAggregations.slices(ctx);
        if (slices == null) {
            return null;
        }
        boolean counted = segCounts != null && segCounts.length == leaves.size();
        if (counted) {
            // The answer only grows with the segments the collectors iterate: when no slice stops
            // even iterating every segment, none stops, whatever Weight.count would say -- and the
            // Lucene weight it would take (a geo filter's builds its whole grid) is not built.
            boolean[] everywhere = new boolean[leaves.size()];
            java.util.Arrays.fill(everywhere, true);
            if (SegmentQueryCache.replayCountTerminated(slices, leaves, everywhere, segCounts, trackUpTo) == false) {
                return false;
            }
        }
        java.io.ByteArrayOutputStream spec = new java.io.ByteArrayOutputStream();
        NativeAggregations.writeInt(spec, trackUpTo);
        NativeAggregations.writeInt(spec, leaves.size());
        boolean[] iterate = new boolean[leaves.size()];
        try {
            Weight weight = ctx.parsedPostFilter() != null || ctx.minimumScore() != null
                ? null
                : ctx.searcher().createWeight(ctx.query(), org.apache.lucene.search.ScoreMode.COMPLETE_NO_SCORES, 1f);
            for (int i = 0; i < leaves.size(); i++) {
                // Under a post_filter or min_score the count collector gets no
                // weight (neither FilteredCollector nor MinimumScoreCollector
                // passes it on): it iterates everywhere.
                iterate[i] = weight == null || weight.count(leaves.get(i)) == -1;
                spec.write(iterate[i] ? 1 : 0);
            }
            if (weight != null) {
                // Weight.count recorded the use (on a segment without deletions; on one with
                // them the cache never answers a count): cache what Lucene's search would.
                used[0] = true;
                SegmentQueryCache.fillQueryCache(ctx, weight, leaves, iterate);
            }
        } catch (IOException e) {
            return null;
        }
        if (counted) {
            // The aggregations' pass already counted each segment's live matches: replay the
            // collectors from those instead of searching again.
            return SegmentQueryCache.replayCountTerminated(slices, leaves, iterate, segCounts, trackUpTo);
        }
        NativeAggregations.writeSlices(spec, slices);
        long[] out = new long[1];
        int rc = NativeSearch.countTerminates(handle, blob, spec.toByteArray(), out);
        if (rc != NativeBridge.OK) {
            NativeMemory.check(rc);
            stats.nativeError();
            logger.warn("accelerated-search: native count replay failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return null;
        }
        return out[0] != 0;
    }

    /**
     * OpenSearch's RescoreProcessor over the native hits: each rescorer in turn ({@link
     * NativeRescore}), the max score then the first hit's.
     */
    private String rescore(SearchContext ctx, long handle, List<NativeRescore> rescores) {
        TopDocsAndMaxScore stored = ctx.queryResult().topDocs();
        TopDocs top = stored.topDocs;
        if (top.scoreDocs.length == 0) {
            return null;
        }
        int n = top.scoreDocs.length;
        int[] docs = new int[n];
        float[] scores = new float[n];
        for (int i = 0; i < n; i++) {
            docs[i] = top.scoreDocs[i].doc;
            scores[i] = top.scoreDocs[i].score;
        }
        for (NativeRescore r : rescores) {
            // QueryRescorer.rescore: the window's documents, for explain.
            java.util.Set<Integer> window = new java.util.HashSet<>();
            for (int i = 0; i < Math.min(n, r.window()); i++) {
                window.add(docs[i]);
            }
            int rc = NativeSearch.rescore(handle, r.blob(), r.window(), r.queryWeight(), r.rescoreQueryWeight(), r.mode(), docs, scores);
            if (rc != NativeBridge.OK) {
                NativeMemory.check(rc);
                stats.nativeError();
                logger.warn("accelerated-search: native rescore failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return FallbackReason.NATIVE_ERROR.key();
            }
            r.context().setRescoredDocs(java.util.Collections.unmodifiableSet(window));
        }
        ScoreDoc[] hits = new ScoreDoc[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        ctx.queryResult().topDocs(new TopDocsAndMaxScore(new TopDocs(top.totalHits, hits), hits[0].score), ctx.queryResult().sortValueFormats());
        return null;
    }

    /**
     * The collapsed half of {@link #searchNative} ({@link NativeCollapse}): {@code numDocs} groups,
     * stored as CollapsingTopDocsCollectorContext stores them, with the sort's formats.
     */
    private String searchCollapsed(SearchContext ctx, long handle, byte[] blob, byte[] sortBlob, byte[] collapseBlob, int numDocs) {
        byte[][] out = new byte[1][];
        int rc = NativeSearch.searchCollapsed(handle, blob, sortBlob, collapseBlob, numDocs, out);
        if (rc != NativeBridge.OK) {
            NativeMemory.check(rc);
            String message = NativeBridge.lastError();
            if (NativeCollapse.refused(message)) {
                // A multi-valued collapse field: Lucene's collector raises the request's error.
                return FallbackReason.COLLAPSE_REFUSED.key();
            }
            stats.nativeError();
            logger.warn("accelerated-search: native collapsed search failed ({}), re-running on Lucene: {}", rc, message);
            return FallbackReason.NATIVE_ERROR.key();
        }
        TopDocsAndMaxScore td;
        try {
            td = NativeCollapse.decode(ctx, out[0], numDocs);
        } catch (RuntimeException e) {
            stats.nativeError();
            logger.warn("accelerated-search: decoding native collapsed hits failed, re-running on Lucene", e);
            return FallbackReason.NATIVE_ERROR.key();
        }
        ctx.queryResult().topDocs(td, ctx.sort() == null ? new DocValueFormat[] { DocValueFormat.RAW } : ctx.sort().formats);
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) {
            // QueryPhase: true when EarlyTerminatingCollector threw, false otherwise.
            ctx.queryResult().terminatedEarly(NativeCollapse.terminated(out[0]));
        }
        return null;
    }

    /**
     * The sorted half of {@link #searchNative}: {@code TopFieldCollectorManager(sort, numDocs,
     * searchAfter, countLimit)}, answered as {@code SimpleTopDocsCollectorContext} answers it -- a
     * {@link TopFieldDocs} of {@link FieldDoc}s with {@code NaN} scores, the max score read off the
     * first hit when the score leads the sort and {@code NaN} otherwise.
     */
    private String searchSorted(
        SearchContext ctx,
        long handle,
        byte[] blob,
        byte[] sortBlob,
        Sort sort,
        boolean scoreDocs,
        int numDocs,
        long countLimit,
        int shortcut
    ) {
        SortField[] fields = sort.getSort();
        int[] docs = new int[numDocs];
        long[] values = new long[numDocs * fields.length];
        long[] counts = new long[5];
        byte[][] terms = new byte[1][];
        int rc = NativeSearch.searchSorted(handle, blob, sortBlob, numDocs, countLimit, docs, values, counts, terms);
        if (rc != NativeBridge.OK) {
            NativeMemory.check(rc);
            stats.nativeError();
            logger.warn("accelerated-search: native sorted search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return FallbackReason.NATIVE_ERROR.key();
        }
        int n = (int) counts[0];
        FieldDoc[] hits = SortEncoder.hits(fields, n, docs, values, terms[0]);
        TotalHits total = shortcut >= 0 ? new TotalHits(shortcut, TotalHits.Relation.EQUAL_TO)
            : countLimit == 0 ? new TotalHits(0, TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO)
            : new TotalHits(counts[1], counts[2] != 0 ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO);
        if (ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER) {
            // QueryPhase: true when EarlyTerminatingCollector threw, false otherwise.
            ctx.queryResult().terminatedEarly(counts[4] != 0);
        }
        if (ctx.size() == 0) {
            // terminate_after without hits: EmptyTopDocsCollectorContext's answer, the count.
            ScoreDoc[] none = new ScoreDoc[0];
            TopDocs empty = ctx.sort() != null ? new TopFieldDocs(total, none, ctx.sort().sort.getSort()) : new TopDocs(total, none);
            ctx.queryResult().topDocs(new TopDocsAndMaxScore(empty, Float.NaN), null);
            return null;
        }
        if (scoreDocs) {
            // By score: the ScoreDocs TopScoreDocCollector gives, the max score its first (a
            // scroll's later page gets the first page's back from store()).
            ScoreDoc[] scored = new ScoreDoc[n];
            for (int i = 0; i < n; i++) {
                scored[i] = new ScoreDoc(hits[i].doc, (float) hits[i].fields[0]);
            }
            store(ctx, new TopDocsAndMaxScore(new TopDocs(total, scored), n == 0 ? Float.NaN : scored[0].score), null);
            return null;
        }
        // The score leading the sort: its first hit's; a tracked max score (track_scores
        // behind another key): the native MaxScoreCollector's, NaN without matches.
        float maxScore = n > 0 && SearchSupport.sortByScore(ctx.sort().sort) ? (float) hits[0].fields[0]
            : ctx.trackScores() ? Float.intBitsToFloat((int) counts[3])
            : Float.NaN;
        store(ctx, new TopDocsAndMaxScore(new TopFieldDocs(total, hits, fields), maxScore), ctx.sort().formats);
        return null;
    }

}
