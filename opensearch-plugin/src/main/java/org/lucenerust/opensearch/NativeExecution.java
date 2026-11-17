/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.lucenerust.opensearch.painless.PainlessScore;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.query.QueryCollectorContext;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Runs a plain request on the native engine over the reader and, with a realtime view, the writer's
 * buffered documents: the stages around the hit collection, and what falls back to Lucene.
 */
final class NativeExecution {
    private static final Logger logger = LogManager.getLogger(NativeExecution.class);

    private final NativeReaders readers;
    private final SearchStats stats;
    private final NativeHits hits;

    NativeExecution(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
        this.hits = new NativeHits(stats);
    }

    /**
     * Whether the native search failed because a Painless script of this request threw (or
     * scored invalidly): the search then fails as the script does in Java, never re-run there.
     * Any other native failure falls back to Lucene as before.
     */
    static boolean scriptFailed() {
        return PainlessScore.ran() && PainlessScore.scriptFailed(NativeBridge.lastError());
    }

    /**
     * The exception a script's failure ({@link #scriptFailed}) raises ({@link
     * PainlessScore#failure}); {@code counted} when the failure was already counted as a native
     * error.
     */
    RuntimeException scriptFailure(IndexReader reader, boolean counted) {
        stats.scriptError(counted);
        return PainlessScore.failure(NativeBridge.lastError(), reader);
    }

    /**
     * Runs {@code p}'s request on the native engine: null when it did, else the reason it did not.
     * The writer's buffered documents are read too when the searcher has a view of them.
     */
    String executeNative(PlainSearch p, NativeAggregations.Plan aggs, NativeAggregationTree.Tree tree, boolean hasTimeout)
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
            reason = hits.searchNative(ctx, acquired.handle(), blob, hitsBlob, sortBlob, collapseBlob, rescores, sort, scoreDocs, aggs, tree, false);
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
                reason = hits.searchNative(ctx, whole[0], blob, hitsBlob, sortBlob, collapseBlob, rescores, sort, scoreDocs, aggs, tree, true);
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
    boolean termsFromDocFreqs(
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
        if (hits.searchUnsorted(ctx, 0L, null, 0, 0, shortcut, null, null, null) != null) {
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
    boolean readByLucene(SearchContext ctx, ContextIndexSearcher searcher, LiveDirectoryReader view) throws IOException {
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
        NativeHits.store(ctx, new TopDocsAndMaxScore(new TopDocs(total, hits), n == 0 ? Float.NaN : scores[0]), null);
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
        NativeHits.store(ctx, new TopDocsAndMaxScore(new TopDocs(total, top), n == 0 ? Float.NaN : top[0].score), null);
        stats.liveQuery();
    }
}
