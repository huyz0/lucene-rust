/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
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
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.internal.ScrollContext;
import org.opensearch.search.internal.SearchContext;
import java.io.IOException;
import java.util.List;

/**
 * The native hit collection of a plain request: unsorted, sorted and collapsed hits, the aggregations
 * beside them, rescore, and the totals OpenSearch's collectors would report.
 */
final class NativeHits {
    private static final Logger logger = LogManager.getLogger(NativeHits.class);

    private final SearchStats stats;

    NativeHits(SearchStats stats) {
        this.stats = stats;
    }

    /**
     * Runs the blob and stores the result; returns null, or the fallback reason when it did not.
     *
     * <p>Cancellation is checked before the call, not during it: a native search runs to completion
     * (see docs/accelerated-queries.md, Known limits).
     */
    String searchNative(
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
                    return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
    String searchUnsorted(
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
                return FallbackReason.ACCELERATED_ERROR.key();
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
            return FallbackReason.ACCELERATED_ERROR.key();
        }
        TopDocsAndMaxScore td;
        try {
            td = NativeCollapse.decode(ctx, out[0], numDocs);
        } catch (RuntimeException e) {
            stats.nativeError();
            logger.warn("accelerated-search: decoding native collapsed hits failed, re-running on Lucene", e);
            return FallbackReason.ACCELERATED_ERROR.key();
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
            return FallbackReason.ACCELERATED_ERROR.key();
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
