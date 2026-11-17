/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.approximated;
import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.buffered;
import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.countSegments;
import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.flags;
import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.postFiltered;
import static org.lucenerust.opensearch.AcceleratedQueryPhaseSearcher.sortByScore;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.ScrollContext;
import org.opensearch.search.internal.SearchContext;

/**
 * What a plain (non-hybrid) search sends the native engine, built stage by stage. Each stage
 * returns null to go on, or the reason the request cannot run natively, which ends the pipeline
 * and leaves the search to Lucene ({@link AcceleratedQueryPhaseSearcher#searchPlain}).
 */
final class PlainSearch {
    final SearchContext ctx;
    final ContextIndexSearcher searcher;
    final Query query;

    /** The query's blob, and the hits' when a post_filter narrows them (the same array otherwise). */
    byte[] blob;
    byte[] hitsBlob;
    List<NativeRescore> rescores;
    /** The sort blob and its keys; null for an unsorted search. */
    byte[] sortBlob;
    Sort sort;
    FieldDoc after;
    /** A search by score run as the score sort, its hits handed back as ScoreDocs. */
    boolean scoreDocs;
    int terminateAfter;
    byte[] collapseBlob;
    /** The reader handle the native search ran over, for a profile to run the query over again. */
    long searchedHandle;
    /** Whether the native calls ran past the request's memory scope's time limit. */
    boolean memoryTimedOut;

    PlainSearch(SearchContext ctx, ContextIndexSearcher searcher, Query query) {
        this.ctx = ctx;
        this.searcher = searcher;
        this.query = query;
    }

    /** The query (and post_filter), with the similarities and min_score in front of its blobs. */
    String encodeQuery() throws IOException {
        String reason = null;
        // A post_filter narrows the hits, not the aggregations (QueryPhase wraps only the top-docs
        // collector in its FilteredCollector): the hits search "query AND filter", the filter a
        // non-scoring clause, so each hit keeps the query's score; the aggregations the query.
        if (reason == null) {
            // Each scored field's similarity, sent in front of the blobs when
            // one is not the default.
            SimilarityEncoder similarities = new SimilarityEncoder(searcher.getSimilarity());
            Query encoding = query;
            QueryEncoder.Encoded enc = QueryEncoder.encode(encoding, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
            reason = enc.fallbackReason();
            blob = enc.blob();
            boolean fast = enc.fast();
            hitsBlob = blob;
            // The post_filter is rewritten here as QueryPhase rewrites it,
            // which is also when an approximate range in it
            // (ApproximateScoreQuery) resolves: approximated, it matches only
            // the first track_total_hits documents it visits in BKD order -- a
            // filter narrower than the range, which the native engine would not
            // reproduce, so OpenSearch answers.
            Query postFilter = reason == null && ctx.parsedPostFilter() != null ? searcher.rewrite(ctx.parsedPostFilter().query()) : null;
            if (postFilter != null && approximated(postFilter) && buffered(searcher) == false && flags(ctx).nativeApproximate() == false) {
                reason = FallbackReason.APPROXIMATE.key();
            }
            if (reason == null && postFilter != null) {
                Query filteredQuery = postFiltered(query, postFilter);
                QueryEncoder.Encoded filtered = QueryEncoder.encode(filteredQuery, similarities, EncodeContext.of(ctx, searcher.getTopReaderContext()));
                reason = filtered.fallbackReason();
                hitsBlob = filtered.blob();
                fast = fast && filtered.fast();
            }
            if (reason == null && fast == false && flags(ctx).allShapes() == false) {
                reason = FallbackReason.SLOWER_SHAPE.key();
            }
            // min_score: OpenSearch's MinimumScoreCollector sits outside every other collector,
            // aggregations included, so every blob carries it.
            if (reason == null && ctx.minimumScore() != null) {
                blob = QueryEncoder.withMinScore(blob, ctx.minimumScore());
                hitsBlob = QueryEncoder.withMinScore(hitsBlob, ctx.minimumScore());
            }
            if (reason == null) {
                boolean sameBlob = blob == hitsBlob;
                blob = NativeDfs.prefix(searcher, similarities.prefix(blob));
                hitsBlob = sameBlob ? blob : NativeDfs.prefix(searcher, similarities.prefix(hitsBlob));
            }
        }
        return reason;
    }

    /** The rescore queries, encoded before anything runs. */
    String encodeRescores() {
        String reason = null;
        // The rescore queries, encoded before anything runs: one that cannot be leaves the whole
        // search to Lucene.
        if (reason == null && NativeRescore.applies(ctx)) {
            rescores = new ArrayList<>();
            reason = NativeRescore.encode(ctx, searcher, new SimilarityEncoder(searcher.getSimilarity()), rescores);
        }
        return reason;
    }

    /** Whether an approximated range must run on OpenSearch's collector. */
    String refuseApproximate() {
        String reason = null;
        // OpenSearch's approximate match_all/range (ApproximateScoreQuery resolved to its
        // ApproximateQuery) collects documents in BKD order and so breaks ties differently from
        // Lucene's exact answer, which is what the native engine gives: native by default
        // (NATIVE_APPROXIMATE), on OpenSearch's when the index turns that off.
        // Not over buffered documents: the approximation walks the reader's points alone, where
        // they have none, so only the native engine finds them.
        if (reason == null && approximated(query) && buffered(searcher) == false && flags(ctx).nativeApproximate() == false) {
            reason = FallbackReason.APPROXIMATE.key();
        }
        return reason;
    }

    /** The sort, scroll, terminate_after and collapse blobs. */
    String encodeSort() {
        String reason = null;
        // A sorted search: the sort blob, unless the hits are not asked for at all (size 0 has no
        // order to keep, and runs as the unsorted count it is).
        // A scroll's later pages search after the last hit it emitted
        // (ScrollingTopDocsCollectorContext); one ordered by score, as the score sort it is:
        // PagingFieldCollector skips what PagingTopScoreDocCollector does, and keeps the same hits.
        sort = ctx.sort() == null ? null : ctx.sort().sort;
        after = ctx.searchAfter();
        // A search by score run as the score sort, its hits handed back as ScoreDocs.
        ScrollContext scroll = ctx.scrollContext();
        if (reason == null && scroll != null && scroll.totalHits != null && scroll.lastEmittedDoc != null) {
            ScoreDoc last = scroll.lastEmittedDoc;
            if (sort == null) {
                sort = new Sort(SortField.FIELD_SCORE);
                after = new FieldDoc(last.doc, last.score, new Object[] { last.score });
                scoreDocs = true;
            } else if (last instanceof FieldDoc fd) {
                after = fd;
            } else {
                reason = FallbackReason.SCROLL_AFTER.key();
            }
        }
        // terminate_after runs on the sorted search, which takes the cut
        // (SortEncoder): by score as the score sort, and without hits (size 0)
        // by _doc, the cheapest order, its one hit dropped.
        terminateAfter = ctx.terminateAfter() == SearchContext.DEFAULT_TERMINATE_AFTER ? 0 : ctx.terminateAfter();
        if (terminateAfter > 0 && ctx.size() == 0) {
            sort = new Sort(SortField.FIELD_DOC);
        } else if (terminateAfter > 0 && sort == null) {
            sort = new Sort(SortField.FIELD_SCORE);
            scoreDocs = true;
        }
        // collapse: CollapsingTopDocsCollectorContext, by the request's sort or
        // the score, from + size groups; its comparators never skip, so no
        // index-sort flags.
        if (reason == null && ctx.collapse() != null && ctx.size() > 0) {
            int[][] slices = NativeAggregations.slices(ctx);
            Sort collapseSort = NativeCollapse.sort(ctx);
            boolean trackMaxScore = ctx.trackScores() && NativeCollapse.sortByScore(ctx) == false;
            SortEncoder.Encoded sorted = slices == null ? new SortEncoder.Encoded(null, FallbackReason.INTRA_SEGMENT.key())
                : SortEncoder.encode(collapseSort, after, trackMaxScore, slices, terminateAfter, false, null);
            reason = sorted.fallbackReason();
            sortBlob = sorted.blob();
            sort = collapseSort;
            collapseBlob = NativeCollapse.blob(ctx.collapse());
        } else if (reason == null && sort != null && (ctx.size() > 0 || terminateAfter > 0)) {
            // track_scores behind another key: OpenSearch's MaxScoreCollector over every match.
            boolean trackMaxScore = ctx.trackScores() && ctx.size() > 0 && sortByScore(sort) == false;
            // A concurrent search collects each slice separately, as Lucene does.
            int[][] slices = NativeAggregations.slices(ctx);
            // TopFieldCollector ends a segment whose index sort begins with the
            // search's sort at its first non-competitive document; the native
            // collector does too, told which segments -- and where the query
            // pins the index sort's first keys and sorts by the ones after.
            byte[] indexSorted = SortEncoder.indexSorted(searcher.getIndexReader(), sort);
            Sort keys = sort;
            FieldDoc from = after;
            boolean counts = countSegments(ctx, terminateAfter);
            int limit = terminateAfter;
            // A script key's script reads fields by the index's mappings.
            SortEncoder.Encoded sorted = slices == null ? new SortEncoder.Encoded(null, FallbackReason.INTRA_SEGMENT.key())
                : SortEncoder.encode(keys, from, trackMaxScore, slices, limit, counts, indexSorted, EncodeContext.of(ctx));
            reason = sorted.fallbackReason();
            sortBlob = sorted.blob();
        }
        return reason;
    }
}
