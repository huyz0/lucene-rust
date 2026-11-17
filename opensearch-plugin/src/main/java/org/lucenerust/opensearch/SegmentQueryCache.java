/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Weight;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import java.io.IOException;
import java.util.List;

/** What Lucene's search leaves in the query cache, for a native search to leave too. */
final class SegmentQueryCache {
    /**
     * What Lucene's search of a {@code size: 0} query leaves in the query cache, for a native one:
     * the policy records the use, and a query it now caches is cached on the segments without
     * deletions -- the only ones whose {@code Weight.count} the cache answers, which is how a later
     * search's {@code terminated_early} (above) sees it.
     */
    static void useQueryCache(SearchContext ctx) throws IOException {
        ContextIndexSearcher searcher = ctx.searcher();
        org.apache.lucene.search.QueryCache cache = searcher.getQueryCache();
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        if (cache == null || leaves.isEmpty()) {
            return;
        }
        Query query = ctx.query();
        org.apache.lucene.search.QueryCachingPolicy policy = searcher.getQueryCachingPolicy();
        policy.onUse(query);
        if (policy.shouldCache(query) == false || leaves.stream().allMatch(l -> l.reader().hasDeletions())) {
            return;
        }
        // The use is recorded: a weight whose policy does not record it again.
        Weight weight = cache.doCache(
            query.createWeight(searcher, org.apache.lucene.search.ScoreMode.COMPLETE_NO_SCORES, 1f),
            new org.apache.lucene.search.QueryCachingPolicy() {
                @Override
                public void onUse(Query q) {}

                @Override
                public boolean shouldCache(Query q) throws IOException {
                    return policy.shouldCache(q);
                }
            }
        );
        boolean[] every = new boolean[leaves.size()];
        java.util.Arrays.fill(every, true);
        fillQueryCache(ctx, weight, leaves, every);
    }

    /** Caches {@code weight}'s query on the iterated segments without deletions, when the policy says to. */
    static void fillQueryCache(SearchContext ctx, Weight weight, List<LeafReaderContext> leaves, boolean[] iterate)
        throws IOException {
        if (ctx.searcher().getQueryCache() == null || ctx.searcher().getQueryCachingPolicy().shouldCache(ctx.query()) == false) {
            return;
        }
        for (int i = 0; i < leaves.size(); i++) {
            if (iterate[i] && leaves.get(i).reader().hasDeletions() == false) {
                // The caching supplier fills the cache when its iterator is asked for, under the
                // top-level lead cost of a search.
                org.apache.lucene.search.ScorerSupplier supplier = weight.scorerSupplier(leaves.get(i));
                if (supplier != null) {
                    supplier.get(Long.MAX_VALUE);
                }
            }
        }
    }

    /**
     * Whether some slice's {@code EarlyTerminatingCollector(TotalHitCountCollector, n)} stops, given
     * each segment's live match count -- the native {@code count_terminates} replay, without the
     * search: each slice's segments in doc-base order, a collector already holding {@code n}
     * documents stops at the next segment, one that iterates a segment ({@code iterate}, where
     * {@code Weight.count} gave -1) stops at its {@code n + 1}th document, and a segment the count
     * answers from {@code Weight.count} reaches it with nothing. {@code
     * the_aggregations_counts_replay_the_count_collector} in {@code reader_search.rs} holds the two
     * to the same answer.
     */
    static boolean replayCountTerminated(
        int[][] slices,
        List<LeafReaderContext> leaves,
        boolean[] iterate,
        long[] segCounts,
        int n
    ) {
        for (int[] slice : slices) {
            Integer[] order = new Integer[slice.length];
            for (int k = 0; k < slice.length; k++) {
                order[k] = slice[k];
            }
            java.util.Arrays.sort(order, java.util.Comparator.comparingInt(i -> leaves.get(i).docBase));
            long collected = 0;
            for (int i : order) {
                if (collected >= n) {
                    return true;
                }
                if (iterate[i] == false) {
                    continue;
                }
                if (collected + segCounts[i] > n) {
                    return true;
                }
                collected += segCounts[i];
            }
        }
        return false;
    }
}
