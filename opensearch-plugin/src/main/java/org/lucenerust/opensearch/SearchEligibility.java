/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.Weight;
import org.opensearch.search.aggregations.GlobalAggCollectorManager;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.approximate.ApproximateScoreQuery;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.action.search.SearchType;
import org.opensearch.search.query.QueryCollectorArguments;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryCollectorContextSpecRegistry;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Why a request cannot run native, or that it can: the request-shape checks {@link AcceleratedQueryPhaseSearcher}
 * makes before it encodes anything, and the ones for the writer's buffered documents.
 */
final class SearchEligibility {
    /**
     * Why a search cannot read the writer's buffered documents, or null when it can: what the
     * buffers do not hold for the fetch phase (highlighting, scripts), a query that
     * was expanded or given term statistics over the reader's terms alone (a multi-term query under
     * a scoring rewrite, a fuzzy query, a blended {@code multi_match}), or nested k-NN hits.
     */
    static String liveIneligible(SearchContext ctx) {
        if (ctx.highlight() != null || ctx.hasScriptFields() || ctx.innerHits() != null
            && ctx.innerHits().getInnerHits().isEmpty() == false) return FallbackReason.FETCH_DOC_VALUES.key();
        // Nested k-NN hits are found over the view's own vectors, which keep no parent filter.
        if (ctx.mapperService().hasNested() && holdsKnnHits(ctx.query())) return FallbackReason.NESTED.key();
        boolean[] expands = new boolean[1];
        org.apache.lucene.search.QueryVisitor visitor = new org.apache.lucene.search.QueryVisitor() {
            @Override
            public void consumeTerms(Query query, org.apache.lucene.index.Term... terms) {
                // Statistics a blended query computed from the reader's terms: the buffers' are not in them.
                if (query instanceof org.opensearch.lucene.queries.BlendedTermQuery
                    || query instanceof org.apache.lucene.search.TermQuery tq && tq.getTermStates() != null) {
                    expands[0] = true;
                }
            }

            @Override
            public void consumeTermsMatching(
                Query query,
                String field,
                java.util.function.Supplier<org.apache.lucene.util.automaton.ByteRunAutomaton> automaton
            ) {
                // A multi-term query the native search expands over every segment, the buffers'
                // too; one a rewrite expands over the reader's terms alone is not.
                if (!(query instanceof org.apache.lucene.search.MultiTermQuery m) || expandsNatively(m) == false) {
                    expands[0] = true;
                }
            }

            @Override
            public org.apache.lucene.search.QueryVisitor getSubVisitor(org.apache.lucene.search.BooleanClause.Occur occur, Query parent) {
                return this;
            }
        };
        // The query as parsed too: a scoring rewrite expanded its terms already, maybe to nothing.
        if (ctx.parsedQuery() != null) {
            ctx.parsedQuery().query().visit(visitor);
        }
        ctx.query().visit(visitor);
        return expands[0] ? "multi_term" : null;
    }

    /** Whether {@code query} is or holds a {@link KnnHitsQuery}, under any wrapper. */
    private static boolean holdsKnnHits(Query query) {
        boolean[] found = new boolean[1];
        query.visit(new org.apache.lucene.search.QueryVisitor() {
            @Override
            public void visitLeaf(Query q) {
                if (q instanceof KnnHitsQuery) {
                    found[0] = true;
                }
            }

            @Override
            public org.apache.lucene.search.QueryVisitor getSubVisitor(org.apache.lucene.search.BooleanClause.Occur occur, Query parent) {
                return this;
            }
        });
        return found[0] || query instanceof KnnHitsQuery;
    }

    /** Whether {@code q}'s rewrite leaves its terms to the search: a constant-score or doc-values one. */
    private static boolean expandsNatively(org.apache.lucene.search.MultiTermQuery q) {
        org.apache.lucene.search.MultiTermQuery.RewriteMethod m = q.getRewriteMethod();
        return m == org.apache.lucene.search.MultiTermQuery.CONSTANT_SCORE_REWRITE
            || m == org.apache.lucene.search.MultiTermQuery.CONSTANT_SCORE_BLENDED_REWRITE
            || m == org.apache.lucene.search.MultiTermQuery.DOC_VALUES_REWRITE;
    }

    /**
     * Whether a plain scored search goes to the buffers written as segments rather than the view's
     * own postings: {@code min_score}, a concurrent count's per-slice early termination, and a
     * scroll or {@code search_after} page, which the view's search does not do, and an index with
     * nested documents, whose blocks the view's postings do not tell apart (the buffers written as
     * segments keep each block whole, its root last). A scroll's or point in time's pages share one
     * searcher supplier, so one view and one numbering of its documents.
     */
    static boolean plainIneligible(SearchContext ctx) {
        return ctx.minimumScore() != null || ctx.size() == 0 && ctx.shouldUseConcurrentSearch()
            || ctx.scrollContext() != null || ctx.searchAfter() != null || ctx.mapperService().hasNested()
            // An explanation is Lucene's, over the postings of the buffers written as segments.
            || ctx.explain();
    }

    /** Why this request cannot run native, or null when it can. */
    static String ineligible(
        SearchContext ctx,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean nativeAggs
    ) {
        if (SearchFlags.of(ctx).enabled() == false) return FallbackReason.DISABLED.key();
        // The specific reasons first: each of these also adds a collector, and "collectors" alone
        // would not tell an operator which request feature to look at.
        // Aggregations only when NativeAggregations plans every one of them, and nothing else (a
        // global aggregation, another plugin's manager) added a collector manager.
        if (ctx.queryCollectorManagers().isEmpty() == false
            && (nativeAggs == false
                || ctx.queryCollectorManagers()
                    .keySet()
                    .stream()
                    .anyMatch(k -> k != NonGlobalAggCollectorManager.class && k != GlobalAggCollectorManager.class))) {
            return NativeAggregationTree.unsupportedReason(ctx);
        }
        boolean minScore = ctx.minimumScore() != null;
        // min_score natively by score or behind a sort (MinimumScoreCollector around the sort's
        // collector: no competitive iterator); behind a scroll's later pages or terminate_after it
        // stays on Lucene.
        if (minScore && (ctx.scrollContext() != null || ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER)) {
            return FallbackReason.MIN_SCORE.key();
        }
        boolean terminating = ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER;
        if (terminating && terminateAfterIneligible(ctx)) return FallbackReason.TERMINATE_AFTER.key();
        // What is left in "collectors": the aggregations' context and the filter collectors
        // (post_filter, terminate_after, min_score).
        boolean postFilter = ctx.parsedPostFilter() != null;
        if (collectors.size() > (nativeAggs ? 1 : 0) + (postFilter ? 1 : 0) + (terminating ? 1 : 0) + (minScore ? 1 : 0)
            || hasFilterCollector != (postFilter || terminating || minScore)) return FallbackReason.COLLECTORS.key();
        // A sort runs native when SortEncoder can encode it (searchWith); search_after only
        // with one. track_scores runs native too: the score leading the sort gives the max
        // score (the first hit), and otherwise the native search tracks it over every match.
        if (ctx.searchAfter() != null && ctx.sort() == null) return FallbackReason.SEARCH_AFTER.key();
        // size 0 behind search_after: ContextIndexSearcher skips segments the after value rules out
        // before any collector sees them, which the native count does not.
        if (ctx.searchAfter() != null && ctx.size() == 0) return FallbackReason.SEARCH_AFTER.key();
        // collapse natively (NativeCollapse) on a keyword or numeric field, behind terminate_after
        // too: EarlyTerminatingCollector lets the first matches in index order through, which the
        // native collector takes as the plain sorted search does (terminateAfterIneligible above).
        if (ctx.collapse() != null && ctx.size() > 0) {
            if (NativeCollapse.blob(ctx.collapse()) == null) return FallbackReason.COLLAPSE_FIELD.key();
            // CollapsingTopDocsCollector refuses search_after unless the sort
            // is the collapse field.
            SortField[] keys = NativeCollapse.sort(ctx).getSort();
            if (ctx.searchAfter() != null
                && (keys.length != 1 || java.util.Objects.equals(keys[0].getField(), ctx.collapse().getFieldName()) == false)) {
                return FallbackReason.COLLAPSE_SEARCH_AFTER.key();
            }
        }
        // rescore natively (NativeRescore) when every rescorer is OpenSearch's
        // QueryRescorer, over the top docs however they were collected (behind
        // terminate_after too); a scroll or a size-0 search never rescores.
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false && ctx.scrollContext() == null && ctx.size() > 0) {
            if (NativeRescore.supported(ctx) == false) return FallbackReason.RESCORE.key();
        }
        // A shard with an @timestamp field sorted by it ascending: ContextIndexSearcher visits
        // each slice's segments last first: NativeAggregations.slices hands the native pass that order.
        // terminate_after cuts the segments in index order, which the reversed visit does not.
        if (ctx.shouldUseTimeSeriesDescSortOptimization() && terminating) return FallbackReason.TIME_SERIES_ORDER.key();
        // dfs_query_then_fetch scores with statistics aggregated across shards
        // (ContextIndexSearcher.setAggregatedDfs), which NativeDfs hands the native statistics pass.
        if (ctx.searchType() == SearchType.DFS_QUERY_THEN_FETCH && NativeDfs.readable() == false) return FallbackReason.DFS.key();
        // Another plugin may replace the top-docs collector (QueryPhase's own first question).
        try {
            if (QueryCollectorContextSpecRegistry.getQueryCollectorContextSpec(
                ctx,
                ctx.query(),
                new QueryCollectorArguments.Builder().hasFilterCollector(hasFilterCollector).build()
            ).isPresent()) return FallbackReason.COLLECTOR_SPEC.key();
        } catch (IOException e) {
            return FallbackReason.COLLECTOR_SPEC.key();
        }
        return null;
    }

    /**
     * Whether a {@code terminate_after} request needs something the native cut does not give it.
     * With aggregations: OpenSearch's collectors let an aggregation see the whole of a segment it
     * answers from index statistics (the points min/max, terms from doc frequencies) past the cut.
     * With {@code search_after}: ContextIndexSearcher skips a segment the after value rules out
     * before EarlyTerminatingCollector counts it. A scroll: each page recounts its prefix. And
     * more documents than {@code track_total_hits} counts exactly: the collector's total there is a
     * lower bound whose value depends on when it stopped counting.
     */
    static boolean terminateAfterIneligible(SearchContext ctx) {
        int trackUpTo = ctx.trackTotalHitsUpTo();
        return ctx.queryCollectorManagers().isEmpty() == false
            || (countSegments(ctx, ctx.terminateAfter()) && segmentCountable(ctx.query()) == false)
            || (ctx.size() > 0
                && ctx.parsedPostFilter() == null
                && ((scoresNeeded(ctx) == false && neverCached(ctx) == false) || rangeCollected(ctx.query())))
            || ctx.searchAfter() != null
            || ctx.scrollContext() != null
            || (trackUpTo != SearchContext.TRACK_TOTAL_HITS_ACCURATE
                && trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED
                && ctx.terminateAfter() > trackUpTo);
    }

    /**
     * Whether an unscored search's query never goes through the query cache, whose cached bulk
     * scorer hands the collector whole ranges once a query has been seen often enough (history
     * no native search reproduces): a plain {@code TermQuery}, which {@code
     * UsageTrackingQueryCachingPolicy.shouldNeverCache} turns away, unless the index caches
     * everything ({@code index.queries.cache.everything}).
     */
    static boolean neverCached(SearchContext ctx) {
        Query q = ctx.query();
        while (q instanceof BoostQuery b) {
            q = b.getQuery();
        }
        return q.getClass() == TermQuery.class
            && org.opensearch.index.IndexModule.INDEX_QUERY_CACHE_EVERYTHING_SETTING.get(ctx.indexShard().indexSettings().getSettings())
                == false;
    }

    /**
     * Whether a {@code terminate_after} search's total is a {@code size: 0} search's: counted by
     * TotalHitCountCollector (EmptyTopDocsCollectorContext), which takes a segment's {@code
     * Weight.count} whole where there is one -- past the cut -- rather than the documents let
     * through.
     */
    static boolean countSegments(SearchContext ctx, int terminateAfter) {
        // FilteredCollector passes no weight on: under a post_filter the count collector counts
        // the documents let through.
        return terminateAfter > 0 && ctx.size() == 0 && ctx.parsedPostFilter() == null
            && ctx.trackTotalHitsUpTo() != SearchContext.TRACK_TOTAL_HITS_DISABLED;
    }

    /**
     * Whether the top-docs collector needs scores, so that the MultiCollector beside
     * EarlyTerminatingCollector (which needs none) runs the search {@code COMPLETE}.
     */
    static boolean scoresNeeded(SearchContext ctx) {
        if (ctx.sort() == null || ctx.trackScores()) {
            return true;
        }
        for (SortField f : ctx.sort().sort.getSort()) {
            if (f.getType() == SortField.Type.SCORE) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether Lucene may hand {@code query}'s matches to the collector in {@code collectRange}
     * batches when searching {@code COMPLETE}: DenseConjunctionBulkScorer (a constant-score
     * query's bulk scorer when dense, a conjunction of filters) and the query cache's. MultiCollector
     * gives a batch to EarlyTerminatingCollector first, which throws inside it, so the top-docs
     * collector loses the whole batch the cut falls in -- which the native cut, document by
     * document, does not reproduce. The positive bulk scorer is BooleanScorerSupplier.booleanScorer's
     * choice: a lone optional or required clause's own, else one that collects by document.
     */
    static boolean rangeCollected(Query query) {
        Query q = query;
        while (q instanceof BoostQuery b) {
            q = b.getQuery();
        }
        if (q instanceof org.apache.lucene.search.BooleanQuery bq) {
            List<Query> must = new ArrayList<>();
            List<Query> filter = new ArrayList<>();
            List<Query> should = new ArrayList<>();
            for (var c : bq.clauses()) {
                switch (c.occur()) {
                    case MUST -> must.add(c.query());
                    case FILTER -> filter.add(c.query());
                    case SHOULD -> should.add(c.query());
                    default -> {
                    }
                }
            }
            int required = must.size() + filter.size();
            if (required == 0) {
                return should.size() == 1 && rangeCollected(should.get(0));
            }
            if (should.isEmpty() && bq.getMinimumNumberShouldMatch() == 0) {
                if (required == 1) {
                    // A lone filter: its non-scoring (possibly cached) bulk scorer.
                    return must.isEmpty() || rangeCollected(must.get(0));
                }
                // Filters only: DenseConjunctionBulkScorer when dense.
                return must.isEmpty();
            }
            return false;
        }
        return q instanceof TermQuery == false
            && q instanceof org.apache.lucene.search.PhraseQuery == false
            && q instanceof org.apache.lucene.search.DisjunctionMaxQuery == false;
    }

    /**
     * Whether {@code query}'s {@code Weight.count} is one the native side has: a term's, a
     * match-all's or a match-none's, through the wrappers whose weights pass {@code count} on.
     * (With a post_filter, FilteredCollector does not pass the weight to the count collector at
     * all; that case falls back too.)
     */
    static boolean segmentCountable(Query query) {
        Query q = query;
        while (true) {
            if (q instanceof BoostQuery b) {
                q = b.getQuery();
            } else if (q instanceof ConstantScoreQuery c) {
                q = c.getQuery();
            } else if (q instanceof ApproximateScoreQuery a) {
                q = a.getOriginalQuery();
            } else {
                break;
            }
        }
        return q.getClass() == TermQuery.class || q.getClass() == MatchAllDocsQuery.class
            || q.getClass() == org.apache.lucene.search.MatchNoDocsQuery.class;
    }
}
