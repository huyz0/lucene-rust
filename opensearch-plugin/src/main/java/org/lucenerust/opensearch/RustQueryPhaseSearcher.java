/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.search.aggregations.AggregationProcessor;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.NonGlobalAggCollectorManager;
import org.opensearch.search.approximate.ApproximateScoreQuery;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.internal.ScrollContext;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.action.search.SearchType;
import org.opensearch.search.query.QueryCollectorArguments;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryCollectorContextSpecRegistry;
import org.opensearch.search.query.QueryPhaseExecutionException;
import org.opensearch.search.query.QueryPhaseSearcher;
import org.opensearch.search.query.QueryPhaseSearcherWrapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Runs the query phase of a shard search natively when the whole request is inside the supported
 * matrix, and on Lucene otherwise -- per query, transparently (M2 T2.4).
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
 * scroll, {@code terminate_after} and (by score) {@code min_score} -- with no collapse, rescore or profile, no aggregation beyond the
 * metrics {@link NativeAggregations} plans, and a query {@link QueryEncoder} can encode over fields
 * using the default {@link BM25Similarity}. What it produces is what OpenSearch's own {@code
 * SimpleTopDocsCollectorContext} would: the top {@code from + size} hits (with their sort values),
 * the max score, and total hits under the same {@code track_total_hits} rules; and, with
 * aggregations, the shard results their aggregators would have built, stored before {@code
 * DefaultAggregationProcessor.postProcess} runs so that it keeps them rather than reducing its own
 * (unfed) collectors.
 *
 * <p>A native failure is logged, counted, and the query is re-run on Lucene: it never fails a
 * search that Lucene could answer.
 */
public final class RustQueryPhaseSearcher implements QueryPhaseSearcher {
    private static final Logger logger = LogManager.getLogger(RustQueryPhaseSearcher.class);

    /** Per-index switch, dynamic: {@code PUT idx/_settings {"index.lucene_rust.search.enabled": false}}. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "index.lucene_rust.search.enabled",
        true,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    /**
     * Which encodable shapes run native: {@code fast} (default) only those measured at least as fast
     * as Lucene ({@link QueryEncoder#isFast}); {@code all} every shape the engine answers correctly.
     * Since read path R1 the two agree -- every encodable shape is measured faster -- and the
     * setting stays so that indices which set it keep opening.
     */
    public static final Setting<String> NATIVE_SHAPES = new Setting<>(
        "index.lucene_rust.search.native_shapes",
        "fast",
        v -> {
            if (v.equals("fast") == false && v.equals("all") == false) {
                throw new IllegalArgumentException("index.lucene_rust.search.native_shapes must be [fast] or [all], got [" + v + "]");
            }
            return v;
        },
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    private final QueryPhaseSearcher fallback = new QueryPhaseSearcherWrapper();
    private final NativeReaders readers;
    private final SearchStats stats;

    public RustQueryPhaseSearcher(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
    }

    @Override
    public AggregationProcessor aggregationProcessor(SearchContext searchContext) {
        return fallback.aggregationProcessor(searchContext);
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
        long start = System.nanoTime();
        // Aggregations OpenSearch's DefaultAggregationProcessor.preProcess registered (their one
        // collector context is the one in "collectors"), when all of them run natively.
        NativeAggregations.Plan aggs = ctx.queryCollectorManagers().isEmpty() ? null : NativeAggregations.plan(ctx);
        String reason = ineligible(ctx, collectors, hasFilterCollector, aggs != null);
        byte[] blob = null;
        // A post_filter narrows the hits, not the aggregations (QueryPhase wraps only the top-docs
        // collector in its FilteredCollector): the hits search "query AND filter", the filter a
        // non-scoring clause, so each hit keeps the query's score; the aggregations the query.
        byte[] hitsBlob = null;
        if (reason == null) {
            QueryEncoder.Encoded enc = QueryEncoder.encode(query, field -> defaultBm25(searcher, field));
            reason = enc.fallbackReason();
            blob = enc.blob();
            boolean fast = enc.fast();
            hitsBlob = blob;
            if (reason == null && ctx.parsedPostFilter() != null) {
                QueryEncoder.Encoded filtered = QueryEncoder.encode(
                    postFiltered(query, searcher.rewrite(ctx.parsedPostFilter().query())),
                    field -> defaultBm25(searcher, field)
                );
                reason = filtered.fallbackReason();
                hitsBlob = filtered.blob();
                fast = fast && filtered.fast();
            }
            if (reason == null && fast == false && flags(ctx).allShapes() == false) {
                reason = "slower_shape";
            }
            // min_score: OpenSearch's MinimumScoreCollector sits outside every other collector,
            // aggregations included, so every blob carries it.
            if (reason == null && ctx.minimumScore() != null) {
                blob = QueryEncoder.withMinScore(blob, ctx.minimumScore());
                hitsBlob = QueryEncoder.withMinScore(hitsBlob, ctx.minimumScore());
            }
        }
        // OpenSearch's approximate match_all/range (ApproximateScoreQuery resolved to its
        // ApproximateQuery) collects documents in BKD order and so breaks ties differently from
        // Lucene's exact answer, which is what the native engine gives: stay on OpenSearch's.
        if (reason == null && approximated(query)) {
            reason = "approximate";
        }
        // A sorted search: the sort blob, unless the hits are not asked for at all (size 0 has no
        // order to keep, and runs as the unsorted count it is).
        byte[] sortBlob = null;
        // A scroll's later pages search after the last hit it emitted
        // (ScrollingTopDocsCollectorContext); one ordered by score, as the score sort it is:
        // PagingFieldCollector skips what PagingTopScoreDocCollector does, and keeps the same hits.
        Sort sort = ctx.sort() == null ? null : ctx.sort().sort;
        FieldDoc after = ctx.searchAfter();
        // A search by score run as the score sort, its hits handed back as ScoreDocs.
        boolean scoreDocs = false;
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
                reason = "scroll_after";
            }
        }
        // terminate_after runs on the sorted search, which takes the cut (SortEncoder): by score as
        // the score sort, and without hits (size 0) by _doc, the cheapest order, its one hit dropped.
        int terminateAfter = ctx.terminateAfter() == SearchContext.DEFAULT_TERMINATE_AFTER ? 0 : ctx.terminateAfter();
        if (terminateAfter > 0 && ctx.size() == 0) {
            sort = new Sort(SortField.FIELD_DOC);
        } else if (terminateAfter > 0 && sort == null) {
            sort = new Sort(SortField.FIELD_SCORE);
            scoreDocs = true;
        }
        if (reason == null && sort != null && (ctx.size() > 0 || terminateAfter > 0)) {
            // The request's own sort only: one put in here (the score, _doc) stands for a
            // collector with no index-sort early exit (QueryPhase.canEarlyTerminate).
            boolean ownSort = ctx.sort() != null && sort == ctx.sort().sort;
            if (ownSort && indexSorted(searcher.getIndexReader())) {
                // TopFieldCollector stops early on an index sorted by the search's sort; the
                // native collector has no such path.
                reason = "index_sort";
            } else {
                // track_scores behind another key: OpenSearch's MaxScoreCollector over every match.
                boolean trackMaxScore = ctx.trackScores() && ctx.size() > 0 && sortByScore(sort) == false;
                // A concurrent search collects each slice separately, as Lucene does.
                int[][] slices = NativeAggregations.slices(ctx);
                SortEncoder.Encoded sorted = slices == null ? new SortEncoder.Encoded(null, "intra_segment")
                    : SortEncoder.encode(sort, after, trackMaxScore, slices, terminateAfter, countSegments(ctx, terminateAfter));
                reason = sorted.fallbackReason();
                sortBlob = sorted.blob();
            }
        }
        NativeReaders.Acquired acquired = null;
        if (reason == null) {
            acquired = readers.acquire(searcher.getIndexReader());
            reason = acquired.fallbackReason();
        }
        if (reason == null) {
            // A timeout already past: Lucene would skip every segment and answer nothing, timed
            // out -- which is what its own path does fastest.
            if (ctx.isCancelled()) {
                reason = "cancelled";
            } else if (hasTimeout && deadlinePassed(ctx)) {
                reason = "timeout";
            } else {
                reason = searchNative(ctx, acquired.handle(), blob, hitsBlob, sortBlob, sort, scoreDocs, aggs);
            }
            if (reason == null) {
                stats.nativeQuery();
                if (hasTimeout && deadlinePassed(ctx)) {
                    timedOut(ctx);
                }
                stats.nativeTime(System.nanoTime() - start);
                return false;
            }
        }
        stats.fallback(reason);
        // Lucene's own time only: not the plugin's checks, nor a native attempt that fell back.
        long luceneStart = System.nanoTime();
        boolean rescore = fallback.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
        stats.luceneTime(System.nanoTime() - luceneStart);
        return rescore;
    }

    /** Why this request cannot run native, or null when it can. */
    static String ineligible(
        SearchContext ctx,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean nativeAggs
    ) {
        if (flags(ctx).enabled() == false) return "disabled";
        // The specific reasons first: each of these also adds a collector, and "collectors" alone
        // would not tell an operator which request feature to look at.
        // Aggregations only when NativeAggregations plans every one of them, and nothing else (a
        // global aggregation, another plugin's manager) added a collector manager.
        if (ctx.queryCollectorManagers().isEmpty() == false
            && (nativeAggs == false
                || ctx.queryCollectorManagers().size() != 1
                || ctx.queryCollectorManagers().containsKey(NonGlobalAggCollectorManager.class) == false)) return "aggregations";
        boolean minScore = ctx.minimumScore() != null;
        // min_score natively by score only: behind a sort, a scroll's later pages or
        // terminate_after it stays on Lucene.
        if (minScore
            && (ctx.sort() != null || ctx.scrollContext() != null || ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER)) {
            return "min_score";
        }
        boolean terminating = ctx.terminateAfter() != SearchContext.DEFAULT_TERMINATE_AFTER;
        if (terminating && terminateAfterIneligible(ctx)) return "terminate_after";
        // What is left in "collectors": the aggregations' context and the filter collectors
        // (post_filter, terminate_after, min_score).
        boolean postFilter = ctx.parsedPostFilter() != null;
        if (collectors.size() > (nativeAggs ? 1 : 0) + (postFilter ? 1 : 0) + (terminating ? 1 : 0) + (minScore ? 1 : 0)
            || hasFilterCollector != (postFilter || terminating || minScore)) return "collectors";
        // A sort runs native when SortEncoder can encode it (searchWith); search_after only
        // with one. track_scores runs native too: the score leading the sort gives the max
        // score (the first hit), and otherwise the native search tracks it over every match.
        if (ctx.searchAfter() != null && ctx.sort() == null) return "search_after";
        // size 0 behind search_after: ContextIndexSearcher skips segments the after value rules out
        // before any collector sees them, which the native count does not.
        if (ctx.searchAfter() != null && ctx.size() == 0) return "search_after";
        if (ctx.collapse() != null) return "collapse";
        if (ctx.rescore() != null && ctx.rescore().isEmpty() == false) return "rescore";
        if (ctx.getProfilers() != null) return "profile";
        // A shard with an @timestamp field sorted by it ascending: ContextIndexSearcher visits
        // each slice's segments last first, which breaks sort ties and orders an aggregation's sums
        // differently from the native pass (ascending); stay on OpenSearch's.
        if (ctx.shouldUseTimeSeriesDescSortOptimization()) return "time_series_order";
        // dfs_query_then_fetch scores with statistics aggregated across shards
        // (ContextIndexSearcher.setAggregatedDfs); the native engine only knows this shard's.
        if (ctx.searchType() == SearchType.DFS_QUERY_THEN_FETCH) return "dfs";
        // Another plugin may replace the top-docs collector (QueryPhase's own first question).
        try {
            if (QueryCollectorContextSpecRegistry.getQueryCollectorContextSpec(
                ctx,
                ctx.query(),
                new QueryCollectorArguments.Builder().hasFilterCollector(hasFilterCollector).build()
            ).isPresent()) return "collector_spec";
        } catch (IOException e) {
            return "collector_spec";
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
            || (ctx.size() > 0 && ctx.parsedPostFilter() == null && (scoresNeeded(ctx) == false || rangeCollected(ctx.query())))
            || ctx.searchAfter() != null
            || ctx.scrollContext() != null
            || (trackUpTo != SearchContext.TRACK_TOTAL_HITS_ACCURATE
                && trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED
                && ctx.terminateAfter() > trackUpTo);
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

    /** The plugin's index settings as last read, and the settings version they were read at. */
    private record Flags(long version, boolean enabled, boolean allShapes) {}

    private static final java.util.concurrent.ConcurrentHashMap<String, Flags> FLAGS = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * The plugin's index settings for {@code ctx}'s index, re-read only when the index's settings
     * version moves: {@code IndexSettings.getValue} parses the setting's string on every call, a
     * cost every search paid twice. One small entry per index ever searched on this node.
     */
    static Flags flags(SearchContext ctx) {
        org.opensearch.index.IndexSettings settings = ctx.indexShard().indexSettings();
        long version = settings.getIndexMetadata().getSettingsVersion();
        String uuid = settings.getUUID();
        Flags f = FLAGS.get(uuid);
        if (f == null || f.version() != version) {
            f = new Flags(version, settings.getValue(ENABLED), "all".equals(settings.getValue(NATIVE_SHAPES)));
            FLAGS.put(uuid, f);
        }
        return f;
    }

    /** True when {@code field} scores with a default-parameter {@link BM25Similarity}. */
    static boolean defaultBm25(ContextIndexSearcher searcher, String field) {
        Similarity sim = searcher.getSimilarity();
        if (sim instanceof PerFieldSimilarityWrapper w) {
            sim = w.get(field);
        }
        return sim != null
            && sim.getClass() == BM25Similarity.class
            && ((BM25Similarity) sim).getK1() == 1.2f
            && ((BM25Similarity) sim).getB() == 0.75f
            && sim.getDiscountOverlaps();
    }

    /**
     * Runs the blob and stores the result; returns null, or the fallback reason when it did not.
     *
     * <p>Cancellation is checked before the call, not during it: a native search runs to completion
     * (see docs/opensearch-native-queries.md, Known limits).
     */
    private String searchNative(
        SearchContext ctx,
        long handle,
        byte[] blob,
        byte[] hitsBlob,
        byte[] sortBlob,
        Sort sort,
        boolean scoreDocs,
        NativeAggregations.Plan aggs
    ) {
        int size = ctx.size();
        ScrollContext scroll = ctx.scrollContext();
        // A concurrent size-0 search reports terminated_early per slice: without whole-segment
        // slices there is nothing to replay.
        if (size == 0 && ctx.shouldUseConcurrentSearch() && NativeAggregations.slices(ctx) == null) {
            return "intra_segment";
        }
        // A scroll page is "size" hits whatever "from" says, and counts them on its first page only
        // (TopDocsCollectorContext.createTopDocsCollectorContext).
        int numDocs = size == 0 ? 0
            : Math.min(scroll != null ? size : ctx.from() + size, Math.max(1, ctx.searcher().getIndexReader().numDocs()));
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
        // TopDocsCollectorContext: "hasFilterCollector ? -1 : shortcutTotalHitCount(...)".
        if (trackUpTo != SearchContext.TRACK_TOTAL_HITS_DISABLED
            && ctx.parsedPostFilter() == null
            && terminating == false
            && ctx.minimumScore() == null) {
            try {
                shortcut = shortcutTotalHitCount(ctx.searcher().getIndexReader(), ctx.query());
            } catch (IOException e) {
                return "native_error";
            }
            if (shortcut >= 0) {
                countLimit = numDocs;
            }
        }
        if (ctx.isCancelled()) {
            return "cancelled";
        }
        // The aggregations first, kept aside until the hits are in too: a failure in either
        // re-runs the whole request on Lucene.
        InternalAggregations aggResult = null;
        if (aggs != null) {
            int[][] slices = NativeAggregations.slices(ctx);
            if (slices == null) {
                return "intra_segment";
            }
            int states = aggs.metrics().size() * Math.max(1, slices.length);
            long[] aggCounts = new long[states];
            double[] aggValues = new double[states * NativeAggregations.VALUES];
            // Every aggregation in one native pass: each segment's matches collected once.
            byte[][] terms = new byte[1][];
            int rc = NativeBridge.aggregate(handle, blob, aggs.blob(slices), aggCounts, aggValues, terms);
            if (rc != NativeBridge.OK) {
                stats.nativeError();
                logger.warn("lucene-rust: native aggregation failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
                return "native_error";
            }
            aggResult = aggs.build(slices, aggCounts, aggValues, terms[0], ctx.partialOnShard());
        }
        String reason = sortBlob != null
            ? searchSorted(ctx, handle, hitsBlob, sortBlob, sort, scoreDocs, Math.max(1, numDocs), countLimit, shortcut)
            : searchUnsorted(ctx, handle, hitsBlob, numDocs, countLimit, shortcut);
        if (reason == null && aggResult != null) {
            // DefaultAggregationProcessor.postProcess keeps a result already there (hasAggs).
            ctx.queryResult().aggregations(aggResult);
        }
        return reason;
    }

    /** The unsorted half of {@link #searchNative}. */
    private String searchUnsorted(SearchContext ctx, long handle, byte[] blob, int numDocs, long countLimit, int shortcut) {
        int[] docs = new int[numDocs];
        float[] scores = new float[numDocs];
        long[] counts = new long[3];
        int rc = NativeBridge.search(handle, blob, numDocs, countLimit, docs, scores, counts);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return "native_error";
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
            Boolean terminated = countTerminatedEarly(ctx, handle, blob, shortcut, total);
            if (terminated == null) {
                return "native_error";
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
    private Boolean countTerminatedEarly(SearchContext ctx, long handle, byte[] blob, int shortcut, TotalHits total) {
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
        // Past the limit, it depends on which segments the count collector answers from Weight.count
        // (nothing reaches the terminating collector) and which it iterates: Lucene's own weight,
        // the query cache included, says; the native side replays each slice.
        int[][] slices = NativeAggregations.slices(ctx);
        if (slices == null) {
            return null;
        }
        java.io.ByteArrayOutputStream spec = new java.io.ByteArrayOutputStream();
        NativeAggregations.writeInt(spec, trackUpTo);
        NativeAggregations.writeInt(spec, leaves.size());
        try {
            Weight weight = ctx.parsedPostFilter() != null || ctx.minimumScore() != null
                ? null
                : ctx.searcher().createWeight(ctx.query(), org.apache.lucene.search.ScoreMode.COMPLETE_NO_SCORES, 1f);
            for (LeafReaderContext leaf : leaves) {
                // Under a post_filter or min_score the count collector gets no weight (neither
                // FilteredCollector nor MinimumScoreCollector passes it on): it iterates everywhere.
                spec.write(weight == null || weight.count(leaf) == -1 ? 1 : 0);
            }
        } catch (IOException e) {
            return null;
        }
        NativeAggregations.writeSlices(spec, slices);
        long[] out = new long[1];
        int rc = NativeBridge.countTerminates(handle, blob, spec.toByteArray(), out);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native count replay failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return null;
        }
        return out[0] != 0;
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
        int rc = NativeBridge.searchSorted(handle, blob, sortBlob, numDocs, countLimit, docs, values, counts, terms);
        if (rc != NativeBridge.OK) {
            stats.nativeError();
            logger.warn("lucene-rust: native sorted search failed ({}), re-running on Lucene: {}", rc, NativeBridge.lastError());
            return "native_error";
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
        float maxScore = n > 0 && sortByScore(ctx.sort().sort) ? (float) hits[0].fields[0]
            : ctx.trackScores() ? Float.intBitsToFloat((int) counts[3])
            : Float.NaN;
        store(ctx, new TopDocsAndMaxScore(new TopFieldDocs(total, hits, fields), maxScore), ctx.sort().formats);
        return null;
    }

    /**
     * Whether the searcher's cancellation checks -- QueryPhase's timeout, and the task's own with
     * {@code low_level_cancellation} -- say stop, as ContextIndexSearcher asks them before each
     * segment.
     */
    static boolean deadlinePassed(SearchContext ctx) {
        QueryTimeout timeout = ctx.searcher().getTimeout();
        return timeout != null && timeout.shouldExit();
    }

    /**
     * A native search that ran past its timeout, answered as QueryPhase answers one: the task
     * cancelled fails it, as the TaskCancelledException ContextIndexSearcher does not catch would;
     * otherwise the request fails without {@code allow_partial_search_results} and is flagged
     * {@code timed_out} with it. Its results are complete -- the native search is not interrupted
     * part way (docs/opensearch-native-queries.md, Known limits) -- which a partial answer allows.
     */
    static void timedOut(SearchContext ctx) {
        // Only a registered low_level_cancellation check sees the task; the timeout's alone does not.
        if (ctx.lowLevelCancellation() && ctx.isCancelled()) {
            throw new TaskCancelledException("cancelled task with reason: " + ctx.getTask().getReasonCancelled());
        }
        if (ctx.request().allowPartialSearchResults() == false) {
            throw new QueryPhaseExecutionException(ctx.shardTarget(), "Time exceeded");
        }
        ctx.queryResult().searchTimedOut(true);
    }

    /** {@code +query #filter}: the documents both match, scored by {@code query} alone. */
    static Query postFiltered(Query query, Query filter) {
        return new org.apache.lucene.search.BooleanQuery.Builder().add(query, org.apache.lucene.search.BooleanClause.Occur.MUST)
            .add(filter, org.apache.lucene.search.BooleanClause.Occur.FILTER)
            .build();
    }

    private static final java.lang.reflect.Field RESOLVED_QUERY = resolvedQueryField();

    private static java.lang.reflect.Field resolvedQueryField() {
        try {
            java.lang.reflect.Field f = ApproximateScoreQuery.class.getDeclaredField("resolvedQuery");
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /**
     * Whether {@code query} holds an {@link ApproximateScoreQuery} that OpenSearch resolved to its
     * approximation for this request ({@code setContext}; the choice is package-private, so it is
     * read reflectively). Unreadable counts as approximated: falling back is always correct.
     */
    static boolean approximated(Query query) {
        if (query instanceof ApproximateScoreQuery a) {
            if (RESOLVED_QUERY == null) {
                return true;
            }
            try {
                return RESOLVED_QUERY.get(a) instanceof org.opensearch.search.approximate.ApproximateQuery;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return true;
            }
        }
        if (query instanceof org.apache.lucene.search.BooleanQuery b) {
            for (var c : b.clauses()) {
                if (approximated(c.query())) {
                    return true;
                }
            }
            return false;
        }
        if (query instanceof ConstantScoreQuery c) {
            return approximated(c.getQuery());
        }
        if (query instanceof BoostQuery b) {
            return approximated(b.getQuery());
        }
        return false;
    }

    /** {@code SortField.FIELD_SCORE.equals(sort.getSort()[0])}: the score, descending, leads. */
    static boolean sortByScore(Sort sort) {
        return SortField.FIELD_SCORE.equals(sort.getSort()[0]);
    }

    /** Whether any segment carries an index sort ({@code index.sort.*}). */
    static boolean indexSorted(IndexReader reader) {
        for (LeafReaderContext leaf : reader.leaves()) {
            if (leaf.reader().getMetaData().sort() != null) {
                return true;
            }
        }
        return false;
    }

    /**
     * {@code TopDocsCollectorContext.shortcutTotalHitCount} (OpenSearch 3.8.0), for the queries the
     * native engine runs: a match-all's count is the reader's {@code numDocs}, and an exact term's
     * the sum of its {@code docFreq}s when nothing is deleted; {@code -1} otherwise.
     */
    static int shortcutTotalHitCount(IndexReader reader, Query query) throws IOException {
        while (true) {
            if (query instanceof ConstantScoreQuery c) {
                query = c.getQuery();
            } else if (query instanceof BoostQuery b) {
                query = b.getQuery();
            } else if (query instanceof ApproximateScoreQuery a) {
                query = a.getOriginalQuery();
            } else {
                break;
            }
        }
        if (query.getClass() == MatchAllDocsQuery.class) {
            return reader.numDocs();
        } else if (query.getClass() == TermQuery.class && reader.hasDeletions() == false) {
            Term term = ((TermQuery) query).getTerm();
            int count = 0;
            for (LeafReaderContext leaf : reader.leaves()) {
                count += leaf.reader().docFreq(term);
            }
            return count;
        }
        return -1;
    }
}
