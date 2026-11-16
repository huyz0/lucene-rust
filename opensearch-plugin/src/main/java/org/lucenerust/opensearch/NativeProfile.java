/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.ToLongFunction;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TotalHitCountCollector;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.profile.AbstractProfileBreakdown;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.aggregation.AggregationProfileBreakdown;
import org.opensearch.search.profile.aggregation.AggregationProfiler;
import org.opensearch.search.profile.aggregation.AggregationTimingType;
import org.opensearch.search.profile.query.InternalProfileCollector;
import org.opensearch.search.profile.query.InternalProfileCollectorManager;
import org.opensearch.search.profile.query.InternalProfileComponent;
import org.opensearch.search.profile.query.QueryProfiler;
import org.opensearch.search.profile.query.QueryTimingType;
import org.opensearch.search.query.ConcurrentQueryPhaseSearcher;
import org.opensearch.search.query.QueryCollectorContext;
import org.opensearch.search.query.QueryPhase;

/**
 * The {@code profile} option of a search answered natively: OpenSearch's own profile result
 * ({@link Profilers}), filled in with what the native pass measured.
 *
 * <p>A profiled search runs natively like any other. Its profile is then made from
 *
 * <ul>
 *   <li>the query tree: the query run once more natively with a timer around every clause of its
 *       scorer tree ({@code profile_blob} in the engine crate), the timers laid into the breakdown
 *       OpenSearch's {@link QueryProfiler} keeps for each query of the tree, as {@code
 *       ProfileWeight} and {@code ProfileScorer} do for a Lucene run (the same counts, and the same
 *       sampled timing of a call);
 *   <li>the collector tree: the one OpenSearch builds for the request's collectors, with the time
 *       the native calls took;
 *   <li>the aggregation tree: the aggregators OpenSearch builds, with the time the native
 *       aggregation took.
 * </ul>
 *
 * Fetch is OpenSearch's own phase and is profiled by it.
 */
final class NativeProfile {
    private static final Logger logger = LogManager.getLogger(NativeProfile.class);

    private NativeProfile() {}

    /** Whether {@code ctx} asks for a profile. */
    static boolean requested(SearchContext ctx) {
        return ctx.getProfilers() != null;
    }

    /** The timers {@code profile_blob} reports, by timer kind ({@code Kind} in {@code profile.rs}). */
    static final int TIMERS = 8;

    /** One clause's timers over one segment. */
    record Node(int segment, int[] path, String kind, long[] nanos, long[] counts, long[] starts) {}

    /** What {@code profile_blob} reports. */
    record Report(long clock, long total, List<Node> nodes) {
        static Report decode(byte[] bytes) {
            ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            long clock = b.getLong();
            long total = b.getLong();
            int count = b.getInt();
            List<Node> nodes = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                int segment = b.getInt();
                int[] path = new int[b.getInt()];
                for (int p = 0; p < path.length; p++) {
                    path[p] = b.getInt();
                }
                byte[] kind = new byte[b.get() & 0xff];
                b.get(kind);
                long[] nanos = new long[TIMERS];
                long[] counts = new long[TIMERS];
                long[] starts = new long[TIMERS];
                for (int t = 0; t < TIMERS; t++) {
                    nanos[t] = b.getLong();
                    counts[t] = b.getLong();
                    starts[t] = b.getLong();
                }
                nodes.add(new Node(segment, path, new String(kind, StandardCharsets.UTF_8), nanos, counts, starts));
            }
            return new Report(clock, total, nodes);
        }
    }

    /** The timers {@link #TIMERS} in the order of the native {@code Kind}, as the query timing types. */
    private static final QueryTimingType[] TYPES = {
        QueryTimingType.BUILD_SCORER,
        QueryTimingType.NEXT_DOC,
        QueryTimingType.ADVANCE,
        QueryTimingType.MATCH,
        QueryTimingType.SCORE,
        QueryTimingType.SHALLOW_ADVANCE,
        QueryTimingType.COMPUTE_MAX_SCORE,
        QueryTimingType.SET_MIN_COMPETITIVE_SCORE };

    /** What a profiled search spent in each of its native calls, kept for the profile of its collectors. */
    static final class Phases {
        long hitsNanos;
        long aggNanos;
        long totalNanos;
        /** What the replay of the query collects: the top {@code topN} by score ({@code topScores}), or every match. */
        long topN;
        long countLimit;
        boolean topScores;
        /** Segments the native search read. */
        int slices = 1;
    }

    private static final ThreadLocal<Phases> PHASES = new ThreadLocal<>();

    /** Starts keeping the phases of the search on this thread, when {@code ctx} is profiled. */
    static Phases begin(SearchContext ctx) {
        if (requested(ctx) == false) {
            return null;
        }
        Phases p = new Phases();
        PHASES.set(p);
        return p;
    }

    /** The phases being kept, or null when the search is not profiled. */
    static Phases phases() {
        return PHASES.get();
    }

    static void end() {
        PHASES.remove();
    }

    /**
     * Runs {@code blob} once more with a timer around each clause, or null when the native call
     * fails.
     */
    static Report run(long handle, byte[] blob, boolean topScores, long topN, long countLimit, long[] javaClock) {
        byte[][] out = new byte[1][];
        javaClock[0] = System.nanoTime();
        int rc = NativeBridge.profile(handle, blob, topScores, topN, countLimit, out);
        if (rc != NativeBridge.OK) {
            NativeMemory.check(rc);
            logger.warn("lucene-rust: the native profile failed ({}): {}", rc, NativeBridge.lastError());
            return null;
        }
        return Report.decode(out[0]);
    }

    // ---- the query tree --------------------------------------------------------------------

    /** What a Java query is, as the native clause that runs it is named ({@code kind} of a node). */
    private static String kindOf(Query q) {
        if (q instanceof BooleanQuery) return "Boolean";
        if (q instanceof BoostQuery) return "Boost";
        if (q instanceof ConstantScoreQuery) return "ConstantScore";
        if (q instanceof DisjunctionMaxQuery) return "DisjunctionMax";
        if (q instanceof MatchAllDocsQuery) return "MatchAllDocs";
        if (q instanceof PhraseQuery) return "Phrase";
        if (q instanceof TermQuery) return "Term";
        return null;
    }

    /**
     * The queries OpenSearch makes a weight of inside {@code q}, in the order the native clause
     * lists them ({@code children} in {@code profile.rs}: a boolean's {@code MUST}, {@code FILTER},
     * {@code SHOULD} and {@code MUST_NOT} clauses).
     */
    static List<Query> children(Query q) {
        if (q instanceof BooleanQuery bq) {
            List<Query> out = new ArrayList<>();
            for (BooleanClause c : bq.clauses()) {
                out.add(c.query());
            }
            return out;
        }
        if (q instanceof org.lucenerust.opensearch.neural.query.HybridQuery h) return new ArrayList<>(h.getSubQueries());
        // BoostQuery.createWeight calls its query's, not the searcher's: no node of the query under it.
        if (q instanceof ConstantScoreQuery c) return List.of(c.getQuery());
        if (q instanceof DisjunctionMaxQuery d) {
            // OpenSearch's profile tree lists a disjunction's first disjunct last.
            List<Query> out = new ArrayList<>(d.getDisjuncts());
            if (out.size() > 1) {
                out.add(out.remove(0));
            }
            return out;
        }
        return List.of();
    }

    /**
     * The place of each of {@link #children}'s queries in the native clause's list of them: a
     * boolean's {@code MUST}, {@code FILTER}, {@code SHOULD} and {@code MUST_NOT} clauses, in that
     * order, each kept in the order it was added.
     */
    static int[] slots(Query q) {
        if (q instanceof BooleanQuery bq) {
            BooleanClause.Occur[] order = { BooleanClause.Occur.MUST, BooleanClause.Occur.FILTER, BooleanClause.Occur.SHOULD, BooleanClause.Occur.MUST_NOT };
            List<BooleanClause> clauses = bq.clauses();
            int[] out = new int[clauses.size()];
            int base = 0;
            for (BooleanClause.Occur occur : order) {
                int seen = 0;
                for (int i = 0; i < out.length; i++) {
                    if (clauses.get(i).occur() == occur) {
                        out[i] = base + seen++;
                    }
                }
                base += seen;
            }
            return out;
        }
        int[] out = new int[children(q).size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = q instanceof DisjunctionMaxQuery ? (i + 1) % out.length : i;
        }
        return out;
    }

    /**
     * One report for a hybrid query: sub-query {@code i}'s nodes under step {@code i}, and a root
     * (the hybrid scorer) that builds as slowly as its slowest sub-query and steps through every
     * sub-query's documents. All starts are {@link System#nanoTime} readings.
     */
    static Report hybridReport(Query hybrid, List<Query> subs, List<Report> reports, long[] javaClocks) {
        List<Node> nodes = new ArrayList<>();
        Map<Integer, long[][]> roots = new HashMap<>();
        long total = 0;
        for (int i = 0; i < subs.size(); i++) {
            Report r = reports.get(i);
            total += r.total();
            int[] root = rootPath(subs.get(i), r);
            for (Node n : r.nodes()) {
                if (n.path().length < root.length || java.util.Arrays.equals(n.path(), 0, root.length, root, 0, root.length) == false) {
                    continue;
                }
                int[] path = new int[n.path().length - root.length + 1];
                path[0] = i;
                System.arraycopy(n.path(), root.length, path, 1, path.length - 1);
                long[] starts = new long[TIMERS];
                for (int t = 0; t < TIMERS; t++) {
                    starts[t] = n.starts()[t] == 0 ? 0 : javaClocks[i] + (n.starts()[t] - r.clock());
                }
                nodes.add(new Node(n.segment(), path, n.kind(), n.nanos(), n.counts(), starts));
                if (path.length == 1) {
                    long[][] agg = roots.computeIfAbsent(n.segment(), k -> new long[3][TIMERS]);
                    for (int t = 0; t < TIMERS; t++) {
                        agg[0][t] = Math.max(agg[0][t], n.nanos()[t]);
                        agg[1][t] += n.counts()[t];
                        agg[2][t] = agg[2][t] == 0 ? starts[t] : starts[t] == 0 ? agg[2][t] : Math.min(agg[2][t], starts[t]);
                    }
                }
            }
        }
        for (Map.Entry<Integer, long[][]> e : roots.entrySet()) {
            long[][] agg = e.getValue();
            long[] nanos = new long[TIMERS];
            long[] counts = new long[TIMERS];
            long[] starts = new long[TIMERS];
            for (int t : new int[] { 0, 1 }) {
                nanos[t] = agg[0][t];
                counts[t] = t == 0 ? 1 : agg[1][t];
                starts[t] = agg[2][t];
            }
            nodes.add(0, new Node(e.getKey(), new int[0], "Hybrid", nanos, counts, starts));
        }
        return new Report(0, total, nodes);
    }

    /** The native nodes by path, over every segment. */
    private static Map<String, List<Node>> byPath(Report report) {
        Map<String, List<Node>> out = new HashMap<>();
        for (Node n : report.nodes()) {
            out.computeIfAbsent(key(n.path()), k -> new ArrayList<>()).add(n);
        }
        return out;
    }

    private static String key(int[] path) {
        StringBuilder b = new StringBuilder();
        for (int p : path) {
            b.append(p).append('.');
        }
        return b.toString();
    }

    private static int[] append(int[] path, int step) {
        int[] out = java.util.Arrays.copyOf(path, path.length + 1);
        out[path.length] = step;
        return out;
    }

    /**
     * The path of the native clause {@code q}, the query tree's root, is: the root itself, or the
     * first clause of that kind under it when the native engine wrapped the query (a lone term
     * scored with its own similarity runs as the one clause of a boolean).
     */
    private static int[] rootPath(Query q, Report report) {
        String want = kindOf(q);
        int[] best = null;
        for (Node n : report.nodes()) {
            if ((want == null || want.equals(n.kind())) && (best == null || n.path().length < best.length)) {
                best = n.path();
            }
        }
        return best == null ? new int[0] : best;
    }

    /**
     * Lays {@code report} into {@code profiler}'s tree for {@code query}. {@code weightNanos} is
     * what making the clause's weight cost (the time to encode it for the native pass), by clause;
     * {@code javaClock} the {@link System#nanoTime} reading at the native clock's {@code
     * report.clock()}.
     */
    static void recordQuery(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        Report report,
        ToLongFunction<Query> weightNanos,
        long javaClock,
        boolean scores,
        boolean advanceOnly
    ) {
        Profilers profilers = ctx.getProfilers();
        QueryProfiler profiler = profilers.getCurrentQueryProfiler();
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        Map<String, List<Node>> nodes = byPath(report);
        // A search that needs no scores runs the query as a ConstantScoreQuery around it
        // (IndexSearcher.rewrite): one node, laid over the native root.
        Query shown = scores ? query : new ConstantScoreQuery(query);
        Layout layout = new Layout(profiler, leaves, nodes, weightNanos, report.clock(), javaClock, shown == query ? null : shown, advanceOnly);
        ContextualProfileBreakdown root = layout.add(shown, rootPath(query, report));
        if (root != null && ctx.shouldUseConcurrentSearch()) {
            // One collector per slice, as Lucene gives a concurrent search: the breakdown reads
            // each slice's time from the leaves associated with it.
            org.apache.lucene.search.IndexSearcher.LeafSlice[] slices = searcher.getSlices();
            if (slices != null) {
                for (org.apache.lucene.search.IndexSearcher.LeafSlice slice : slices) {
                    Collector sliceCollector = new TotalHitCountCollector();
                    for (LeafReaderContext leaf : slice.partitions == null ? List.<LeafReaderContext>of() : leafsOf(slice)) {
                        root.associateCollectorToLeaves(sliceCollector, leaf);
                    }
                }
            }
        }
    }

    private static List<LeafReaderContext> leafsOf(org.apache.lucene.search.IndexSearcher.LeafSlice slice) {
        List<LeafReaderContext> out = new ArrayList<>();
        for (org.apache.lucene.search.IndexSearcher.LeafReaderContextPartition p : slice.partitions) {
            out.add(p.ctx);
        }
        return out;
    }

    /** The profile tree being built. */
    private static final class Layout {
        private final QueryProfiler profiler;
        private final List<LeafReaderContext> leaves;
        private final Map<String, List<Node>> nodes;
        private final ToLongFunction<Query> weightNanos;
        private final long nativeClock;
        private final long javaClock;
        /** The query standing over the native root with no clause of its own, if any. */
        private final Query alias;
        /** Time stepping as advances, as a rescorer steps through its window. */
        private final boolean advanceOnly;

        Layout(
            QueryProfiler profiler,
            List<LeafReaderContext> leaves,
            Map<String, List<Node>> nodes,
            ToLongFunction<Query> weightNanos,
            long nativeClock,
            long javaClock,
            Query alias,
            boolean advanceOnly
        ) {
            this.alias = alias;
            this.advanceOnly = advanceOnly;
            this.profiler = profiler;
            this.leaves = leaves;
            this.nodes = nodes;
            this.weightNanos = weightNanos;
            this.nativeClock = nativeClock;
            this.javaClock = javaClock;
        }

        ContextualProfileBreakdown add(Query q, int[] path) {
            ContextualProfileBreakdown breakdown = profiler.getQueryBreakdown(q);
            try {
                long created = Math.max(1, weightNanos.applyAsLong(q));
                set(breakdown.getTimer(QueryTimingType.CREATE_WEIGHT), created, 1, System.nanoTime() - created, false);
                String kind = kindOf(q);
                for (Node n : nodes.getOrDefault(key(path), List.of())) {
                    if (q != alias && kind != null && kind.equals(n.kind()) == false) {
                        continue;
                    }
                    if (n.segment() >= leaves.size()) {
                        continue;
                    }
                    AbstractProfileBreakdown into = breakdown.context(leaves.get(n.segment()));
                    // Without slices every leaf lands on the one breakdown: its timers add up.
                    boolean accumulate = into == breakdown;
                    for (int t = 0; t < TIMERS; t++) {
                        QueryTimingType type = TYPES[t];
                        if (advanceOnly && type == QueryTimingType.NEXT_DOC) {
                            continue;
                        }
                        long start = n.starts()[t] == 0 ? 0 : javaClock + (n.starts()[t] - nativeClock);
                        if (advanceOnly && type == QueryTimingType.ADVANCE) {
                            int next = java.util.Arrays.asList(TYPES).indexOf(QueryTimingType.NEXT_DOC);
                            long steps = n.counts()[t] + n.counts()[next];
                            long nanos = n.nanos()[t] + n.nanos()[next];
                            long from = n.starts()[next] == 0 ? start : javaClock + (n.starts()[next] - nativeClock);
                            set(into.getTimer(type), nanos, steps, start == 0 ? from : start, accumulate);
                            continue;
                        }
                        set(into.getTimer(type), n.nanos()[t], n.counts()[t], start, accumulate);
                    }
                }
                List<Query> kids = children(q);
                int[] slots = slots(q);
                for (int i = 0; i < kids.size(); i++) {
                    add(kids.get(i), q == alias ? path : append(path, slots[i]));
                }
            } finally {
                profiler.pollLastElement();
            }
            return breakdown;
        }
    }

    /**
     * Sets a timer to {@code nanos} over {@code count} calls first started at {@code start} (a
     * {@link System#nanoTime} reading, 0 for none), or adds them to what it holds. A {@link
     * Timer} only runs on the clock; its fields are what its breakdown reads.
     */
    static void set(Timer timer, long nanos, long count, long start, boolean add) {
        Field timing = Reflect.declared(Timer.class, "timing");
        Field counted = Reflect.declared(Timer.class, "count");
        Field lastCount = Reflect.declared(Timer.class, "lastCount");
        Field earliest = Reflect.declared(Timer.class, "earliestTimerStartTime");
        if (timing == null || counted == null || lastCount == null || earliest == null) {
            throw new IllegalStateException("the profile Timer of this OpenSearch has other fields than the plugin's");
        }
        try {
            long newCount = count;
            long newTiming = nanos;
            long newStart = start;
            if (add) {
                newCount += counted.getLong(timer);
                newTiming += timing.getLong(timer);
                long was = earliest.getLong(timer);
                newStart = was == 0 ? start : start == 0 ? was : Math.min(was, start);
            }
            timing.setLong(timer, newTiming);
            counted.setLong(timer, newCount);
            lastCount.setLong(timer, newCount);
            earliest.setLong(timer, newStart);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The live list of the top-level queries the current thread's profile tree holds. */
    @SuppressWarnings("unchecked")
    private static List<Integer> rootsOf(SearchContext ctx) throws ReflectiveOperationException {
        QueryProfiler profiler = ctx.getProfilers().getCurrentQueryProfiler();
        Object tree;
        if (profiler instanceof org.opensearch.search.profile.query.ConcurrentQueryProfiler) {
            Field threads = Reflect.declared(org.opensearch.search.profile.query.ConcurrentQueryProfiler.class, "threadToProfileTree");
            tree = ((Map<Long, Object>) threads.get(profiler)).get(Thread.currentThread().threadId());
        } else {
            tree = Reflect.declared(org.opensearch.search.profile.AbstractProfiler.class, "profileTree").get(profiler);
        }
        if (tree == null) {
            return null;
        }
        return (List<Integer>) Reflect.declared(org.opensearch.search.profile.AbstractInternalProfileTree.class, "roots").get(tree);
    }

    /** How many top-level queries the profile holds. */
    static int rootCount(SearchContext ctx) {
        try {
            List<Integer> roots = rootsOf(ctx);
            return roots == null ? 0 : roots.size();
        } catch (ReflectiveOperationException | RuntimeException e) {
            return -1;
        }
    }

    /** Drops the top-level queries added after the first {@code keep}: weights the collectors made over what was laid already. */
    static void dropRootsAfter(SearchContext ctx, int keep) {
        if (keep < 0) {
            return;
        }
        try {
            List<Integer> roots = rootsOf(ctx);
            while (roots != null && roots.size() > keep) {
                roots.remove(roots.size() - 1);
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.debug("lucene-rust: profile roots left as they were", e);
        }
    }

    // ---- the aggregation profile -------------------------------------------------------------

    /**
     * Gives the timers of the aggregators the native pass did not run a start time, so that the
     * profile's reduction over slices ({@code ConcurrentAggregationProfiler}) takes a span from
     * them, and puts the native aggregation call's time on the collection of each aggregation
     * that collected no documents through Java.
     */
    private static final Map<SearchContext, long[]> AGGREGATED = java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /** Keeps when the native aggregation call began and how long it took, for {@link #finishAggregations}. */
    static void aggregated(SearchContext ctx, long start, long nanos) {
        AGGREGATED.put(ctx, new long[] { start, nanos });
    }

    /**
     * After the aggregations of a profiled search ran: when the native pass answered them, its
     * time is the aggregators' (split over their phases, as the native pass does not step them
     * apart), and their timers get the start the reduction over slices needs; when Lucene ran
     * them, the aggregators planning built and dropped leave the profile.
     */
    static void finishAggregations(SearchContext ctx) {
        if (ctx.getProfilers() == null || ctx.getProfilers().getAggregationProfiler() == null) {
            return;
        }
        long[] native_ = AGGREGATED.remove(ctx);
        AggregationProfiler profiler = ctx.getProfilers().getAggregationProfiler();
        Field lookup = Reflect.declared(AggregationProfiler.class, "profileBreakdownLookup");
        if (lookup == null) {
            return;
        }
        try {
            @SuppressWarnings("unchecked")
            Map<Object, AggregationProfileBreakdown> all = (Map<Object, AggregationProfileBreakdown>) lookup.get(profiler);
            if (native_ == null) {
                dropUnrun(profiler, all);
                return;
            }
            long hits = ctx.queryResult().topDocs() == null ? 0 : ctx.queryResult().topDocs().topDocs.totalHits.value();
            int leaves = Math.max(1, ctx.searcher().getIndexReader().leaves().size());
            long anchor = Math.max(1, native_[0]);
            long perAgg = all.isEmpty() ? 0 : native_[1] / all.size();
            for (AggregationProfileBreakdown b : all.values()) {
                long seen = 0;
                for (AggregationTimingType t : AggregationTimingType.values()) {
                    Timer timer = b.getTimer(t);
                    long s = timer.getCount() == 0 ? 0 : timer.getEarliestTimerStartTime();
                    seen = s == 0 ? seen : seen == 0 ? s : Math.min(seen, s);
                }
                long start = seen == 0 ? anchor : seen;
                setIfUnrun(b, AggregationTimingType.BUILD_LEAF_COLLECTOR, perAgg / 10, leaves, start);
                setIfUnrun(b, AggregationTimingType.COLLECT, perAgg * 7 / 10, hits, start);
                setIfUnrun(b, AggregationTimingType.POST_COLLECTION, perAgg / 20, 1, start);
                setIfUnrun(b, AggregationTimingType.BUILD_AGGREGATION, perAgg / 10, 1, start);
                for (AggregationTimingType t : AggregationTimingType.values()) {
                    Timer timer = b.getTimer(t);
                    if (timer.getCount() == 0) {
                        set(timer, 0, 0, start, false);
                    }
                }
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.debug("lucene-rust: aggregation profile left as it was", e);
        }
    }

    private static void setIfUnrun(AggregationProfileBreakdown b, AggregationTimingType type, long nanos, long count, long start) {
        Timer timer = b.getTimer(type);
        if (timer.getCount() == 0 && count > 0) {
            set(timer, Math.max(1, nanos), count, start, false);
        }
    }

    /** Removes the top-level aggregators that never collected nor built a result from the profile tree. */
    private static void dropUnrun(AggregationProfiler profiler, Map<Object, AggregationProfileBreakdown> all)
        throws ReflectiveOperationException {
        Field treeField = Reflect.declared(org.opensearch.search.profile.AbstractProfiler.class, "profileTree");
        Field rootsField = Reflect.declared(org.opensearch.search.profile.AbstractInternalProfileTree.class, "roots");
        Field breakdownsField = Reflect.declared(org.opensearch.search.profile.AbstractInternalProfileTree.class, "breakdowns");
        if (treeField == null || rootsField == null || breakdownsField == null) {
            return;
        }
        Object tree = treeField.get(profiler);
        @SuppressWarnings("unchecked")
        List<Integer> roots = (List<Integer>) rootsField.get(tree);
        @SuppressWarnings("unchecked")
        List<AggregationProfileBreakdown> breakdowns = (List<AggregationProfileBreakdown>) breakdownsField.get(tree);
        roots.removeIf(token -> {
            AggregationProfileBreakdown b = breakdowns.get(token);
            return b.getTimer(AggregationTimingType.COLLECT).getCount() == 0
                && b.getTimer(AggregationTimingType.BUILD_AGGREGATION).getCount() == 0
                && b.getTimer(AggregationTimingType.POST_COLLECTION).getCount() == 0;
        });
    }

    // ---- the collector tree ------------------------------------------------------------------

    /**
     * Builds the collector tree OpenSearch makes for this request's collectors, sets the time the
     * native calls took on it, and gives it to the query profiler: the shape ({@code
     * QueryCollectorContext.createWithProfiler}) is OpenSearch's, built over the same contexts.
     */
    static void recordCollectors(
        SearchContext ctx,
        ContextIndexSearcher searcher,
        Query query,
        LinkedList<QueryCollectorContext> collectors,
        boolean hasFilterCollector,
        boolean hasTimeout,
        Phases phases
    ) throws IOException {
        QueryPhase.DefaultQueryPhaseSearcher shape = ctx.shouldUseConcurrentSearch()
            ? new ConcurrentShape(phases)
            : new SerialShape(phases);
        shape.searchWith(ctx, searcher, query, collectors, hasFilterCollector, hasTimeout);
    }

    /** What a collector of {@code reason} spent: its own native call's time, or the sum of its children's. */
    private static long timeOf(String reason, List<? extends InternalProfileComponent> children, Phases p) {
        if (children.isEmpty() == false) {
            long sum = 0;
            for (InternalProfileComponent c : children) {
                sum += c instanceof InternalProfileCollector ic ? ic.getTime()
                    : c instanceof InternalProfileCollectorManager m ? m.getTime() : 0;
            }
            return sum;
        }
        return switch (reason) {
            case "aggregation", "aggregation_global" -> p.aggNanos;
            case "search_top_hits", "search_count", "search_terminate_after_count" -> p.hitsNanos;
            default -> 0;
        };
    }

    /** OpenSearch's non-concurrent profile collector chain, with times. */
    private static final class SerialShape extends QueryPhase.DefaultQueryPhaseSearcher {
        private final Phases phases;

        SerialShape(Phases phases) {
            this.phases = phases;
        }

        @Override
        protected boolean searchWithCollector(
            SearchContext searchContext,
            ContextIndexSearcher searcher,
            Query query,
            LinkedList<QueryCollectorContext> collectors,
            QueryCollectorContext queryCollectorContext,
            boolean hasFilterCollector,
            boolean hasTimeout
        ) throws IOException {
            collectors.addFirst(queryCollectorContext);
            try {
                Method create = QueryCollectorContext.class.getDeclaredMethod("createQueryCollectorWithProfiler", List.class);
                create.setAccessible(true);
                InternalProfileCollector collector = (InternalProfileCollector) create.invoke(null, collectors);
                time(collector, phases);
                searchContext.getProfilers().getCurrentQueryProfiler().setCollector(collector);
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof IOException io) {
                    throw io;
                }
                throw new IllegalStateException(e.getCause());
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("the collector profile of this OpenSearch is not the plugin's", e);
            }
            return false;
        }

        private static void time(InternalProfileCollector c, Phases p) throws ReflectiveOperationException {
            for (InternalProfileComponent child : c.children()) {
                time((InternalProfileCollector) child, p);
            }
            long nanos = Math.max(1, timeOf(c.getReason(), new ArrayList<>(c.children()), p));
            Field inner = Reflect.declared(InternalProfileCollector.class, "collector");
            Object profile = inner.get(c);
            Field time = Reflect.declared(profile.getClass(), "time");
            Field start = Reflect.declared(profile.getClass(), "sliceStartTime");
            Field started = Reflect.declared(profile.getClass(), "isStarted");
            time.setLong(profile, nanos);
            start.setLong(profile, System.nanoTime() - nanos);
            started.setBoolean(profile, true);
        }
    }

    /** OpenSearch's concurrent profile collector-manager tree, with times. */
    private static final class ConcurrentShape extends ConcurrentQueryPhaseSearcher {
        private final Phases phases;

        ConcurrentShape(Phases phases) {
            this.phases = phases;
        }

        @Override
        protected boolean searchWithCollector(
            SearchContext searchContext,
            ContextIndexSearcher searcher,
            Query query,
            LinkedList<QueryCollectorContext> collectors,
            QueryCollectorContext queryCollectorContext,
            boolean hasFilterCollector,
            boolean hasTimeout
        ) throws IOException {
            collectors.addFirst(queryCollectorContext);
            var manager = org.opensearch.search.query.QueryCollectorManagerContext.createQueryCollectorManagerWithProfiler(collectors);
            try {
                time((InternalProfileCollectorManager) manager, phases);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("the collector profile of this OpenSearch is not the plugin's", e);
            }
            searchContext.getProfilers().getCurrentQueryProfiler().setCollector(manager);
            return false;
        }

        private static void time(InternalProfileCollectorManager m, Phases p) throws ReflectiveOperationException {
            for (InternalProfileComponent child : m.children()) {
                time((InternalProfileCollectorManager) child, p);
            }
            long nanos = Math.max(1, timeOf(m.getReason(), new ArrayList<>(m.children()), p));
            set(m, "time", nanos);
            set(m, "reduceTime", 1);
            set(m, "maxSliceTime", nanos);
            set(m, "minSliceTime", nanos);
            set(m, "avgSliceTime", nanos);
            Field slices = Reflect.declared(InternalProfileCollectorManager.class, "sliceCount");
            slices.setInt(m, Math.max(1, p.slices));
        }

        private static void set(InternalProfileCollectorManager m, String name, long value) throws ReflectiveOperationException {
            Reflect.declared(InternalProfileCollectorManager.class, name).setLong(m, value);
        }
    }
}
