/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.lucenerust.opensearch.tensor.MaxSimQuery;
import org.apache.lucene.queries.intervals.IntervalQuery;
import org.apache.lucene.queries.intervals.IntervalsSource;
import org.apache.lucene.queries.spans.FieldMaskingSpanQuery;
import org.apache.lucene.queries.spans.SpanContainingQuery;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.opensearch.lucene.queries.SpanMatchNoDocsQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanPositionRangeQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWithinQuery;
import org.apache.lucene.search.CombinedFieldQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.search.join.ToParentBlockJoinQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Transition;
import org.opensearch.common.lucene.search.function.CombineFunction;
import org.opensearch.common.lucene.search.function.FieldValueFactorFunction;
import org.opensearch.common.lucene.search.function.FunctionScoreQuery;
import org.opensearch.common.lucene.search.function.ScriptScoreQuery;
import org.lucenerust.opensearch.js.JsScripts;
import org.lucenerust.opensearch.painless.PainlessIr;
import org.lucenerust.opensearch.painless.PainlessScore;
import org.lucenerust.opensearch.knn.script.KnnScoringScriptEngine;
import org.lucenerust.opensearch.knn.script.PainlessKnnScript;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.IntervalFilterScript;
import org.opensearch.script.AggregationScript;
import org.opensearch.script.FilterScript;
import org.opensearch.script.NumberSortScript;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TermsSetQueryScript;
import org.opensearch.common.lucene.search.function.RandomScoreFunction;
import org.opensearch.common.lucene.search.function.ScoreFunction;
import org.opensearch.common.lucene.search.function.ScriptScoreFunction;
import org.opensearch.common.lucene.search.function.WeightFactorFunction;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.IndexNumericFieldData;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.search.MultiValueMode;

import org.apache.lucene.analysis.Analyzer;
import org.lucenerust.opensearch.patterntext.PatternPhraseQuery;
import org.lucenerust.opensearch.substring.RustWildcardQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.function.Function;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * The extension of {@link QueryEncoder} for the OpenSearch shapes {@code lucene-join} and
 * {@code lucene-queries} back natively, as the nodes {@code crates/lucene-engine/src/nodes.rs}
 * reads.
 *
 * <ul>
 *   <li>{@code nested}: {@link OpenSearchToParentBlockJoinQuery} (or a plain {@link
 *       ToParentBlockJoinQuery}) over a {@link QueryBitSetProducer} or OpenSearch's {@code
 *       BitsetFilterCache} producer, any {@code score_mode} (node 20);
 *   <li>{@code span_term}, {@code span_near} (without gaps), {@code span_or}, {@code span_not},
 *       {@code span_first}, {@code span_containing}, {@code span_within}, {@code
 *       field_masking_span}, and {@code span_multi} once the searcher has rewritten it (node 21);
 *   <li>{@code intervals}: {@link IntervalQuery} over any of Lucene's {@code IntervalsSource}s,
 *       read field by field (node 22);
 *   <li>{@code combined_fields}: {@link CombinedFieldQuery} (node 23);
 *   <li>{@code function_score}: OpenSearch's {@link FunctionScoreQuery} with {@code weight}, {@code
 *       field_value_factor}, {@code random_score} and the numeric and date decay functions, each
 *       optionally filtered, every {@code score_mode} and {@code boost_mode}, {@code max_boost}
 *       (node 24);
 *   <li>{@code script_score} running k-NN's {@code knn_score} over a {@code knn_vector} space:
 *       OpenSearch's {@link ScriptScoreQuery} (node 42), under a {@code min_score} node;
 *   <li>{@code max_sim} over a {@code multi_vector} field: {@link MaxSimQuery} (node 44);
 *   <li>an exact phrase over a {@code pattern_text} field: {@link PatternPhraseQuery} (node 47).
 * </ul>
 *
 * <p>The classes are read through their getters, or through {@link Reflect} where Lucene and
 * OpenSearch keep the fields private (the fields their constructors were given, which are what
 * their {@code equals} compares). Not encoded, each under a named reason: a {@code span_near} with
 * a {@code span_gap} ({@code span_gap}), a payload span query ({@code span_<Class>}), an interval
 * source with a payload filter or of a class this does not know ({@code interval_<Class>}), a
 * script ({@code function_score_script}), a geo or other non-numeric decay ({@code
 * function_score_<Class>}), field data other than plain sorted-numeric (or keyword, for {@code
 * random_score}) doc values ({@code function_score_field_data}), and fields the reflection cannot
 * read ({@code extended_reflect}).
 */
final class ExtendedEncoder {
    static final byte NODE_TO_PARENT = 20;
    static final byte NODE_SPAN = 21;
    static final byte NODE_INTERVAL = 22;
    static final byte NODE_COMBINED_FIELD = 23;
    static final byte NODE_FUNCTION_SCORE = 24;
    /** {@code NODE_MIN_SCORE} in {@code queries.rs}: {@code min_score}, then the scored query. */
    static final byte NODE_MIN_SCORE = 56;
    static final byte NODE_TO_CHILD = 41;
    static final byte NODE_KNN_SCORE = 42;
    static final byte NODE_PAINLESS_SCORE = 43;
    static final byte NODE_MAX_SIM = 44;
    static final byte NODE_PAINLESS_FILTER = 45;
    static final byte NODE_JOIN = 46;
    /** {@code NODE_MATCH_NONE} of the query tree (query_blob.rs). */
    static final byte NODE_MATCH_NONE = 6;
    static final byte NODE_PATTERN_PHRASE = 47;
    static final byte NODE_SUBSTRING = 49;
    static final byte SUBSTRING_WILDCARD = 0;
    static final byte SUBSTRING_REGEXP = 1;
    static final byte SUBSTRING_CONTAINS = 2;
    static final byte SUBSTRING_EXACT = 3;
    /** {@code pattern_phrase.rs}'s {@code MAX_TERMS}. */
    static final int MAX_PATTERN_TERMS = 1 << 16;

    /** What {@link #node} returns for a query that is none of these. */
    static final String NOT_EXTENDED = "not_extended";

    /** Writes a child query as a node one level down; returns a fallback reason, or null. */
    interface Child {
        String node(Query q);
    }

    private ExtendedEncoder() {}

    /**
     * Appends {@code q} when it is one of these shapes: null when it was written, a fallback reason
     * when it cannot be, {@link #NOT_EXTENDED} when it is none of them.
     */
    static String node(Query q, ByteArrayOutputStream out, Predicate<String> fieldOk, Child child) {
        try {
            if (q instanceof FunctionScoreQuery fs) {
                return functionScore(fs, out, fieldOk, child);
            }
            if (q instanceof ScriptScoreQuery ss) {
                minScore(field(ss, "minScore"), out);
                return knnScore(ss, out, child);
            }
            if (q instanceof MaxSimQuery ms) {
                return maxSim(ms, out, child);
            }
            if (q instanceof PatternPhraseQuery pp) {
                return patternPhrase(pp, out, child);
            }
            if (q instanceof RustWildcardQuery rw) {
                return substring(rw, out);
            }
            if (q instanceof OpenSearchToParentBlockJoinQuery os) {
                Field f = Reflect.declared(OpenSearchToParentBlockJoinQuery.class, "query");
                if (f == null) {
                    return "extended_reflect";
                }
                return toParent((ToParentBlockJoinQuery) f.get(os), out, child);
            }
            if (q.getClass() == ToParentBlockJoinQuery.class) {
                return toParent((ToParentBlockJoinQuery) q, out, child);
            }
            if (q.getClass() == ToChildBlockJoinQuery.class) {
                return toChild((ToChildBlockJoinQuery) q, out, child);
            }
            if (q instanceof org.lucenerust.opensearch.join.DeferredJoinQuery dj) {
                return join(dj, out, child);
            }
            if (q instanceof SpanQuery sq) {
                out.write(NODE_SPAN);
                return span(sq, out, fieldOk);
            }
            if (q instanceof IntervalQuery iq) {
                return interval(iq, out);
            }
            if (q instanceof CombinedFieldQuery cf) {
                return combinedField(cf, out, fieldOk);
            }
            if (q.getClass().getName().equals("org.opensearch.index.query.ScriptQueryBuilder$ScriptQuery")) {
                return painlessFilter(q, out);
            }
            return NOT_EXTENDED;
        } catch (ReflectiveOperationException | ClassCastException e) {
            return "extended_reflect";
        }
    }

    // ---------------------------------------------------------------------------------------------
    // nested
    // ---------------------------------------------------------------------------------------------

    private static String toParent(ToParentBlockJoinQuery q, ByteArrayOutputStream out, Child child)
        throws ReflectiveOperationException {
        Field parentsF = Reflect.declared(ToParentBlockJoinQuery.class, "parentsFilter");
        Field modeF = Reflect.declared(ToParentBlockJoinQuery.class, "scoreMode");
        if (parentsF == null || modeF == null) {
            return "extended_reflect";
        }
        Query parents = parentsQuery((BitSetProducer) parentsF.get(q));
        if (parents == null) {
            return "join_parents";
        }
        out.write(NODE_TO_PARENT);
        out.write((byte) ((ScoreMode) modeF.get(q)).ordinal());
        String reason = child.node(parents);
        if (reason != null) {
            return reason;
        }
        return child.node(q.getChildQuery());
    }

    /**
     * {@code has_child}/{@code has_parent}: the join field, score mode, {@code min_children} and
     * {@code max_children}, the to side and the from side, both rewritten.
     */
    private static String join(org.lucenerust.opensearch.join.DeferredJoinQuery q, ByteArrayOutputStream out, Child child)
        throws ReflectiveOperationException {
        if (q.from() == null || q.to() == null) {
            return "join_unrewritten";
        }
        out.write(NODE_JOIN);
        writeString(out, q.joinField());
        out.write((byte) q.scoreMode().ordinal());
        writeInt(out, q.minChildren());
        writeInt(out, q.maxChildren());
        String reason = child.node(q.to());
        if (reason != null) {
            return reason;
        }
        return child.node(q.from());
    }

    private static String toChild(ToChildBlockJoinQuery q, ByteArrayOutputStream out, Child child)
        throws ReflectiveOperationException {
        Field parentsF = Reflect.declared(ToChildBlockJoinQuery.class, "parentsFilter");
        if (parentsF == null) {
            return "extended_reflect";
        }
        Query parents = parentsQuery((BitSetProducer) parentsF.get(q));
        if (parents == null) {
            return "join_parents";
        }
        out.write(NODE_TO_CHILD);
        String reason = child.node(parents);
        if (reason != null) {
            return reason;
        }
        return child.node(q.getParentQuery());
    }

    /** The query a parent filter's bit sets are built from, or null for a producer this cannot read. */
    static Query parentsQuery(BitSetProducer p) throws ReflectiveOperationException {
        if (p instanceof QueryBitSetProducer qp) {
            return qp.getQuery();
        }
        // OpenSearch's IndicesBitsetFilterCache.QueryWrapperBitSetProducer (package-private).
        if (p != null && p.getClass().getName().equals("org.opensearch.indices.IndicesBitsetFilterCache$QueryWrapperBitSetProducer")) {
            Field f = Reflect.declared(p.getClass(), "query");
            return f == null ? null : (Query) f.get(p);
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // spans
    // ---------------------------------------------------------------------------------------------

    private static String span(SpanQuery q, ByteArrayOutputStream out, Predicate<String> fieldOk) throws ReflectiveOperationException {
        Class<?> c = q.getClass();
        if (c == SpanTermQuery.class) {
            Term t = ((SpanTermQuery) q).getTerm();
            if (fieldOk.test(t.field()) == false) {
                return "field_similarity";
            }
            out.write(0);
            writeString(out, t.field());
            writeBytes(out, t.bytes());
            return null;
        }
        if (c == SpanNearQuery.class) {
            SpanNearQuery n = (SpanNearQuery) q;
            out.write(1);
            writeInt(out, n.getSlop());
            out.write(n.isInOrder() ? 1 : 0);
            return spans(n.getClauses(), out, fieldOk);
        }
        if (c == SpanOrQuery.class) {
            out.write(2);
            return spans(((SpanOrQuery) q).getClauses(), out, fieldOk);
        }
        if (c == SpanFirstQuery.class) {
            SpanFirstQuery f = (SpanFirstQuery) q;
            out.write(3);
            writeInt(out, f.getEnd());
            return span(f.getMatch(), out, fieldOk);
        }
        if (c == SpanPositionRangeQuery.class) {
            SpanPositionRangeQuery r = (SpanPositionRangeQuery) q;
            out.write(4);
            writeInt(out, r.getStart());
            writeInt(out, r.getEnd());
            return span(r.getMatch(), out, fieldOk);
        }
        if (c == SpanNotQuery.class) {
            SpanNotQuery n = (SpanNotQuery) q;
            Field pre = Reflect.declared(SpanNotQuery.class, "pre");
            Field post = Reflect.declared(SpanNotQuery.class, "post");
            if (pre == null || post == null) {
                return "extended_reflect";
            }
            out.write(5);
            writeInt(out, pre.getInt(n));
            writeInt(out, post.getInt(n));
            String reason = span(n.getInclude(), out, fieldOk);
            return reason != null ? reason : span(n.getExclude(), out, fieldOk);
        }
        if (c == SpanContainingQuery.class || c == SpanWithinQuery.class) {
            // SpanContainQuery is package-private: its public getters through each subclass.
            boolean containing = c == SpanContainingQuery.class;
            SpanQuery big = containing ? ((SpanContainingQuery) q).getBig() : ((SpanWithinQuery) q).getBig();
            SpanQuery little = containing ? ((SpanContainingQuery) q).getLittle() : ((SpanWithinQuery) q).getLittle();
            out.write(containing ? 6 : 7);
            String reason = span(big, out, fieldOk);
            return reason != null ? reason : span(little, out, fieldOk);
        }
        if (c == FieldMaskingSpanQuery.class) {
            FieldMaskingSpanQuery m = (FieldMaskingSpanQuery) q;
            if (fieldOk.test(m.getField()) == false) {
                return "field_similarity";
            }
            out.write(8);
            writeString(out, m.getField());
            return span(m.getMaskedQuery(), out, fieldOk);
        }
        if (c.getName().equals("org.apache.lucene.queries.spans.SpanNearQuery$SpanGapQuery")) {
            out.write(9);
            writeString(out, q.getField());
            writeInt(out, (int) field(q, "width"));
            return null;
        }
        if (c == SpanMatchNoDocsQuery.class) {
            out.write(10);
            writeString(out, q.getField());
            return null;
        }
        return "span_" + c.getSimpleName();
    }

    private static String spans(SpanQuery[] clauses, ByteArrayOutputStream out, Predicate<String> fieldOk)
        throws ReflectiveOperationException {
        writeInt(out, clauses.length);
        for (SpanQuery s : clauses) {
            String reason = span(s, out, fieldOk);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // intervals
    // ---------------------------------------------------------------------------------------------

    private static final String INTERVALS = "org.apache.lucene.queries.intervals.";

    private static String interval(IntervalQuery q, ByteArrayOutputStream out) throws ReflectiveOperationException {
        Field fieldF = Reflect.declared(IntervalQuery.class, "field");
        Field sourceF = Reflect.declared(IntervalQuery.class, "intervalsSource");
        Field scoringF = Reflect.declared(IntervalQuery.class, "scoreFunction");
        if (fieldF == null || sourceF == null || scoringF == null) {
            return "extended_reflect";
        }
        Object scoring = scoringF.get(q);
        out.write(NODE_INTERVAL);
        writeString(out, (String) fieldF.get(q));
        String sc = scoring.getClass().getName();
        if (sc.equals(INTERVALS + "IntervalScoreFunction$SaturationFunction")) {
            out.write(0);
            writeInt(out, Float.floatToIntBits(floatField(scoring, "pivot")));
        } else if (sc.equals(INTERVALS + "IntervalScoreFunction$SigmoidFunction")) {
            out.write(1);
            writeInt(out, Float.floatToIntBits(floatField(scoring, "pivot")));
            writeInt(out, Float.floatToIntBits(floatField(scoring, "a")));
        } else {
            return "interval_scoring";
        }
        return source((IntervalsSource) sourceF.get(q), out);
    }

    private static float floatField(Object o, String name) throws ReflectiveOperationException {
        Field f = Reflect.declared(o.getClass(), name);
        if (f == null) {
            throw new NoSuchFieldException(name);
        }
        return f.getFloat(o);
    }

    static Object field(Object o, String name) throws ReflectiveOperationException {
        Field f = Reflect.inHierarchy(o.getClass(), name);
        if (f == null) {
            throw new NoSuchFieldException(name);
        }
        return f.get(o);
    }

    @SuppressWarnings("unchecked")
    private static String source(IntervalsSource s, ByteArrayOutputStream out) throws ReflectiveOperationException {
        String name = s.getClass().getName();
        if (name.equals("org.opensearch.index.query.IntervalsSourceProvider$ScriptFilterSource")) {
            out.write(6);
            out.write(2);
            writeString(out, (String) field(s, "name"));
            String why = painlessInterval((IntervalFilterScript) field(s, "script"), out);
            return why != null ? why : source((IntervalsSource) field(s, "in"), out);
        }
        if (name.startsWith(INTERVALS) == false) {
            return "interval_" + s.getClass().getSimpleName();
        }
        switch (name.substring(INTERVALS.length())) {
            case "TermIntervalsSource" -> {
                out.write(0);
                writeBytes(out, (BytesRef) field(s, "term"));
                return null;
            }
            case "BlockIntervalsSource" -> {
                out.write(1);
                return sources((Collection<IntervalsSource>) field(s, "subSources"), out);
            }
            case "DisjunctionIntervalsSource" -> {
                out.write(2);
                out.write((boolean) field(s, "pullUpDisjunctions") ? 1 : 0);
                return sources((Collection<IntervalsSource>) field(s, "subSources"), out);
            }
            case "OrderedIntervalsSource", "UnorderedIntervalsSource" -> {
                out.write(name.endsWith("UnorderedIntervalsSource") ? 4 : 3);
                return sources((Collection<IntervalsSource>) field(s, "subSources"), out);
            }
            case "RepeatingIntervalsSource" -> {
                int count = (int) field(s, "childCount");
                if (count > MAX_REPEATS) {
                    // The native decoder's bound (the clause limit), which Lucene's own
                    // deduplication of equal clauses never passes under OpenSearch's limit.
                    return "interval_repeating_count";
                }
                out.write(5);
                writeInt(out, count);
                Object n = field(s, "name");
                out.write(n == null ? 0 : "ORDERED".equals(n) ? 1 : "UNORDERED".equals(n) ? 2 : 3);
                if (n != null && "ORDERED".equals(n) == false && "UNORDERED".equals(n) == false) {
                    // Lucene names a repeating source only ORDERED or UNORDERED (its own setName callers).
                    throw new ClassCastException("repeating source name " + n);
                }
                return source((IntervalsSource) field(s, "in"), out);
            }
            case "FilteredIntervalsSource$MaxGaps", "FilteredIntervalsSource$MaxWidth" -> {
                boolean gaps = name.endsWith("MaxGaps");
                out.write(6);
                out.write(gaps ? 0 : 1);
                writeInt(out, (int) field(s, gaps ? "maxGaps" : "maxWidth"));
                return source((IntervalsSource) field(s, "in"), out);
            }
            case "ExtendedIntervalsSource" -> {
                out.write(7);
                writeInt(out, (int) field(s, "before"));
                writeInt(out, (int) field(s, "after"));
                return source((IntervalsSource) field(s, "source"), out);
            }
            case "OffsetIntervalsSource" -> {
                out.write(8);
                out.write((boolean) field(s, "before") ? 1 : 0);
                return source((IntervalsSource) field(s, "in"), out);
            }
            case "FixedFieldIntervalsSource" -> {
                out.write(9);
                writeString(out, (String) field(s, "field"));
                return source((IntervalsSource) field(s, "source"), out);
            }
            case "NoMatchIntervalsSource" -> {
                out.write(10);
                writeString(out, (String) field(s, "reason"));
                return null;
            }
            case "ContainingIntervalsSource" -> {
                return pair(11, field(s, "big"), field(s, "small"), out);
            }
            case "ContainedByIntervalsSource" -> {
                return pair(12, field(s, "small"), field(s, "big"), out);
            }
            case "NotContainingIntervalsSource" -> {
                return pair(13, field(s, "minuend"), field(s, "subtrahend"), out);
            }
            case "NotContainedByIntervalsSource" -> {
                return pair(14, field(s, "minuend"), field(s, "subtrahend"), out);
            }
            case "OverlappingIntervalsSource" -> {
                return pair(15, field(s, "source"), field(s, "reference"), out);
            }
            case "NonOverlappingIntervalsSource" -> {
                return pair(16, field(s, "minuend"), field(s, "subtrahend"), out);
            }
            case "MinimumShouldMatchIntervalsSource" -> {
                out.write(17);
                writeInt(out, (int) field(s, "minShouldMatch"));
                return sources(List.of((IntervalsSource[]) field(s, "sources")), out);
            }
            case "MultiTermIntervalsSource" -> {
                out.write(18);
                writeInt(out, (int) field(s, "maxExpansions"));
                writeString(out, (String) field(s, "pattern"));
                return automaton((CompiledAutomaton) field(s, "automaton"), out, "interval_automaton");
            }
            default -> {
                return "interval_" + s.getClass().getSimpleName();
            }
        }
    }

    /** At most this many copies in a repeating source ({@code MAX_CLAUSE_COUNT} natively). */
    private static final int MAX_REPEATS = 1 << 20;

    private static String pair(int tag, Object a, Object b, ByteArrayOutputStream out) throws ReflectiveOperationException {
        out.write(tag);
        String reason = source((IntervalsSource) a, out);
        return reason != null ? reason : source((IntervalsSource) b, out);
    }

    private static String sources(Collection<IntervalsSource> list, ByteArrayOutputStream out) throws ReflectiveOperationException {
        writeInt(out, list.size());
        for (IntervalsSource s : list) {
            String reason = source(s, out);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    /** At most this many states of a multi-term source's automaton are sent (the native limit). */
    private static final int MAX_AUTOMATON_STATES = 1 << 20;

    /**
     * Writes {@code c} as {@code decode_compiled_automaton} in {@code nodes.rs} reads it: its
     * type, then the term (single) or the byte automaton (normal). Returns {@code tooLarge} for an
     * automaton of more states than the native side accepts, or null.
     */
    static String automaton(CompiledAutomaton c, ByteArrayOutputStream out, String tooLarge) {
        switch (c.type) {
            case NONE -> out.write(0);
            case ALL -> out.write(1);
            case SINGLE -> {
                out.write(2);
                writeBytes(out, c.term);
            }
            case NORMAL -> {
                Automaton a = c.automaton;
                if (a.getNumStates() > MAX_AUTOMATON_STATES) {
                    return tooLarge;
                }
                out.write(3);
                writeInt(out, a.getNumStates());
                Transition t = new Transition();
                for (int s = 0; s < a.getNumStates(); s++) {
                    out.write(a.isAccept(s) ? 1 : 0);
                    int n = a.initTransition(s, t);
                    writeInt(out, n);
                    for (int i = 0; i < n; i++) {
                        a.getNextTransition(t);
                        writeInt(out, t.dest);
                        writeInt(out, t.min);
                        writeInt(out, t.max);
                    }
                }
            }
            default -> {
                return tooLarge;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // combined_fields
    // ---------------------------------------------------------------------------------------------

    private static String combinedField(CombinedFieldQuery q, ByteArrayOutputStream out, Predicate<String> fieldOk)
        throws ReflectiveOperationException {
        Map<?, ?> fields = (Map<?, ?>) field(q, "fieldAndWeights");
        BytesRef term = (BytesRef) field(q, "term");
        if (fields.isEmpty()) {
            // CombinedFieldQuery.rewrite: no field is the empty boolean query, which matches nothing.
            out.write(NODE_MATCH_NONE);
            return null;
        }
        out.write(NODE_COMBINED_FIELD);
        writeBytes(out, term);
        writeInt(out, fields.size());
        for (Object fw : fields.values()) {
            String f = (String) field(fw, "field");
            // BM25F is BM25's own: a field the searcher scores otherwise is not native.
            if (fieldOk.test(f) == false || (fieldOk instanceof SimilarityEncoder s && s.isDefault(f) == false)) {
                return "field_similarity";
            }
            writeString(out, f);
            writeInt(out, Float.floatToIntBits((float) field(fw, "weight")));
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // max_sim
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link MaxSimQuery}: the sub-query's documents, each scored by its {@code MaxSim} against the
     * query's tokens (0 without a value), as {@code crates/lucene-engine/src/max_sim_query.rs} reads it.
     */
    private static String maxSim(MaxSimQuery q, ByteArrayOutputStream out, Child child) {
        out.write(NODE_MAX_SIM);
        String reason = child.node(q.sub());
        if (reason != null) {
            return reason;
        }
        writeString(out, q.field());
        float[][] tokens = q.tokens();
        writeInt(out, tokens[0].length);
        writeInt(out, tokens.length);
        for (float[] t : tokens) {
            for (float v : t) {
                writeInt(out, Float.floatToIntBits(v));
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // compact_wildcard
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link RustWildcardQuery}: its field, kind, case flag and pattern, as {@code NODE_SUBSTRING} in
     * {@code crates/lucene-engine/src/queries.rs} reads them. A regexp with other than {@code
     * RegExp.ALL} syntax runs in Java.
     */
    private static String substring(RustWildcardQuery q, ByteArrayOutputStream out) {
        if (q.nativePattern() == false) {
            return "substring_regexp_flags";
        }
        out.write(NODE_SUBSTRING);
        writeString(out, q.field());
        out.write(switch (q.kind()) {
            case WILDCARD -> SUBSTRING_WILDCARD;
            case REGEXP -> SUBSTRING_REGEXP;
            case CONTAINS -> SUBSTRING_CONTAINS;
            case EXACT -> SUBSTRING_EXACT;
        });
        out.write(q.caseInsensitive() ? 1 : 0);
        writeBytes(out, new BytesRef(q.pattern()));
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // pattern_text phrase
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link PatternPhraseQuery}, an exact phrase: its field, its index analyzer's spec, whether a
     * template's static text may decide it, and its terms by position, as {@code
     * crates/lucene-engine/src/pattern_phrase.rs} reads them. A phrase with slop or a prefix, or an
     * analyzer {@link AnalyzerSpec} cannot describe, runs in Java.
     */
    private static String patternPhrase(PatternPhraseQuery q, ByteArrayOutputStream out, Child child) {
        AnalyzerSpec.Spec spec = AnalyzerSpec.cached(q.analyzer(), q.field());
        if (spec.ok() == false) {
            return "pattern_phrase_" + spec.reason();
        }
        if (q.positions() == null) {
            // Slop or a prefix: the native side runs the phrase over each candidate message.
            out.write(NODE_PATTERN_PHRASE);
            writeString(out, q.field());
            writeBytes(out, spec.blob());
            out.write(q.templateParts() ? 1 : 0);
            out.write(1);
            String reason = child.node(q.approximation());
            return reason != null ? reason : child.node(q.positional());
        }
        out.write(NODE_PATTERN_PHRASE);
        writeString(out, q.field());
        writeBytes(out, spec.blob());
        out.write(q.templateParts() ? 1 : 0);
        out.write(0);
        int[] positions = q.positions();
        BytesRef[][] terms = q.terms();
        writeInt(out, positions.length);
        for (int i = 0; i < positions.length; i++) {
            writeInt(out, positions[i]);
            writeInt(out, terms[i].length);
            for (BytesRef t : terms[i]) {
                writeBytes(out, t);
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // script_score with knn_score
    // ---------------------------------------------------------------------------------------------

    /** The {@code min_score} of a scored query, as the node that wraps it ({@code minScore} null for none). */
    private static void minScore(Object minScore, ByteArrayOutputStream out) {
        if (minScore != null) {
            out.write(NODE_MIN_SCORE);
            writeInt(out, Float.floatToIntBits((Float) minScore));
        }
    }

    /**
     * OpenSearch's {@code ScriptScoreQuery} running k-NN's {@code knn_score} over a {@code
     * knn_vector} space: the sub-query's documents, each scored by the script's distance (0 without
     * a value), times the boost. Other scripts run in Java, as does a {@code hammingbit} over a field without doc values (which fails there).
     */
    private static String knnScore(ScriptScoreQuery q, ByteArrayOutputStream out, Child child) throws ReflectiveOperationException {
        Object builder = field(q, "scriptBuilder");
        if ((builder instanceof KnnScoringScriptEngine.KnnLeafFactory) == false) {
            return painlessKnn(q, out, child);
        }
        KnnScoringScriptEngine.Native spec = ((KnnScoringScriptEngine.KnnLeafFactory) builder).nativeSpec();
        if (spec == null) {
            return "script_score_hammingbit";
        }
        out.write(NODE_KNN_SCORE);
        String reason = child.node((Query) field(q, "subQuery"));
        if (reason != null) {
            return reason;
        }
        writeString(out, spec.field());
        out.write(spec.space());
        out.write(spec.dataType());
        out.write(spec.halfCosine() ? 1 : 0);
        if (spec.floats() != null) {
            out.write(0);
            writeInt(out, spec.floats().length);
            for (float v : spec.floats()) {
                writeInt(out, Float.floatToIntBits(v));
            }
        } else {
            out.write(1);
            writeBytes(out, spec.bytes());
        }
        // knn_score's own formula.
        out.write(0);
        return null;
    }

    /**
     * The field types of the index being searched, for scripts that name a field: set by the
     * searcher around encoding ({@link #withFieldTypes}); without them such scripts run in Java.
     */
    private static final ThreadLocal<Function<String, MappedFieldType>> FIELD_TYPES = new ThreadLocal<>();

    /** Runs {@code encode} with {@code fieldTypes} as the index's field types. */
    static <T> T withFieldTypes(
        Function<String, MappedFieldType> fieldTypes,
        org.opensearch.search.internal.SearchContext ctx,
        java.util.function.Supplier<T> encode
    ) {
        Function<String, MappedFieldType> previous = FIELD_TYPES.get();
        FIELD_TYPES.set(fieldTypes);
        ExtendedScripts.set(ExtendedScripts.ENABLED.get(ctx.getQueryShardContext().getIndexSettings().getSettings()));
        try {
            return encode.get();
        } finally {
            ExtendedScripts.clear();
            if (previous == null) {
                FIELD_TYPES.remove();
            } else {
                FIELD_TYPES.set(previous);
            }
        }
    }

    /**
     * A Painless {@code script_score} script made of one k-NN Painless function call inside
     * arithmetic ({@link PainlessKnnScript}), as the {@code knn_score} node with the script's
     * formula. Other scripts run in Java.
     */
    private static String painlessKnn(ScriptScoreQuery q, ByteArrayOutputStream out, Child child) throws ReflectiveOperationException {
        Script script = PainlessScore.inline((Script) field(q, "script"));
        if (field(q, "scriptBuilder") instanceof JsScripts.Leaf js) {
            return jsScore(q, js, out, child);
        }
        if (isExpression(field(q, "scriptBuilder"))) {
            return expressionScore(q, (ScoreScript.LeafFactory) field(q, "scriptBuilder"), out, child, "script_score_script");
        }
        if (script == null || "painless".equals(script.getLang()) == false) {
            return "script_score_script";
        }
        PainlessKnnScript.Compiled c = PainlessKnnScript.compile(script.getIdOrCode(), script.getParams(), FIELD_TYPES.get());
        if (c == null) {
            return painlessScore(q, script, out, child);
        }
        out.write(NODE_KNN_SCORE);
        String reason = child.node((Query) field(q, "subQuery"));
        if (reason != null) {
            return reason;
        }
        writeString(out, c.field());
        // The space and cosine flag are unused under this formula.
        out.write(0);
        out.write(c.dataType());
        out.write(0);
        if (c.floats() != null) {
            out.write(0);
            writeInt(out, c.floats().length);
            for (float v : c.floats()) {
                writeInt(out, Float.floatToIntBits(v));
            }
        } else {
            out.write(1);
            writeBytes(out, c.bytes());
        }
        out.write(1);
        out.write(c.function());
        writeInt(out, Float.floatToRawIntBits(c.norm()));
        out.writeBytes(c.program());
        return null;
    }

    /**
     * A Painless {@code script_score} script run by the native Painless runtime: its IR, params,
     * and how each field it names by a constant reads in {@code doc} (node 43).
     */
    private static String painlessScore(ScriptScoreQuery q, Script script, ByteArrayOutputStream out, Child child)
        throws ReflectiveOperationException {
        ScoreScript.LeafFactory factory = (ScoreScript.LeafFactory) field(q, "scriptBuilder");
        Object prepared = painless(script, factory.needs_score() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE, "script_score");
        if (prepared instanceof String reason) {
            return reason;
        }
        out.write(NODE_PAINLESS_SCORE);
        String reason = child.node((Query) field(q, "subQuery"));
        if (reason != null) {
            return reason;
        }
        painlessScoreSpec(out, (Painless) prepared, factory, (String) field(q, "indexName"), (int) field(q, "shardId"), (org.opensearch.Version) field(q, "indexVersion"));
        return null;
    }

    /** A score script's spec, registered as this request's next run for its errors. */
    private static void painlessScoreSpec(
        ByteArrayOutputStream out,
        Painless p,
        ScoreScript.LeafFactory factory,
        String index,
        int shard,
        org.opensearch.Version version
    ) {
        int run = PainlessScore.run(factory, index, shard, version);
        painlessSpec(out, p, index == null ? "" : index, shard, run);
    }

    /**
     * A {@code script} query ({@code ScriptQuery}) with an inline Painless script run by the
     * native runtime: node 45, its script's spec. Other scripts run in Java.
     */
    private static String painlessFilter(Query q, ByteArrayOutputStream out) throws ReflectiveOperationException {
        Script script = PainlessScore.inline((Script) field(q, "script"));
        if (field(q, "filterScript") instanceof JsScripts.Leaf js) {
            Object prepared = js(js, PainlessScore.FILTER, "script_query");
            if (prepared instanceof String reason) {
                return reason;
            }
            out.write(NODE_PAINLESS_FILTER);
            jsSpec(out, (Js) prepared, "", 0);
            return null;
        }
        Object wrapped = wrappedExpression(field(q, "filterScript"));
        if (wrapped != null) {
            if (((ScoreScript.LeafFactory) wrapped).needs_score()) {
                return "script_query_script";
            }
            ByteArrayOutputStream spec = new ByteArrayOutputStream();
            String why = expressionSpec(wrapped, PainlessScore.FILTER, "", 0, spec, "script_query_script");
            if (why != null) {
                return why;
            }
            out.write(NODE_PAINLESS_FILTER);
            out.writeBytes(spec.toByteArray());
            return null;
        }
        if (script == null || "painless".equals(script.getLang()) == false) {
            return "script_query_script";
        }
        Object prepared = painless(script, PainlessScore.FILTER, "script_query");
        if (prepared instanceof String reason) {
            return reason;
        }
        out.write(NODE_PAINLESS_FILTER);
        int run = PainlessScore.run((FilterScript.LeafFactory) field(q, "filterScript"));
        painlessSpec(out, (Painless) prepared, "", 0, run);
        return null;
    }

    /**
     * A {@code _script} sort key's number sort script, made by {@code factory}, as the native
     * runtime runs it: its spec (context {@link PainlessScore#SORT}, or {@link
     * PainlessScore#SORT_WITH_SCORE} for a script reading {@code _score}), or why it stays in Java.
     */
    static String painlessSort(NumberSortScript.LeafFactory factory, ByteArrayOutputStream out) {
        byte context = factory.needs_score() ? PainlessScore.SORT_WITH_SCORE : PainlessScore.SORT;
        if (factory instanceof JsScripts.Leaf js) {
            return jsSpec(out, js, context, "sort_script");
        }
        if (isExpression(factory)) {
            return expressionSpec(factory, context, "", 0, out, "sort_script");
        }
        if (isExpression(factory)) {
            return expressionSpec(factory, PainlessScore.SORT, "", 0, out, "sort_script");
        }
        Object[] sp = PainlessScore.sourceAndParams(factory);
        if (sp == null || (sp[0] instanceof String source) == false || (sp[1] instanceof Map<?, ?>) == false) {
            return "sort_script";
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) sp[1];
        if (PainlessScore.compiledWithDefaults((String) sp[0]) == false) {
            return "sort_script_options";
        }
        Object prepared = painless(new Script(ScriptType.INLINE, "painless", (String) sp[0], params), context, "sort_script");
        if (prepared instanceof String reason) {
            return reason;
        }
        painlessSpec(out, (Painless) prepared, "", 0, PainlessScore.run(factory));
        return null;
    }

    /**
     * An aggregation's script, made by {@code factory}, as the native runtime runs it: its spec
     * (context {@link PainlessScore#AGGREGATION}), or why it stays in Java. A script reading
     * {@code _score} stays in Java.
     */
    static String painlessAggregation(AggregationScript.LeafFactory factory, ByteArrayOutputStream out) {
        if (factory == null || factory.needs_score()) {
            return "aggregation_script_score";
        }
        if (factory instanceof JsScripts.Leaf js) {
            return jsSpec(out, js, PainlessScore.AGGREGATION, "aggregation_script");
        }
        if (isExpression(factory)) {
            return expressionSpec(factory, PainlessScore.AGGREGATION, "", 0, out, "aggregation_script");
        }
        Object[] sp = PainlessScore.sourceAndParams(factory);
        if (sp == null || (sp[0] instanceof String source) == false || (sp[1] instanceof Map<?, ?>) == false) {
            return "aggregation_script";
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) sp[1];
        if (PainlessScore.compiledWithDefaults((String) sp[0]) == false) {
            return "aggregation_script_options";
        }
        Object prepared = painless(new Script(ScriptType.INLINE, "painless", (String) sp[0], params), PainlessScore.AGGREGATION, "aggregation_script");
        if (prepared instanceof String reason) {
            return reason;
        }
        painlessSpec(out, (Painless) prepared, "", 0, PainlessScore.run(factory));
        return null;
    }

    /**
     * A {@code terms_set} query's {@code minimum_should_match_script}, made by {@code factory}, as
     * the native runtime runs it: its spec (context {@link PainlessScore#TERMS_SET}), or why it
     * stays in Java.
     */
    static String painlessTermsSet(TermsSetQueryScript.LeafFactory factory, ByteArrayOutputStream out) {
        if (factory instanceof JsScripts.Leaf js) {
            return jsSpec(out, js, PainlessScore.TERMS_SET, "covering");
        }
        if (isExpression(factory)) {
            return expressionSpec(factory, PainlessScore.TERMS_SET, "", 0, out, "covering_script");
        }
        Object[] sp = PainlessScore.sourceAndParams(factory);
        if (sp == null || (sp[0] instanceof String) == false || (sp[1] instanceof Map<?, ?>) == false) {
            return "covering_script";
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) sp[1];
        if (PainlessScore.compiledWithDefaults((String) sp[0]) == false) {
            return "covering_options";
        }
        Object prepared = painless(new Script(ScriptType.INLINE, "painless", (String) sp[0], params), PainlessScore.TERMS_SET, "covering");
        if (prepared instanceof String reason) {
            return reason;
        }
        painlessSpec(out, (Painless) prepared, "", 0, PainlessScore.runTermsSet(factory));
        return null;
    }

    /**
     * An {@code intervals} query's script filter, as the native runtime runs it: its spec (context
     * {@link PainlessScore#INTERVAL}), or why it stays in Java.
     */
    private static String painlessInterval(IntervalFilterScript script, ByteArrayOutputStream out) {
        String source = PainlessScore.sourceOf(script);
        if (source == null) {
            return "interval_script";
        }
        if (PainlessScore.compiledWithDefaults(source) == false) {
            return "interval_options";
        }
        Object prepared = painless(new Script(ScriptType.INLINE, "painless", source, Map.of()), PainlessScore.INTERVAL, "interval");
        if (prepared instanceof String reason) {
            return reason;
        }
        if (((Painless) prepared).names().length != 0) {
            return "interval_painless";
        }
        painlessSpec(out, (Painless) prepared, "", 0, PainlessScore.runInterval(script));
        return null;
    }

    /**
     * OpenSearch's scripted similarity as the native runtime scores with it (similarity kind 9
     * of {@code similarity.rs}, after the kind), or null when it stays in Java: a script that
     * is not inline Painless compiled with no options or params of its own, or one the static
     * check refuses. {@code similarity} is the scripted similarity, or the {@code
     * NonNegativeScoresSimilarity} around it.
     */
    static byte[] painlessSimilarity(Similarity similarity) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            org.opensearch.index.similarity.ScriptedSimilarity sim;
            int negative;
            if (similarity instanceof org.opensearch.index.similarity.NonNegativeScoresSimilarity n) {
                sim = (org.opensearch.index.similarity.ScriptedSimilarity) n.getDelegate();
                Field enforce = org.opensearch.index.similarity.NonNegativeScoresSimilarity.class.getDeclaredField(
                    "ENFORCE_POSITIVE_SCORES"
                );
                enforce.setAccessible(true);
                negative = enforce.getBoolean(null) ? 1 : 2;
            } else {
                sim = (org.opensearch.index.similarity.ScriptedSimilarity) similarity;
                negative = 0;
            }
            out.write(field(sim, "discountOverlaps") instanceof Boolean b && b ? 1 : 0);
            out.write(negative);
            writeString(out, sim.toString());
            Object weightFactory = field(sim, "weightScriptFactory");
            if (weightFactory == null) {
                out.write(0);
            } else {
                out.write(1);
                Object w = ((org.opensearch.script.SimilarityWeightScript.Factory) weightFactory).newInstance();
                if (similarityScript(out, w, (String) field(sim, "weightScriptSource"), true, sim) == false) {
                    return null;
                }
            }
            Object script = ((org.opensearch.script.SimilarityScript.Factory) field(sim, "scriptFactory")).newInstance();
            if (similarityScript(out, script, (String) field(sim, "scriptSource"), false, sim) == false) {
                return null;
            }
            return out.toByteArray();
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** Writes a scripted similarity's {@code script}, whose {@code Script} printed {@code printed}: false when it stays in Java. */
    private static boolean similarityScript(
        ByteArrayOutputStream out,
        Object script,
        String printed,
        boolean weight,
        org.opensearch.index.similarity.ScriptedSimilarity sim
    ) {
        String source = PainlessScore.sourceOf(script);
        // Compiled as the native runtime compiles it: inline Painless, no options, no params.
        if (source == null || new Script(ScriptType.INLINE, "painless", source, Map.of()).toString().equals(printed) == false) {
            return false;
        }
        Object compiledOrWhy = PainlessScore.similarity(source, weight);
        if ((compiledOrWhy instanceof PainlessIr.Compiled) == false) {
            return false;
        }
        // No docFields check: here `doc` is the typed ScriptedSimilarity.Doc, which Painless's
        // DocFieldsPhase still lists by name (doc.freq) though no doc values are read.
        PainlessIr.Compiled compiled = (PainlessIr.Compiled) compiledOrWhy;
        byte[] params;
        try {
            params = PainlessIr.constant(Map.of());
        } catch (PainlessIr.Unsupported e) {
            return false;
        }
        byte context = weight ? PainlessScore.SIMILARITY_WEIGHT : PainlessScore.SIMILARITY;
        Painless p = new Painless(context, compiled, params, new String[0], new byte[0]);
        String refused = PainlessScore.check(compiled, context, params, new byte[0], () -> {
            ByteArrayOutputStream checked = new ByteArrayOutputStream();
            painlessSpec(checked, p, "", 0, 0);
            return checked.toByteArray();
        });
        if (refused != null) {
            return false;
        }
        painlessSpec(out, p, "", 0, PainlessScore.runSimilarity(sim, weight));
        return true;
    }

    /** A script the native runtime runs: what {@link #painlessSpec} writes, but the run. */
    private record Painless(byte context, PainlessIr.Compiled compiled, byte[] params, String[] names, byte[] kinds) {}

    /**
     * {@code script} in {@code context} ({@link PainlessScore#SCORE}, {@link
     * PainlessScore#SCORE_WITH_SCORE} or {@link PainlessScore#FILTER}) as the native runtime runs
     * it: a {@link Painless}, or why it stays in Java ({@code reason} and why).
     */
    private static Object painless(Script script, byte context, String reason) {
        Function<String, MappedFieldType> types = FIELD_TYPES.get();
        if (types == null || script.getOptions().keySet().stream().anyMatch(k -> k.equals(Script.CONTENT_TYPE_OPTION) == false)) {
            return reason + "_script";
        }
        Object compiledOrWhy = switch (context) {
            case PainlessScore.FILTER -> PainlessScore.filter(script.getIdOrCode());
            case PainlessScore.SORT, PainlessScore.SORT_WITH_SCORE -> PainlessScore.sort(script.getIdOrCode());
            case PainlessScore.AGGREGATION -> PainlessScore.aggregation(script.getIdOrCode());
            case PainlessScore.TERMS_SET -> PainlessScore.termsSet(script.getIdOrCode());
            case PainlessScore.INTERVAL -> PainlessScore.interval(script.getIdOrCode());
            default -> PainlessScore.score(script.getIdOrCode());
        };
        if ((compiledOrWhy instanceof PainlessIr.Compiled) == false) {
            return reason + "_painless";
        }
        PainlessIr.Compiled compiled = (PainlessIr.Compiled) compiledOrWhy;
        byte[] params;
        try {
            params = PainlessIr.constant(script.getParams());
        } catch (PainlessIr.Unsupported e) {
            return reason + "_params";
        }
        byte[] kinds = new byte[compiled.docFields().size()];
        // The doc values a field's name reads: an alias's are its target's.
        String[] names = new String[kinds.length];
        for (int i = 0; i < kinds.length; i++) {
            MappedFieldType ft = types.apply(compiled.docFields().get(i));
            kinds[i] = PainlessScore.fieldKind(ft);
            names[i] = ft == null ? compiled.docFields().get(i) : ft.name();
            if ((kinds[i] == PainlessScore.DATE || kinds[i] == PainlessScore.DATE_NANOS) && ExtendedScripts.on() == false) {
                // Dates reach the script as ZonedDateTime through `def` calls: 2 to 5 times slower than
                // OpenSearch's own Painless on a whole-index scan.
                return "script_date_values";
            }
        }
        if (context == PainlessScore.SORT_WITH_SCORE && ExtendedScripts.on() == false) {
            // A sort script that reads the score is evaluated one hit at a time through the interpreter:
            // 1.5 to 2 times slower than OpenSearch's own compiled Painless.
            return "script_sort_score";
        }
        // The whole script checked before anything runs: one the native runtime would not run
        // exactly as Painless does stays in Java, never tried natively first.
        Painless p = new Painless(context, compiled, params, names, kinds);
        String refused = PainlessScore.check(compiled, context, params, kinds, () -> {
            ByteArrayOutputStream checked = new ByteArrayOutputStream();
            painlessSpec(checked, p, "", 0, 0);
            return checked.toByteArray();
        });
        return refused != null ? reason + "_painless" : p;
    }

    /** A painless node's script, after any sub-query, as {@code painless_score.rs} reads it. */
    private static void painlessSpec(ByteArrayOutputStream out, Painless p, String index, int shard, int run) {
        out.write(p.context());
        writeBytes(out, p.compiled().ir());
        writeBytes(out, p.params());
        writeString(out, index);
        writeInt(out, shard);
        writeInt(out, run);
        writeInt(out, p.kinds().length);
        for (int i = 0; i < p.kinds().length; i++) {
            // The name the script reads it by, then the one its doc values are under.
            writeString(out, p.compiled().docFields().get(i));
            writeString(out, p.names()[i]);
            out.write(p.kinds()[i]);
        }
    }

    /** A JavaScript script the native runtime runs: what {@link #jsSpec} writes, but the place and run. */
    private record Js(byte context, String source, byte[] params, List<String> names, List<String> reads, byte[] kinds) {}

    /**
     * A {@code js} {@code script_score} query as node 43: its sub-query, then its script. A
     * search the native runtime does not run fails before the script runs ({@link JsScripts}).
     */
    private static String jsScore(ScriptScoreQuery q, JsScripts.Leaf js, ByteArrayOutputStream out, Child child)
        throws ReflectiveOperationException {
        Object prepared = js(js, js.needsScore() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE, "script_score");
        if (prepared instanceof String reason) {
            return reason;
        }
        out.write(NODE_PAINLESS_SCORE);
        String reason = child.node((Query) field(q, "subQuery"));
        if (reason != null) {
            return reason;
        }
        String index = (String) field(q, "indexName");
        jsSpec(out, (Js) prepared, index == null ? "" : index, (int) field(q, "shardId"));
        return null;
    }

    /** {@code js} in {@code context} written to {@code out}, or why not ({@code reason} and why). */
    private static String jsSpec(ByteArrayOutputStream out, JsScripts.Leaf js, byte context, String reason) {
        Object prepared = js(js, context, reason);
        if (prepared instanceof String why) {
            return why;
        }
        jsSpec(out, (Js) prepared, "", 0);
        return null;
    }

    /**
     * {@code js} in {@code context} as the native runtime runs it: a {@link Js} with the fields
     * its source names that the mapping has, or why not.
     */
    private static Object js(JsScripts.Leaf js, byte context, String reason) {
        Function<String, MappedFieldType> types = FIELD_TYPES.get();
        if (types == null) {
            return reason + "_script";
        }
        byte[] params;
        try {
            params = PainlessIr.constant(js.params());
        } catch (PainlessIr.Unsupported e) {
            return reason + "_params";
        }
        List<String> names = JsScripts.fields(js.source(), types);
        List<String> reads = new java.util.ArrayList<>(names.size());
        byte[] kinds = new byte[names.size()];
        for (int i = 0; i < kinds.length; i++) {
            MappedFieldType ft = types.apply(names.get(i));
            kinds[i] = jsFieldKind(ft);
            reads.add(ft.name());
        }
        if (JsScripts.check(js.source(), context) != null) {
            return reason + "_js";
        }
        return new Js(context, js.source(), params, names, reads, kinds);
    }

    /** How a field reads in a js script's {@code doc}: as in Painless, and a millisecond date as its epoch millis. */
    private static byte jsFieldKind(MappedFieldType ft) {
        if (ft instanceof org.opensearch.index.mapper.DateFieldMapper.DateFieldType d) {
            return d.hasDocValues() && d.resolution() == org.opensearch.index.mapper.DateFieldMapper.Resolution.MILLISECONDS
                ? PainlessScore.LONG
                : PainlessScore.UNSUPPORTED;
        }
        return PainlessScore.fieldKind(ft);
    }

    /** A js script's node fields, after any sub-query, registered as this request's next run. */
    private static void jsSpec(ByteArrayOutputStream out, Js js, String index, int shard) {
        int run = PainlessScore.runJs(js.source());
        JsScripts.writeSpec(out, js.context(), js.source(), js.params(), js.names(), js.reads(), js.kinds(), index, shard, run);
    }


    // ---------------------------------------------------------------------------------------------
    // Lucene expressions (lang: expression)
    // ---------------------------------------------------------------------------------------------

    private static final String EXPRESSION_PACKAGE = "org.opensearch.script.expression.";

    /** The Joda date getters an expression's {@code date} object serves, in the native runtime's order. */
    private static final List<String> JODA_PARTS = List.of(
        "centuryOfEra", "dayOfMonth", "dayOfWeek", "dayOfYear", "era", "hourOfDay", "millisOfDay", "millisOfSecond",
        "minuteOfDay", "minuteOfHour", "monthOfYear", "secondOfDay", "secondOfMinute", "weekOfWeekyear", "weekyear",
        "year", "yearOfCentury", "yearOfEra"
    );

    /** Whether {@code o} is a script of the expression language module. */
    static boolean isExpression(Object o) {
        return o != null && o.getClass().getName().startsWith(EXPRESSION_PACKAGE);
    }

    /**
     * The expression behind a {@code script} query's filter factory, which wraps the score
     * script's factory, or null.
     */
    private static Object wrappedExpression(Object filterFactory) {
        if (filterFactory == null || filterFactory.getClass().getClassLoader() == null) {
            return null;
        }
        for (Field f : filterFactory.getClass().getDeclaredFields()) {
            try {
                f.setAccessible(true);
                Object v = f.get(filterFactory);
                if (v instanceof ScoreScript.LeafFactory && isExpression(v)) {
                    return v;
                }
            } catch (ReflectiveOperationException | RuntimeException e) {
                return null;
            }
        }
        return null;
    }

    /**
     * An expression script's spec ({@code painless_score.rs}'s {@code ExprSource}) written to {@code
     * out}: its source, the fields its variables read, and what each variable is. Returns null, or
     * {@code reason} when the script binds a variable the native runtime does not read.
     */
    private static String expressionSpec(Object factory, byte context, String index, int shard, ByteArrayOutputStream out, String reason) {
        Function<String, MappedFieldType> types = FIELD_TYPES.get();
        if (types == null) {
            return reason;
        }
        if (context != PainlessScore.AGGREGATION && context != PainlessScore.TERMS_SET && ExtendedScripts.on() == false) {
            // One value read per document costs more here than in Lucene's expression engine.
            return "script_expression_per_document";
        }
        try {
            Object expr = field(factory, "exprScript");
            String source = (String) field(expr, "sourceText");
            String[] variables = (String[]) field(expr, "variables");
            // The score script's factory keeps only the expression's value source, whose variable
            // sources are in the order of the expression's variables.
            Object bindings = Reflect.inHierarchy(factory.getClass(), "bindings") != null ? field(factory, "bindings") : null;
            DoubleValuesSource[] bound = bindings == null ? (DoubleValuesSource[]) field(field(factory, "source"), "variables") : null;
            java.lang.reflect.Method get = bindings == null ? null : bindings.getClass().getMethod("getDoubleValuesSource", String.class);
            java.util.LinkedHashMap<String, Integer> fields = new java.util.LinkedHashMap<>();
            ByteArrayOutputStream vars = new ByteArrayOutputStream();
            for (int v = 0; v < variables.length; v++) {
                String variable = variables[v];
                Object src = get != null ? get.invoke(bindings, variable) : bound[v];
                String cn = src.getClass().getName();
                writeString(vars, variable);
                if (src == DoubleValuesSource.SCORES) {
                    vars.write(0);
                } else if (cn.endsWith("PerThreadReplaceableConstDoubleValueSource") && context == PainlessScore.AGGREGATION) {
                    // `_value`: the field value the aggregator maps.
                    vars.write(10);
                } else if (cn.endsWith("ConstantValuesSource")) {
                    vars.write(1);
                    writeLong(vars, Double.doubleToRawLongBits(((DoubleValuesSource) src).getValues(null, null).doubleValue()));
                } else if (cn.startsWith(EXPRESSION_PACKAGE) && Reflect.inHierarchy(src.getClass(), "fieldData") != null) {
                    Object data = field(src, "fieldData");
                    String name = ((IndexFieldData<?>) data).getFieldName();
                    MappedFieldType ft = types.apply(name);
                    if (ft == null || ft.hasDocValues() == false) {
                        return reason;
                    }
                    int at = fields.computeIfAbsent(ft.name(), n -> fields.size());
                    String kind = cn.substring(EXPRESSION_PACKAGE.length());
                    boolean geo = kind.startsWith("Geo");
                    if (geo && data.getClass().getName().equals(LAT_LON_FIELD_DATA) == false) {
                        return reason;
                    }
                    byte[] numeric = geo ? null : scoreNumeric(data);
                    if (geo == false && numeric == null && kind.equals("EmptyMemberValueSource") == false && kind.equals("CountMethodValueSource") == false) {
                        return reason;
                    }
                    switch (kind) {
                        case "FieldDataValueSource" -> {
                            int m = switch (((Enum<?>) field(src, "multiValueMode")).name()) {
                                case "MIN" -> 0;
                                case "MAX" -> 1;
                                case "AVG" -> 2;
                                case "MEDIAN" -> 3;
                                case "SUM" -> 4;
                                default -> -1;
                            };
                            if (m < 0) {
                                return reason;
                            }
                            vars.write(2);
                            writeInt(vars, at);
                            vars.writeBytes(numeric);
                            vars.write(m);
                        }
                        case "EmptyMemberValueSource" -> {
                            vars.write(3);
                            writeInt(vars, at);
                        }
                        case "CountMethodValueSource" -> {
                            vars.write(4);
                            writeInt(vars, at);
                        }
                        case "DateMethodValueSource" -> {
                            int calendarField = (int) field(src, "calendarType");
                            if (calendarField != 1 && calendarField != 2 && calendarField != 5 && calendarField != 11
                                && calendarField != 12 && calendarField != 13) {
                                return reason;
                            }
                            vars.write(5);
                            writeInt(vars, at);
                            vars.writeBytes(numeric);
                            vars.write(calendarField);
                        }
                        case "DateObjectValueSource" -> {
                            String method = (String) field(src, "methodName");
                            if (method.startsWith("get") && method.length() > 3) {
                                method = Character.toLowerCase(method.charAt(3)) + method.substring(4);
                            }
                            int part = JODA_PARTS.indexOf(method);
                            if (part < 0) {
                                return reason;
                            }
                            vars.write(6);
                            writeInt(vars, at);
                            vars.writeBytes(numeric);
                            vars.write(part);
                        }
                        case "GeoLatitudeValueSource" -> {
                            vars.write(7);
                            writeInt(vars, at);
                        }
                        case "GeoLongitudeValueSource" -> {
                            vars.write(8);
                            writeInt(vars, at);
                        }
                        case "GeoEmptyValueSource" -> {
                            vars.write(9);
                            writeInt(vars, at);
                        }
                        default -> {
                            return reason;
                        }
                    }
                } else {
                    return reason;
                }
            }
            out.write(context | 0x40);
            writeBytes(out, source.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            writeBytes(out, PainlessIr.constant(Map.of()));
            writeString(out, index == null ? "" : index);
            writeInt(out, shard);
            writeInt(out, 0);
            writeInt(out, fields.size());
            for (String name : fields.keySet()) {
                writeString(out, name);
                writeString(out, name);
                // The stored longs, which each binding reads as its field type says.
                out.write(PainlessScore.LONG);
            }
            writeInt(out, variables.length);
            out.writeBytes(vars.toByteArray());
            return null;
        } catch (ReflectiveOperationException | java.io.IOException | RuntimeException | PainlessIr.Unsupported e) {
            return reason;
        }
    }

    /** A {@code script_score} query's expression as node 43: its sub-query, then its spec. */
    private static String expressionScore(ScriptScoreQuery q, ScoreScript.LeafFactory factory, ByteArrayOutputStream out, Child child, String reason)
        throws ReflectiveOperationException {
        ByteArrayOutputStream spec = new ByteArrayOutputStream();
        String why = expressionSpec(
            factory, factory.needs_score() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE,
            (String) field(q, "indexName"), (int) field(q, "shardId"), spec, reason
        );
        if (why != null) {
            return why;
        }
        out.write(NODE_PAINLESS_SCORE);
        String childReason = child.node((Query) field(q, "subQuery"));
        if (childReason != null) {
            return childReason;
        }
        out.writeBytes(spec.toByteArray());
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // function_score
    // ---------------------------------------------------------------------------------------------

    private static String functionScore(
        FunctionScoreQuery q,
        ByteArrayOutputStream out,
        Predicate<String> fieldOk,
        Child child
    ) throws ReflectiveOperationException {
        minScore(q.getMinScore(), out);
        // OpenSearch's weight passes any boost from above to the sub-query's weight alone; the
        // native source does the same (`boosts_wrapped_query`), whatever the ancestor.
        Query sub = q.getSubQuery();
        ScoreFunction[] functions = q.getFunctions();
        if (functions.length == 0) {
            // FunctionFactorScorer.score(): the sub-query's score, unchanged -- but with boost_mode
            // replace the weight never scores the sub-query (needsScores is false), so it is 0.
            if (q.getCombineFunction() == CombineFunction.REPLACE) {
                return child.node(new org.apache.lucene.search.BoostQuery(new org.apache.lucene.search.ConstantScoreQuery(sub), 0f));
            }
            return child.node(sub);
        }
        out.write(NODE_FUNCTION_SCORE);
        out.write((byte) q.getCombineFunction().ordinal());
        out.write((byte) ((Enum<?>) field(q, "scoreMode")).ordinal());
        writeInt(out, Float.floatToIntBits((float) field(q, "maxBoost")));
        String reason = child.node(sub);
        if (reason != null) {
            return reason;
        }
        writeInt(out, functions.length);
        for (ScoreFunction f : functions) {
            if (f instanceof FunctionScoreQuery.FilterScoreFunction fsf) {
                out.write(1);
                reason = child.node(fsf.filter);
                if (reason != null) {
                    return reason;
                }
                f = fsf.function;
            } else {
                out.write(0);
            }
            reason = function(f, out, true);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    private static final String FUNCTIONS = "org.opensearch.index.query.functionscore.";

    private static String function(ScoreFunction f, ByteArrayOutputStream out, boolean outer) throws ReflectiveOperationException {
        if (f instanceof WeightFactorFunction w && outer) {
            ScoreFunction inner = w.getScoreFunction();
            out.write(1);
            writeInt(out, Float.floatToIntBits(w.getWeight()));
            return function(inner, out, false);
        }
        String name = f.getClass().getName();
        if (name.equals(WeightFactorFunction.class.getName() + "$ScoreOne")) {
            out.write(0);
            return null;
        }
        if (f.getClass() == FieldValueFactorFunction.class) {
            Object data = field(f, "indexFieldData");
            byte[] numeric = scoreNumeric(data);
            if (numeric == null) {
                return "function_score_field_data";
            }
            Double missing = (Double) field(f, "missing");
            FieldValueFactorFunction.Modifier modifier = (FieldValueFactorFunction.Modifier) field(f, "modifier");
            out.write(2);
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            // The requested name, an alias's own, as FieldValueFactorFunction's errors name it.
            writeString(out, (String) field(f, "field"));
            out.writeBytes(numeric);
            writeInt(out, Float.floatToIntBits((float) field(f, "boostFactor")));
            out.write((byte) modifier.ordinal());
            out.write(missing == null ? 0 : 1);
            writeLong(out, Double.doubleToRawLongBits(missing == null ? 0d : missing));
            return null;
        }
        if (f.getClass() == RandomScoreFunction.class) {
            Object data = field(f, "fieldData");
            out.write(3);
            writeInt(out, (int) field(f, "saltedSeed"));
            if (data == null) {
                out.write(0);
                return null;
            }
            String dataClass = data.getClass().getName();
            byte[] numeric = null;
            if (dataClass.equals("org.opensearch.index.mapper.IdFieldMapper$IdFieldType$1$1")) {
                if (ExtendedScripts.on() == false) {
                    // The stored `_id` of every scored document is decompressed: twice stock's time.
                    return "random_score_id";
                }
                out.write(3);
                return null;
            }
            if (dataClass.equals("org.opensearch.index.fielddata.plain.SortedSetOrdinalsIndexFieldData")) {
                out.write(1);
            } else if ((numeric = scoreNumeric(data)) != null) {
                out.write(2);
            } else {
                return "function_score_field_data";
            }
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            if (numeric != null) {
                out.writeBytes(numeric);
            }
            return null;
        }
        if (name.equals(FUNCTIONS + "DecayFunctionBuilder$NumericFieldDataScoreFunction")) {
            Object data = field(f, "fieldData");
            byte[] numeric = scoreNumeric(data);
            if (numeric == null) {
                return "function_score_field_data";
            }
            byte decay = decayKind(f);
            if (decay < 0) {
                return "function_score_decay";
            }
            byte m = multiValueMode(f);
            out.write(4);
            out.write(decay);
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            out.writeBytes(numeric);
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "origin")));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "scale")));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "offset")));
            out.write(m);
            return null;
        }
        if (name.equals(FUNCTIONS + "DecayFunctionBuilder$GeoFieldDataScoreFunction")) {
            byte decay = decayKind(f);
            if (decay < 0) {
                return "function_score_decay";
            }
            org.opensearch.common.geo.GeoPoint origin = (org.opensearch.common.geo.GeoPoint) field(f, "origin");
            Object data = field(f, "fieldData");
            if (data == null || data.getClass().getName().equals(LAT_LON_FIELD_DATA) == false) {
                return "function_score_field_data";
            }
            out.write(6);
            out.write(decay);
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            writeLong(out, Double.doubleToRawLongBits(origin.lat()));
            writeLong(out, Double.doubleToRawLongBits(origin.lon()));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "scale")));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "offset")));
            out.write(multiValueMode(f));
            return null;
        }
        if (f.getClass() == ScriptScoreFunction.class) {
            Script script = PainlessScore.inline((Script) field(f, "sScript"));
            ScoreScript.LeafFactory factory = (ScoreScript.LeafFactory) field(f, "script");
            if (factory instanceof JsScripts.Leaf js) {
                Object prepared = js(js, js.needsScore() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE, "function_score");
                if (prepared instanceof String) {
                    return "function_score_script";
                }
                out.write(5);
                jsSpec(out, (Js) prepared, (String) field(f, "indexName"), (int) field(f, "shardId"));
                return null;
            }
            if (isExpression(factory)) {
                ByteArrayOutputStream spec = new ByteArrayOutputStream();
                String why = expressionSpec(
                    factory, factory.needs_score() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE,
                    (String) field(f, "indexName"), (int) field(f, "shardId"), spec, "function_score_script"
                );
                if (why != null) {
                    return why;
                }
                out.write(5);
                out.writeBytes(spec.toByteArray());
                return null;
            }
            if (script == null || "painless".equals(script.getLang()) == false) {
                return "function_score_script";
            }
            Object prepared = painless(script, factory.needs_score() ? PainlessScore.SCORE_WITH_SCORE : PainlessScore.SCORE, "function_score");
            if (prepared instanceof String why) {
                return "script_date_values".equals(why) ? why : "function_score_script";
            }
            out.write(5);
            painlessScoreSpec(out, (Painless) prepared, factory, (String) field(f, "indexName"), (int) field(f, "shardId"), (org.opensearch.Version) field(f, "indexVersion"));
            return null;
        }
        if (name.endsWith("ScriptScoreFunction")) {
            return "function_score_script";
        }
        return "function_score_" + f.getClass().getSimpleName();
    }

    /** The {@code geo_point} field data a geo decay reads: {@code LatLonPointDVLeafFieldData}'s index side. */
    private static final String LAT_LON_FIELD_DATA = "org.opensearch.index.fielddata.plain.AbstractLatLonPointIndexFieldData$LatLonPointIndexFieldData";

    /** A decay function's {@code decay} byte of its {@code func}, or -1 for a decay this does not know. */
    private static byte decayKind(ScoreFunction f) throws ReflectiveOperationException {
        String func = field(f, "func").getClass().getName();
        return func.equals(FUNCTIONS + "GaussDecayFunctionBuilder$GaussScoreFunction") ? (byte) 0
            : func.equals(FUNCTIONS + "ExponentialDecayFunctionBuilder$ExponentialDecayScoreFunction") ? (byte) 1
            : func.equals(FUNCTIONS + "LinearDecayFunctionBuilder$LinearDecayScoreFunction") ? (byte) 2
            : (byte) -1;
    }

    /** A decay function's {@code MultiValueMode} as the node's byte. */
    private static byte multiValueMode(ScoreFunction f) throws ReflectiveOperationException {
        return switch ((MultiValueMode) field(f, "mode")) {
            case SUM -> 0;
            case AVG -> 1;
            case MEDIAN -> 2;
            case MIN -> 3;
            case MAX -> 4;
        };
    }

    /**
     * How {@code function_score} reads a field data's doubles, as the node writes it ({@code
     * function_score.rs}'s {@code numeric}): 0 an integral type, boolean or date as a long, 1 a
     * double, 2 a float, 3 a half float, 4 an unsigned long, 5 a nanosecond date (its
     * milliseconds), 6 a scaled float and the inverse of its scaling factor; null for field data
     * that is none of these.
     */
    static byte[] scoreNumeric(Object data) {
        if (data == null) {
            return null;
        }
        String name = data.getClass().getName();
        if (name.equals("org.opensearch.index.mapper.ScaledFloatFieldMapper$ScaledFloatIndexFieldData")) {
            try {
                Object scaled = field(data, "scaledFieldData");
                if (scaled == null || numericType(scaled) != 0) {
                    return null;
                }
                double inverse = 1d / (double) field(data, "scalingFactor");
                java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(9).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                b.put((byte) 6).putLong(Double.doubleToRawLongBits(inverse));
                return b.array();
            } catch (ReflectiveOperationException | RuntimeException e) {
                return null;
            }
        }
        if (name.equals("org.opensearch.index.fielddata.plain.SortedNumericIndexFieldData") == false) {
            return null;
        }
        return switch (((IndexNumericFieldData) data).getNumericType()) {
            case BOOLEAN, BYTE, SHORT, INT, LONG, DATE -> new byte[] { 0 };
            case DOUBLE -> new byte[] { 1 };
            case FLOAT -> new byte[] { 2 };
            case HALF_FLOAT -> new byte[] { 3 };
            case UNSIGNED_LONG -> new byte[] { 4 };
            case DATE_NANOSECONDS -> new byte[] { 5 };
        };
    }

    /**
     * How the native side reads a field data's doubles: 0 a long ({@code (double) v}) -- the
     * integral types, {@code boolean} (0 and 1) and {@code date} (epoch milliseconds), all of which
     * OpenSearch's {@code SortedNumericLongFieldData} serves, whose {@code getBytesValues} prints
     * each value with {@code Long.toString} -- 1 a double, 2 a float; -1 for field data that is not
     * plain sorted-numeric doc values of those types.
     */
    static byte numericType(Object data) {
        if (data == null || data.getClass().getName().equals("org.opensearch.index.fielddata.plain.SortedNumericIndexFieldData") == false) {
            return -1;
        }
        return switch (((IndexNumericFieldData) data).getNumericType()) {
            case BOOLEAN, BYTE, SHORT, INT, LONG, DATE -> 0;
            case DOUBLE -> 1;
            case FLOAT -> 2;
            default -> -1;
        };
    }

    // ---------------------------------------------------------------------------------------------

    private static void writeString(ByteArrayOutputStream out, String s) {
        writeBytes(out, s.getBytes(StandardCharsets.UTF_8));
    }

    private static void writeBytes(ByteArrayOutputStream out, BytesRef b) {
        writeInt(out, b.length);
        out.write(b.bytes, b.offset, b.length);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        writeInt(out, (int) v);
        writeInt(out, (int) (v >>> 32));
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
