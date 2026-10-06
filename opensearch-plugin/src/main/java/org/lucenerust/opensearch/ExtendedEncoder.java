/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.queries.intervals.IntervalQuery;
import org.apache.lucene.queries.intervals.IntervalsSource;
import org.apache.lucene.queries.spans.FieldMaskingSpanQuery;
import org.apache.lucene.queries.spans.SpanContainingQuery;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanPositionRangeQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWithinQuery;
import org.apache.lucene.search.CombinedFieldQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.search.join.ToParentBlockJoinQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Transition;
import org.opensearch.common.lucene.search.function.CombineFunction;
import org.opensearch.common.lucene.search.function.FieldValueFactorFunction;
import org.opensearch.common.lucene.search.function.FunctionScoreQuery;
import org.opensearch.common.lucene.search.function.RandomScoreFunction;
import org.opensearch.common.lucene.search.function.ScoreFunction;
import org.opensearch.common.lucene.search.function.WeightFactorFunction;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.IndexNumericFieldData;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.search.MultiValueMode;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * The M10 half of {@link QueryEncoder} (T10.7): the OpenSearch shapes {@code lucene-join} and
 * {@code lucene-queries} back natively, as the nodes {@code crates/lucene-ffi/src/jvm_nodes.rs}
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
 *       (node 24).
 * </ul>
 *
 * <p>The classes are read through their getters, or through {@link Reflect} where Lucene and
 * OpenSearch keep the fields private (the fields their constructors were given, which are what
 * their {@code equals} compares). Not encoded, each under a named reason: a {@code span_near} with
 * a {@code span_gap} ({@code span_gap}), a payload span query ({@code span_<Class>}), an interval
 * source with a payload filter or of a class this does not know ({@code interval_<Class>}), a
 * {@code function_score} with {@code min_score} ({@code function_score_min_score}), a script
 * ({@code function_score_script}), a geo or other non-numeric decay ({@code
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

    /** What {@link #node} returns for a query that is none of these. */
    static final String NOT_EXTENDED = "not_extended";

    /** Writes a child query as a node one level down; returns a fallback reason, or null. */
    interface Child {
        String node(Query q);
    }

    private ExtendedEncoder() {}

    /**
     * Appends {@code q} when it is one of these shapes: null when it was written, a fallback reason
     * when it cannot be, {@link #NOT_EXTENDED} when it is none of them. {@code boost} is a boost
     * around it that the shape takes itself (OpenSearch's function score passes it to its
     * sub-query); 1 otherwise.
     */
    static String node(Query q, float boost, ByteArrayOutputStream out, Predicate<String> fieldOk, Child child) {
        try {
            if (q instanceof FunctionScoreQuery fs) {
                return functionScore(fs, boost, out, fieldOk, child);
            }
            if (boost != 1f) {
                return NOT_EXTENDED;
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

    /** The query a parent filter's bit sets are built from, or null for a producer this cannot read. */
    private static Query parentsQuery(BitSetProducer p) throws ReflectiveOperationException {
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
            for (SpanQuery s : n.getClauses()) {
                if (s.getClass().getSimpleName().equals("SpanGapQuery")) {
                    return "span_gap";
                }
            }
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

    private static Object field(Object o, String name) throws ReflectiveOperationException {
        Field f = Reflect.inHierarchy(o.getClass(), name);
        if (f == null) {
            throw new NoSuchFieldException(name);
        }
        return f.get(o);
    }

    @SuppressWarnings("unchecked")
    private static String source(IntervalsSource s, ByteArrayOutputStream out) throws ReflectiveOperationException {
        String name = s.getClass().getName();
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
                out.write(5);
                writeInt(out, (int) field(s, "childCount"));
                Object n = field(s, "name");
                out.write(n == null ? 0 : "ORDERED".equals(n) ? 1 : "UNORDERED".equals(n) ? 2 : 3);
                if (n != null && "ORDERED".equals(n) == false && "UNORDERED".equals(n) == false) {
                    return "interval_repeating_name";
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
                return automaton((CompiledAutomaton) field(s, "automaton"), out);
            }
            default -> {
                return "interval_" + s.getClass().getSimpleName();
            }
        }
    }

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
    private static final int MAX_AUTOMATON_STATES = 1 << 16;

    private static String automaton(CompiledAutomaton c, ByteArrayOutputStream out) {
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
                    return "interval_automaton";
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
                return "interval_automaton";
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
            return "combined_fields_empty";
        }
        out.write(NODE_COMBINED_FIELD);
        writeBytes(out, term);
        writeInt(out, fields.size());
        for (Object fw : fields.values()) {
            String f = (String) field(fw, "field");
            if (fieldOk.test(f) == false) {
                return "field_similarity";
            }
            writeString(out, f);
            writeInt(out, Float.floatToIntBits((float) field(fw, "weight")));
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------
    // function_score
    // ---------------------------------------------------------------------------------------------

    private static String functionScore(
        FunctionScoreQuery q,
        float boost,
        ByteArrayOutputStream out,
        Predicate<String> fieldOk,
        Child child
    ) throws ReflectiveOperationException {
        if (q.getMinScore() != null) {
            return "function_score_min_score";
        }
        // OpenSearch's weight passes the boost to the sub-query's weight alone.
        Query sub = boost == 1f ? q.getSubQuery() : new org.apache.lucene.search.BoostQuery(q.getSubQuery(), boost);
        ScoreFunction[] functions = q.getFunctions();
        if (functions.length == 0) {
            // FunctionFactorScorer.score(): the sub-query's score, unchanged.
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
            byte numeric = numericType(data);
            if (numeric < 0) {
                return "function_score_field_data";
            }
            Double missing = (Double) field(f, "missing");
            FieldValueFactorFunction.Modifier modifier = (FieldValueFactorFunction.Modifier) field(f, "modifier");
            out.write(2);
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            out.write(numeric);
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
            if (dataClass.equals("org.opensearch.index.fielddata.plain.SortedSetOrdinalsIndexFieldData")) {
                out.write(1);
            } else if (numericType(data) == 0) {
                out.write(2);
            } else {
                return "function_score_field_data";
            }
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            return null;
        }
        if (name.equals(FUNCTIONS + "DecayFunctionBuilder$NumericFieldDataScoreFunction")) {
            Object data = field(f, "fieldData");
            byte numeric = numericType(data);
            if (numeric < 0) {
                return "function_score_field_data";
            }
            String func = field(f, "func").getClass().getName();
            byte decay = func.equals(FUNCTIONS + "GaussDecayFunctionBuilder$GaussScoreFunction") ? (byte) 0
                : func.equals(FUNCTIONS + "ExponentialDecayFunctionBuilder$ExponentialDecayScoreFunction") ? (byte) 1
                : func.equals(FUNCTIONS + "LinearDecayFunctionBuilder$LinearDecayScoreFunction") ? (byte) 2
                : (byte) -1;
            if (decay < 0) {
                return "function_score_decay";
            }
            MultiValueMode mode = (MultiValueMode) field(f, "mode");
            byte m = switch (mode) {
                case SUM -> 0;
                case AVG -> 1;
                case MEDIAN -> 2;
                case MIN -> 3;
                case MAX -> 4;
            };
            out.write(4);
            out.write(decay);
            writeString(out, ((IndexFieldData<?>) data).getFieldName());
            out.write(numeric);
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "origin")));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "scale")));
            writeLong(out, Double.doubleToRawLongBits((double) field(f, "offset")));
            out.write(m);
            return null;
        }
        if (name.endsWith("ScriptScoreFunction")) {
            return "function_score_script";
        }
        return "function_score_" + f.getClass().getSimpleName();
    }

    /**
     * How the native side reads a field data's doubles: 0 a long ({@code (double) v}), 1 a double,
     * 2 a float; -1 for field data that is not plain sorted-numeric doc values of those types.
     */
    private static byte numericType(Object data) {
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
