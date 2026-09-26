/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.IndexOrDocValuesQuery;
import org.apache.lucene.search.PointRangeQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.util.BytesRefIterator;
import org.apache.lucene.util.automaton.ByteRunAutomaton;
import java.io.IOException;
import java.util.function.Supplier;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.BytesRef;
import org.opensearch.search.approximate.ApproximateScoreQuery;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.function.Predicate;

/**
 * Translates a rewritten Lucene {@link Query} into the query blob {@code decode_query} in {@code
 * crates/lucene-ffi/src/jvm_reader.rs} reads, or says why it cannot.
 *
 * <p>This is the <em>supported matrix</em> of the native path, and it is deliberately exact rather
 * than generous: a query is encoded only when every node of it is one the Rust engine answers with
 * Lucene's hits and scores. Everything else returns an {@link Encoded} with a {@code
 * fallbackReason}, and runs on Lucene. Encodable today, recursively in any combination:
 *
 * <ul>
 *   <li>{@link TermQuery} (the plain class, scoring from the reader's own statistics);
 *   <li>{@link BooleanQuery}, any {@link BooleanClause.Occur} and {@code minimumNumberShouldMatch};
 *   <li>{@link ConstantScoreQuery} and {@link BoostQuery} -- OpenSearch builds the first for every
 *       {@code term} query on a {@code keyword} field, and the second for a boosted query and for
 *       a filter-only {@code bool} (boost 0);
 *   <li>{@link DisjunctionMaxQuery} -- {@code multi_match} {@code best_fields} and {@code dis_max};
 *   <li>{@link MatchAllDocsQuery} and {@link MatchNoDocsQuery};
 *   <li>{@link ApproximateScoreQuery} -- OpenSearch 3.x's wrapper around {@code match_all} and
 *       {@code range}, which only substitutes its approximation for sorted searches (never native):
 *       encoded as the original query it wraps, which scores identically.
 * </ul>
 *
 * <p>The blob is {@code QUERY_TREE}: one node per query, each a kind byte and its payload (the
 * layout is {@code decode_node}'s doc). A root {@link TermQuery} is sent as {@code QUERY_TERM}, the
 * native engine's single-term entry point.
 */
public final class QueryEncoder {
    /**
     * An encoded query, or the reason there is none (exactly one of {@code blob} and {@code
     * fallbackReason} is non-null). {@code fast} says whether the shape is one the native engine is
     * measured to run at least as fast as Lucene ({@link #isFast}).
     */
    public record Encoded(byte[] blob, String fallbackReason, boolean fast) {
        static Encoded fallback(String reason) {
            return new Encoded(null, reason, false);
        }
    }

    /** Blob tag for {@code min_score} in front of a query blob (the native {@code QUERY_MIN_SCORE}). */
    static final byte QUERY_MIN_SCORE = 3;

    /**
     * {@code blob} behind OpenSearch's {@code min_score}: the tag, the minimum as a float's bits,
     * then the query blob itself.
     */
    static byte[] withMinScore(byte[] blob, float min) {
        byte[] out = new byte[blob.length + 5];
        out[0] = QUERY_MIN_SCORE;
        int bits = Float.floatToIntBits(min);
        for (int i = 0; i < 4; i++) {
            out[1 + i] = (byte) (bits >>> (8 * i));
        }
        System.arraycopy(blob, 0, out, 5, blob.length);
        return out;
    }

    /**
     * Whether the native engine is measured at least as fast as Lucene on {@code q}'s shape. Since
     * read path R1 (the scorer tree and bulk scorers, {@code docs/milestones/m5-6-native-read.md})
     * that is every shape this class encodes: the boosts, {@code must_not}s, mixed booleans and
     * {@code constant_score} wrappers M2 measured at 0.13-0.28x and routed to Lucene now run
     * 1.2-6.7x Lucene in process. So an encodable query is a fast one; the {@code native_shapes}
     * setting no longer has anything to exclude.
     */
    public static boolean isFast(Query q) {
        return encode(q, field -> true).blob() != null;
    }

    private QueryEncoder() {}

    /**
     * Encodes {@code query}; {@code fieldOk} is asked about every field the query touches, and a
     * field it rejects (a non-default similarity, say) falls the whole query back.
     */
    public static Encoded encode(Query query, Predicate<String> fieldOk) {
        query = unwrapUnitBoost(query);
        if (query instanceof TermQuery tq) {
            if (plainTerm(tq) == false) {
                return Encoded.fallback("term_states");
            }
            Term t = tq.getTerm();
            if (fieldOk.test(t.field()) == false) {
                return Encoded.fallback("field_similarity");
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(NativeBridge.QUERY_TERM);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return new Encoded(out.toByteArray(), null, true);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(NativeBridge.QUERY_TREE);
        String reason = node(query, out, fieldOk, 0, new int[1]);
        if (reason != null) {
            return Encoded.fallback(reason);
        }
        return new Encoded(out.toByteArray(), null, true);
    }

    /** The native decoder's depth limit ({@code MAX_CLAUSE_DEPTH} in {@code query.rs}). */
    private static final int MAX_DEPTH = 32;

    /**
     * The native decoder's node limit ({@code MAX_CLAUSE_COUNT} in {@code query.rs}), over every
     * node, wrappers included. Lucene counts only leaves and OpenSearch's {@code
     * indices.query.bool.max_clause_count} may raise its limit, so a larger query is legal: it runs
     * on Lucene, under a named reason rather than as a native error.
     */
    private static final int MAX_NODES = 1024;

    private static final byte NODE_TERM = 0;
    private static final byte NODE_BOOLEAN = 1;
    private static final byte NODE_CONSTANT_SCORE = 2;
    private static final byte NODE_BOOST = 3;
    private static final byte NODE_DISMAX = 4;
    private static final byte NODE_MATCH_ALL = 5;
    private static final byte NODE_MATCH_NONE = 6;
    private static final byte NODE_PHRASE = 7;
    private static final byte NODE_TERM_SET = 8;
    private static final byte NODE_PREFIX = 9;
    private static final byte NODE_WILDCARD = 10;
    private static final byte NODE_POINT_RANGE = 11;

    /** Appends one node (and its children); returns a fallback reason, or null. */
    private static String node(Query q, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        if (depth >= MAX_DEPTH) {
            return "query_too_deep";
        }
        if (++nodes[0] > MAX_NODES) {
            return "query_too_large";
        }
        q = unwrapUnitBoost(q);
        if (q instanceof TermQuery tq) {
            if (plainTerm(tq) == false) {
                return "term_states";
            }
            Term t = tq.getTerm();
            if (fieldOk.test(t.field()) == false) {
                return "field_similarity";
            }
            out.write(NODE_TERM);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return null;
        }
        if (q instanceof BooleanQuery bq) {
            if (bq.getMinimumNumberShouldMatch() < 0) {
                // BooleanQuery.Builder does not reject it; the native decoder does.
                return "boolean_msm_negative";
            }
            out.write(NODE_BOOLEAN);
            writeInt(out, bq.getMinimumNumberShouldMatch());
            writeInt(out, bq.clauses().size());
            for (BooleanClause c : bq.clauses()) {
                out.write((byte) c.occur().ordinal());
                String reason = node(c.query(), out, fieldOk, depth + 1, nodes);
                if (reason != null) {
                    return reason;
                }
            }
            return null;
        }
        if (q instanceof ConstantScoreQuery cs) {
            // ConstantScoreQuery scores its boost, which is 1 unless a BoostQuery wraps it.
            out.write(NODE_CONSTANT_SCORE);
            writeInt(out, Float.floatToIntBits(1f));
            return node(cs.getQuery(), out, fieldOk, depth + 1, nodes);
        }
        if (q instanceof BoostQuery b) {
            float boost = b.getBoost();
            if (Float.isFinite(boost) == false || boost < 0) {
                return "boost_invalid";
            }
            out.write(NODE_BOOST);
            writeInt(out, Float.floatToIntBits(boost));
            return node(b.getQuery(), out, fieldOk, depth + 1, nodes);
        }
        if (q instanceof DisjunctionMaxQuery dm) {
            out.write(NODE_DISMAX);
            writeInt(out, Float.floatToIntBits(dm.getTieBreakerMultiplier()));
            writeInt(out, dm.getDisjuncts().size());
            for (Query d : dm.getDisjuncts()) {
                String reason = node(d, out, fieldOk, depth + 1, nodes);
                if (reason != null) {
                    return reason;
                }
            }
            return null;
        }
        if (q instanceof ApproximateScoreQuery a) {
            // A wrapper, not a level (nor a node): the range it approximates is still the root
            // when it is.
            nodes[0]--;
            return node(a.getOriginalQuery(), out, fieldOk, depth, nodes);
        }
        if (q.getClass() == PhraseQuery.class) {
            PhraseQuery pq = (PhraseQuery) q;
            Term[] terms = pq.getTerms();
            int[] positions = pq.getPositions();
            if (terms.length == 0) {
                out.write(NODE_MATCH_NONE);
                return null;
            }
            if (fieldOk.test(pq.getField()) == false) {
                return "field_similarity";
            }
            // The native phrase has no position gaps (a stopword the analyzer removed).
            for (int i = 0; i < positions.length; i++) {
                if (positions[i] - positions[0] != i) {
                    return "phrase_positions";
                }
            }
            nodes[0] += terms.length;
            if (nodes[0] > MAX_NODES) {
                return "query_too_large";
            }
            out.write(NODE_PHRASE);
            writeBytes(out, pq.getField().getBytes(StandardCharsets.UTF_8));
            writeInt(out, pq.getSlop());
            writeInt(out, terms.length);
            for (int i = 0; i < terms.length; i++) {
                writeInt(out, i);
                writeBytes(out, terms[i].bytes());
            }
            return null;
        }
        if (q instanceof IndexOrDocValuesQuery iodv) {
            // Two equivalent ways to run one query; the native side runs the index one. A
            // wrapper, not a level (nor a node).
            nodes[0]--;
            return node(iodv.getIndexQuery(), out, fieldOk, depth, nodes);
        }
        if (q instanceof PointRangeQuery pr) {
            // `LongPoint`/`DoublePoint`/date ranges: one dimension of 8 bytes, sent as the sortable
            // longs their packed bytes encode. A 4-byte point (`integer`, `float`) falls back.
            if (pr.getNumDims() != 1 || pr.getBytesPerDim() != Long.BYTES) {
                return "points_width";
            }
            out.write(NODE_POINT_RANGE);
            writeBytes(out, pr.getField().getBytes(StandardCharsets.UTF_8));
            writeLong(out, NumericUtils.sortableBytesToLong(pr.getLowerPoint(), 0));
            writeLong(out, NumericUtils.sortableBytesToLong(pr.getUpperPoint(), 0));
            return null;
        }
        String wrapper = q.getClass().getSimpleName();
        if (wrapper.equals("MultiTermQueryConstantScoreBlendedWrapper") || wrapper.equals("MultiTermQueryConstantScoreWrapper")) {
            // A MultiTermQuery under a constant-score rewrite (package-private classes, so found
            // by name; their query by the visitor, which hands it over as it is).
            Query inner = wrappedMultiTermQuery(q);
            if (inner == null) {
                return "clause_" + wrapper;
            }
            return multiTerm(inner, out, fieldOk, depth, nodes);
        }
        if (q.getClass() == MatchAllDocsQuery.class) {
            out.write(NODE_MATCH_ALL);
            return null;
        }
        if (q.getClass() == MatchNoDocsQuery.class) {
            out.write(NODE_MATCH_NONE);
            return null;
        }
        // The whole query is unsupported ("query_"), or one clause of an otherwise native tree is.
        return (depth == 0 ? "query_" : "clause_") + name(q);
    }

    /** The query a constant-score multi-term wrapper wraps, as its visitor reports it, or null. */
    private static Query wrappedMultiTermQuery(Query wrapper) {
        Query[] inner = new Query[1];
        wrapper.visit(new QueryVisitor() {
            @Override
            public void consumeTerms(Query query, Term... terms) {
                if (inner[0] == null) inner[0] = query;
            }

            @Override
            public void consumeTermsMatching(Query query, String field, Supplier<ByteRunAutomaton> automaton) {
                if (inner[0] == null) inner[0] = query;
            }

            @Override
            public void visitLeaf(Query query) {
                if (inner[0] == null) inner[0] = query;
            }

            @Override
            public QueryVisitor getSubVisitor(BooleanClause.Occur occur, Query parent) {
                return this;
            }
        });
        return inner[0];
    }

    /** A term set, prefix or wildcard under a constant-score rewrite: scored as its boost. */
    private static String multiTerm(Query q, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        if (q.getClass() == TermInSetQuery.class) {
            TermInSetQuery ts = (TermInSetQuery) q;
            nodes[0] += (int) Math.min(Integer.MAX_VALUE, ts.getTermsCount());
            if (nodes[0] > MAX_NODES) {
                return "query_too_large";
            }
            out.write(NODE_TERM_SET);
            writeBytes(out, ts.getField().getBytes(StandardCharsets.UTF_8));
            writeInt(out, (int) ts.getTermsCount());
            try {
                BytesRefIterator it = ts.getBytesRefIterator();
                for (BytesRef t = it.next(); t != null; t = it.next()) {
                    writeBytes(out, BytesRef.deepCopyOf(t));
                }
            } catch (IOException e) {
                return "clause_TermInSetQuery";
            }
            return null;
        }
        if (q.getClass() == PrefixQuery.class) {
            Term t = ((PrefixQuery) q).getPrefix();
            out.write(NODE_PREFIX);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return null;
        }
        if (q.getClass() == WildcardQuery.class) {
            Term t = ((WildcardQuery) q).getTerm();
            if (t.text().indexOf(WildcardQuery.WILDCARD_ESCAPE) >= 0) {
                // The native matcher has no escape syntax.
                return "wildcard_escape";
            }
            out.write(NODE_WILDCARD);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return null;
        }
        return (depth == 0 ? "query_" : "clause_") + name(q);
    }

    /**
     * A {@link TermQuery} that scores from the reader's own statistics: exactly that class (not a
     * subclass that may score differently) and no caller-supplied {@code TermStates}. OpenSearch
     * builds term queries with blended {@code TermStates} for {@code multi_match} {@code
     * cross_fields}; the native engine would score those with the unblended ones.
     */
    static boolean plainTerm(TermQuery tq) {
        return tq.getClass() == TermQuery.class && tq.getTermStates() == null;
    }

    /** A query class's name for a fallback reason; an anonymous class reports its enclosing class. */
    private static String name(Query q) {
        Class<?> c = q.getClass();
        while (c.getSimpleName().isEmpty() && c.getEnclosingClass() != null) {
            c = c.getEnclosingClass();
        }
        return c.getSimpleName();
    }

    private static Query unwrapUnitBoost(Query q) {
        while (q instanceof BoostQuery b && b.getBoost() == 1f) {
            q = b.getQuery();
        }
        return q;
    }

    private static byte[] toArray(BytesRef b) {
        byte[] out = new byte[b.length];
        System.arraycopy(b.bytes, b.offset, out, 0, b.length);
        return out;
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        writeInt(out, (int) v);
        writeInt(out, (int) (v >>> 32));
    }

    private static void writeBytes(ByteArrayOutputStream out, BytesRef b) {
        writeBytes(out, toArray(b));
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
