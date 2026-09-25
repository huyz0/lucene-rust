/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Translates a rewritten Lucene {@link Query} into the query blob {@code decode_query} in {@code
 * crates/lucene-ffi/src/jvm_reader.rs} reads, or says why it cannot.
 *
 * <p>This is the <em>supported matrix</em> of the native path, and it is deliberately exact rather
 * than generous: a shape is encoded only when the Rust searcher produces the same hits and scores
 * Lucene does for it. Everything else returns an {@link Encoded} with a {@code fallbackReason},
 * and runs on Lucene.
 *
 * <ul>
 *   <li>{@link TermQuery};
 *   <li>{@link BooleanQuery} whose clauses are, recursively, {@link TermQuery} or {@link
 *       BooleanQuery}, with any {@link BooleanClause.Occur} and {@code minimumNumberShouldMatch};
 *   <li>{@link ConstantScoreQuery} and {@link BoostQuery} around any of these -- OpenSearch builds
 *       the first for every {@code term} query on a {@code keyword} field, and the second for a
 *       boosted query and for a filter-only {@code bool} (boost 0).
 * </ul>
 *
 * <p>A wrapper at the root is sent as a boolean with that one {@code MUST} clause, which scores
 * identically.
 *
 * <p>A boolean with no clauses, or none that can match (only {@code MUST_NOT}), falls back:
 * Lucene rewrites those to match nothing, and that is not worth a native path.
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

    /**
     * The shapes the native engine runs at least as fast as Lucene, measured through the REST layer
     * ({@code docs/benchmarks/m2-opensearch-e2e.md}): a term; a constant-score term (a {@code
     * keyword} {@code term} query); and a boolean of those that is either a pure disjunction (only
     * {@code SHOULD}, {@code minimumNumberShouldMatch <= 1}) or a pure conjunction (only {@code
     * MUST}/{@code FILTER}). Every other encodable shape is answered correctly natively but slower,
     * because the Rust engine prunes only these shapes; those are routed to Lucene unless the index
     * sets {@code index.lucene_rust.search.native_shapes: all}.
     */
    public static boolean isFast(Query q) {
        q = unwrapUnitBoost(q);
        if (isTermLeaf(q)) {
            return true;
        }
        if (!(q instanceof BooleanQuery bq) || bq.getMinimumNumberShouldMatch() > 1) {
            return false;
        }
        boolean should = false;
        boolean required = false;
        for (BooleanClause c : bq.clauses()) {
            if (isTermLeaf(unwrapUnitBoost(c.query())) == false || c.occur() == BooleanClause.Occur.MUST_NOT) {
                return false;
            }
            should |= c.occur() == BooleanClause.Occur.SHOULD;
            required |= c.occur() != BooleanClause.Occur.SHOULD;
        }
        return should != required;
    }

    private static boolean isTermLeaf(Query q) {
        return q instanceof TermQuery || (q instanceof ConstantScoreQuery cs && cs.getQuery() instanceof TermQuery);
    }

    private QueryEncoder() {}

    /**
     * Encodes {@code query}; {@code fieldOk} is asked about every field the query touches, and a
     * field it rejects (a non-default similarity, say) falls the whole query back.
     */
    public static Encoded encode(Query query, Predicate<String> fieldOk) {
        query = unwrapUnitBoost(query);
        boolean fast = isFast(query);
        if (query instanceof ConstantScoreQuery || query instanceof BoostQuery) {
            query = new BooleanQuery.Builder().add(query, BooleanClause.Occur.MUST).build();
        }
        if (query instanceof TermQuery tq) {
            Term t = tq.getTerm();
            if (fieldOk.test(t.field()) == false) {
                return Encoded.fallback("field_similarity");
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(NativeBridge.QUERY_TERM);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return new Encoded(out.toByteArray(), null, fast);
        }
        if (query instanceof BooleanQuery bq) {
            List<Clause> clauses = new ArrayList<>();
            String reason = flatten(bq, -1, clauses, fieldOk);
            if (reason != null) {
                return Encoded.fallback(reason);
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(NativeBridge.QUERY_BOOLEAN);
            writeInt(out, bq.getMinimumNumberShouldMatch());
            writeInt(out, clauses.size());
            for (Clause c : clauses) {
                out.write(c.occur);
                out.write(c.kind);
                writeInt(out, c.parent);
                writeInt(out, c.param);
                writeBytes(out, c.field);
                writeBytes(out, c.term);
            }
            return new Encoded(out.toByteArray(), null, fast);
        }
        return Encoded.fallback("query_" + name(query));
    }

    private record Clause(byte occur, byte kind, int parent, int param, byte[] field, byte[] term) {}

    private static final byte[] EMPTY = new byte[0];

    /** Appends {@code bq}'s clauses under {@code parent}, depth first; returns a fallback reason or null. */
    private static String flatten(BooleanQuery bq, int parent, List<Clause> out, Predicate<String> fieldOk) {
        List<BooleanClause> clauses = bq.clauses();
        if (clauses.isEmpty()) {
            return "boolean_empty";
        }
        boolean positive = false;
        for (BooleanClause c : clauses) {
            positive |= c.occur() != BooleanClause.Occur.MUST_NOT;
        }
        if (positive == false) {
            return "boolean_pure_negative";
        }
        for (BooleanClause c : clauses) {
            String reason = clause(c.query(), (byte) c.occur().ordinal(), parent, out, fieldOk);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    /** Appends one clause (and, for a container, its children); returns a fallback reason or null. */
    private static String clause(Query q, byte occur, int parent, List<Clause> out, Predicate<String> fieldOk) {
        q = unwrapUnitBoost(q);
        int me = out.size();
        if (q instanceof TermQuery tq) {
            Term t = tq.getTerm();
            if (fieldOk.test(t.field()) == false) {
                return "field_similarity";
            }
            out.add(new Clause(occur, KIND_TERM, parent, 0, t.field().getBytes(StandardCharsets.UTF_8), toArray(t.bytes())));
            return null;
        }
        if (q instanceof BooleanQuery nested) {
            out.add(new Clause(occur, KIND_BOOLEAN, parent, nested.getMinimumNumberShouldMatch(), EMPTY, EMPTY));
            return flatten(nested, me, out, fieldOk);
        }
        if (q instanceof ConstantScoreQuery cs) {
            // ConstantScoreQuery scores its boost, which is 1 unless a BoostQuery wraps it.
            out.add(new Clause(occur, KIND_CONSTANT_SCORE, parent, Float.floatToIntBits(1f), EMPTY, EMPTY));
            return clause(cs.getQuery(), MUST, me, out, fieldOk);
        }
        if (q instanceof BoostQuery b) {
            float boost = b.getBoost();
            if (Float.isFinite(boost) == false || boost < 0) {
                return "boost_invalid";
            }
            out.add(new Clause(occur, KIND_BOOST, parent, Float.floatToIntBits(boost), EMPTY, EMPTY));
            return clause(b.getQuery(), MUST, me, out, fieldOk);
        }
        return "clause_" + name(q);
    }

    private static final byte MUST = (byte) BooleanClause.Occur.MUST.ordinal();
    private static final byte KIND_TERM = 0;
    private static final byte KIND_BOOLEAN = 1;
    private static final byte KIND_CONSTANT_SCORE = 2;
    private static final byte KIND_BOOST = 3;

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
