/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
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
 *   <li>{@link BoostQuery} with boost exactly 1, which scores as its inner query.
 * </ul>
 *
 * <p>A boolean with no clauses, or none that can match (only {@code MUST_NOT}), falls back:
 * Lucene rewrites those to match nothing, and that is not worth a native path.
 */
public final class QueryEncoder {
    /** An encoded query, or the reason there is none. Exactly one field is non-null. */
    public record Encoded(byte[] blob, String fallbackReason) {
        static Encoded fallback(String reason) {
            return new Encoded(null, reason);
        }
    }

    private QueryEncoder() {}

    /**
     * Encodes {@code query}; {@code fieldOk} is asked about every field the query touches, and a
     * field it rejects (a non-default similarity, say) falls the whole query back.
     */
    public static Encoded encode(Query query, Predicate<String> fieldOk) {
        query = unwrapUnitBoost(query);
        if (query instanceof TermQuery tq) {
            Term t = tq.getTerm();
            if (fieldOk.test(t.field()) == false) {
                return Encoded.fallback("field_similarity");
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            out.write(NativeBridge.QUERY_TERM);
            writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, t.bytes());
            return new Encoded(out.toByteArray(), null);
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
            return new Encoded(out.toByteArray(), null);
        }
        return Encoded.fallback("query_" + query.getClass().getSimpleName());
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
            byte occur = (byte) c.occur().ordinal();
            Query q = unwrapUnitBoost(c.query());
            if (q instanceof TermQuery tq) {
                Term t = tq.getTerm();
                if (fieldOk.test(t.field()) == false) {
                    return "field_similarity";
                }
                out.add(new Clause(occur, (byte) 0, parent, 0, t.field().getBytes(StandardCharsets.UTF_8), toArray(t.bytes())));
            } else if (q instanceof BooleanQuery nested) {
                int me = out.size();
                out.add(new Clause(occur, (byte) 1, parent, nested.getMinimumNumberShouldMatch(), EMPTY, EMPTY));
                String reason = flatten(nested, me, out, fieldOk);
                if (reason != null) {
                    return reason;
                }
            } else {
                return "clause_" + q.getClass().getSimpleName();
            }
        }
        return null;
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
