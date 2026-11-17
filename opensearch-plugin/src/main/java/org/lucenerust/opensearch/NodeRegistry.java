/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiPredicate;
import org.apache.lucene.search.Query;

/**
 * An ordered table of query encoders: one row per query class (or per class and condition). The
 * first row whose class matches the query, and whose condition holds, writes it; a query no row
 * claims gets {@link #NOT_REGISTERED}, so the caller can try the next table.
 *
 * <p>Adding a query type is one {@link #add} line plus the method the row points at. Rows are
 * tried in the order they were added, so a more specific row (an exact class, a row with a
 * condition) goes before a general one over the same class.
 *
 * @param <W> what an encoder is handed besides the query: the walk's output, depth and context
 */
final class NodeRegistry<W> {
    /** What {@link #encode} returns for a query no row claims. */
    static final String NOT_REGISTERED = "not_registered";

    /** Writes one query: null when written, a fallback reason when it cannot be. */
    @FunctionalInterface
    interface Encoder<Q extends Query, W> {
        String encode(Q query, W walk) throws ReflectiveOperationException;
    }

    private record Row<W>(Class<? extends Query> type, boolean exact, BiPredicate<Query, W> when, Encoder<Query, W> encoder) {
        boolean claims(Query q, W w) {
            return (exact ? q.getClass() == type : type.isInstance(q)) && (when == null || when.test(q, w));
        }
    }

    private final List<Row<W>> rows = new ArrayList<>();

    /** A row for {@code type} and its subclasses. */
    <Q extends Query> NodeRegistry<W> add(Class<Q> type, Encoder<Q, W> encoder) {
        return row(type, false, null, encoder);
    }

    /** A row for {@code type} and its subclasses, taken only when {@code when} holds. */
    <Q extends Query> NodeRegistry<W> addWhen(Class<Q> type, BiPredicate<Q, W> when, Encoder<Q, W> encoder) {
        return row(type, false, (q, w) -> when.test(type.cast(q), w), encoder);
    }

    /** A row for exactly {@code type}, not its subclasses. */
    <Q extends Query> NodeRegistry<W> addExact(Class<Q> type, Encoder<Q, W> encoder) {
        return row(type, true, null, encoder);
    }

    @SuppressWarnings("unchecked")
    private <Q extends Query> NodeRegistry<W> row(Class<Q> type, boolean exact, BiPredicate<Query, W> when, Encoder<Q, W> encoder) {
        rows.add(new Row<>(type, exact, when, (Encoder<Query, W>) encoder));
        return this;
    }

    /** Encodes {@code q} with the first row that claims it, or returns {@link #NOT_REGISTERED}. */
    String encode(Query q, W walk) throws ReflectiveOperationException {
        for (Row<W> r : rows) {
            if (r.claims(q, walk)) {
                return r.encoder().encode(q, walk);
            }
        }
        return NOT_REGISTERED;
    }

    /** The number of rows, for the tests that pin the table. */
    int size() {
        return rows.size();
    }
}
