/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.join;

import org.apache.lucene.search.FilterWeight;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.index.LeafReaderContext;
import org.lucenerust.opensearch.LiveDirectoryReader;
import org.opensearch.search.internal.ContextIndexSearcher;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Objects;

/**
 * OpenSearch's {@code HasChildQueryBuilder.LateParsingQuery} (what {@code has_child} and {@code
 * has_parent} build), kept unrewritten until it is searched. The late-parsing query's rewrite runs
 * the whole join -- its from side searched over the reader's global ordinals -- while the search
 * context is prepared, before the native engine could take the query; this one rewrites only the
 * two sides, so the native engine runs the join itself (the {@code join} node of {@code
 * ExtendedEncoder}). Searched by Lucene, it runs the late-parsing query's rewrite then, against
 * the same searcher: the same join, the same scores.
 *
 * <p>A search profiled, or one of the writer's buffered documents, gets the late-parsing query's
 * own rewrite at once: the profile names Lucene's queries, and Lucene answers those searches.
 */
public final class DeferredJoinQuery extends Query {
    private final Query late;
    /** The late-parsing query's from and to sides, rewritten; null until {@link #rewrite}. */
    private final Query from;
    private final Query to;

    DeferredJoinQuery(Query late) {
        this(late, null, null);
    }

    private DeferredJoinQuery(Query late, Query from, Query to) {
        this.late = late;
        this.from = from;
        this.to = to;
    }

    /** The late-parsing query this stands for. */
    public Query late() {
        return late;
    }

    /** The from side, rewritten as {@code JoinUtil.createJoinQuery} rewrites it; null before. */
    public Query from() {
        return from;
    }

    /** The to side, rewritten; null before. */
    public Query to() {
        return to;
    }

    /** {@code joinField}. */
    public String joinField() throws ReflectiveOperationException {
        return (String) field("joinField").get(late);
    }

    /** {@code scoreMode}. */
    public org.apache.lucene.search.join.ScoreMode scoreMode() throws ReflectiveOperationException {
        return (org.apache.lucene.search.join.ScoreMode) field("scoreMode").get(late);
    }

    public int minChildren() throws ReflectiveOperationException {
        return field("minChildren").getInt(late);
    }

    public int maxChildren() throws ReflectiveOperationException {
        return field("maxChildren").getInt(late);
    }

    private Field field(String name) throws ReflectiveOperationException {
        Field f = JoinQueries.lateField(late.getClass(), name);
        if (f == null) {
            throw new NoSuchFieldException(name);
        }
        return f;
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        if (searcher instanceof ContextIndexSearcher c && c.getProfiler() != null
            || LiveDirectoryReader.find(searcher.getIndexReader()) != null) {
            return late;
        }
        if (from != null) {
            return this;
        }
        // What LateParsingQuery.rewrite searches the from side with, and what JoinUtil rewrites
        // both sides with first.
        IndexSearcher sides = new IndexSearcher(searcher.getIndexReader());
        sides.setQueryCache(null);
        try {
            Similarity similarity = (Similarity) field("similarity").get(late);
            if (similarity != null) {
                sides.setSimilarity(similarity);
            }
            return new DeferredJoinQuery(
                late,
                sides.rewrite((Query) field("innerQuery").get(late)),
                sides.rewrite((Query) field("toQuery").get(late))
            );
        } catch (ReflectiveOperationException | ClassCastException e) {
            return late;
        }
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Weight w = searcher.createWeight(searcher.rewrite(late), scoreMode, boost);
        // The join's matches in one segment depend on the others: never cached under this query,
        // whose equality has no reader (the joins Lucene builds compare their reader's id).
        return new FilterWeight(this, w) {
            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return false;
            }
        };
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(String field) {
        return late.toString(field);
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && late.equals(((DeferredJoinQuery) o).late)
            && Objects.equals(from, ((DeferredJoinQuery) o).from)
            && Objects.equals(to, ((DeferredJoinQuery) o).to);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), late, from, to);
    }
}
