/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * neural-search's {@code SparseVectorQuery}: the approximate {@code neural_sparse} search of a
 * {@code sparse_vector} field. The Rust engine runs it ({@code lucene_sparse::SparseAnnQuery}): the
 * clustered postings of each segment at or over the field's threshold, under {@link #filter} AND
 * the field exists, and {@link #fallback} on every other segment. It has no Java weight.
 */
public final class SparseVectorQuery extends Query {
    final String field;
    /** The query tokens, folded duplicates kept at their highest weight. */
    final Map<Integer, Float> tokens;
    /** {@code top_n}: the heaviest tokens whose postings are walked. */
    final int topN;
    /** {@code k}: the results kept per segment. */
    final int k;
    final float heapFactor;
    final Query filter;
    /** The sum of linear {@code FeatureField} queries, plus the filter, for small segments. */
    final Query fallback;

    public SparseVectorQuery(String field, Map<Integer, Float> tokens, int topN, int k, float heapFactor, Query filter, Query fallback) {
        this.field = Objects.requireNonNull(field);
        this.tokens = Map.copyOf(tokens);
        this.topN = topN;
        this.k = k;
        this.heapFactor = heapFactor;
        this.filter = filter;
        this.fallback = Objects.requireNonNull(fallback);
    }

    public String field() {
        return field;
    }

    public Map<Integer, Float> tokens() {
        return tokens;
    }

    public int topN() {
        return topN;
    }

    public int k() {
        return k;
    }

    public float heapFactor() {
        return heapFactor;
    }

    public Query filter() {
        return filter;
    }

    public Query fallback() {
        return fallback;
    }

    /** The filter and fallback rewritten, so what reaches the Rust engine is primitive queries. */
    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        Query f = filter == null ? null : searcher.rewrite(filter);
        Query fb = searcher.rewrite(fallback);
        if (f == filter && fb == fallback) {
            return this;
        }
        return new SparseVectorQuery(field, tokens, topN, k, heapFactor, f, fb);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        throw new IllegalStateException("[neural_sparse] on a [sparse_vector] field is searched by the Rust engine only");
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
        if (filter != null) {
            filter.visit(visitor.getSubVisitor(BooleanClause.Occur.FILTER, this));
        }
    }

    @Override
    public String toString(String defaultField) {
        return "neural_sparse(" + field + ", top_n=" + topN + ", k=" + k + ", heap_factor=" + heapFactor + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && o instanceof SparseVectorQuery q
            && field.equals(q.field)
            && tokens.equals(q.tokens)
            && topN == q.topN
            && k == q.k
            && Float.compare(heapFactor, q.heapFactor) == 0
            && Objects.equals(filter, q.filter)
            && fallback.equals(q.fallback);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, tokens, topN, k, heapFactor, filter, fallback);
    }
}
