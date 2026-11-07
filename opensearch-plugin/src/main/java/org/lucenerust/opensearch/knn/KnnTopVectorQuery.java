/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;

import java.io.IOException;
import java.util.Objects;

/**
 * The shard's top {@code k} documents (the {@code lucene} engine's {@code KnnFloatVectorQuery},
 * searching {@code candidates = max(k, ef_search)} and keeping {@code k}, as the stock plugin's
 * {@code OSKnnFloatVectorQuery} does). Like Lucene's, it is a rewrite: into the {@link KnnHitsQuery}
 * of its hits, found by the Rust engine over the reader's native segments ({@link #setNative}) or,
 * when the reader has none, by Lucene.
 */
public final class KnnTopVectorQuery extends Query {
    /** Searches a query natively over a searcher's reader: its hits, or null when the reader is not native. */
    public interface NativeSearch {
        KnnHitsQuery search(IndexSearcher searcher, KnnTopVectorQuery query) throws IOException;
    }

    private static volatile NativeSearch nativeSearch;

    /** Installs the Rust engine's search (by the plugin, once). */
    public static void setNative(NativeSearch search) {
        nativeSearch = search;
    }

    private final String field;
    private final KnnTarget target;
    private final int k;
    private final int candidates;
    private final Query filter;

    public KnnTopVectorQuery(String field, KnnTarget target, int k, int candidates, Query filter) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        this.field = Objects.requireNonNull(field);
        this.target = Objects.requireNonNull(target);
        this.k = k;
        this.candidates = Math.max(k, candidates);
        this.filter = filter;
    }

    public String field() {
        return field;
    }

    public KnnTarget target() {
        return target;
    }

    public int k() {
        return k;
    }

    public int candidates() {
        return candidates;
    }

    public Query filter() {
        return filter;
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        KnnTopVectorQuery q = this;
        if (filter != null) {
            Query f = searcher.rewrite(filter);
            if (f != filter) {
                q = new KnnTopVectorQuery(field, target, k, candidates, f);
            }
        }
        NativeSearch n = nativeSearch;
        if (n != null) {
            KnnHitsQuery hits = n.search(searcher, q);
            if (hits != null) {
                return hits;
            }
        }
        Query lucene = target.floats() != null
            ? new KnnFloatVectorQuery(field, target.floats(), candidates, q.filter)
            : new KnnByteVectorQuery(field, target.bytes(), candidates, q.filter);
        TopDocs top = searcher.search(lucene, k);
        return KnnHitsQuery.of(searcher.getIndexReader(), top.scoreDocs);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        throw new IllegalStateException("[knn] must be rewritten before it is searched");
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
        return "KnnTopVectorQuery(" + field + ", k=" + k + ", candidates=" + candidates + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && o instanceof KnnTopVectorQuery q
            && field.equals(q.field)
            && target.equals(q.target)
            && k == q.k
            && candidates == q.candidates
            && Objects.equals(filter, q.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, target, k, candidates, filter);
    }
}
