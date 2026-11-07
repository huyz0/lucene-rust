/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * A KNN search run segment by segment (the Rust engine's {@code KnnLeafQuery}, node 39): each
 * segment's top {@code k} under the filter, its graph walked with a beam of {@code max(k,
 * ef_search)} -- the {@code faiss} engine's {@code KNNQuery}, which returns {@code k} hits per
 * segment -- or, radially, every document scoring at least {@link #minScore}. The Java weight runs
 * the same searches through Lucene's vector readers.
 */
public final class KnnLeafVectorQuery extends Query {
    private final String field;
    private final KnnTarget target;
    /** The hits per segment; 0 for a radial search. */
    private final int k;
    private final int efSearch;
    private final float minScore;
    private final Query filter;

    private KnnLeafVectorQuery(String field, KnnTarget target, int k, int efSearch, float minScore, Query filter) {
        this.field = Objects.requireNonNull(field);
        this.target = Objects.requireNonNull(target);
        this.k = k;
        this.efSearch = efSearch;
        this.minScore = minScore;
        this.filter = filter;
    }

    public static KnnLeafVectorQuery topK(String field, KnnTarget target, int k, int efSearch, Query filter) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        return new KnnLeafVectorQuery(field, target, k, efSearch, Float.NaN, filter);
    }

    public static KnnLeafVectorQuery radial(String field, KnnTarget target, float minScore, Query filter) {
        return new KnnLeafVectorQuery(field, target, 0, 0, minScore, filter);
    }

    public String field() {
        return field;
    }

    public KnnTarget target() {
        return target;
    }

    public boolean isRadial() {
        return k == 0;
    }

    public int k() {
        return k;
    }

    public int efSearch() {
        return efSearch;
    }

    public float minScore() {
        return minScore;
    }

    public Query filter() {
        return filter;
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        if (filter == null) {
            return this;
        }
        Query f = searcher.rewrite(filter);
        return f == filter ? this : new KnnLeafVectorQuery(field, target, k, efSearch, minScore, f);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        if (isRadial()) {
            Query q = target.floats() != null
                ? new FloatVectorSimilarityQuery(field, target.floats(), minScore, filter)
                : new ByteVectorSimilarityQuery(field, target.bytes(), minScore, filter);
            return searcher.createWeight(searcher.rewrite(q), scoreMode, boost);
        }
        Weight filterWeight = filter == null ? null : searcher.createWeight(searcher.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1);
        int beam = Math.max(k, efSearch);
        return KnnHits.weight(this, ctx -> {
            LeafReader reader = ctx.reader();
            Bits live = reader.getLiveDocs();
            Bits accept = live;
            if (filterWeight != null) {
                Scorer s = filterWeight.scorer(ctx);
                if (s == null) {
                    return KnnHits.Leaf.EMPTY;
                }
                FixedBitSet bits = new FixedBitSet(reader.maxDoc());
                DocIdSetIterator it = s.iterator();
                for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                    if (live == null || live.get(d)) {
                        bits.set(d);
                    }
                }
                accept = bits;
            }
            AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(accept, reader.maxDoc());
            TopDocs top = target.floats() != null
                ? reader.searchNearestVectors(field, target.floats(), beam, acceptDocs, Integer.MAX_VALUE)
                : reader.searchNearestVectors(field, target.bytes(), beam, acceptDocs, Integer.MAX_VALUE);
            ScoreDoc[] hits = top.scoreDocs.length > k ? Arrays.copyOf(top.scoreDocs, k) : top.scoreDocs;
            Arrays.sort(hits, (a, b) -> Integer.compare(a.doc, b.doc));
            int[] docs = new int[hits.length];
            float[] scores = new float[hits.length];
            for (int i = 0; i < hits.length; i++) {
                docs[i] = hits[i].doc;
                scores[i] = hits[i].score;
            }
            return new KnnHits.Leaf(docs, scores);
        }, boost);
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
        return isRadial()
            ? "KnnLeafVectorQuery(" + field + ", min_score=" + minScore + ")"
            : "KnnLeafVectorQuery(" + field + ", k=" + k + ", ef_search=" + efSearch + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && o instanceof KnnLeafVectorQuery q
            && field.equals(q.field)
            && target.equals(q.target)
            && k == q.k
            && efSearch == q.efSearch
            && Float.compare(minScore, q.minScore) == 0
            && Objects.equals(filter, q.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, target, k, efSearch, minScore, filter);
    }
}
