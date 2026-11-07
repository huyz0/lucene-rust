/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Lucene's {@code DocAndScoreQuery}: a whole-reader KNN search's hits, global doc ids with their
 * scores, matched as they are. Built for one reader ({@link #contextIdentity}) and only valid on it.
 */
public final class KnnHitsQuery extends Query {
    private final int[] docs;
    private final float[] scores;
    private final int[] docBases;
    private final Object contextIdentity;

    private KnnHitsQuery(int[] docs, float[] scores, int[] docBases, Object contextIdentity) {
        this.docs = docs;
        this.scores = scores;
        this.docBases = docBases;
        this.contextIdentity = contextIdentity;
    }

    /** The hits, in any order, over {@code reader}. */
    public static KnnHitsQuery of(IndexReader reader, ScoreDoc[] hits) {
        ScoreDoc[] sorted = hits.clone();
        Arrays.sort(sorted, (a, b) -> Integer.compare(a.doc, b.doc));
        int[] docs = new int[sorted.length];
        float[] scores = new float[sorted.length];
        for (int i = 0; i < sorted.length; i++) {
            docs[i] = sorted[i].doc;
            scores[i] = sorted[i].score;
        }
        List<LeafReaderContext> leaves = reader.leaves();
        int[] bases = new int[leaves.size()];
        for (int i = 0; i < bases.length; i++) {
            bases[i] = leaves.get(i).docBase;
        }
        return new KnnHitsQuery(docs, scores, bases, reader.getContext().id());
    }

    /** Global doc ids, ascending. */
    public int[] docs() {
        return docs;
    }

    public float[] scores() {
        return scores;
    }

    /** The reader's segment doc bases, ascending. */
    public int[] docBases() {
        return docBases;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        if (searcher.getIndexReader().getContext().id() != contextIdentity) {
            throw new IllegalStateException("This KNN result was built for a different reader");
        }
        return KnnHits.weight(this, ctx -> {
            int lo = lowerBound(ctx.docBase);
            int hi = lowerBound(ctx.docBase + ctx.reader().maxDoc());
            if (lo == hi) {
                return KnnHits.Leaf.EMPTY;
            }
            int[] local = new int[hi - lo];
            for (int i = lo; i < hi; i++) {
                local[i - lo] = docs[i] - ctx.docBase;
            }
            return new KnnHits.Leaf(local, Arrays.copyOfRange(scores, lo, hi));
        }, boost);
    }

    private int lowerBound(int doc) {
        int i = Arrays.binarySearch(docs, doc);
        if (i < 0) {
            return -i - 1;
        }
        while (i > 0 && docs[i - 1] == doc) {
            i--;
        }
        return i;
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(String field) {
        return "KnnHits(" + docs.length + " docs)";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && o instanceof KnnHitsQuery q
            && contextIdentity == q.contextIdentity
            && Arrays.equals(docs, q.docs)
            && Arrays.equals(scores, q.scores);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), contextIdentity, Arrays.hashCode(docs), Arrays.hashCode(scores));
    }
}
