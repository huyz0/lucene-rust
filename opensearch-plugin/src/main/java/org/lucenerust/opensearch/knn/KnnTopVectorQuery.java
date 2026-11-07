/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenByteKnnVectorQuery;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * The shard's top {@code k} documents (the {@code lucene} engine's {@code KnnFloatVectorQuery},
 * searching {@code candidates = max(k, ef_search)} and keeping {@code k}, as the stock plugin's
 * {@code OSKnnFloatVectorQuery} does). Like Lucene's, it is a rewrite: into the {@link KnnHitsQuery}
 * of its hits, found by the Rust engine over the reader's native segments ({@link #setNative}) or,
 * when the reader has none, by Lucene.
 *
 * <p>On a nested field ({@code parents} set) it is k-NN's {@code
 * OSDiversifyingChildrenFloatKnnVectorQuery}: the best child of each of the best {@code k} parents;
 * with {@code expand} ({@code expand_nested_docs}, k-NN's {@code ExpandNestedDocsQuery}) every child
 * of those parents that passes the filter and has a vector, each scored.
 *
 * <p>{@code nprobes} ({@code method_parameters.nprobes}) is the clusters probed in each segment
 * whose field has an IVF index ({@code LuceneRustIvfVectorsFormat}), {@code 0} for the depth each
 * segment calibrated; other formats, and a nested search (which scans an IVF field exactly), ignore
 * it.
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
    private final BitSetProducer parents;
    private final boolean expand;
    private final int nprobes;

    public KnnTopVectorQuery(String field, KnnTarget target, int k, int candidates, Query filter) {
        this(field, target, k, candidates, filter, null, false);
    }

    public KnnTopVectorQuery(String field, KnnTarget target, int k, int candidates, Query filter, BitSetProducer parents, boolean expand) {
        this(field, target, k, candidates, filter, parents, expand, 0);
    }

    private KnnTopVectorQuery(
        String field,
        KnnTarget target,
        int k,
        int candidates,
        Query filter,
        BitSetProducer parents,
        boolean expand,
        int nprobes
    ) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        this.field = Objects.requireNonNull(field);
        this.target = Objects.requireNonNull(target);
        this.k = k;
        this.candidates = Math.max(k, candidates);
        this.filter = filter;
        if (expand && parents == null) {
            throw new IllegalArgumentException("expand needs a parent filter");
        }
        this.parents = parents;
        this.expand = expand;
        if (nprobes < 0) {
            throw new IllegalArgumentException("nprobes must not be negative");
        }
        this.nprobes = nprobes;
    }

    /** This query probing {@code nprobes} IVF clusters per segment. */
    public KnnTopVectorQuery withNprobes(int nprobes) {
        return new KnnTopVectorQuery(field, target, k, candidates, filter, parents, expand, nprobes);
    }

    public int nprobes() {
        return nprobes;
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

    /** A nested field's parent filter, or null. */
    public BitSetProducer parents() {
        return parents;
    }

    public boolean expand() {
        return expand;
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        KnnTopVectorQuery q = this;
        if (filter != null) {
            Query f = searcher.rewrite(filter);
            if (f != filter) {
                q = new KnnTopVectorQuery(field, target, k, candidates, f, parents, expand, nprobes);
            }
        }
        NativeSearch n = nativeSearch;
        if (n != null) {
            KnnHitsQuery hits = n.search(searcher, q);
            if (hits != null) {
                return hits;
            }
        }
        Query lucene;
        if (parents == null) {
            lucene = target.floats() != null
                ? new KnnFloatVectorQuery(field, target.floats(), candidates, q.filter)
                : new KnnByteVectorQuery(field, target.bytes(), candidates, q.filter);
        } else {
            lucene = target.floats() != null
                ? new DiversifyingChildrenFloatKnnVectorQuery(field, target.floats(), q.filter, candidates, parents)
                : new DiversifyingChildrenByteKnnVectorQuery(field, target.bytes(), q.filter, candidates, parents);
        }
        TopDocs top = searcher.search(lucene, k);
        ScoreDoc[] hits = expand ? expanded(searcher, q.filter, top.scoreDocs) : top.scoreDocs;
        return KnnHitsQuery.of(searcher.getIndexReader(), hits);
    }

    /**
     * {@code ExpandNestedDocsQuery}: every child of the hits' parents that passes {@code filter} and
     * has a vector, exactly scored, best first.
     */
    private ScoreDoc[] expanded(IndexSearcher searcher, Query filter, ScoreDoc[] hits) throws IOException {
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        Weight filterWeight = filter == null ? null : searcher.createWeight(searcher.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        List<ScoreDoc> out = new ArrayList<>();
        ScoreDoc[] byDoc = hits.clone();
        java.util.Arrays.sort(byDoc, Comparator.comparingInt(h -> h.doc));
        int i = 0;
        while (i < byDoc.length) {
            LeafReaderContext leaf = leaves.get(ReaderUtil.subIndex(byDoc[i].doc, leaves));
            int end = leaf.docBase + leaf.reader().maxDoc();
            FixedBitSet siblings = new FixedBitSet(leaf.reader().maxDoc());
            BitSet parentBits = parents.getBitSet(leaf);
            for (; i < byDoc.length && byDoc[i].doc < end; i++) {
                int local = byDoc[i].doc - leaf.docBase;
                if (parentBits == null || local >= parentBits.length()) {
                    continue;
                }
                int parent = parentBits.nextSetBit(local);
                if (parent == DocIdSetIterator.NO_MORE_DOCS) {
                    continue;
                }
                int first = local == 0 ? 0 : parentBits.prevSetBit(local - 1) + 1;
                siblings.set(first, parent);
            }
            if (filterWeight != null) {
                Scorer scorer = filterWeight.scorer(leaf);
                FixedBitSet passing = new FixedBitSet(leaf.reader().maxDoc());
                if (scorer != null) {
                    passing.or(scorer.iterator());
                }
                siblings.and(passing);
            }
            score(leaf, siblings, out);
        }
        out.sort((a, b) -> a.score != b.score ? Float.compare(b.score, a.score) : Integer.compare(a.doc, b.doc));
        return out.toArray(ScoreDoc[]::new);
    }

    private void score(LeafReaderContext leaf, FixedBitSet docs, List<ScoreDoc> out) throws IOException {
        VectorScorer scorer;
        if (target.floats() != null) {
            FloatVectorValues values = leaf.reader().getFloatVectorValues(field);
            scorer = values == null ? null : values.scorer(target.floats());
        } else {
            ByteVectorValues values = leaf.reader().getByteVectorValues(field);
            scorer = values == null ? null : values.scorer(target.bytes());
        }
        if (scorer == null) {
            return;
        }
        Bits live = leaf.reader().getLiveDocs();
        DocIdSetIterator it = scorer.iterator();
        for (int d = docs.nextSetBit(0); d != DocIdSetIterator.NO_MORE_DOCS; d = d + 1 < docs.length() ? docs.nextSetBit(d + 1) : DocIdSetIterator.NO_MORE_DOCS) {
            if (it.docID() < d) {
                it.advance(d);
            }
            if (it.docID() == DocIdSetIterator.NO_MORE_DOCS) {
                return;
            }
            if (it.docID() == d && (live == null || live.get(d))) {
                out.add(new ScoreDoc(leaf.docBase + d, scorer.score()));
            }
        }
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
        return "KnnTopVectorQuery(" + field + ", k=" + k + ", candidates=" + candidates + (parents == null ? "" : ", nested") + (expand ? ", expand" : "") + (nprobes == 0 ? "" : ", nprobes=" + nprobes) + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && o instanceof KnnTopVectorQuery q
            && field.equals(q.field)
            && target.equals(q.target)
            && k == q.k
            && candidates == q.candidates
            && Objects.equals(filter, q.filter)
            && Objects.equals(parents, q.parents)
            && expand == q.expand
            && nprobes == q.nprobes;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, target, k, candidates, filter, parents, expand, nprobes);
    }
}
