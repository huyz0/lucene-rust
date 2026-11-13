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
 * whose field has an IVF index ({@code IvfVectorsFormat}), {@code 0} for the depth each
 * segment calibrated, nested searches included; other formats ignore it.
 *
 * <p>With a {@link Grouping} ({@code group_by}) it matches the best {@link Grouping#size} documents of
 * each of the shard's best {@code k} groups of a keyword or numeric field: the Rust engine's walks
 * collect by group; Lucene, for a reader the engine does not hold, searches ever more candidates
 * until they cover {@code k} groups.
 */
public final class KnnTopVectorQuery extends Query {
    /** Searches a query natively over a searcher's reader: its hits, or null when the reader is not native. */
    public interface NativeSearch {
        KnnHitsQuery search(IndexSearcher searcher, KnnTopVectorQuery query) throws IOException;

        /**
         * Before Lucene answers over a realtime view: gives Lucene the view's buffered documents,
         * so it finds their vectors too. Throws when they cannot be read; does nothing over a
         * reader that is no view.
         */
        default void readBuffersByLucene(IndexSearcher searcher) throws IOException {}
    }

    /**
     * What a grouped search groups by: the concrete field (an alias resolved), whether it is a keyword
     * (else a number), and the documents kept per group.
     */
    public record Grouping(String field, boolean keyword, int size) {
        public Grouping {
            Objects.requireNonNull(field);
            if (size < 1) {
                throw new IllegalArgumentException("group size must be at least 1");
            }
        }
    }

    private static volatile NativeSearch nativeSearch;

    /** The group of the documents without a value, as collapse groups them. */
    private static final Object MISSING = new Object();

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
    private final Grouping grouping;
    /** The graph walk's adaptive beam stop over the best {@code k}, or null for Lucene's stop. */
    private final Float explorationSlack;
    private final RecallTarget recall;

    /**
     * A recall-targeted search ({@code index.knn.target_recall}): the target, the
     * sample rate and check budget, and the key naming the tuner (index, shard and field) whose
     * tuned depth the search runs at.
     */
    public record RecallTarget(float recall, float sampleRate, float budget, String key) {
        public RecallTarget {
            Objects.requireNonNull(key);
        }
    }

    public KnnTopVectorQuery(String field, KnnTarget target, int k, int candidates, Query filter) {
        this(field, target, k, candidates, filter, null, false);
    }

    public KnnTopVectorQuery(String field, KnnTarget target, int k, int candidates, Query filter, BitSetProducer parents, boolean expand) {
        this(field, target, k, candidates, filter, parents, expand, 0, null, null, null);
    }

    private KnnTopVectorQuery(
        String field,
        KnnTarget target,
        int k,
        int candidates,
        Query filter,
        BitSetProducer parents,
        boolean expand,
        int nprobes,
        Float explorationSlack,
        RecallTarget recall,
        Grouping grouping
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
        if (explorationSlack != null && (Float.isFinite(explorationSlack) == false || explorationSlack < 0)) {
            throw new IllegalArgumentException("exploration_slack must be a finite number >= 0");
        }
        this.explorationSlack = explorationSlack;
        this.recall = recall;
        if (grouping != null && parents != null) {
            throw new IllegalArgumentException("a grouped search cannot be nested");
        }
        this.grouping = grouping;
    }

    /** This query probing {@code nprobes} IVF clusters per segment. */
    public KnnTopVectorQuery withNprobes(int nprobes) {
        return new KnnTopVectorQuery(field, target, k, candidates, filter, parents, expand, nprobes, explorationSlack, recall, grouping);
    }

    /** This query searching at the depth tuned for {@code recall} (null: as asked). */
    public KnnTopVectorQuery withRecallTarget(RecallTarget recall) {
        return new KnnTopVectorQuery(field, target, k, candidates, filter, parents, expand, nprobes, explorationSlack, recall, grouping);
    }

    /** This query grouped by {@code grouping}: {@code k} counts groups. */
    public KnnTopVectorQuery withGrouping(Grouping grouping) {
        return new KnnTopVectorQuery(field, target, k, candidates, filter, parents, expand, nprobes, explorationSlack, recall, Objects.requireNonNull(grouping));
    }

    /** The grouping, or null for a search that is not grouped. */
    public Grouping grouping() {
        return grouping;
    }

    public int nprobes() {
        return nprobes;
    }

    /**
     * This query with the graph walk's opt-in adaptive beam stop ({@code method_parameters.exploration_slack}):
     * a walk stops once its best candidate is farther than {@code 1 + slack} times the {@code k}-th best
     * found. A nested search and the {@code lucene} engine's Java fallback ignore it.
     */
    public KnnTopVectorQuery withExplorationSlack(float slack) {
        return new KnnTopVectorQuery(field, target, k, candidates, filter, parents, expand, nprobes, slack, recall, grouping);
    }

    /** The adaptive beam stop's slack, or null. */
    public Float explorationSlack() {
        return explorationSlack;
    }

    /** The recall target, or null. */
    public RecallTarget recallTarget() {
        return recall;
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
                q = new KnnTopVectorQuery(field, target, k, candidates, f, parents, expand, nprobes, explorationSlack, recall, grouping);
            }
        }
        NativeSearch n = nativeSearch;
        if (n != null) {
            // A later phase of the same request (the fetch phase, a scroll page) rewrites the query
            // again: the hits its first rewrite found.
            KnnHitsQuery hits = KnnHitsMemo.find(searcher, q);
            if (hits != null) {
                return hits;
            }
            hits = n.search(searcher, q);
            if (hits != null) {
                KnnHitsMemo.keep(searcher, q, hits);
                return hits;
            }
        }
        if (n != null) {
            n.readBuffersByLucene(searcher);
        }
        if (grouping != null) {
            return KnnHitsQuery.of(searcher.getIndexReader(), groupedByLucene(searcher, q.filter));
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
     * The grouped answer by Lucene: the top {@code n} documents for ever larger {@code n} (from {@code
     * candidates}, four times more each round) until they cover {@code k} groups, the reader's every
     * document, or a round brings no new group, then grouped.
     */
    private ScoreDoc[] groupedByLucene(IndexSearcher searcher, Query filter) throws IOException {
        int maxDoc = Math.max(1, searcher.getIndexReader().maxDoc());
        int n = Math.min(maxDoc, (int) Math.min(Integer.MAX_VALUE, Math.max((long) candidates, (long) k * grouping.size())));
        int before = -1;
        while (true) {
            Query lucene = target.floats() != null
                ? new KnnFloatVectorQuery(field, target.floats(), n, filter)
                : new KnnByteVectorQuery(field, target.bytes(), n, filter);
            ScoreDoc[] top = searcher.search(lucene, n).scoreDocs;
            List<ScoreDoc> grouped = new ArrayList<>();
            int groups = group(searcher, top, grouped);
            if (groups >= k || groups == before || n >= maxDoc || top.length < n) {
                return grouped.toArray(ScoreDoc[]::new);
            }
            before = groups;
            n = (int) Math.min(maxDoc, (long) n * 4);
        }
    }

    /**
     * Groups {@code hits} (best first) by the group field: the best {@link Grouping#size} of each of
     * the first {@code k} groups go to {@code out}, best first. Returns how many groups the hits hold.
     */
    private int group(IndexSearcher searcher, ScoreDoc[] hits, List<ScoreDoc> out) throws IOException {
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        Object[] keys = new Object[hits.length];
        Integer[] byDoc = new Integer[hits.length];
        for (int i = 0; i < hits.length; i++) {
            byDoc[i] = i;
        }
        java.util.Arrays.sort(byDoc, Comparator.comparingInt(i -> hits[i].doc));
        int leafAt = -1;
        org.apache.lucene.index.SortedSetDocValues terms = null;
        org.apache.lucene.index.SortedNumericDocValues numbers = null;
        LeafReaderContext leaf = null;
        for (int i : byDoc) {
            int at = ReaderUtil.subIndex(hits[i].doc, leaves);
            if (at != leafAt) {
                leafAt = at;
                leaf = leaves.get(at);
                if (grouping.keyword()) {
                    terms = org.apache.lucene.index.DocValues.getSortedSet(leaf.reader(), grouping.field());
                } else {
                    numbers = org.apache.lucene.index.DocValues.getSortedNumeric(leaf.reader(), grouping.field());
                }
            }
            int local = hits[i].doc - leaf.docBase;
            Object key = null;
            if (grouping.keyword()) {
                if (terms.advanceExact(local)) {
                    if (terms.docValueCount() > 1) {
                        throw new IllegalStateException("failed to group " + local + ", the group_by field must be single valued");
                    }
                    key = org.apache.lucene.util.BytesRef.deepCopyOf(terms.lookupOrd(terms.nextOrd()));
                }
            } else if (numbers.advanceExact(local)) {
                if (numbers.docValueCount() > 1) {
                    throw new IllegalStateException("failed to group " + local + ", the group_by field must be single valued");
                }
                key = numbers.nextValue();
            }
            keys[i] = key;
        }
        java.util.Map<Object, Integer> kept = new java.util.HashMap<>();
        java.util.Set<Object> seen = new java.util.HashSet<>();
        for (int i = 0; i < hits.length; i++) {
            Object key = keys[i] == null ? MISSING : keys[i];
            seen.add(key);
            Integer held = kept.get(key);
            if (held == null && kept.size() >= k) {
                continue;
            }
            int count = held == null ? 0 : held;
            if (count < grouping.size()) {
                kept.put(key, count + 1);
                out.add(hits[i]);
            }
        }
        return seen.size();
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
        return "KnnTopVectorQuery(" + field + ", k=" + k + ", candidates=" + candidates + (parents == null ? "" : ", nested") + (expand ? ", expand" : "") + (nprobes == 0 ? "" : ", nprobes=" + nprobes) + (explorationSlack == null ? "" : ", exploration_slack=" + explorationSlack) + (recall == null ? "" : ", target_recall=" + recall.recall()) + (grouping == null ? "" : ", group_by=" + grouping.field() + "/" + grouping.size()) + ")";
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
            && nprobes == q.nprobes
            && Objects.equals(explorationSlack, q.explorationSlack)
            && Objects.equals(recall, q.recall)
            && Objects.equals(grouping, q.grouping);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, target, k, candidates, filter, parents, expand, nprobes, explorationSlack, recall, grouping);
    }
}
