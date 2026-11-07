/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;

import java.io.IOException;

/** A weight over hits found up front, one segment's ({@code docs} local and ascending) at a time. */
final class KnnHits {
    /** A segment's hits: local doc ids, ascending, and their scores. */
    record Leaf(int[] docs, float[] scores) {
        static final Leaf EMPTY = new Leaf(new int[0], new float[0]);
    }

    interface Source {
        Leaf hits(LeafReaderContext context) throws IOException;
    }

    private KnnHits() {}

    static Weight weight(Query query, Source source, float boost) {
        return new Weight(query) {
            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                Leaf leaf = source.hits(context);
                if (leaf.docs.length == 0) {
                    return null;
                }
                Scorer scorer = new HitsScorer(leaf, boost);
                return new ScorerSupplier() {
                    @Override
                    public Scorer get(long leadCost) {
                        return scorer;
                    }

                    @Override
                    public long cost() {
                        return leaf.docs.length;
                    }
                };
            }

            @Override
            public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                Leaf leaf = source.hits(context);
                int i = java.util.Arrays.binarySearch(leaf.docs, doc);
                if (i < 0) {
                    return Explanation.noMatch("not in the top hits");
                }
                return Explanation.match(leaf.scores[i] * boost, "within top hits");
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return true;
            }
        };
    }

    private static final class HitsScorer extends Scorer {
        private final Leaf leaf;
        private final float boost;
        private int i = -1;
        private final DocIdSetIterator it = new DocIdSetIterator() {
            @Override
            public int docID() {
                return i < 0 ? -1 : i >= leaf.docs.length ? NO_MORE_DOCS : leaf.docs[i];
            }

            @Override
            public int nextDoc() {
                i++;
                return docID();
            }

            @Override
            public int advance(int target) {
                if (i >= leaf.docs.length) {
                    return NO_MORE_DOCS;
                }
                int at = java.util.Arrays.binarySearch(leaf.docs, Math.max(i + 1, 0), leaf.docs.length, target);
                i = at >= 0 ? at : -at - 1;
                return docID();
            }

            @Override
            public long cost() {
                return leaf.docs.length;
            }
        };

        HitsScorer(Leaf leaf, float boost) {
            this.leaf = leaf;
            this.boost = boost;
        }

        @Override
        public int docID() {
            return it.docID();
        }

        @Override
        public DocIdSetIterator iterator() {
            return it;
        }

        @Override
        public float getMaxScore(int upTo) {
            float max = 0;
            for (float s : leaf.scores) {
                max = Math.max(max, s);
            }
            return max * boost;
        }

        @Override
        public float score() {
            return leaf.scores[i] * boost;
        }
    }
}
