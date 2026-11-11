/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.tensor;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * {@code sub}'s documents scored by the {@code MaxSim} of {@code query}'s tokens against the {@code
 * multi_vector} field {@code field} ({@link MultiVectorEncoding#maxSim}): Lucene's {@link
 * FunctionScoreQuery}, so a document without a value, or with a negative {@code MaxSim}, scores 0.
 * Encoded natively as node 44 of the query tree ({@code crates/lucene-ffi/src/jvm_max_sim.rs}),
 * which scores the same.
 */
public final class MaxSimQuery extends Query {
    private final Query sub;
    private final String field;
    private final float[][] query;

    public MaxSimQuery(Query sub, String field, float[][] query) {
        if (query.length == 0) {
            throw new IllegalArgumentException("queryVector must not be null or empty");
        }
        for (float[] t : query) {
            if (t.length == 0) {
                throw new IllegalArgumentException("composing token vectors in provided query vector should not be null or empty");
            }
            if (t.length != query[0].length) {
                throw new IllegalArgumentException("all composing token vectors in provided query vector should have the same length");
            }
        }
        this.sub = Objects.requireNonNull(sub);
        this.field = Objects.requireNonNull(field);
        this.query = query;
    }

    public Query sub() {
        return sub;
    }

    public String field() {
        return field;
    }

    /** The query's tokens; not to be modified. */
    public float[][] tokens() {
        return query;
    }

    private FunctionScoreQuery delegate(Query sub) {
        return new FunctionScoreQuery(sub, new Source(field, query));
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        Query rewritten = sub.rewrite(searcher);
        return rewritten == sub ? this : new MaxSimQuery(rewritten, field, query);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        return delegate(sub).createWeight(searcher, scoreMode, boost);
    }

    @Override
    public void visit(QueryVisitor visitor) {
        sub.visit(visitor.getSubVisitor(org.apache.lucene.search.BooleanClause.Occur.MUST, this));
    }

    @Override
    public String toString(String f) {
        return "max_sim(" + field + ", " + query.length + " tokens of " + query[0].length + ", " + sub.toString(f) + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && sub.equals(((MaxSimQuery) o).sub)
            && field.equals(((MaxSimQuery) o).field)
            && Arrays.deepEquals(query, ((MaxSimQuery) o).query);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), sub, field, Arrays.deepHashCode(query));
    }

    /** Each document's {@code MaxSim} from the field's binary doc values. */
    static final class Source extends DoubleValuesSource {
        private final String field;
        private final float[][] query;

        Source(String field, float[][] query) {
            this.field = field;
            this.query = query;
        }

        @Override
        public DoubleValues getValues(LeafReaderContext ctx, DoubleValues scores) throws IOException {
            BinaryDocValues values = DocValues.getBinary(ctx.reader(), field);
            return new DoubleValues() {
                @Override
                public double doubleValue() throws IOException {
                    BytesRef v = values.binaryValue();
                    return MultiVectorEncoding.maxSim(query, MultiVectorEncoding.decode(v.bytes, v.offset, v.length));
                }

                @Override
                public boolean advanceExact(int doc) throws IOException {
                    return values.advanceExact(doc);
                }
            };
        }

        @Override
        public boolean needsScores() {
            return false;
        }

        @Override
        public DoubleValuesSource rewrite(IndexSearcher searcher) {
            return this;
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return DocValues.isCacheable(ctx, field);
        }

        @Override
        public Explanation explain(LeafReaderContext ctx, int docId, Explanation scoreExplanation) throws IOException {
            DoubleValues v = getValues(ctx, null);
            if (v.advanceExact(docId)) {
                return Explanation.match((float) v.doubleValue(), "max_sim over field [" + field + "]");
            }
            return Explanation.noMatch("no value for field [" + field + "]");
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Source s && field.equals(s.field) && Arrays.deepEquals(query, s.query);
        }

        @Override
        public int hashCode() {
            return Objects.hash(field, Arrays.deepHashCode(query));
        }

        @Override
        public String toString() {
            return "max_sim(" + field + ")";
        }
    }
}
