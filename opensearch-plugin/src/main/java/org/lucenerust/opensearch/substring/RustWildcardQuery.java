/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.substring;

import java.io.IOException;
import java.util.Objects;
import java.util.function.Predicate;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.ConstantScoreWeight;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BytesRef;

/**
 * A query on a {@code compact_wildcard} field: the documents whose value passes {@link #matcher}, the
 * test OpenSearch's {@code wildcard} field runs on each candidate (its {@code WildcardMatchingQuery}'s
 * second phase), constant-scored. The native engine answers it from the field's {@code Substring90}
 * blocks and trigram index ({@code NODE_SUBSTRING}); Lucene, here, tests every value.
 *
 * <p>{@link #kind} and {@link #pattern} are what the native side runs ({@code SubstringQuery}); a
 * query that has none of them ({@link #nativePattern} false) runs on Lucene only.
 */
public final class RustWildcardQuery extends Query {
    /** How {@link #pattern} reads: {@code SubstringKind} in {@code extended_query.rs}. */
    public enum Kind {
        WILDCARD,
        REGEXP,
        CONTAINS,
        EXACT
    }

    private final String field;
    private final Kind kind;
    private final String pattern;
    private final boolean caseInsensitive;
    private final boolean nativePattern;
    /** A regexp's syntax and match flags, which {@link #matcher} was built with; 0 for the other kinds. */
    private final long flags;
    private final Predicate<String> matcher;

    RustWildcardQuery(
        String field,
        Kind kind,
        String pattern,
        boolean caseInsensitive,
        boolean nativePattern,
        long flags,
        Predicate<String> matcher
    ) {
        this.field = Objects.requireNonNull(field);
        this.kind = Objects.requireNonNull(kind);
        this.pattern = Objects.requireNonNull(pattern);
        this.caseInsensitive = caseInsensitive;
        this.nativePattern = nativePattern;
        this.flags = flags;
        this.matcher = Objects.requireNonNull(matcher);
    }

    public String field() {
        return field;
    }

    public Kind kind() {
        return kind;
    }

    public String pattern() {
        return pattern;
    }

    public boolean caseInsensitive() {
        return caseInsensitive;
    }

    /** Whether the native engine's {@code SubstringQuery} of {@link #kind} and {@link #pattern} matches what {@link #matcher} does. */
    public boolean nativePattern() {
        return nativePattern;
    }

    /** Whether {@code value} matches. */
    public boolean matches(String value) {
        return matcher.test(value);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        return new ConstantScoreWeight(this, boost) {
            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                if (context.reader().getFieldInfos().fieldInfo(field) == null) {
                    return null;
                }
                BinaryDocValues values = DocValues.getBinary(context.reader(), field);
                TwoPhaseIterator twoPhase = new TwoPhaseIterator(values) {
                    @Override
                    public boolean matches() throws IOException {
                        BytesRef v = values.binaryValue();
                        return matcher.test(v.utf8ToString());
                    }

                    @Override
                    public float matchCost() {
                        return 1000f;
                    }
                };
                float score = score();
                return new ScorerSupplier() {
                    @Override
                    public org.apache.lucene.search.Scorer get(long leadCost) {
                        return new ConstantScoreScorer(score, scoreMode, twoPhase);
                    }

                    @Override
                    public long cost() {
                        return values.cost();
                    }
                };
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return DocValues.isCacheable(ctx, field);
            }
        };
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public String toString(String f) {
        return "RustWildcardQuery(" + field + ":" + kind.name().toLowerCase(java.util.Locale.ROOT) + " \"" + pattern + "\"" + (caseInsensitive ? ", case_insensitive" : "") + ")";
    }

    @Override
    public boolean equals(Object o) {
        return sameClassAs(o)
            && field.equals(((RustWildcardQuery) o).field)
            && kind == ((RustWildcardQuery) o).kind
            && pattern.equals(((RustWildcardQuery) o).pattern)
            && caseInsensitive == ((RustWildcardQuery) o).caseInsensitive
            && nativePattern == ((RustWildcardQuery) o).nativePattern
            && flags == ((RustWildcardQuery) o).flags;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, kind, pattern, caseInsensitive, nativePattern, flags);
    }
}
