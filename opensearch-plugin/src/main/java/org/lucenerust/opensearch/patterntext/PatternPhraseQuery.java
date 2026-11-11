/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.patterntext;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.ConstantScoreWeight;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A phrase over a {@code pattern_text} field: the documents of {@code approximation} (its terms,
 * filter-only) whose message holds {@code positional}, each scoring 0 as a {@code match_only_text}
 * phrase does.
 *
 * <p>A candidate is decided once per template ordinal and segment where the template settles it:
 * a template without variables by itself, and (for an exact phrase, when {@code templateParts})
 * one whose static text between two variables holds the phrase matches outright. The rest rebuild
 * the message from the template and the stored variables and run {@code positional} over it in a
 * {@link MemoryIndex} analyzed by the field's index analyzer, as OpenSearch's {@code
 * SourceFieldMatchQuery} runs it over {@code _source}.
 *
 * <p>An exact phrase ({@link #positions()} not null) is encoded natively as node 47 of the query
 * tree ({@code crates/lucene-ffi/src/jvm_pattern_phrase.rs}), which matches and scores the same.
 */
public final class PatternPhraseQuery extends Query {
    private final String field;
    private final Query approximation;
    private final Query positional;
    private final Analyzer analyzer;
    private final int[] positions;
    private final BytesRef[][] terms;
    private final boolean templateParts;

    /**
     * @param positions the exact phrase's positions, or null for a phrase with slop or a prefix
     * @param terms each position's terms (any of them matches), or null with {@code positions}
     */
    public PatternPhraseQuery(
        String field,
        Query approximation,
        Query positional,
        Analyzer analyzer,
        int[] positions,
        BytesRef[][] terms,
        boolean templateParts
    ) {
        if ((positions == null) != (terms == null) || (positions != null && positions.length != terms.length)) {
            throw new IllegalArgumentException("positions and terms must come together");
        }
        this.field = Objects.requireNonNull(field);
        this.approximation = Objects.requireNonNull(approximation);
        this.positional = Objects.requireNonNull(positional);
        this.analyzer = Objects.requireNonNull(analyzer);
        this.positions = positions;
        this.terms = terms;
        this.templateParts = templateParts;
    }

    /**
     * {@code positional}, a {@link PhraseQuery} or {@link MultiPhraseQuery} on {@code field}, as a
     * {@link PatternPhraseQuery} (exact when {@code exact}, so encodable natively); any other query
     * as it is.
     */
    public static Query of(String field, Query positional, Analyzer analyzer, boolean exact, boolean templateParts) {
        int[] positions;
        Term[][] terms;
        if (positional instanceof PhraseQuery pq) {
            positions = pq.getPositions();
            Term[] t = pq.getTerms();
            terms = new Term[t.length][];
            for (int i = 0; i < t.length; i++) {
                terms[i] = new Term[] { t[i] };
            }
        } else if (positional instanceof MultiPhraseQuery mq) {
            positions = mq.getPositions();
            terms = mq.getTermArrays();
        } else {
            return positional;
        }
        if (terms.length == 0) {
            return positional;
        }
        BooleanQuery.Builder approximation = new BooleanQuery.Builder();
        BytesRef[][] bytes = new BytesRef[terms.length][];
        for (int i = 0; i < terms.length; i++) {
            bytes[i] = new BytesRef[terms[i].length];
            if (terms[i].length == 1) {
                approximation.add(new TermQuery(terms[i][0]), Occur.FILTER);
            } else {
                BooleanQuery.Builder any = new BooleanQuery.Builder();
                for (Term t : terms[i]) {
                    any.add(new TermQuery(t), Occur.SHOULD);
                }
                approximation.add(any.build(), Occur.FILTER);
            }
            for (int k = 0; k < terms[i].length; k++) {
                bytes[i][k] = terms[i][k].bytes();
            }
        }
        return new PatternPhraseQuery(
            field,
            approximation.build(),
            positional,
            analyzer,
            exact ? positions : null,
            exact ? bytes : null,
            templateParts
        );
    }

    public String field() {
        return field;
    }

    public Analyzer analyzer() {
        return analyzer;
    }

    /** The exact phrase's positions, or null when it has slop or a prefix. */
    public int[] positions() {
        return positions;
    }

    /** Each position's terms, or null with {@link #positions()}. */
    public BytesRef[][] terms() {
        return terms;
    }

    public boolean templateParts() {
        return templateParts;
    }

    public Query positional() {
        return positional;
    }

    private boolean onePosition() {
        return positions != null && Arrays.stream(positions).distinct().count() <= 1;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Weight approx = searcher.createWeight(searcher.rewrite(approximation), ScoreMode.COMPLETE_NO_SCORES, 1f);
        String templateField = field + PatternTextFieldMapper.TEMPLATE_SUFFIX;
        String argsField = field + PatternTextFieldMapper.ARGS_SUFFIX;
        boolean check = onePosition() == false;
        return new ConstantScoreWeight(this, 0f) {
            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext ctx) throws IOException {
                ScorerSupplier inner = approx.scorerSupplier(ctx);
                if (inner == null) {
                    return null;
                }
                float score = score();
                return new ScorerSupplier() {
                    @Override
                    public Scorer get(long leadCost) throws IOException {
                        DocIdSetIterator it = inner.get(leadCost).iterator();
                        if (check == false) {
                            return new ConstantScoreScorer(score, scoreMode, it);
                        }
                        return new ConstantScoreScorer(score, scoreMode, new Verifier(it, ctx, templateField, argsField));
                    }

                    @Override
                    public long cost() {
                        return inner.cost();
                    }
                };
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return DocValues.isCacheable(ctx, templateField) && approx.isCacheable(ctx);
            }
        };
    }

    /** The second phase: whether a candidate's message holds the phrase. */
    private final class Verifier extends TwoPhaseIterator {
        private final SortedDocValues templates;
        private final StoredFields stored;
        private final String argsField;
        private final Map<Integer, Object> verdicts = new HashMap<>();
        private final MemoryIndex memory = new MemoryIndex();
        private byte[] args;

        Verifier(DocIdSetIterator approximation, LeafReaderContext ctx, String templateField, String argsField) throws IOException {
            super(approximation);
            this.templates = DocValues.getSorted(ctx.reader(), templateField);
            this.stored = ctx.reader().storedFields();
            this.argsField = argsField;
        }

        @Override
        public boolean matches() throws IOException {
            int doc = approximation.docID();
            if (templates.advanceExact(doc) == false) {
                return false;
            }
            int ord = templates.ordValue();
            Object verdict = verdicts.get(ord);
            if (verdict == null) {
                verdict = verdict(templates.lookupOrd(ord).utf8ToString());
                verdicts.put(ord, verdict);
            }
            if (verdict instanceof Boolean b) {
                return b;
            }
            args = null;
            stored.document(doc, new StoredFieldVisitor() {
                @Override
                public Status needsField(FieldInfo fieldInfo) {
                    if (args != null) {
                        return Status.STOP;
                    }
                    return fieldInfo.name.equals(argsField) ? Status.YES : Status.NO;
                }

                @Override
                public void binaryField(FieldInfo fieldInfo, byte[] value) {
                    args = value;
                }
            });
            List<String> values = args == null ? List.of() : PatternTextFormat.decodeArgs(args, 0, args.length);
            return holds(PatternTextFormat.join((String) verdict, values));
        }

        /** {@code Boolean} when {@code template} settles its documents, else the template itself. */
        private Object verdict(String template) {
            if (template.indexOf(PatternTextFormat.VARIABLE) < 0) {
                return holds(template);
            }
            if (templateParts && positions != null) {
                for (String part : template.split(String.valueOf(PatternTextFormat.VARIABLE), -1)) {
                    if (holds(part)) {
                        return Boolean.TRUE;
                    }
                }
            }
            return template;
        }

        private boolean holds(String text) {
            memory.reset();
            memory.addField(field, text, analyzer);
            try {
                return memory.createSearcher().count(positional) > 0;
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public float matchCost() {
            return 100f;
        }
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            approximation.visit(visitor.getSubVisitor(BooleanClause.Occur.FILTER, this));
        }
    }

    @Override
    public String toString(String f) {
        return "pattern_phrase(" + positional.toString(f) + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (sameClassAs(o) == false) {
            return false;
        }
        PatternPhraseQuery q = (PatternPhraseQuery) o;
        return field.equals(q.field)
            && approximation.equals(q.approximation)
            && positional.equals(q.positional)
            && analyzer == q.analyzer
            && templateParts == q.templateParts;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, approximation, positional, System.identityHashCode(analyzer), templateParts);
    }
}
