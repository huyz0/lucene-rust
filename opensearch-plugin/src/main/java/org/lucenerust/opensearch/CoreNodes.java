/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.function.Predicate;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermStates;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexOrDocValuesQuery;
import org.apache.lucene.search.IndexSortSortedNumericDocValuesRangeQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PointInSetQuery;
import org.apache.lucene.search.PointRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SynonymQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.NumericUtils;
import org.opensearch.search.approximate.ApproximateScoreQuery;

/** The rows of the core query classes of {@link QueryEncoder}, one per class, and the methods they run. */
final class CoreNodes {
    private CoreNodes() {}

    /** The core query classes, in the order they are tried. */
    static final NodeRegistry<Walk> ROWS = new NodeRegistry<Walk>()
        .add(TermQuery.class, (tq, w) -> term(w.cx(), tq, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(SynonymQuery.class, (sq, w) -> QueryEncoder.synonym(w.cx(), sq, w.out(), w.fieldOk(), w.nodes()))
        .add(BooleanQuery.class, (bq, w) -> boolQuery(w.cx(), bq, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(ConstantScoreQuery.class, (cs, w) -> constantScore(w.cx(), cs, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(BoostQuery.class, (b, w) -> boost(w.cx(), b, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(DisjunctionMaxQuery.class, (dm, w) -> disMax(w.cx(), dm, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(ApproximateScoreQuery.class, (a, w) -> approximate(w.cx(), a, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(org.apache.lucene.search.MultiPhraseQuery.class, (q, w) -> QueryEncoder.plainMultiPhrase(w.cx(), q, w.out(), w.fieldOk(), w.nodes()))
        .add(org.opensearch.common.lucene.search.MultiPhrasePrefixQuery.class, (q, w) -> QueryEncoder.plainMultiPhrase(w.cx(), q, w.out(), w.fieldOk(), w.nodes()))
        .addExact(PhraseQuery.class, (pq, w) -> phrase(w.cx(), pq, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(IndexOrDocValuesQuery.class, (q, w) -> indexOrDocValues(w.cx(), q, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(IndexSortSortedNumericDocValuesRangeQuery.class, (q, w) -> indexSortRange(w.cx(), q, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(PointRangeQuery.class, (pr, w) -> pointRange(w.cx(), pr, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .addWhen(PointInSetQuery.class,
            (ps, w) -> ps.getNumDims() == 1 && ps.getBytesPerDim() == Long.BYTES && QueryEncoder.seqNoOnDocValues(w.cx(), ps.getField()),
            (ps, w) -> pointSetDocValues(w.cx(), ps, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .add(PointInSetQuery.class, (ps, w) -> pointSet(w.cx(), ps, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .addWhen(Query.class, (q, w) -> q.getClass().getPackageName().equals("org.apache.lucene.search") && wrapperRewrite(q) >= 0,
            (q, w) -> multiTermWrapper(w.cx(), q, w.out(), w.fieldOk(), w.depth(), w.nodes()))
        .addExact(MatchAllDocsQuery.class, (q, w) -> {
            w.out().write(QueryEncoder.NODE_MATCH_ALL);
            return null;
        })
        .addExact(FieldExistsQuery.class, (q, w) -> {
            // `exists` on a field with norms or doc values (one on every document was rewritten to
            // a match-all already). A vector field is the native side's error, and Lucene's answer.
            w.out().write(QueryEncoder.NODE_EXISTS);
            QueryEncoder.writeBytes(w.out(), q.getField().getBytes(StandardCharsets.UTF_8));
            return null;
        })
        .addExact(MatchNoDocsQuery.class, (q, w) -> {
            w.out().write(QueryEncoder.NODE_MATCH_NONE);
            return null;
        });

    /** Where a node is written: the request, the output, and the position in the tree. */
    record Walk(EncodeContext cx, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {}

    /** The rewrite a package-private {@code MultiTermQuery} wrapper class stands for, or -1. */
    static byte wrapperRewrite(Query q) {
        return switch (q.getClass().getSimpleName()) {
            case "MultiTermQueryConstantScoreBlendedWrapper" -> QueryEncoder.REWRITE_CONSTANT_SCORE_BLENDED;
            case "MultiTermQueryConstantScoreWrapper" -> QueryEncoder.REWRITE_CONSTANT_SCORE;
            case "MultiTermQueryDocValuesWrapper" -> QueryEncoder.REWRITE_DOC_VALUES;
            default -> -1;
        };
    }

    private static String term(EncodeContext cx, TermQuery tq, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        if (tq.getClass() != TermQuery.class) {
            return FallbackReason.TERM_STATES.key();
        }
        Term t = tq.getTerm();
        if (fieldOk.test(t.field()) == false) {
            return FallbackReason.FIELD_SIMILARITY.key();
        }
        TermStates states = tq.getTermStates();
        if (states == null) {
            out.write(QueryEncoder.NODE_TERM);
            QueryEncoder.writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            QueryEncoder.writeBytes(out, t.bytes());
            return null;
        }
        IndexReaderContext top = cx.top();
        if (top == null) {
            return FallbackReason.TERM_STATES.key();
        }
        if (states.wasBuiltFor(top) == false) {
            // TermQuery.createWeight ignores states built for another reader and rebuilds them
            // from the searcher's: the plain term ("terms" lookup, more_like_this, percolate).
            out.write(QueryEncoder.NODE_TERM);
            QueryEncoder.writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
            QueryEncoder.writeBytes(out, t.bytes());
            return null;
        }
        int docFreq;
        try {
            docFreq = states.docFreq();
        } catch (IllegalStateException | AssertionError e) {
            // States built without statistics: nothing to score from.
            return FallbackReason.TERM_STATES.key();
        }
        if (docFreq < 1) {
            // No statistics, so no scorer (TermWeight's termStates.get(leaf) is null on every
            // leaf): the term matches nothing, whatever the index holds.
            return QueryEncoder.noLeafState(states, top) ? QueryEncoder.matchNone(out) : "term_states";
        }
        out.write(QueryEncoder.NODE_TERM_STATS);
        QueryEncoder.writeBytes(out, t.field().getBytes(StandardCharsets.UTF_8));
        QueryEncoder.writeBytes(out, t.bytes());
        QueryEncoder.writeLong(out, docFreq);
        return null;
    }

    private static String boolQuery(EncodeContext cx, BooleanQuery bq, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        if (bq.getMinimumNumberShouldMatch() < 0) {
            // BooleanQuery.Builder does not reject it; the native decoder does.
            return FallbackReason.BOOLEAN_MSM_NEGATIVE.key();
        }
        if (cx.plain() && QueryEncoder.inExtractorOrder(bq) == false) {
            if (cx.extendedHighlight() == false) {
                return FallbackReason.HIGHLIGHT_CLAUSE_ORDER.key();
            }
            // The plain highlighter's extractor reads a bool query's clauses in the order they
            // were added, whatever their occurrence, and skips the prohibited ones: the native
            // reader walks musts, then shoulds, then filters, so the clauses go as musts in
            // that order (the highlighter only reads the query's terms, it never runs it).
            int kept = 0;
            for (BooleanClause c : bq.clauses()) {
                kept += c.occur() == BooleanClause.Occur.MUST_NOT ? 0 : 1;
            }
            out.write(QueryEncoder.NODE_BOOLEAN);
            QueryEncoder.writeInt(out, 0);
            QueryEncoder.writeInt(out, kept);
            for (BooleanClause c : bq.clauses()) {
                if (c.occur() == BooleanClause.Occur.MUST_NOT) {
                    continue;
                }
                out.write((byte) BooleanClause.Occur.MUST.ordinal());
                String reason = QueryEncoder.node(cx, c.query(), out, fieldOk, depth + 1, nodes);
                if (reason != null) {
                    return reason;
                }
            }
            return null;
        }
        out.write(QueryEncoder.NODE_BOOLEAN);
        QueryEncoder.writeInt(out, bq.getMinimumNumberShouldMatch());
        QueryEncoder.writeInt(out, bq.clauses().size());
        for (BooleanClause c : bq.clauses()) {
            out.write((byte) c.occur().ordinal());
            String reason = QueryEncoder.node(cx, c.query(), out, fieldOk, depth + 1, nodes);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    private static String constantScore(EncodeContext cx, ConstantScoreQuery cs, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // ConstantScoreQuery scores its boost, which is 1 unless a BoostQuery wraps it.
        out.write(QueryEncoder.NODE_CONSTANT_SCORE);
        QueryEncoder.writeInt(out, Float.floatToIntBits(1f));
        return QueryEncoder.node(cx, cs.getQuery(), out, fieldOk, depth + 1, nodes);
    }

    private static String boost(EncodeContext cx, BoostQuery b, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        float boost = b.getBoost();
        if (Float.isFinite(boost) == false || boost < 0) {
            return FallbackReason.BOOST_INVALID.key();
        }
        out.write(QueryEncoder.NODE_BOOST);
        QueryEncoder.writeInt(out, Float.floatToIntBits(boost));
        return QueryEncoder.node(cx, b.getQuery(), out, fieldOk, depth + 1, nodes);
    }

    private static String disMax(EncodeContext cx, DisjunctionMaxQuery dm, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        out.write(QueryEncoder.NODE_DISMAX);
        QueryEncoder.writeInt(out, Float.floatToIntBits(dm.getTieBreakerMultiplier()));
        QueryEncoder.writeInt(out, dm.getDisjuncts().size());
        for (Query d : dm.getDisjuncts()) {
            String reason = QueryEncoder.node(cx, d, out, fieldOk, depth + 1, nodes);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    private static String approximate(EncodeContext cx, ApproximateScoreQuery a, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // A wrapper, not a level (nor a node): the range it approximates is still the root
        // when it is.
        nodes[0]--;
        return QueryEncoder.node(cx, a.getOriginalQuery(), out, fieldOk, depth, nodes);
    }

    private static String phrase(EncodeContext cx, PhraseQuery pq, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        Term[] terms = pq.getTerms();
        int[] positions = pq.getPositions();
        if (terms.length == 0) {
            out.write(QueryEncoder.NODE_MATCH_NONE);
            return null;
        }
        if (fieldOk.test(pq.getField()) == false) {
            return FallbackReason.FIELD_SIMILARITY.key();
        }
        // A gap in the positions is a stopword the analyzer removed: the node then carries them.
        boolean gaps = false;
        for (int i = 0; i < positions.length; i++) {
            gaps |= positions[i] - positions[0] != i;
        }
        nodes[0] += terms.length;
        if (nodes[0] > QueryEncoder.MAX_NODES) {
            return FallbackReason.QUERY_TOO_LARGE.key();
        }
        out.write(gaps ? QueryEncoder.NODE_PHRASE_POSITIONS : QueryEncoder.NODE_PHRASE);
        QueryEncoder.writeBytes(out, pq.getField().getBytes(StandardCharsets.UTF_8));
        QueryEncoder.writeInt(out, pq.getSlop());
        QueryEncoder.writeInt(out, terms.length);
        for (int i = 0; i < terms.length; i++) {
            QueryEncoder.writeInt(out, gaps ? positions[i] : i);
            QueryEncoder.writeBytes(out, terms[i].bytes());
        }
        return null;
    }

    private static String indexOrDocValues(EncodeContext cx, IndexOrDocValuesQuery iodv, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // A long, date or double range on a field with doc values: both sides over the same
        // field and bounds, so the native side can pick per segment as Lucene does (doc
        // values when a cheaper clause leads).
        if (iodv.getIndexQuery() instanceof PointRangeQuery pr
                && pr.getNumDims() == 1
                && pr.getBytesPerDim() == Long.BYTES) {
            long min = NumericUtils.sortableBytesToLong(pr.getLowerPoint(), 0);
            long max = NumericUtils.sortableBytesToLong(pr.getUpperPoint(), 0);
            if (iodv.getRandomAccessQuery().equals(SortedNumericDocValuesField.newSlowRangeQuery(pr.getField(), min, max))) {
                out.write(QueryEncoder.NODE_POINT_RANGE_OR_DV);
                QueryEncoder.writeBytes(out, pr.getField().getBytes(StandardCharsets.UTF_8));
                QueryEncoder.writeLong(out, min);
                QueryEncoder.writeLong(out, max);
                return null;
            }
        }
        // Otherwise two equivalent ways to run one query; the native side runs the index
        // one. A wrapper, not a level (nor a node).
        nodes[0]--;
        return QueryEncoder.node(cx, iodv.getIndexQuery(), out, fieldOk, depth, nodes);
    }

    private static String indexSortRange(EncodeContext cx, IndexSortSortedNumericDocValuesRangeQuery is, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // A `range` on an index-sorted shard's primary sort field: Lucene binary-searches the
        // sorted doc values where it can and otherwise runs its fallback query, which matches
        // the same documents with the same constant score. A wrapper, not a level.
        nodes[0]--;
        return QueryEncoder.node(cx, is.getFallbackQuery(), out, fieldOk, depth, nodes);
    }

    private static String pointRange(EncodeContext cx, PointRangeQuery pr, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // `LongPoint`/`DoublePoint`/date ranges: one dimension of 8 bytes, sent as the sortable
        // longs their packed bytes encode. Any other width or dimension count (`integer`,
        // `float`, `half_float`, `ip`, `unsigned_long`, `LatLonPoint.newBoxQuery`) is a points
        // box: its packed corners as they are.
        if (pr.getNumDims() != 1 || pr.getBytesPerDim() != Long.BYTES) {
            return GeoEncoder.node(pr, out);
        }
        if (QueryEncoder.seqNoOnDocValues(cx, pr.getField())) {
            out.write(DocValuesEncoder.NODE_DV_NUMERIC_RANGE);
            QueryEncoder.writeBytes(out, pr.getField().getBytes(StandardCharsets.UTF_8));
            QueryEncoder.writeLong(out, NumericUtils.sortableBytesToLong(pr.getLowerPoint(), 0));
            QueryEncoder.writeLong(out, NumericUtils.sortableBytesToLong(pr.getUpperPoint(), 0));
            return null;
        }
        out.write(QueryEncoder.NODE_POINT_RANGE);
        QueryEncoder.writeBytes(out, pr.getField().getBytes(StandardCharsets.UTF_8));
        QueryEncoder.writeLong(out, NumericUtils.sortableBytesToLong(pr.getLowerPoint(), 0));
        QueryEncoder.writeLong(out, NumericUtils.sortableBytesToLong(pr.getUpperPoint(), 0));
        return null;
    }

    private static String pointSetDocValues(EncodeContext cx, PointInSetQuery ps, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        java.util.Collection<byte[]> points = ps.getPackedPoints();
        out.write(DocValuesEncoder.NODE_DV_NUMERIC_SET);
        QueryEncoder.writeBytes(out, ps.getField().getBytes(StandardCharsets.UTF_8));
        QueryEncoder.writeInt(out, points.size());
        for (byte[] p : points) {
            QueryEncoder.writeLong(out, NumericUtils.sortableBytesToLong(p, 0));
        }
        return null;
    }

    private static String pointSet(EncodeContext cx, PointInSetQuery ps, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        // `terms` (and `IntPoint.newSetQuery` and the like): one leaf, its packed points.
        out.write(QueryEncoder.NODE_POINT_SET);
        QueryEncoder.writeBytes(out, ps.getField().getBytes(StandardCharsets.UTF_8));
        QueryEncoder.writeInt(out, ps.getNumDims());
        QueryEncoder.writeInt(out, ps.getBytesPerDim());
        java.util.Collection<byte[]> points = ps.getPackedPoints();
        QueryEncoder.writeInt(out, points.size());
        for (byte[] p : points) {
            QueryEncoder.writeBytes(out, p);
        }
        return null;
    }

    private static String multiTermWrapper(EncodeContext cx, Query q, ByteArrayOutputStream out, Predicate<String> fieldOk, int depth, int[] nodes) {
        String wrapper = q.getClass().getSimpleName();
        byte rewrite = wrapperRewrite(q);
        // A MultiTermQuery under a constant-score or doc-values rewrite (package-private
        // classes, so found by name; their query by the visitor, which hands it over as it is).
        Query inner = QueryEncoder.wrappedMultiTermQuery(q);
        if (inner == null) {
            return FallbackReason.CLAUSE_ANY.key() + wrapper;
        }
        return QueryEncoder.multiTerm(cx, inner, rewrite, out, fieldOk, depth, nodes);
    }
}
