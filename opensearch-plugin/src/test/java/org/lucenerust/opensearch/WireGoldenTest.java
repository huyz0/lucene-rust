/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.junit.Test;

/**
 * Golden blobs of the hand-written wire format, one test per node kind: the query is encoded and
 * the bytes compared with ones written out by {@link WireWriter}. The kind numbers are literals
 * here on purpose (see {@code NODE_*} in {@code query_blob.rs}).
 */
public class WireGoldenTest {
    private static byte[] encode(Query q) {
        QueryEncoder.Encoded e = QueryEncoder.encode(q, f -> true);
        assertNull("fallback: " + e.fallbackReason(), e.fallbackReason());
        assertNotNull(e.blob());
        return e.blob();
    }

    private static WireWriter term(String field, String text) {
        return new WireWriter().u8(0).str(field).str(text);
    }

    @Test
    public void term() {
        // A root term is the single-term entry point, not a tree node.
        byte[] blob = encode(new TermQuery(new Term("title", "fox")));
        assertArrayEquals(new WireWriter().u8(0).str("title").str("fox").build(), blob);
    }

    @Test
    public void booleanNode() {
        Query q = new BooleanQuery.Builder()
            .add(new TermQuery(new Term("f", "a")), Occur.MUST)
            .add(new TermQuery(new Term("f", "b")), Occur.SHOULD)
            .add(new TermQuery(new Term("f", "c")), Occur.MUST_NOT)
            .setMinimumNumberShouldMatch(1)
            .build();
        byte[] expected = tree(new WireWriter().u8(1).i32(1).i32(3)
            .u8(0).raw(term("f", "a").build())
            .u8(2).raw(term("f", "b").build())
            .u8(3).raw(term("f", "c").build()));
        assertArrayEquals(expected, encode(q));
    }

    @Test
    public void constantScoreAndBoost() {
        Query q = new BoostQuery(new ConstantScoreQuery(new TermQuery(new Term("f", "a"))), 2.5f);
        byte[] expected = tree(new WireWriter().u8(3).f32(2.5f).u8(2).f32(1f).raw(term("f", "a").build()));
        assertArrayEquals(expected, encode(q));
    }

    @Test
    public void disMax() {
        Query q = new DisjunctionMaxQuery(java.util.List.of(new TermQuery(new Term("f", "a")), new TermQuery(new Term("g", "b"))), 0.3f);
        byte[] expected = tree(new WireWriter().u8(4).f32(0.3f).i32(2)
            .raw(term("f", "a").build())
            .raw(term("g", "b").build()));
        assertArrayEquals(expected, encode(q));
    }

    @Test
    public void matchAllAndNone() {
        assertArrayEquals(tree(new WireWriter().u8(5)), encode(MatchAllDocsQuery.INSTANCE));
        assertArrayEquals(tree(new WireWriter().u8(6)), encode(new MatchNoDocsQuery("none")));
    }

    @Test
    public void phrase() {
        Query q = new PhraseQuery(2, "body", "quick", "fox");
        byte[] expected = tree(new WireWriter().u8(7).str("body").i32(2).i32(2).i32(0).str("quick").i32(1).str("fox"));
        assertArrayEquals(expected, encode(q));
    }

    @Test
    public void phraseWithPositionGap() {
        PhraseQuery.Builder b = new PhraseQuery.Builder();
        b.add(new Term("body", "quick"), 0);
        b.add(new Term("body", "fox"), 2);
        byte[] expected = tree(new WireWriter().u8(50).str("body").i32(0).i32(2).i32(0).str("quick").i32(2).str("fox"));
        assertArrayEquals(expected, encode(b.build()));
    }

    @Test
    public void pointRange() {
        // LongPoint's sortable bytes decode to the long itself.
        assertArrayEquals(
            tree(new WireWriter().u8(11).str("n").i64(5).i64(9)),
            encode(LongPoint.newRangeQuery("n", 5, 9)));
    }

    @Test
    public void pointSet() {
        byte[] expected = tree(new WireWriter().u8(26).str("n").i32(1).i32(8).i32(2)
            .bytes(sortable(3)).bytes(sortable(7)));
        assertArrayEquals(expected, encode(LongPoint.newSetQuery("n", 3L, 7L)));
    }

    @Test
    public void exists() {
        assertArrayEquals(tree(new WireWriter().u8(13).str("tag")), encode(new FieldExistsQuery("tag")));
    }

    private static byte[] sortable(long v) {
        byte[] b = new byte[8];
        org.apache.lucene.util.NumericUtils.longToSortableBytes(v, b, 0);
        return b;
    }

    /** The blob of a tree query: the tree tag, then the node. */
    private static byte[] tree(WireWriter node) {
        byte[] body = node.build();
        byte[] out = new byte[body.length + 1];
        out[0] = TREE;
        System.arraycopy(body, 0, out, 1, body.length);
        return out;
    }

    /** {@code QUERY_TREE}. */
    private static final byte TREE = 2;
}
