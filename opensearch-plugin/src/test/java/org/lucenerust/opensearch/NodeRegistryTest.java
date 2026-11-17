/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.junit.Test;

public class NodeRegistryTest {
    private static final Query TERM = new TermQuery(new Term("f", "a"));

    @Test
    public void firstClaimingRowWins() throws Exception {
        NodeRegistry<String> r = new NodeRegistry<String>()
            .add(TermQuery.class, (q, w) -> "term")
            .add(Query.class, (q, w) -> "any");
        assertEquals("term", r.encode(TERM, ""));
        assertEquals("any", r.encode(MatchAllDocsQuery.INSTANCE, ""));
    }

    @Test
    public void conditionAndExactClass() throws Exception {
        NodeRegistry<String> r = new NodeRegistry<String>()
            .addWhen(TermQuery.class, (q, w) -> w.equals("on"), (q, w) -> "when")
            .addExact(MatchAllDocsQuery.class, (q, w) -> "exact");
        assertEquals("when", r.encode(TERM, "on"));
        assertSame(NodeRegistry.NOT_REGISTERED, r.encode(TERM, "off"));
        assertEquals("exact", r.encode(MatchAllDocsQuery.INSTANCE, ""));
        assertEquals(2, r.size());
    }

    @Test
    public void unclaimedQuery() throws Exception {
        NodeRegistry<String> r = new NodeRegistry<String>().add(TermQuery.class, (q, w) -> "term");
        assertSame(NodeRegistry.NOT_REGISTERED, r.encode(new BoostQuery(TERM, 2f), ""));
    }
}
