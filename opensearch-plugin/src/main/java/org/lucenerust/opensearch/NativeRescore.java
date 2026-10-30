/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.Query;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.rescore.QueryRescorer;
import org.opensearch.search.rescore.RescoreContext;

import java.io.IOException;
import java.util.List;

/**
 * One {@code rescore} entry run natively: OpenSearch's {@link QueryRescorer} (Lucene's {@code
 * QueryRescorer} with OpenSearch's {@code QueryRescoreMode} combination), the rescore query scored
 * by {@code rescore_hits} in {@code crates/lucene-ffi/src/jvm_reader.rs}. RescoreProcessor's loop
 * runs in {@code RustQueryPhaseSearcher.rescore}.
 */
record NativeRescore(
    byte[] blob,
    int window,
    float queryWeight,
    float rescoreQueryWeight,
    int mode,
    QueryRescorer.QueryRescoreContext context
) {
    /** Whether {@code ctx} rescores: a non-scroll search that asks for hits and has rescorers. */
    static boolean applies(SearchContext ctx) {
        return ctx.rescore() != null && ctx.rescore().isEmpty() == false && ctx.scrollContext() == null && ctx.size() > 0;
    }

    /** Whether every rescorer of {@code ctx} is OpenSearch's {@link QueryRescorer}. */
    static boolean supported(SearchContext ctx) {
        for (RescoreContext r : ctx.rescore()) {
            if (r.rescorer() != QueryRescorer.INSTANCE
                || !(r instanceof QueryRescorer.QueryRescoreContext q)
                || q.parsedQuery() == null
                || q.scoreMode() == null) {
                return false;
            }
        }
        return true;
    }

    /**
     * Encodes each rescorer of {@code ctx} into {@code out}, its query rewritten as Lucene's
     * QueryRescorer rewrites it ({@code searcher.rewrite}); returns null, or why one cannot run.
     */
    static String encode(SearchContext ctx, ContextIndexSearcher searcher, SimilarityEncoder similarities, List<NativeRescore> out) {
        for (RescoreContext r : ctx.rescore()) {
            QueryRescorer.QueryRescoreContext q = (QueryRescorer.QueryRescoreContext) r;
            Query query;
            try {
                query = searcher.rewrite(q.parsedQuery().query());
            } catch (IOException | RuntimeException e) {
                return "rescore_query";
            }
            // A root range or match_all that ContextIndexSearcher.rewrite resolved to OpenSearch's
            // approximation (for the search's own size) matches only the documents it visits
            // first: the rescore then scales the rest by query_weight alone, which the exact native
            // query does not reproduce.
            if (RustQueryPhaseSearcher.approximated(query)) {
                return "rescore_approximate";
            }
            QueryEncoder.Encoded enc = QueryEncoder.encode(query, similarities, searcher.getTopReaderContext());
            if (enc.fallbackReason() != null) {
                return "rescore_" + enc.fallbackReason();
            }
            out.add(new NativeRescore(similarities.prefix(enc.blob()), q.getWindowSize(), q.queryWeight(), q.rescoreQueryWeight(), q.scoreMode().ordinal(), q));
        }
        return null;
    }
}
