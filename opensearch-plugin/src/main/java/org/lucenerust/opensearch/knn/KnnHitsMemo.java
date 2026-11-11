/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.ReaderContext;
import org.opensearch.search.internal.SearchContext;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * The native hits of each {@link KnnTopVectorQuery} a shard's reader context searched, for the
 * later phases of the same request to reuse.
 *
 * <p>OpenSearch builds a new search context for each phase of a request over the shard's reader
 * context (DFS, query, fetch, each scroll page), and each rewrites the query again: without this,
 * the fetch phase of a multi-shard search ran the shard's k-NN search a second time only to rewrite
 * a query it does not search. The hits are those of the same query over the same reader, so
 * reusing them changes nothing the request returns. They are dropped with the reader context.
 */
public final class KnnHitsMemo {
    private static final Logger logger = LogManager.getLogger(KnnHitsMemo.class);

    /** The queries a reader context keeps hits for: a scroll or point in time runs many. */
    static final int PER_CONTEXT = 16;

    /** {@code ContextIndexSearcher.searchContext}, which OpenSearch keeps private; null when unreadable. */
    private static final VarHandle SEARCH_CONTEXT = searchContextHandle();

    private static final Map<ReaderContext, Kept> KEPT = new ConcurrentHashMap<>();

    private static final LongAdder KEPT_COUNT = new LongAdder();
    private static final LongAdder REUSED_COUNT = new LongAdder();

    private KnnHitsMemo() {}

    private static VarHandle searchContextHandle() {
        try {
            return MethodHandles.privateLookupIn(ContextIndexSearcher.class, MethodHandles.lookup())
                .findVarHandle(ContextIndexSearcher.class, "searchContext", SearchContext.class);
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.warn("lucene-rust: cannot read a searcher's search context; [knn] searches again in each phase", e);
            return null;
        }
    }

    /** One query's hits and the reader they are of. */
    private record Hits(Object[] reader, ScoreDoc[] hits) {}

    /** A reader context's entries, most recently used last. */
    private static final class Kept {
        private final LinkedHashMap<KnnTopVectorQuery, Hits> entries = new LinkedHashMap<>(4, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<KnnTopVectorQuery, Hits> eldest) {
                return size() > PER_CONTEXT;
            }
        };

        synchronized Hits get(KnnTopVectorQuery q) {
            return entries.get(q);
        }

        synchronized void put(KnnTopVectorQuery q, Hits e) {
            entries.put(q, e);
        }
    }

    /** The hits kept for {@code q} over the searcher's reader, rebuilt for it; null when none are. */
    static KnnHitsQuery find(IndexSearcher searcher, KnnTopVectorQuery q) {
        ReaderContext rc = readerContext(searcher);
        Kept kept = rc == null ? null : KEPT.get(rc);
        if (kept == null) {
            return null;
        }
        Hits e = kept.get(q);
        if (e == null) {
            return null;
        }
        IndexReader reader = searcher.getIndexReader();
        Object[] identity = identity(reader);
        if (identity == null || Arrays.equals(identity, e.reader) == false) {
            return null;
        }
        REUSED_COUNT.increment();
        return KnnHitsQuery.of(reader, e.hits);
    }

    /** Keeps {@code hits}, found for {@code q} over the searcher's reader, for the request's later phases. */
    static void keep(IndexSearcher searcher, KnnTopVectorQuery q, KnnHitsQuery hits) {
        ReaderContext rc = readerContext(searcher);
        if (rc == null) {
            return;
        }
        Object[] identity = identity(searcher.getIndexReader());
        if (identity == null) {
            return;
        }
        ScoreDoc[] sd = new ScoreDoc[hits.docs().length];
        for (int i = 0; i < sd.length; i++) {
            sd[i] = new ScoreDoc(hits.docs()[i], hits.scores()[i]);
        }
        Kept kept = KEPT.computeIfAbsent(rc, c -> {
            c.addOnClose(() -> KEPT.remove(c));
            return new Kept();
        });
        kept.put(q, new Hits(identity, sd));
        KEPT_COUNT.increment();
    }

    /** How many native k-NN searches' hits were kept for a later phase. */
    public static long keptCount() {
        return KEPT_COUNT.sum();
    }

    /** How many rewrites took kept hits instead of searching again. */
    public static long reusedCount() {
        return REUSED_COUNT.sum();
    }

    /** How many reader contexts keep hits now. */
    public static int contexts() {
        return KEPT.size();
    }

    private static ReaderContext readerContext(IndexSearcher searcher) {
        if (SEARCH_CONTEXT == null || searcher instanceof ContextIndexSearcher == false) {
            return null;
        }
        SearchContext sc = (SearchContext) SEARCH_CONTEXT.get((ContextIndexSearcher) searcher);
        return sc == null ? null : sc.readerContext();
    }

    /**
     * What the hits' doc ids depend on: the reader's cache key and each leaf's, which change as its
     * documents do (a realtime view's when Lucene is given its buffers, a segment's when it has new
     * deletes); null for a reader without a key, whose hits are not kept.
     */
    private static Object[] identity(IndexReader reader) {
        IndexReader.CacheHelper top = reader.getReaderCacheHelper();
        if (top == null) {
            return null;
        }
        List<LeafReaderContext> leaves = reader.leaves();
        Object[] out = new Object[1 + leaves.size()];
        out[0] = top.getKey();
        for (int i = 0; i < leaves.size(); i++) {
            IndexReader.CacheHelper h = leaves.get(i).reader().getReaderCacheHelper();
            out[i + 1] = h != null ? h.getKey() : FilterLeafReader.unwrap(leaves.get(i).reader());
        }
        return out;
    }
}
