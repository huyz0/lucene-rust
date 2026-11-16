/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.opensearch.search.dfs.AggregatedDfs;
import org.opensearch.search.internal.ContextIndexSearcher;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * The statistics of a {@code dfs_query_then_fetch} search, in front of a query blob: what {@link
 * ContextIndexSearcher#termStatistics} and {@code collectionStatistics} answer once the DFS phase
 * aggregated every shard's ({@code AggregatedDfs}), as the native {@code QUERY_DFS} prefix reads
 * them. The native statistics pass scores with them in place of the shard's, a term or field
 * the DFS does not hold keeping the shard's own, as the searcher does.
 */
final class NativeDfs {
    /** Blob tag ({@code QUERY_DFS}). */
    static final byte QUERY_DFS = 5;

    private static final Field AGGREGATED = field();

    private NativeDfs() {}

    private static Field field() {
        try {
            Field f = ContextIndexSearcher.class.getDeclaredField("aggregatedDfs");
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** Whether the searcher's aggregated DFS can be read. */
    static boolean readable() {
        return AGGREGATED != null;
    }

    /** {@code blob} behind the searcher's DFS statistics, or as it is when the search has none. */
    static byte[] prefix(ContextIndexSearcher searcher, byte[] blob) {
        if (blob == null || AGGREGATED == null) {
            return blob;
        }
        AggregatedDfs dfs;
        try {
            dfs = (AggregatedDfs) AGGREGATED.get(searcher);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return blob;
        }
        if (dfs == null) {
            return blob;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(QUERY_DFS);
        writeInt(out, dfs.termStatistics().size());
        for (Map.Entry<Term, TermStatistics> e : dfs.termStatistics().entrySet()) {
            writeBytes(out, e.getKey().field().getBytes(StandardCharsets.UTF_8));
            writeBytes(out, e.getKey().bytes().bytes, e.getKey().bytes().offset, e.getKey().bytes().length);
            writeLong(out, e.getValue().docFreq());
            writeLong(out, e.getValue().totalTermFreq());
        }
        writeInt(out, dfs.fieldStatistics().size());
        for (Map.Entry<String, CollectionStatistics> e : dfs.fieldStatistics().entrySet()) {
            writeBytes(out, e.getKey().getBytes(StandardCharsets.UTF_8));
            CollectionStatistics s = e.getValue();
            writeLong(out, s.maxDoc());
            writeLong(out, s.docCount());
            writeLong(out, s.sumTotalTermFreq());
            writeLong(out, s.sumDocFreq());
        }
        out.writeBytes(blob);
        return out.toByteArray();
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeBytes(out, b, 0, b.length);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b, int offset, int length) {
        writeInt(out, length);
        out.write(b, offset, length);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        for (int i = 0; i < 4; i++) {
            out.write(v >>> (8 * i));
        }
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        for (int i = 0; i < 8; i++) {
            out.write((int) (v >>> (8 * i)));
        }
    }
}
