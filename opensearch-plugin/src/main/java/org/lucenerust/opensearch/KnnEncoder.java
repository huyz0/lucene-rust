/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.lucenerust.opensearch.knn.KnnLeafVectorQuery;
import org.lucenerust.opensearch.knn.KnnTarget;
import org.lucenerust.opensearch.knn.KnnTopVectorQuery;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * The KNN nodes of the query tree ({@code jvm_queries.rs}) and the whole-reader KNN request
 * ({@code jvm_knn.rs}):
 *
 * <ul>
 *   <li>{@link KnnLeafVectorQuery}, node 39: the field, the target, the kind ({@link #KNN_TOP_K}
 *       with {@code k} and {@code ef_search}, or {@link #KNN_RADIAL} with the least score), then
 *       the filter when there is one;
 *   <li>{@link KnnHitsQuery}, node 40: the reader's segment doc bases, then the hits as global doc
 *       id and score pairs.
 * </ul>
 *
 * {@link #topHits} is {@link KnnTopVectorQuery}'s native rewrite.
 */
final class KnnEncoder {
    private static final Logger logger = LogManager.getLogger(KnnEncoder.class);

    static final byte NODE_KNN = 39;
    static final byte NODE_DOC_AND_SCORE = 40;
    static final byte KNN_TOP_K = 0;
    static final byte KNN_RADIAL = 1;
    static final byte KNN_TARGET_FLOAT = 0;
    static final byte KNN_TARGET_BYTE = 1;

    /** What {@link #node} returns for a query that is none of these. */
    static final String NOT_KNN = "not_knn";

    private KnnEncoder() {}

    /** Appends {@code q} when it is a KNN query: null when written, a fallback reason when not, {@link #NOT_KNN} otherwise. */
    static String node(Query q, ByteArrayOutputStream out, ExtendedEncoder.Child child) {
        if (q instanceof KnnLeafVectorQuery k) {
            out.write(NODE_KNN);
            writeString(out, k.field());
            target(out, k.target());
            if (k.isRadial()) {
                out.write(KNN_RADIAL);
                writeInt(out, Float.floatToIntBits(k.minScore()));
            } else {
                out.write(KNN_TOP_K);
                writeInt(out, k.k());
                writeInt(out, k.efSearch());
            }
            if (k.filter() == null) {
                out.write(0);
                return null;
            }
            out.write(1);
            return child.node(k.filter());
        }
        if (q instanceof KnnHitsQuery h) {
            out.write(NODE_DOC_AND_SCORE);
            writeInt(out, h.docBases().length);
            for (int b : h.docBases()) {
                writeInt(out, b);
            }
            writeInt(out, h.docs().length);
            for (int i = 0; i < h.docs().length; i++) {
                writeInt(out, h.docs()[i]);
                writeInt(out, Float.floatToIntBits(h.scores()[i]));
            }
            return null;
        }
        if (q instanceof KnnTopVectorQuery) {
            return "clause_KnnTopVectorQuery_unrewritten";
        }
        return NOT_KNN;
    }

    /**
     * {@code q}'s hits over the searcher's native reader, or null when the reader has none (or its
     * filter cannot be written, or the native search fails), for Lucene to find.
     */
    static KnnHitsQuery topHits(NativeReaders readers, IndexSearcher searcher, KnnTopVectorQuery q) {
        IndexReader reader = searcher.getIndexReader();
        // A view of the writer's buffers has segments the native reader does not.
        if (LiveDirectoryReader.baseOf(reader) != null) {
            return null;
        }
        NativeReaders.Acquired acquired = readers.acquire(reader);
        if (acquired.handle() == 0) {
            return null;
        }
        ByteArrayOutputStream request = new ByteArrayOutputStream();
        writeString(request, q.field());
        target(request, q.target());
        writeInt(request, q.k());
        writeInt(request, q.candidates());
        if (q.filter() == null) {
            request.write(0);
        } else {
            request.write(1);
            if (QueryEncoder.encodeNode(q.filter(), request, searcher.getTopReaderContext()) != null) {
                return null;
            }
        }
        if (q.parents() == null) {
            request.write(0);
        } else {
            Query parents;
            try {
                parents = ExtendedEncoder.parentsQuery(q.parents());
            } catch (ReflectiveOperationException e) {
                parents = null;
            }
            if (parents == null) {
                return null;
            }
            request.write(1);
            if (QueryEncoder.encodeNode(parents, request, searcher.getTopReaderContext()) != null) {
                return null;
            }
            request.write(q.expand() ? 1 : 0);
        }
        writeInt(request, q.nprobes());
        byte[] blob = request.toByteArray();
        // An expanded search returns every scored child of the k parents: a few per parent, usually.
        int room = q.expand() ? (int) Math.min((long) q.k() * 8, Integer.MAX_VALUE - 8) : q.k();
        int[] docs = new int[room];
        float[] scores = new float[room];
        int[] count = new int[1];
        int rc = NativeBridge.readerKnn(acquired.handle(), blob, room, docs, scores, count);
        if (rc == NativeBridge.BUFFER_TOO_SMALL && count[0] > room) {
            room = count[0];
            docs = new int[room];
            scores = new float[room];
            rc = NativeBridge.readerKnn(acquired.handle(), blob, room, docs, scores, count);
        }
        if (rc != NativeBridge.OK) {
            // As every native search does: a failure falls back to Lucene rather than failing the query.
            logger.debug("lucene-rust: [knn] native search failed ({}): {}", rc, NativeBridge.lastError());
            return null;
        }
        ScoreDoc[] hits = new ScoreDoc[count[0]];
        for (int i = 0; i < hits.length; i++) {
            hits[i] = new ScoreDoc(docs[i], scores[i]);
        }
        return KnnHitsQuery.of(reader, hits);
    }

    private static void target(ByteArrayOutputStream out, KnnTarget t) {
        if (t.floats() != null) {
            out.write(KNN_TARGET_FLOAT);
            writeInt(out, t.floats().length);
            for (float f : t.floats()) {
                writeInt(out, Float.floatToIntBits(f));
            }
        } else {
            out.write(KNN_TARGET_BYTE);
            writeInt(out, t.bytes().length);
            out.writeBytes(t.bytes());
        }
    }

    private static void writeString(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
