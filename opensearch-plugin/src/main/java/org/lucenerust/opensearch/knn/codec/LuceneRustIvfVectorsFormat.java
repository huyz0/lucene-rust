/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import java.io.IOException;
import java.util.Map;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * The Rust engine's IVF vectors format, as the JVM reads it: a field's raw vectors are {@code
 * Lucene99FlatVectorsFormat}'s {@code .vec}/{@code .vemf}, and the IVF index the Rust writer puts
 * beside them ({@code .ivm}, {@code .ivp}, {@code .ivv}) is searched by the Rust engine only. Here a
 * search scores every accepted vector, so Lucene's own paths (and a segment this side wrote, which
 * has the raw vectors alone) still return exact results.
 */
public final class LuceneRustIvfVectorsFormat extends KnnVectorsFormat {
    public static final String NAME = "LuceneRustIvfVectorsFormat";

    private static final FlatVectorsFormat FLAT = new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);

    public LuceneRustIvfVectorsFormat() {
        super(NAME);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return FLAT.fieldsWriter(state);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Reader(FLAT.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return 16_000;
    }

    @Override
    public String toString() {
        return NAME + "(flat=" + FLAT + ")";
    }

    /** The flat vectors, searched exhaustively. */
    private static final class Reader extends KnnVectorsReader {
        private final FlatVectorsReader flat;

        Reader(FlatVectorsReader flat) {
            this.flat = flat;
        }

        @Override
        public void checkIntegrity() throws IOException {
            flat.checkIntegrity();
        }

        @Override
        public FloatVectorValues getFloatVectorValues(String field) throws IOException {
            return flat.getFloatVectorValues(field);
        }

        @Override
        public ByteVectorValues getByteVectorValues(String field) throws IOException {
            return flat.getByteVectorValues(field);
        }

        @Override
        public void search(String field, float[] target, KnnCollector collector, AcceptDocs acceptDocs) throws IOException {
            collect(flat.getRandomVectorScorer(field, target), collector, acceptDocs);
        }

        @Override
        public void search(String field, byte[] target, KnnCollector collector, AcceptDocs acceptDocs) throws IOException {
            collect(flat.getRandomVectorScorer(field, target), collector, acceptDocs);
        }

        private static void collect(RandomVectorScorer scorer, KnnCollector collector, AcceptDocs acceptDocs) throws IOException {
            Bits accepted = scorer.getAcceptOrds(acceptDocs == null ? null : acceptDocs.bits());
            for (int ord = 0; ord < scorer.maxOrd(); ord++) {
                if (collector.earlyTerminated()) {
                    break;
                }
                if (accepted == null || accepted.get(ord)) {
                    collector.incVisitedCount(1);
                    collector.collect(scorer.ordToDoc(ord), scorer.score(ord));
                }
            }
        }

        @Override
        public Map<String, Long> getOffHeapByteSize(FieldInfo info) {
            return flat.getOffHeapByteSize(info);
        }

        @Override
        public void close() throws IOException {
            flat.close();
        }
    }
}
