/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import java.io.IOException;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

/**
 * The k-NN plugin's format for packed bit vectors on its lucene engine: {@code
 * Lucene99HnswVectorsFormat}'s files, with byte vectors scored as {@code 1 / (1 + hamming
 * distance)} whatever similarity the field records (k-NN records {@code EUCLIDEAN}). The Rust
 * engine writes and reads the same files ({@code vectors.rs}'s {@code HAMMING_FORMAT_NAME}).
 */
public final class KNN990HnswBinaryVectorsFormat extends KnnVectorsFormat {
    public static final String NAME = "KNN990HnswBinaryVectorsFormat";

    private static final FlatVectorsFormat FLAT = new Lucene99FlatVectorsFormat(new HammingScorer());

    private final int maxConn;
    private final int beamWidth;

    public KNN990HnswBinaryVectorsFormat() {
        this(Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN, Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH);
    }

    public KNN990HnswBinaryVectorsFormat(int maxConn, int beamWidth) {
        super(NAME);
        if (maxConn <= 0 || maxConn > Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN) {
            throw new IllegalArgumentException(
                "maxConn must be positive and less than or equal to " + Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN
                    + "; maxConn=" + maxConn);
        }
        if (beamWidth <= 0 || beamWidth > Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH) {
            throw new IllegalArgumentException(
                "beamWidth must be positive and less than or equal to " + Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH
                    + "; beamWidth=" + beamWidth);
        }
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, FLAT, FLAT.fieldsWriter(state), 1, null);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, FLAT.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return 16_000;
    }

    @Override
    public String toString() {
        return NAME + "(name=" + NAME + ", maxConn=" + maxConn + ", beamWidth=" + beamWidth + ")";
    }

    /** {@code 1 / (1 + popcount(a ^ b))} over byte vectors; float targets are refused. */
    static final class HammingScorer implements FlatVectorsScorer {
        static float score(byte[] a, byte[] b) {
            return 1.0f / (1 + VectorUtil.xorBitCount(a, b));
        }

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction similarity, KnnVectorValues values) throws IOException {
            return new Supplier(bytes(values));
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction similarity, KnnVectorValues values, float[] target) {
            throw new IllegalArgumentException("binary vectors do not support float[] targets");
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction similarity, KnnVectorValues values, byte[] target) {
            ByteVectorValues v = bytes(values);
            return new RandomVectorScorer.AbstractRandomVectorScorer(v) {
                @Override
                public float score(int node) throws IOException {
                    return HammingScorer.score(target, v.vectorValue(node));
                }
            };
        }

        private static ByteVectorValues bytes(KnnVectorValues values) {
            if (values instanceof ByteVectorValues b) {
                return b;
            }
            throw new IllegalArgumentException("binary vectors must be byte vector values");
        }
    }

    /** Graph-build scorers: each reads candidates through its own copy of the values. */
    private static final class Supplier implements RandomVectorScorerSupplier {
        private final ByteVectorValues values;
        private final ByteVectorValues targets;

        Supplier(ByteVectorValues values) throws IOException {
            this.values = values;
            this.targets = values.copy();
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
            byte[] query = new byte[values.dimension()];
            return new UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer(values) {
                @Override
                public float score(int node) throws IOException {
                    return HammingScorer.score(query, targets.vectorValue(node));
                }

                @Override
                public void setScoringOrdinal(int node) throws IOException {
                    System.arraycopy(targets.vectorValue(node), 0, query, 0, query.length);
                }
            };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new Supplier(values.copy());
        }
    }
}
