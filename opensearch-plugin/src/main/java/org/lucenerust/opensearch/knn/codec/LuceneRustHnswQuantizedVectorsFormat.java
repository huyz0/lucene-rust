/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import java.io.IOException;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.lucenerust.opensearch.knn.KnnFieldConfig;

/**
 * The Rust engine's quantized HNSW format, as the JVM reads it: a field's files are {@code
 * Lucene99HnswVectorsFormat}'s (raw vectors and the graph, built on them), and the quantized codes
 * the Rust writer puts beside them ({@code .vqm}, {@code .vqd}) are walked by the Rust engine only.
 * Here a search walks the graph on the raw vectors, so Lucene's own paths return exact scores, and a
 * segment this side writes has the graph alone, which the Rust engine searches on the raw vectors.
 */
public final class LuceneRustHnswQuantizedVectorsFormat extends KnnVectorsFormat {
    public static final String NAME = "LuceneRustHnswQuantizedVectorsFormat";

    private final Lucene99HnswVectorsFormat delegate;

    /** The SPI instance: a reader takes the graph parameters from the files. */
    public LuceneRustHnswQuantizedVectorsFormat() {
        this(KnnFieldConfig.DEFAULT_M, KnnFieldConfig.DEFAULT_EF_CONSTRUCTION);
    }

    public LuceneRustHnswQuantizedVectorsFormat(int maxConn, int beamWidth) {
        super(NAME);
        this.delegate = new Lucene99HnswVectorsFormat(maxConn, beamWidth);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return delegate.fieldsWriter(state);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return delegate.fieldsReader(state);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KnnFieldConfig.MAX_DIMENSION;
    }

    @Override
    public String toString() {
        return NAME + "(delegate=" + delegate + ")";
    }
}
