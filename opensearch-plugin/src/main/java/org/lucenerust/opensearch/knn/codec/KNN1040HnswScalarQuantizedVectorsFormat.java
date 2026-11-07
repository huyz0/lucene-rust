/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

/**
 * The k-NN plugin's name for {@code Lucene104HnswScalarQuantizedVectorsFormat}, which its lucene
 * engine writes for {@code sq} encoders and compression levels: the same files under another
 * format name.
 */
public final class KNN1040HnswScalarQuantizedVectorsFormat extends Lucene104HnswScalarQuantizedVectorsFormat {
    public static final String NAME = "KNN1040HnswScalarQuantizedVectorsFormat";

    /** The default instance, which the Lucene SPI reads a segment's fields with. */
    public KNN1040HnswScalarQuantizedVectorsFormat() {}

    public KNN1040HnswScalarQuantizedVectorsFormat(ScalarEncoding encoding, int maxConn, int beamWidth) {
        super(encoding, maxConn, beamWidth);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
