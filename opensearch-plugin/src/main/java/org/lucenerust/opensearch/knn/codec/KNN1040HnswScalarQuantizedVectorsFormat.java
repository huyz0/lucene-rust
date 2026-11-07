/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;

/**
 * The k-NN plugin's name for {@code Lucene104HnswScalarQuantizedVectorsFormat}, which its lucene
 * engine writes for {@code sq} encoders and compression levels: the same files under another
 * format name.
 */
public final class KNN1040HnswScalarQuantizedVectorsFormat extends Lucene104HnswScalarQuantizedVectorsFormat {
    public static final String NAME = "KNN1040HnswScalarQuantizedVectorsFormat";

    public KNN1040HnswScalarQuantizedVectorsFormat() {
        super();
    }

    @Override
    public String getName() {
        return NAME;
    }
}
