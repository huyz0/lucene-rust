/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;

/**
 * The k-NN plugin's name for {@code Lucene104ScalarQuantizedVectorsFormat}, the flat quantized
 * vectors under {@link KNN1040HnswScalarQuantizedVectorsFormat}'s graph: the same files under
 * another format name.
 */
public final class KNN1040ScalarQuantizedVectorsFormat extends Lucene104ScalarQuantizedVectorsFormat {
    public static final String NAME = "KNN1040ScalarQuantizedVectorsFormat";

    public KNN1040ScalarQuantizedVectorsFormat() {
        super();
    }

    @Override
    public String getName() {
        return NAME;
    }
}
