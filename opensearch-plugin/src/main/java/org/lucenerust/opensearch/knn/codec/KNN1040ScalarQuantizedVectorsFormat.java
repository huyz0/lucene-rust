/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

/**
 * The k-NN plugin's name for {@code Lucene104ScalarQuantizedVectorsFormat}, the flat quantized
 * vectors under {@link KNN1040HnswScalarQuantizedVectorsFormat}'s graph: the same files under
 * another format name.
 */
public final class KNN1040ScalarQuantizedVectorsFormat extends Lucene104ScalarQuantizedVectorsFormat {
    public static final String NAME = "KNN1040ScalarQuantizedVectorsFormat";

    /** The default instance, which the Lucene SPI reads a segment's fields with. */
    public KNN1040ScalarQuantizedVectorsFormat() {}

    public KNN1040ScalarQuantizedVectorsFormat(ScalarEncoding encoding) {
        super(encoding);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
