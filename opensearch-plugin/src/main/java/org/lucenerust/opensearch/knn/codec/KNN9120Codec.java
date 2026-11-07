/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene912.Lucene912Codec;

/** k-NN's codec name over {@code Lucene912Codec}. */
public final class KNN9120Codec extends KnnFilterCodec {
    public KNN9120Codec() {
        super("KNN9120Codec", new Lucene912Codec());
    }
}
