/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene912.Lucene912Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene912Codec}. Its
 * files are Lucene912Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN9120Codec extends FilterCodec {
    public KNN9120Codec() {
        super("KNN9120Codec", new Lucene912Codec());
    }
}
