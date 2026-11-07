/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene99Codec}. Its
 * files are Lucene99Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN990Codec extends FilterCodec {
    public KNN990Codec() {
        super("KNN990Codec", new Lucene99Codec());
    }
}
