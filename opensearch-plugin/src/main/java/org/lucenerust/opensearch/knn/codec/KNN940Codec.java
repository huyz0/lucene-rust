/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene94.Lucene94Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene94Codec}. Its
 * files are Lucene94Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN940Codec extends FilterCodec {
    public KNN940Codec() {
        super("KNN940Codec", new Lucene94Codec());
    }
}
