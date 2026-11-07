/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene104Codec}. Its
 * files are Lucene104Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN1040Codec extends FilterCodec {
    public KNN1040Codec() {
        super("KNN1040Codec", new Lucene104Codec());
    }
}
