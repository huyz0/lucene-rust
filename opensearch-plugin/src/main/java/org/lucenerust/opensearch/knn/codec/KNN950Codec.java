/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene95Codec}. Its
 * files are Lucene95Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN950Codec extends FilterCodec {
    public KNN950Codec() {
        super("KNN950Codec", new Lucene95Codec());
    }
}
