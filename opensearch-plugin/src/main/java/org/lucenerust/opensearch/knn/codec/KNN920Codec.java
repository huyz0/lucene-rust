/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene92Codec}. Its
 * files are Lucene92Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN920Codec extends FilterCodec {
    public KNN920Codec() {
        super("KNN920Codec", new Lucene92Codec());
    }
}
