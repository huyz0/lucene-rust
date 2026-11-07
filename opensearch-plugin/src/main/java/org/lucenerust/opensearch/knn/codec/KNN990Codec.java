/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99Codec;

/** k-NN's codec name over {@code Lucene99Codec}. */
public final class KNN990Codec extends KnnFilterCodec {
    public KNN990Codec() {
        super("KNN990Codec", new Lucene99Codec());
    }
}
