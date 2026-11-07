/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene95.Lucene95Codec;

/** k-NN's codec name over {@code Lucene95Codec}. */
public final class KNN950Codec extends KnnFilterCodec {
    public KNN950Codec() {
        super("KNN950Codec", new Lucene95Codec());
    }
}
