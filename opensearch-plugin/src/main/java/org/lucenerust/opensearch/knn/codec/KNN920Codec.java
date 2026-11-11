/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene92.Lucene92Codec;

/** k-NN's codec name over {@code Lucene92Codec}. */
public final class KNN920Codec extends KnnFilterCodec {
    public KNN920Codec() {
        super("KNN920Codec", new Lucene92Codec());
    }
}
