/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;

/** k-NN's codec name over {@code Lucene91Codec}. */
public final class KNN910Codec extends KnnFilterCodec {
    public KNN910Codec() {
        super("KNN910Codec", new Lucene91Codec());
    }
}
