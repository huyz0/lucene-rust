/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene103.Lucene103Codec;

/** k-NN's codec name over {@code Lucene103Codec}. */
public final class KNN1030Codec extends KnnFilterCodec {
    public KNN1030Codec() {
        super("KNN1030Codec", new Lucene103Codec());
    }
}
