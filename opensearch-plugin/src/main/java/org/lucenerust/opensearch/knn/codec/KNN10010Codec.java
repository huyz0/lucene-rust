/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;

/** k-NN's codec name over {@code Lucene101Codec}. */
public final class KNN10010Codec extends KnnFilterCodec {
    public KNN10010Codec() {
        super("KNN10010Codec", new Lucene101Codec());
    }
}
