/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.lucene104.Lucene104Codec;

/** k-NN's codec name over {@code Lucene104Codec}. */
public final class KNN1040Codec extends KnnFilterCodec {
    public KNN1040Codec() {
        super("KNN1040Codec", new Lucene104Codec());
    }
}
