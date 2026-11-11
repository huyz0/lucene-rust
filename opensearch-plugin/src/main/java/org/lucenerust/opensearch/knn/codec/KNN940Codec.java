/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene94.Lucene94Codec;

/** k-NN's codec name over {@code Lucene94Codec}. */
public final class KNN940Codec extends KnnFilterCodec {
    public KNN940Codec() {
        super("KNN940Codec", new Lucene94Codec());
    }
}
