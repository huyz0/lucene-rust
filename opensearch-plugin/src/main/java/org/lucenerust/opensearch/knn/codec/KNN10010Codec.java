/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene101Codec}. Its
 * files are Lucene101Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN10010Codec extends FilterCodec {
    public KNN10010Codec() {
        super("KNN10010Codec", new Lucene101Codec());
    }
}
