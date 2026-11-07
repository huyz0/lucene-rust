/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene103.Lucene103Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene103Codec}. Its
 * files are Lucene103Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN1030Codec extends FilterCodec {
    public KNN1030Codec() {
        super("KNN1030Codec", new Lucene103Codec());
    }
}
