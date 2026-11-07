/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.backward_codecs.lucene91.Lucene91Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec name the k-NN plugin records in the segments it writes over {@code Lucene91Codec}. Its
 * files are Lucene91Codec's; each vector field's format is resolved from the field's attributes.
 */
public final class KNN910Codec extends FilterCodec {
    public KNN910Codec() {
        super("KNN910Codec", new Lucene91Codec());
    }
}
