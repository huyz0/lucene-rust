/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.FilterCodec;

/**
 * A codec name the k-NN plugin records in the segments it writes: its delegate's files, each
 * vector field's format resolved from the field's attributes, and the k-NN compound format.
 */
abstract class KnnFilterCodec extends FilterCodec {
    private final CompoundFormat compound;

    KnnFilterCodec(String name, Codec delegate) {
        super(name, delegate);
        this.compound = new KnnCompoundFormat(delegate.compoundFormat());
    }

    @Override
    public final CompoundFormat compoundFormat() {
        return compound;
    }
}
