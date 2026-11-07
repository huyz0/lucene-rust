/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.StoredFieldsFormat;

/**
 * A codec name the k-NN plugin records in the segments it writes: its delegate's files, each
 * vector field's format resolved from the field's attributes, the k-NN compound format, and its
 * derived source ({@link DerivedSourceStoredFieldsFormat}).
 */
abstract class KnnFilterCodec extends FilterCodec {
    private final CompoundFormat compound;
    private final StoredFieldsFormat storedFields;

    KnnFilterCodec(String name, Codec delegate) {
        super(name, delegate);
        this.compound = new KnnCompoundFormat(delegate.compoundFormat());
        this.storedFields = new DerivedSourceStoredFieldsFormat(this, delegate);
    }

    @Override
    public final StoredFieldsFormat storedFieldsFormat() {
        return storedFields;
    }

    @Override
    public final CompoundFormat compoundFormat() {
        return compound;
    }
}
