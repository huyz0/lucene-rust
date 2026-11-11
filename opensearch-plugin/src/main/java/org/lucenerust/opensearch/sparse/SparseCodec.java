/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec the segments of an {@code index.sparse} index record, which neural-search names
 * {@code Sparse10010Codec}: Lucene 10.1's formats on read, each field's postings and doc values
 * format resolved from its attributes. Every segment carries the clustered postings
 * ({@code .sit}/{@code .sip}) of its {@code sparse_vector} fields at or over their
 * {@code approximate_threshold}: the Rust engine writes them itself, and on OpenSearch's engine
 * {@link SparseCodecService} writes with this codec over the index's own, which clusters the fields
 * in Rust as their doc values are flushed and merged ({@link ClusteringDocValuesFormat}).
 */
public final class SparseCodec extends FilterCodec {
    public static final String NAME = "Sparse10010Codec";

    private final DocValuesFormat docValues;

    /** The codec segments are read with. */
    public SparseCodec() {
        super(NAME, new Lucene101Codec());
        this.docValues = delegate.docValuesFormat();
    }

    /** The codec OpenSearch's engine writes with: {@code delegate}'s formats, the fields clustered. */
    public SparseCodec(Codec delegate) {
        super(NAME, delegate);
        this.docValues = new ClusteringDocValuesFormat(delegate.docValuesFormat());
    }

    /** The codec this one writes and reads the segments' formats with. */
    public Codec delegate() {
        return delegate;
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return docValues;
    }
}
