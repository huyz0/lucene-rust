/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.FilterCodec;

/**
 * The codec the segments of an {@code index.sparse} index record, which neural-search names
 * {@code Sparse10010Codec}: Lucene 10.1's formats on read, each field's postings and doc values
 * format resolved from its attributes. The Rust engine writes these segments, with the clustered
 * postings ({@code .sit}/{@code .sip}) it searches; this side only opens them.
 */
public final class SparseCodec extends FilterCodec {
    public static final String NAME = "Sparse10010Codec";

    public SparseCodec() {
        super(NAME, new Lucene101Codec());
    }
}
