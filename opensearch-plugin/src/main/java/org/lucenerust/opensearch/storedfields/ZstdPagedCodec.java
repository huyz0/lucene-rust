/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.storedfields;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;

/**
 * The codec the segments of an index with {@code index.codec.stored_fields.page_aligned} record
 * ({@code ZSTD_PAGED_CODEC} in {@code codecs.rs}): {@code Lucene104}'s formats, with the Rust
 * engine's paged zstd stored fields. The Rust engine writes these segments; this side opens them.
 */
public final class ZstdPagedCodec extends FilterCodec {
    public static final String NAME = "PagedZstd104";

    private final StoredFieldsFormat storedFields = new ZstdPagedStoredFieldsFormat();

    public ZstdPagedCodec() {
        super(NAME, new Lucene104Codec());
    }

    @Override
    public StoredFieldsFormat storedFieldsFormat() {
        return storedFields;
    }
}
