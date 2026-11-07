/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.terms;

import java.io.IOException;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.PostingsReaderBase;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104PostingsReader;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;

/**
 * The {@code RustZstd104} postings format the Rust engine writes under {@code
 * index.lucene_rust.zstd_terms} ({@code Lucene104PostingsFormat::rust_zstd} in {@code
 * per_field_postings.rs}): {@code Lucene104PostingsFormat}'s postings under a block-tree terms
 * dictionary whose blocks may be zstd-compressed, a large field's against a dictionary trained on
 * its own blocks. Its files carry their own header names, so stock Lucene refuses them rather than
 * misreading a zstd block; {@link RustZstdBlockTreeTermsReader} reads them.
 *
 * <p>Java writes it as plain {@code Lucene104} files (zstd blocks are optional), which this format
 * reads with Lucene's own reader. A Java merge writes its own codec's formats, so it rewrites the
 * fields stock-readable.
 */
public final class RustZstd104PostingsFormat extends PostingsFormat {
    /** {@code RUST_ZSTD_FORMAT_NAME} in {@code per_field_postings.rs}. */
    public static final String NAME = "RustZstd104";

    private final Lucene104PostingsFormat lucene104 = new Lucene104PostingsFormat();

    public RustZstd104PostingsFormat() {
        super(NAME);
    }

    @Override
    public FieldsConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
        return lucene104.fieldsConsumer(state);
    }

    @Override
    public FieldsProducer fieldsProducer(SegmentReadState state) throws IOException {
        if (zstdTerms(state) == false) {
            return lucene104.fieldsProducer(state);
        }
        PostingsReaderBase postings = new Lucene104PostingsReader(state);
        try {
            return new RustZstdBlockTreeTermsReader(postings, state);
        } catch (Throwable t) {
            IOUtils.closeWhileSuppressingExceptions(t, postings);
            throw t;
        }
    }

    /** Whether the segment's {@code .tim} carries the zstd terms dictionary's header. */
    private static boolean zstdTerms(SegmentReadState state) throws IOException {
        String name = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            RustZstdBlockTreeTermsReader.TERMS_EXTENSION
        );
        try (IndexInput in = state.directory.openInput(name, IOContext.READONCE)) {
            if (CodecUtil.readBEInt(in) != CodecUtil.CODEC_MAGIC) {
                return false;
            }
            return RustZstdBlockTreeTermsReader.TERMS_CODEC_NAME.equals(in.readString());
        }
    }
}
