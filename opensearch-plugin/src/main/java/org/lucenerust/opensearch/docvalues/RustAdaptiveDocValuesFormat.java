/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.docvalues;

import java.io.IOException;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * The {@code RustAdaptive} doc-values format the Rust engine writes under {@code
 * index.rust.doc_values.adaptive}: {@code Lucene90DocValuesFormat}'s files, in which a NUMERIC or
 * SORTED_NUMERIC field's values may be stored as adaptive blocks (delta, offset, GCD and decimal
 * transforms, then bit-packing; {@code lucene-codecs/src/adaptive_values.rs}). Every other field,
 * the skip indexes and the checksums are Lucene90's own, read by Lucene90's producer.
 *
 * <p>Read-only: only the native engine writes it. A Java writer (a merge of these segments outside
 * the Rust engine) writes its own codec's formats, so merging rewrites the fields stock-readable.
 */
public final class RustAdaptiveDocValuesFormat extends DocValuesFormat {
    /** {@code Lucene90DocValuesFormat::RUST_ADAPTIVE_NAME} in {@code doc_values.rs}. */
    public static final String NAME = "RustAdaptive";

    private final Lucene90DocValuesFormat lucene90 = new Lucene90DocValuesFormat();

    public RustAdaptiveDocValuesFormat() {
        super(NAME);
    }

    @Override
    public DocValuesConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
        throw new UnsupportedOperationException("the RustAdaptive doc-values format is written by the native engine only");
    }

    @Override
    public DocValuesProducer fieldsProducer(SegmentReadState state) throws IOException {
        DocValuesProducer delegate = lucene90.fieldsProducer(state);
        try {
            return new RustAdaptiveDocValuesProducer(state, delegate);
        } catch (Throwable t) {
            org.apache.lucene.util.IOUtils.closeWhileSuppressingExceptions(t, delegate);
            throw t;
        }
    }
}
