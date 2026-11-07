/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.storedfields;

import java.io.IOException;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.codecs.lucene90.compressing.Lucene90CompressingStoredFieldsFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;

/**
 * The Rust engine's paged zstd stored fields ({@code Mode::ZstdPaged} in {@code stored_fields.rs}):
 * {@code Lucene90CompressingStoredFieldsFormat}'s three files under the data codec name {@value
 * #FORMAT_NAME}, with 16 KiB chunks of one zstd frame each, compressed against an optional segment
 * dictionary that the {@code .fdt} holds right after its header ({@code vint length, bytes}).
 *
 * <p>A segment records {@value #MODE_KEY} = {@value #MODE}; one without it (which Java wrote) holds
 * {@code Lucene90StoredFieldsFormat} stored fields, and Java writes those.
 */
public final class ZstdPagedStoredFieldsFormat extends StoredFieldsFormat {
    /** {@code ZSTD_PAGED_ATTRIBUTE} in {@code segment_writer.rs}. */
    public static final String MODE_KEY = "LuceneRustStoredFieldsFormat.mode";
    /** {@code ZSTD_PAGED_VALUE} in {@code segment_writer.rs}. */
    public static final String MODE = "ZSTD_PAGED";
    /** {@code DATA_CODEC_ZSTD_PAGED} in {@code stored_fields.rs}. */
    public static final String FORMAT_NAME = "RustZstdPagedStoredFieldsData";
    /** {@code ZSTD_PAGED_CHUNK_SIZE}, {@code ZSTD_PAGED_MAX_DOCS_PER_CHUNK} and the block shift. */
    static final int CHUNK_SIZE = 16 * 1024;
    static final int MAX_DOCS_PER_CHUNK = 1024;
    static final int BLOCK_SHIFT = 10;
    /** {@code ZSTD_PAGED_MAX_DICTIONARY} in {@code stored_fields.rs}. */
    static final int MAX_DICTIONARY = 1 << 20;
    private static final int VERSION = 1;

    private final Lucene90StoredFieldsFormat lucene90 = new Lucene90StoredFieldsFormat();

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo si, FieldInfos fn, IOContext context) throws IOException {
        if (MODE.equals(si.getAttribute(MODE_KEY)) == false) {
            return lucene90.fieldsReader(directory, si, fn, context);
        }
        byte[] dictionary = readDictionary(directory, si);
        ZstdDictionary dict = dictionary.length == 0 ? null : ZstdDictionary.open(dictionary);
        boolean success = false;
        try {
            StoredFieldsReader reader = new Lucene90CompressingStoredFieldsFormat(
                FORMAT_NAME,
                new ZstdPagedCompressionMode(dict == null ? 0 : dict.handle()),
                CHUNK_SIZE,
                MAX_DOCS_PER_CHUNK,
                BLOCK_SHIFT
            ).fieldsReader(directory, si, fn, context);
            success = true;
            return dict == null ? reader : new DictionaryStoredFieldsReader(reader, dict);
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(dict);
            }
        }
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo si, IOContext context) throws IOException {
        return lucene90.fieldsWriter(directory, si, context);
    }

    /** The {@code .fdt}'s dictionary, empty when the segment has none. */
    static byte[] readDictionary(Directory directory, SegmentInfo si) throws IOException {
        String name = IndexFileNames.segmentFileName(si.name, "", "fdt");
        try (IndexInput in = directory.openInput(name, IOContext.READONCE)) {
            CodecUtil.checkIndexHeader(in, FORMAT_NAME, VERSION, VERSION, si.getId(), "");
            int length = in.readVInt();
            if (length < 0 || length > MAX_DICTIONARY) {
                throw new CorruptIndexException("invalid zstd dictionary length " + length, in);
            }
            byte[] dictionary = new byte[length];
            in.readBytes(dictionary, 0, length);
            return dictionary;
        }
    }
}
