/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.storedfields;

import java.io.IOException;
import org.apache.lucene.codecs.compressing.CompressionMode;
import org.apache.lucene.codecs.compressing.Compressor;
import org.apache.lucene.codecs.compressing.Decompressor;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeBridge;

/**
 * One segment's paged zstd units ({@code Compression::ZstdPaged} in {@code compression.rs}): a vint
 * compressed length, then one zstd frame, compressed against the segment's dictionary when it has
 * one. Only the Rust engine writes them.
 */
final class ZstdPagedCompressionMode extends CompressionMode {
    /** The dictionary's native handle, 0 for none. */
    private final long dictionary;

    ZstdPagedCompressionMode(long dictionary) {
        this.dictionary = dictionary;
    }

    @Override
    public Compressor newCompressor() {
        throw new UnsupportedOperationException("only the Rust engine writes paged zstd stored fields");
    }

    @Override
    public Decompressor newDecompressor() {
        return new PagedDecompressor(dictionary);
    }

    @Override
    public String toString() {
        return "ZSTD_PAGED";
    }

    private static final class PagedDecompressor extends Decompressor {
        private final long dictionary;
        private byte[] compressed = BytesRef.EMPTY_BYTES;

        PagedDecompressor(long dictionary) {
            this.dictionary = dictionary;
        }

        @Override
        public void decompress(DataInput in, int originalLength, int offset, int length, BytesRef bytes) throws IOException {
            // The Rust writer writes no unit for a chunk of empty documents.
            if (originalLength == 0) {
                bytes.offset = 0;
                bytes.length = 0;
                return;
            }
            int n = in.readVInt();
            if (n < 0) {
                throw new CorruptIndexException("negative zstd frame length " + n, in);
            }
            compressed = ArrayUtil.growNoCopy(compressed, n);
            in.readBytes(compressed, 0, n);
            bytes.bytes = ArrayUtil.growNoCopy(bytes.bytes, originalLength);
            int rc = NativeBridge.zstdPagedDecompress(dictionary, compressed, n, bytes.bytes, originalLength);
            if (rc != NativeBridge.OK) {
                throw new CorruptIndexException("paged zstd stored fields: " + NativeBridge.lastError(), in);
            }
            bytes.offset = offset;
            bytes.length = length;
        }

        @Override
        public Decompressor clone() {
            return new PagedDecompressor(dictionary);
        }
    }
}
