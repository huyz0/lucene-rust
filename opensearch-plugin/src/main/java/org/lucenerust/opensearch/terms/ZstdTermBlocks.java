/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.terms;

import java.io.IOException;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;

/**
 * Reads one zstd term block ({@code blocktree_writer::BlockZstd} in {@code lucene-codecs}): after
 * the block's token ({@code vLong(suffixBytes << 3 | isLeaf << 2 | 3)}) come {@code
 * vInt(plainLength)}, {@code vInt(frameLength)} and a bare zstd frame of {@code plainLength} bytes,
 * which hold the raw suffix bytes and then the suffix-lengths, stats and metadata sections exactly
 * as an ordinary block writes them. {@link #load} decodes the frame and returns those plain bytes
 * for the frame to read on; {@link #finish} checks it read all of them.
 */
final class ZstdTermBlocks {
    /** The block token's compression code for a zstd block; stock Lucene has no algorithm with it. */
    static final int CODE = 3;

    /**
     * The most bytes per entry the suffix-lengths, stats and metadata sections can take together,
     * so a corrupt plain length cannot buy a large allocation ({@code ZSTD_SECTION_BYTES_PER_ENTRY}
     * in {@code blocktree.rs}).
     */
    private static final int SECTION_BYTES_PER_ENTRY = 256;

    private byte[] frame = new byte[64];
    private byte[] plain = new byte[64];
    private final ByteArrayDataInput plainIn = new ByteArrayDataInput();

    /** Decodes the zstd block whose token {@code in} has just read; the plain bytes to read on. */
    DataInput load(IndexInput in, ZstdTermsDictionary dictionary, int numSuffixBytes, int entCount) throws IOException {
        if (dictionary == null) {
            throw new CorruptIndexException("a zstd term block in a dictionary without them", in);
        }
        int plainLength = in.readVInt();
        int frameLength = in.readVInt();
        long maxPlain = numSuffixBytes + (long) entCount * SECTION_BYTES_PER_ENTRY;
        if (plainLength < numSuffixBytes || plainLength > maxPlain) {
            throw new CorruptIndexException(
                "zstd term block of " + plainLength + " bytes for " + numSuffixBytes + " suffix bytes and " + entCount + " entries",
                in
            );
        }
        if (frameLength < 0 || frameLength > in.length() - in.getFilePointer()) {
            throw new CorruptIndexException("zstd term block frame of " + frameLength + " bytes", in);
        }
        if (frame.length < frameLength) {
            frame = new byte[ArrayUtil.oversize(frameLength, 1)];
        }
        if (plain.length < plainLength) {
            plain = new byte[ArrayUtil.oversize(plainLength, 1)];
        }
        in.readBytes(frame, 0, frameLength);
        dictionary.decompress(frame, frameLength, plain, plainLength, in);
        plainIn.reset(plain, 0, plainLength);
        return plainIn;
    }

    /** Checks the frame read every plain byte of the block {@link #load} decoded. */
    void finish() throws IOException {
        if (plainIn.eof() == false) {
            throw new CorruptIndexException("bytes left after a zstd term block's sections", "zstd term block");
        }
    }
}
