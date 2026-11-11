/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.terms;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.IndexInput;
import org.lucenerust.opensearch.NativeBridge;

/**
 * A field's zstd dictionary for its term blocks, prepared once in native memory; a field without
 * one decodes its blocks without a dictionary. Its {@code .tmd} record ends with it: {@code
 * vInt(length)} then the bytes, length 0 for none.
 */
final class ZstdTermsDictionary implements Closeable {
    private final long handle;
    private boolean closed;

    private ZstdTermsDictionary(long handle) {
        this.handle = handle;
    }

    /** The dictionary at the end of a field's {@code .tmd} record. */
    static ZstdTermsDictionary read(IndexInput meta) throws IOException {
        int length = meta.readVInt();
        // Bounded by what is left before the checksum is verified, so a corrupt length cannot buy a
        // large allocation.
        if (length < 0 || length > meta.length() - meta.getFilePointer()) {
            throw new CorruptIndexException("invalid zstd terms dictionary length: " + length, meta.toString());
        }
        if (length == 0) {
            return new ZstdTermsDictionary(0);
        }
        byte[] bytes = new byte[length];
        meta.readBytes(bytes, 0, length);
        long[] out = new long[1];
        int rc = NativeBridge.zstdTermsDictionaryOpen(bytes, out);
        if (rc != NativeBridge.OK) {
            throw new IOException("cannot prepare a zstd terms dictionary: " + NativeBridge.lastError());
        }
        return new ZstdTermsDictionary(out[0]);
    }

    /** Decodes {@code frame[0..frameLength)} into exactly {@code plain[0..plainLength)}. */
    void decompress(byte[] frame, int frameLength, byte[] plain, int plainLength, IndexInput in) throws IOException {
        int rc = NativeBridge.zstdTermsDecompress(handle, frame, frameLength, plain, plainLength);
        if (rc != NativeBridge.OK) {
            throw new CorruptIndexException(NativeBridge.lastError(), in);
        }
    }

    @Override
    public synchronized void close() {
        if (closed == false) {
            closed = true;
            if (handle != 0) {
                NativeBridge.zstdTermsDictionaryClose(handle);
            }
        }
    }
}
