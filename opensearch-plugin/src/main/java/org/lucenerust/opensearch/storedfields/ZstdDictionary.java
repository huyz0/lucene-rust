/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.storedfields;

import java.io.Closeable;
import java.io.IOException;
import org.lucenerust.opensearch.NativeBridge;

/** A segment's zstd dictionary, prepared once in native memory for decoding its chunks. */
final class ZstdDictionary implements Closeable {
    private final long handle;
    private boolean closed;

    private ZstdDictionary(long handle) {
        this.handle = handle;
    }

    static ZstdDictionary open(byte[] dictionary) throws IOException {
        long[] out = new long[1];
        int rc = NativeBridge.zstdDictionaryOpen(dictionary, out);
        if (rc != NativeBridge.OK) {
            throw new IOException("cannot prepare a zstd dictionary: " + NativeBridge.lastError());
        }
        return new ZstdDictionary(out[0]);
    }

    long handle() {
        return handle;
    }

    @Override
    public synchronized void close() {
        if (closed == false) {
            closed = true;
            NativeBridge.zstdDictionaryClose(handle);
        }
    }
}
