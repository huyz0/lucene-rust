/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.util.zip.CRC32;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

import org.apache.lucene.store.IndexOutput;

/**
 * Writes one file in the LRSE format ({@link Lrse}) over the underlying directory's output:
 * plaintext is buffered a chunk at a time, and a full chunk is sealed once more bytes arrive (so the
 * final chunk, sealed on {@link #close}, is the only one that may be short). The file pointer and
 * checksum are the plaintext's, as Lucene's codecs expect.
 */
final class EncryptingIndexOutput extends IndexOutput {
    private final IndexOutput out;
    private final StoreCrypto crypto;
    private final Cipher cipher;
    private final SecretKeySpec key;
    private final byte[] aad;
    private final byte[] buf;
    private final byte[] nonce = new byte[Lrse.NONCE_LEN];
    private final CRC32 crc = new CRC32();
    private int used;
    private long index;
    private long written;
    private boolean closed;

    EncryptingIndexOutput(IndexOutput out, String name, KeySource.DataKey dataKey, int chunkSize, StoreCrypto crypto)
        throws IOException {
        super("EncryptingIndexOutput(" + out + ")", name);
        this.out = out;
        this.crypto = crypto;
        this.cipher = crypto.newCipher();
        this.key = new SecretKeySpec(dataKey.raw(), "AES");
        Lrse.Header header = Lrse.Header.of(chunkSize, dataKey.keyId(), dataKey.wrapped());
        this.aad = Lrse.aad(header.bytes(), 0, false);
        this.buf = new byte[chunkSize];
        out.writeBytes(header.bytes(), header.length());
    }

    @Override
    public void writeByte(byte b) throws IOException {
        if (used == buf.length) {
            seal(false);
        }
        buf[used++] = b;
        crc.update(b);
        written++;
    }

    @Override
    public void writeBytes(byte[] b, int offset, int length) throws IOException {
        crc.update(b, offset, length);
        written += length;
        while (length > 0) {
            if (used == buf.length) {
                seal(false);
            }
            int n = Math.min(length, buf.length - used);
            System.arraycopy(b, offset, buf, used, n);
            used += n;
            offset += n;
            length -= n;
        }
    }

    private void seal(boolean last) throws IOException {
        if (index > 0xFFFF_FFFFL) {
            throw new IOException("encrypted file " + getName() + " exceeds 2^32 chunks");
        }
        Lrse.setAad(aad, index, last);
        byte[] sealed = crypto.seal(cipher, key, aad, buf, 0, used, nonce);
        out.writeBytes(nonce, nonce.length);
        out.writeBytes(sealed, sealed.length);
        index++;
        used = 0;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        try {
            seal(true);
        } finally {
            out.close();
        }
    }

    @Override
    public long getFilePointer() {
        return written;
    }

    @Override
    public long getChecksum() {
        return crc.getValue();
    }
}
