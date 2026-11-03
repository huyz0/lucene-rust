/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.EOFException;
import java.io.IOException;
import java.security.GeneralSecurityException;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.IndexInput;

/**
 * Reads one LRSE file ({@link Lrse}) with random access: the chunk holding the position is read
 * from the underlying input, authenticated and decrypted into this input's buffer. Clones and slices
 * share the underlying input and key, and each decrypts into its own buffer. A chunk that fails
 * authentication -- a changed header, a reordered, dropped or truncated chunk, the wrong key -- is a
 * {@link CorruptIndexException}.
 */
final class DecryptingIndexInput extends IndexInput {
    /** Shared by an input, its clones and slices. */
    private record File(
        IndexInput raw,
        String name,
        int headerLen,
        int chunkSize,
        long chunks,
        long plainLen,
        byte[] aadTemplate,
        SecretKeySpec key,
        StoreCrypto crypto
    ) {}

    private final File file;
    /** This input's window of the plaintext: [offset, offset + length). */
    private final long offset;
    private final long length;
    private final boolean isClone;
    private IndexInput raw;
    private Cipher cipher;
    private byte[] aad;
    private byte[] sealed;
    private byte[] plain;
    private byte[] nonce;
    /** Which chunk {@code plain} holds, or -1; and its plaintext length. */
    private long bufChunk = -1;
    private int bufLen;
    /** Position within this input's window. */
    private long pos;

    private DecryptingIndexInput(String desc, File file, long offset, long length, boolean isClone) {
        super(desc);
        this.file = file;
        this.offset = offset;
        this.length = length;
        this.isClone = isClone;
        this.raw = isClone ? file.raw.clone() : file.raw;
    }

    /** Opens {@code raw} (a whole LRSE file named {@code name}), unwrapping its key with {@code keys}. */
    static DecryptingIndexInput open(IndexInput raw, String name, KeySource keys, StoreCrypto crypto) throws IOException {
        boolean ok = false;
        try {
            raw.seek(0);
            Lrse.Header h = Lrse.Header.read(raw, name);
            long body = raw.length() - h.length();
            long plainLen = Lrse.plaintextLength(body, h.chunkSize());
            if (plainLen < 0) {
                throw new CorruptIndexException("encrypted body of " + body + " bytes is truncated", name);
            }
            byte[] rawKey = keys.unwrap(h.keyId(), h.wrappedKey());
            File f = new File(
                raw,
                name,
                h.length(),
                h.chunkSize(),
                Lrse.chunkCount(body, h.chunkSize()),
                plainLen,
                Lrse.aad(h.bytes(), 0, false),
                new SecretKeySpec(rawKey, "AES"),
                crypto
            );
            DecryptingIndexInput in = new DecryptingIndexInput("DecryptingIndexInput(" + raw + ")", f, 0, plainLen, false);
            ok = true;
            return in;
        } finally {
            if (ok == false) {
                raw.close();
            }
        }
    }

    /** The plaintext length of an LRSE file whose raw input is {@code raw}, from its header alone. */
    static long plaintextLength(IndexInput raw, String name) throws IOException {
        Lrse.Header h = Lrse.Header.read(raw, name);
        long len = Lrse.plaintextLength(raw.length() - h.length(), h.chunkSize());
        if (len < 0) {
            throw new CorruptIndexException("encrypted body of " + (raw.length() - h.length()) + " bytes is truncated", name);
        }
        return len;
    }

    /**
     * The plaintext length of {@code raw} as {@link #plaintextLength} gives it, or, for a file that
     * ends inside its header or inside a chunk (one still being written), {@code -n - 1} where
     * {@code n} is the plaintext of its complete chunks. A file that is not LRSE still fails.
     */
    static long plaintextLengthSoFar(IndexInput raw, String name) throws IOException {
        Lrse.Header h;
        try {
            h = Lrse.Header.read(raw, name);
        } catch (EOFException e) {
            return -1;
        }
        long body = raw.length() - h.length();
        long len = Lrse.plaintextLength(body, h.chunkSize());
        if (len >= 0) {
            return len;
        }
        long complete = body / ((long) h.chunkSize() + Lrse.CHUNK_OVERHEAD) * h.chunkSize();
        return -complete - 1;
    }

    private void load(long chunk) throws IOException {
        if (cipher == null) {
            cipher = file.crypto.newCipher();
            aad = file.aadTemplate.clone();
            nonce = new byte[Lrse.NONCE_LEN];
            sealed = new byte[file.chunkSize + Lrse.TAG_LEN];
            plain = new byte[file.chunkSize];
        }
        long per = (long) file.chunkSize + Lrse.CHUNK_OVERHEAD;
        boolean last = chunk == file.chunks - 1;
        long start = file.headerLen + chunk * per;
        int sealedLen = (int) Math.min(per, raw.length() - start);
        raw.seek(start);
        raw.readBytes(nonce, 0, Lrse.NONCE_LEN);
        int ctAndTag = sealedLen - Lrse.NONCE_LEN;
        raw.readBytes(sealed, 0, ctAndTag);
        Lrse.setAad(aad, chunk, last);
        try {
            bufLen = file.crypto.open(cipher, file.key, nonce, aad, sealed, 0, ctAndTag, plain, 0);
        } catch (GeneralSecurityException e) {
            bufChunk = -1;
            throw new CorruptIndexException("chunk " + chunk + " fails authentication (wrong key, or a changed file)", file.name, e);
        }
        if (last == false && bufLen != file.chunkSize) {
            throw new CorruptIndexException("chunk " + chunk + " is short", file.name);
        }
        bufChunk = chunk;
    }

    @Override
    public byte readByte() throws IOException {
        if (pos >= length) {
            throw new EOFException("read past EOF: " + this);
        }
        long abs = offset + pos;
        long chunk = abs / file.chunkSize;
        if (chunk != bufChunk) {
            load(chunk);
        }
        byte b = plain[(int) (abs - chunk * file.chunkSize)];
        pos++;
        return b;
    }

    @Override
    public void readBytes(byte[] b, int off, int len) throws IOException {
        if (len > length - pos) {
            throw new EOFException("read past EOF: " + this + " (pos " + pos + " + " + len + " > " + length + ")");
        }
        while (len > 0) {
            long abs = offset + pos;
            long chunk = abs / file.chunkSize;
            if (chunk != bufChunk) {
                load(chunk);
            }
            int at = (int) (abs - chunk * file.chunkSize);
            int n = Math.min(len, bufLen - at);
            if (n <= 0) {
                throw new EOFException("read past EOF: " + this);
            }
            System.arraycopy(plain, at, b, off, n);
            pos += n;
            off += n;
            len -= n;
        }
    }

    @Override
    public long getFilePointer() {
        return pos;
    }

    @Override
    public void seek(long p) throws IOException {
        if (p < 0 || p > length) {
            throw new EOFException("seek to " + p + " outside " + this + " of length " + length);
        }
        pos = p;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public IndexInput slice(String sliceDescription, long off, long len) throws IOException {
        if (off < 0 || len < 0 || off + len > length) {
            throw new IllegalArgumentException("slice(" + sliceDescription + ", " + off + ", " + len + ") out of bounds: " + this);
        }
        return new DecryptingIndexInput(getFullSliceDescription(sliceDescription), file, offset + off, len, true);
    }

    @Override
    public DecryptingIndexInput clone() {
        DecryptingIndexInput c = new DecryptingIndexInput(toString(), file, offset, length, true);
        c.pos = pos;
        return c;
    }

    @Override
    public void close() throws IOException {
        if (isClone == false) {
            file.raw.close();
        }
    }
}
