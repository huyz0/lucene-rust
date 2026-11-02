/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.DataInput;

/**
 * The encrypted file format both sides read and write ({@code lucene-store-crypto/src/format.rs}):
 *
 * <pre>
 * header  magic "LRSE" | version u8 = 1 | algorithm u8 = 1 (AES-256-GCM)
 *         chunk_size u32 BE | key_id_len u16 BE | key_id (UTF-8)
 *         wrapped_len u16 BE | wrapped data key
 * chunk i nonce [12] | ciphertext (chunk_size bytes, the last 0..=chunk_size) | tag [16]
 * </pre>
 *
 * Chunk {@code i}'s associated data is the header's bytes, then {@code i} (u32 BE), then 1 for the
 * final chunk and 0 otherwise. Every file has a final chunk (an empty file's is empty), and only it
 * may be short.
 */
public final class Lrse {
    public static final byte[] MAGIC = { 'L', 'R', 'S', 'E' };
    public static final int VERSION = 1;
    public static final int ALG_AES_256_GCM = 1;
    public static final int DATA_KEY_LEN = 32;
    public static final int NONCE_LEN = 12;
    public static final int TAG_LEN = 16;
    public static final int CHUNK_OVERHEAD = NONCE_LEN + TAG_LEN;
    public static final int DEFAULT_CHUNK_SIZE = 64 * 1024;
    public static final int MAX_CHUNK_SIZE = 16 * 1024 * 1024;

    private Lrse() {}

    /** A parsed header: {@code bytes} is its encoding, every chunk's associated data prefix. */
    public record Header(int chunkSize, String keyId, byte[] wrappedKey, byte[] bytes) {
        public static Header of(int chunkSize, String keyId, byte[] wrappedKey) {
            byte[] id = keyId.getBytes(StandardCharsets.UTF_8);
            if (id.length > 0xFFFF || wrappedKey.length > 0xFFFF) {
                throw new IllegalArgumentException("key id or wrapped key longer than 65535 bytes");
            }
            if (chunkSize <= 0 || chunkSize > MAX_CHUNK_SIZE) {
                throw new IllegalArgumentException("chunk size " + chunkSize + " out of range");
            }
            byte[] out = new byte[4 + 1 + 1 + 4 + 2 + id.length + 2 + wrappedKey.length];
            int p = 0;
            System.arraycopy(MAGIC, 0, out, p, 4);
            p += 4;
            out[p++] = VERSION;
            out[p++] = ALG_AES_256_GCM;
            p = putInt(out, p, chunkSize);
            p = putShort(out, p, id.length);
            System.arraycopy(id, 0, out, p, id.length);
            p += id.length;
            p = putShort(out, p, wrappedKey.length);
            System.arraycopy(wrappedKey, 0, out, p, wrappedKey.length);
            return new Header(chunkSize, keyId, wrappedKey.clone(), out);
        }

        public int length() {
            return bytes.length;
        }

        /** Reads a header from the start of {@code in}, a file named {@code name}. */
        public static Header read(DataInput in, String name) throws IOException {
            byte[] magic = new byte[4];
            in.readBytes(magic, 0, 4);
            if (Arrays.equals(magic, MAGIC) == false) {
                throw new CorruptIndexException("not an encrypted file (no LRSE magic); is it a plaintext index?", name);
            }
            int version = in.readByte() & 0xFF;
            if (version != VERSION) {
                throw new CorruptIndexException("unsupported encrypted format version " + version, name);
            }
            int alg = in.readByte() & 0xFF;
            if (alg != ALG_AES_256_GCM) {
                throw new CorruptIndexException("unsupported encryption algorithm " + alg, name);
            }
            int chunkSize = readInt(in);
            if (chunkSize <= 0 || chunkSize > MAX_CHUNK_SIZE) {
                throw new CorruptIndexException("chunk size " + Integer.toUnsignedString(chunkSize) + " out of range", name);
            }
            byte[] id = new byte[readShort(in)];
            in.readBytes(id, 0, id.length);
            byte[] wrapped = new byte[readShort(in)];
            in.readBytes(wrapped, 0, wrapped.length);
            String keyId = StandardCharsets.UTF_8.newDecoder()
                .decode(java.nio.ByteBuffer.wrap(id))
                .toString();
            return of(chunkSize, keyId, wrapped);
        }
    }

    /**
     * The plaintext length of an encrypted body (everything after the header) of {@code bodyLen}
     * bytes in chunks of {@code chunkSize}, or -1 when no well-formed body has that length.
     */
    public static long plaintextLength(long bodyLen, int chunkSize) {
        long perChunk = (long) chunkSize + CHUNK_OVERHEAD;
        long chunks = chunkCount(bodyLen, chunkSize);
        long lastSealed = bodyLen - (chunks - 1) * perChunk;
        if (bodyLen < 0 || lastSealed < CHUNK_OVERHEAD) {
            return -1;
        }
        return bodyLen - chunks * CHUNK_OVERHEAD;
    }

    /** How many chunks a body of {@code bodyLen} bytes holds (at least one). */
    public static long chunkCount(long bodyLen, int chunkSize) {
        long perChunk = (long) chunkSize + CHUNK_OVERHEAD;
        return Math.max(1, (bodyLen + perChunk - 1) / perChunk);
    }

    /** Chunk {@code index}'s associated data: {@code header}, the index, the final flag. */
    static byte[] aad(byte[] header, long index, boolean last) {
        byte[] aad = Arrays.copyOf(header, header.length + 5);
        putInt(aad, header.length, (int) index);
        aad[header.length + 4] = (byte) (last ? 1 : 0);
        return aad;
    }

    /** Fills {@code aad}'s last five bytes (see {@link #aad}). */
    static void setAad(byte[] aad, long index, boolean last) {
        putInt(aad, aad.length - 5, (int) index);
        aad[aad.length - 1] = (byte) (last ? 1 : 0);
    }

    private static int putInt(byte[] b, int p, int v) {
        b[p] = (byte) (v >>> 24);
        b[p + 1] = (byte) (v >>> 16);
        b[p + 2] = (byte) (v >>> 8);
        b[p + 3] = (byte) v;
        return p + 4;
    }

    private static int putShort(byte[] b, int p, int v) {
        b[p] = (byte) (v >>> 8);
        b[p + 1] = (byte) v;
        return p + 2;
    }

    private static int readInt(DataInput in) throws IOException {
        return ((in.readByte() & 0xFF) << 24) | ((in.readByte() & 0xFF) << 16) | ((in.readByte() & 0xFF) << 8) | (in.readByte() & 0xFF);
    }

    private static int readShort(DataInput in) throws IOException {
        return ((in.readByte() & 0xFF) << 8) | (in.readByte() & 0xFF);
    }
}
