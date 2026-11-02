/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.lucenerust.opensearch.NativeBridge;

/**
 * The {@code rust_encrypted} store's directory: every file is encrypted at rest in the LRSE format
 * ({@link Lrse}), with data keys from a {@link KeySource} (a KMS, for BYOK).
 *
 * <p>The native engine reads and writes the same shard directory itself, so creating this directory
 * binds its path in the native library to the same key source ({@code ffi_store_bind_encrypted}):
 * the Rust writer and readers encrypt and decrypt the same format, with the same keys, and nothing
 * the engine writes reaches disk in plaintext. {@link #close} unbinds it.
 *
 * <p>Commit metadata ({@link #isCommitMetadata}: {@code segments_N}, {@code .si} files and
 * {@code corrupted_*} markers) stays in plaintext, on both sides. OpenSearch reads those files
 * through a plain {@code NIOFSDirectory}, without the store's directory, when it allocates a primary
 * after a restart ({@code Store.tryOpenIndex}). They hold segment and file names, document counts,
 * codec names and commit user data, never field contents.
 */
public final class RustEncryptedDirectory extends FilterDirectory {
    private final KeySource keys;
    private final StoreCrypto crypto;
    private final int chunkSize;
    private final long handle;
    private final String boundPath;
    /** Plaintext lengths of files this directory wrote or read the header of. */
    private final Map<String, Long> lengths = new ConcurrentHashMap<>();
    private volatile boolean closed;

    /**
     * Encrypts {@code inner}'s files with keys from {@code keys} (owned: closed with this directory),
     * sealing {@code chunkSize} bytes per chunk, and binds {@code inner}'s path for the native engine.
     */
    public RustEncryptedDirectory(FSDirectory inner, KeySource keys, StoreCrypto crypto, int chunkSize) throws IOException {
        super(inner);
        if (chunkSize <= 0 || chunkSize > Lrse.MAX_CHUNK_SIZE) {
            throw new IllegalArgumentException("chunk size " + chunkSize + " out of range");
        }
        this.keys = keys;
        this.crypto = crypto;
        this.chunkSize = chunkSize;
        KeyHandles.install();
        this.handle = KeyHandles.register(keys);
        this.boundPath = nativePath(inner.getDirectory());
        int rc = NativeBridge.storeBindEncrypted(boundPath, handle, chunkSize);
        if (rc != NativeBridge.OK) {
            KeyHandles.release(handle);
            throw new IOException("lucene-rust: binding the encrypted store at [" + boundPath + "] failed: " + NativeBridge.lastError());
        }
    }

    /**
     * Whether {@code name} is stored in plaintext; {@code is_commit_metadata} in
     * {@code lucene-store-crypto}'s {@code directory.rs}. Matches recovery temp names
     * ({@code recovery.<uuid>.<name>}) too, so a rename into place never changes the format.
     */
    public static boolean isCommitMetadata(String name) {
        return name.contains("segments_") || name.endsWith(".si") || name.startsWith("corrupted_");
    }

    /** The path string the native engine is handed for {@code dir}: what {@code shard_directory} is keyed by. */
    public static String nativePath(Path dir) {
        return dir.toAbsolutePath().toString();
    }

    public StoreCrypto crypto() {
        return crypto;
    }

    @Override
    public IndexOutput createOutput(String name, IOContext context) throws IOException {
        return encrypting(in.createOutput(name, context));
    }

    @Override
    public IndexOutput createTempOutput(String prefix, String suffix, IOContext context) throws IOException {
        return encrypting(in.createTempOutput(prefix, suffix, context));
    }

    private IndexOutput encrypting(IndexOutput raw) throws IOException {
        boolean ok = false;
        try {
            String name = raw.getName();
            lengths.remove(name);
            if (isCommitMetadata(name)) {
                ok = true;
                return raw;
            }
            EncryptingIndexOutput out = new EncryptingIndexOutput(raw, name, keys.generate(), chunkSize, crypto);
            ok = true;
            return new LengthRecordingOutput(out, lengths);
        } finally {
            if (ok == false) {
                raw.close();
            }
        }
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
        if (isCommitMetadata(name)) {
            return in.openInput(name, context);
        }
        DecryptingIndexInput input = DecryptingIndexInput.open(in.openInput(name, context), name, keys, crypto);
        lengths.put(name, input.length());
        return input;
    }

    @Override
    public long fileLength(String name) throws IOException {
        if (isCommitMetadata(name)) {
            return in.fileLength(name);
        }
        // The native engine writes and deletes files behind this directory's back, so a cached
        // length counts only for a file that still exists (the raw length is a stat).
        long rawLength = in.fileLength(name);
        Long known = lengths.get(name);
        if (known != null && rawLength > known) {
            return known;
        }
        try (IndexInput raw = in.openInput(name, IOContext.READONCE)) {
            long len = DecryptingIndexInput.plaintextLength(raw, name);
            lengths.put(name, len);
            return len;
        }
    }

    @Override
    public void deleteFile(String name) throws IOException {
        lengths.remove(name);
        in.deleteFile(name);
    }

    @Override
    public void rename(String source, String dest) throws IOException {
        if (isCommitMetadata(source) != isCommitMetadata(dest)) {
            throw new IOException("cannot rename [" + source + "] to [" + dest + "]: one is stored encrypted, the other in plaintext");
        }
        in.rename(source, dest);
        Long len = lengths.remove(source);
        lengths.remove(dest);
        if (len != null) {
            lengths.put(dest, len);
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        try {
            NativeBridge.storeUnbind(boundPath);
            KeyHandles.release(handle);
        } finally {
            try {
                keys.close();
            } finally {
                in.close();
            }
        }
    }

    @Override
    public String toString() {
        return "RustEncryptedDirectory(" + in + ")";
    }

    /** Records an output's plaintext length once it closes, for {@link #fileLength}. */
    private static final class LengthRecordingOutput extends IndexOutput {
        private final EncryptingIndexOutput out;
        private final Map<String, Long> lengths;

        LengthRecordingOutput(EncryptingIndexOutput out, Map<String, Long> lengths) {
            super(out.toString(), out.getName());
            this.out = out;
            this.lengths = lengths;
        }

        @Override
        public void writeByte(byte b) throws IOException {
            out.writeByte(b);
        }

        @Override
        public void writeBytes(byte[] b, int offset, int length) throws IOException {
            out.writeBytes(b, offset, length);
        }

        @Override
        public long getFilePointer() {
            return out.getFilePointer();
        }

        @Override
        public long getChecksum() throws IOException {
            return out.getChecksum();
        }

        @Override
        public void close() throws IOException {
            out.close();
            lengths.put(getName(), out.getFilePointer());
        }
    }
}
