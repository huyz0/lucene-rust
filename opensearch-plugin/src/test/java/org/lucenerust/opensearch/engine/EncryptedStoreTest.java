/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KeepOnlyLastCommitDeletionPolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.NativeLibrary;
import org.lucenerust.opensearch.NativeReaders;
import org.lucenerust.opensearch.store.CachingKeySource;
import org.lucenerust.opensearch.store.KeySource;
import org.lucenerust.opensearch.store.LocalKeySource;
import org.lucenerust.opensearch.store.Lrse;
import org.lucenerust.opensearch.store.NativeStores;
import org.lucenerust.opensearch.store.NativeStoresAccess;
import org.lucenerust.opensearch.store.RustEncryptedDirectory;
import org.lucenerust.opensearch.store.StoreCrypto;
import org.opensearch.index.mapper.ParseContext;

/**
 * The {@code rust_encrypted} store end to end in one JVM: Lucene writes an encrypted index and the
 * native reader searches it; the Rust engine writes one and Lucene's CheckIndex reads it; nothing
 * reaches disk in plaintext, the wrong key is refused, and a changed byte is caught.
 */
public final class EncryptedStoreTest {
    private static int checks;
    private static int failures;
    private static final String[] WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split(" ");

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        StoreCrypto crypto = StoreCrypto.get();
        System.out.println("EncryptedStoreTest: crypto is " + crypto.description() + (crypto.fips() ? ", FIPS" : ""));
        check(crypto.fips() == NativeBridge.storeIsFips(), "the JVM and native crypto agree on FIPS mode");
        lrse();
        javaWritesRustReads(crypto);
        rustWritesJavaReads(crypto);
        keyCache();
        storeSizing(crypto);
        refusals();
        System.out.printf("EncryptedStoreTest: %d checks, %d failures%n", checks, failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static KeySource keys(StoreCrypto crypto, byte fill) {
        byte[] kek = new byte[32];
        java.util.Arrays.fill(kek, fill);
        return new CachingKeySource(new LocalKeySource("test-kek", kek, crypto), 3_600_000_000_000L, 300_000_000_000L, 64);
    }

    private static Document doc(int i) {
        Document d = new Document();
        d.add(new StringField("id", String.format("doc-%05d", i), Field.Store.YES));
        d.add(new TextField("body", WORDS[i % WORDS.length] + " " + WORDS[(i * 7) % WORDS.length] + " secretword" + (i % 3), Field.Store.NO));
        return d;
    }

    /** Lucene writes; the native reader opens the bound path and counts the same postings. */
    private static void javaWritesRustReads(StoreCrypto crypto) throws Exception {
        Path root = Files.createTempDirectory("encrypted-store-java");
        try {
            try (RustEncryptedDirectory dir = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 7), crypto, 4096)) {
                try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()).setMaxBufferedDocs(97))) {
                    for (int i = 0; i < 1000; i++) {
                        w.addDocument(doc(i));
                    }
                    w.deleteDocuments(new Term("id", "doc-00010"));
                    w.commit();
                }
                checkIndex("Lucene-written encrypted index", dir);
                NativeReaders readers = new NativeReaders();
                try (DirectoryReader r = DirectoryReader.open(dir)) {
                    NativeReaders.Acquired a = readers.acquire(r);
                    check(a.handle() != 0, "the native reader opens the encrypted index: " + a.fallbackReason() + " " + NativeBridge.lastError());
                    if (a.handle() != 0) {
                        for (String t : List.of("alpha", "kappa", "secretword1", "nothere")) {
                            long[] out = new long[1];
                            int rc = NativeBridge.docFreq(a.handle(), "body".getBytes(StandardCharsets.UTF_8), t.getBytes(StandardCharsets.UTF_8), out);
                            check(rc == NativeBridge.OK, "native docFreq(" + t + "): " + NativeBridge.lastError());
                            check(out[0] == r.docFreq(new Term("body", t)), "native docFreq(" + t + ") " + out[0] + " == Lucene's " + r.docFreq(new Term("body", t)));
                        }
                    }
                } finally {
                    readers.closeAll();
                }
                noPlaintext(root, "Lucene-written");
            }
            // The wrong key opens nothing.
            try (RustEncryptedDirectory wrong = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 8), crypto, 4096)) {
                try (DirectoryReader r = DirectoryReader.open(wrong)) {
                    check(false, "the wrong key opened the index: " + r);
                } catch (IOException e) {
                    check(e.getMessage().contains("unwrap"), "the wrong key is refused: " + e);
                }
            }
            // A changed byte fails authentication.
            try (RustEncryptedDirectory dir = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 7), crypto, 4096)) {
                String victim = null;
                for (String f : dir.listAll()) {
                    if (f.endsWith(".tim")) {
                        victim = f;
                    }
                }
                check(victim != null, "a terms file to change");
                if (victim != null) {
                    Path p = root.resolve(victim);
                    byte[] raw = Files.readAllBytes(p);
                    raw[raw.length - 20] ^= 1;
                    Files.write(p, raw, StandardOpenOption.TRUNCATE_EXISTING);
                    try (IndexInput in = dir.openInput(victim, IOContext.DEFAULT)) {
                        byte[] all = new byte[(int) in.length()];
                        in.readBytes(all, 0, all.length);
                        check(false, "a changed chunk decrypted");
                    } catch (org.apache.lucene.index.CorruptIndexException e) {
                        check(e.getMessage().contains("authentication"), "a changed chunk is caught: " + e.getMessage());
                    }
                }
            }
        } finally {
            delete(root);
        }
    }

    /**
     * OpenSearch sizes a store from every file it lists (Store.stats, also when it allocates another
     * shard of the index): the lock file and files still being written must have a length, not fail
     * the call -- a failure there fails the shard being allocated.
     */
    private static void storeSizing(StoreCrypto crypto) throws Exception {
        Path root = Files.createTempDirectory("encrypted-store-sizing");
        int chunk = 4096;
        try (RustEncryptedDirectory dir = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 5), crypto, chunk)) {
            try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()))) {
                for (int i = 0; i < 300; i++) {
                    w.addDocument(doc(i));
                }
                w.commit();
                check(Files.exists(root.resolve(IndexWriter.WRITE_LOCK_NAME)), "the open writer holds write.lock");
                check(RustEncryptedDirectory.isPlaintext(IndexWriter.WRITE_LOCK_NAME), "write.lock is stored as-is");
                for (String name : dir.listAll()) {
                    try {
                        long len = dir.fileLength(name);
                        check(len >= 0, "length of " + name + ": " + len);
                    } catch (IOException e) {
                        check(false, "every listed file has a length while the writer is open: " + name + ": " + e);
                    }
                }
                check(dir.fileLength(IndexWriter.WRITE_LOCK_NAME) == 0, "write.lock has length 0");
            }
            // A file another writer is still writing: its raw bytes so far, cut inside the header,
            // at its end, and inside a chunk. A cut that could be a short last chunk counts as one.
            byte[] plain = new byte[3 * chunk + 100];
            for (int i = 0; i < plain.length; i++) {
                plain[i] = (byte) (i * 31);
            }
            try (IndexOutput out = dir.createOutput("_full.bin", IOContext.DEFAULT)) {
                out.writeBytes(plain, 0, plain.length);
            }
            byte[] raw = Files.readAllBytes(root.resolve("_full.bin"));
            int per = chunk + Lrse.CHUNK_OVERHEAD;
            int header = raw.length - (3 * per + 100 + Lrse.CHUNK_OVERHEAD);
            int[][] cuts = { { 0, 0 }, { 3, 0 }, { header - 1, 0 }, { header, 0 }, { header + per / 2, per / 2 - Lrse.CHUNK_OVERHEAD }, { header + 2 * per + 10, 2 * chunk } };
            for (int[] cut : cuts) {
                String name = "_partial" + cut[0] + ".bin";
                Files.write(root.resolve(name), java.util.Arrays.copyOf(raw, cut[0]));
                try {
                    long len = dir.fileLength(name);
                    check(len == cut[1], "a file cut at " + cut[0] + " raw bytes counts " + cut[1] + " complete plaintext bytes: " + len);
                } catch (IOException e) {
                    check(false, "a file cut at " + cut[0] + " raw bytes has a length: " + e);
                }
                try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
                    byte[] all = new byte[(int) in.length()];
                    in.readBytes(all, 0, all.length);
                    check(false, "a file cut at " + cut[0] + " raw bytes opened");
                } catch (IOException expected) {
                    // Opening still refuses what is not a whole file.
                }
            }
            // A length taken while the file was shorter is not served once it has grown.
            String growing = "_growing.bin";
            Files.write(root.resolve(growing), java.util.Arrays.copyOf(raw, header + per));
            long early = dir.fileLength(growing);
            Files.write(root.resolve(growing), raw, StandardOpenOption.TRUNCATE_EXISTING);
            check(early == chunk && dir.fileLength(growing) == plain.length, "a grown file's length follows it: " + early + " then " + dir.fileLength(growing));
            check(dir.fileLength("_full.bin") == plain.length, "a closed output's length is its plaintext");
        } finally {
            delete(root);
        }
    }

    /** The Rust engine writes through the bound path; Lucene reads and checks what it wrote. */
    private static void rustWritesJavaReads(StoreCrypto crypto) throws Exception {
        Path root = Files.createTempDirectory("encrypted-store-rust");
        try {
            try (RustEncryptedDirectory dir = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 9), crypto, 1 << 16)) {
                try (IndexWriter empty = new IndexWriter(dir, new IndexWriterConfig())) {
                    empty.commit();
                }
                RustIndexWriter w = RustIndexWriter.open(
                    dir,
                    root.toRealPath(),
                    16,
                    false,
                    false,
                    IndexWriter.MAX_DOCS,
                    new StandardAnalyzer(),
                    new BM25Similarity(),
                    10,
                    "__soft_deletes",
                    null,
                    new KeepOnlyLastCommitDeletionPolicy(),
                    null,
                    null,
                    null
                );
                w.setLiveCommitData(Map.<String, String>of().entrySet());
                for (int i = 0; i < 700; i++) {
                    ParseContext.Document d = new ParseContext.Document();
                    for (var f : doc(i)) {
                        d.add(f);
                    }
                    w.addDocument(d, null);
                    if (i % 250 == 249) {
                        w.commit();
                    }
                }
                w.commit();
                w.close();
                checkIndex("Rust-written encrypted index", dir);
                try (DirectoryReader r = DirectoryReader.open(dir)) {
                    check(r.numDocs() == 700, "Lucene reads the Rust-written docs: " + r.numDocs());
                    check(r.docFreq(new Term("body", "secretword2")) == 233, "Lucene reads Rust's postings: " + r.docFreq(new Term("body", "secretword2")));
                }
                noPlaintext(root, "Rust-written");
            }
        } finally {
            delete(root);
        }
    }

    private static void noPlaintext(Path root, String what) throws IOException {
        int files = 0;
        int metadata = 0;
        try (Stream<Path> s = Files.list(root)) {
            for (Path p : s.toList()) {
                String name = p.getFileName().toString();
                if (name.equals("write.lock")) {
                    continue;
                }
                byte[] raw = Files.readAllBytes(p);
                boolean lrse = raw.length >= 4 && raw[0] == 'L' && raw[1] == 'R' && raw[2] == 'S' && raw[3] == 'E';
                // Commit metadata stays readable for OpenSearch's gateway; it never holds field contents.
                boolean commit = RustEncryptedDirectory.isCommitMetadata(name);
                check(lrse != commit, what + ": " + name + (commit ? " is plaintext commit metadata" : " is encrypted"));
                String text = new String(raw, StandardCharsets.ISO_8859_1);
                List<String> needles = commit ? List.of("secretword", "doc-0004") : List.of("secretword", "doc-0004", "Lucene", "segments");
                for (String needle : needles) {
                    check(text.contains(needle) == false, what + ": " + name + " leaks [" + needle + "]");
                }
                files++;
                metadata += commit ? 1 : 0;
            }
        }
        check(files > 3 && metadata >= 2, what + ": an index was written (" + files + " files, " + metadata + " commit metadata)");
        // What OpenSearch's gateway does before the store opens (Store.tryOpenIndex): read the
        // latest commit through a plain directory.
        try (NIOFSDirectory plain = new NIOFSDirectory(root)) {
            SegmentInfos infos = SegmentInfos.readLatestCommit(plain);
            check(infos.size() > 0, what + ": the latest commit reads through a plain directory (" + infos.size() + " segments)");
        }
    }

    /** The format's edges: every length around chunk boundaries round-trips, and lengths are known from headers. */
    private static void lrse() throws IOException {
        StoreCrypto crypto = StoreCrypto.get();
        Path root = Files.createTempDirectory("encrypted-store-lrse");
        try (RustEncryptedDirectory dir = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 3), crypto, 4096)) {
            for (int len : new int[] { 0, 1, 4095, 4096, 4097, 8192, 20_000 }) {
                byte[] data = new byte[len];
                for (int i = 0; i < len; i++) {
                    data[i] = (byte) (i * 31 + len);
                }
                String name = "f" + len;
                try (IndexOutput out = dir.createOutput(name, IOContext.DEFAULT)) {
                    out.writeBytes(data, data.length);
                    check(out.getFilePointer() == len, "file pointer is the plaintext's");
                }
                long raw = Files.size(root.resolve(name));
                check(raw > expectedBody(len, 4096), "a header precedes the body");
                check(Lrse.plaintextLength(expectedBody(len, 4096), 4096) == len, "plaintext length of " + len);
                try (Directory again = new RustEncryptedDirectory(new MMapDirectory(root), keys(crypto, (byte) 3), crypto, 4096)) {
                    check(again.fileLength(name) == len, "fileLength(" + len + ") from the header: " + again.fileLength(name));
                    try (IndexInput in = again.openInput(name, IOContext.DEFAULT)) {
                        byte[] back = new byte[len];
                        in.readBytes(back, 0, len);
                        check(java.util.Arrays.equals(back, data), "round trip of " + len + " bytes");
                        if (len > 5000) {
                            IndexInput slice = in.slice("s", 4090, 20);
                            byte[] part = new byte[20];
                            slice.readBytes(part, 0, 20);
                            check(java.util.Arrays.equals(part, java.util.Arrays.copyOfRange(data, 4090, 4110)), "a slice across a chunk boundary");
                            IndexInput clone = in.clone();
                            clone.seek(len - 1);
                            check(clone.readByte() == data[len - 1], "a clone seeks to the last byte");
                        }
                    }
                }
            }
        } finally {
            delete(root);
        }
    }

    private static long expectedBody(int len, int chunk) {
        long chunks = Math.max(1, (len + chunk - 1) / chunk);
        return len + chunks * Lrse.CHUNK_OVERHEAD;
    }

    /** One generated key per period, and unwrapped keys from the cache. */
    private static void keyCache() throws IOException {
        int[] calls = new int[2];
        KeySource counting = new KeySource() {
            final KeySource inner = new LocalKeySource("k", new byte[32], StoreCrypto.get());

            @Override
            public DataKey generate() throws IOException {
                calls[0]++;
                return inner.generate();
            }

            @Override
            public byte[] unwrap(String keyId, byte[] wrapped) throws IOException {
                calls[1]++;
                return inner.unwrap(keyId, wrapped);
            }
        };
        try (CachingKeySource c = new CachingKeySource(counting, 3_600_000_000_000L, 300_000_000_000L, 16)) {
            KeySource.DataKey a = c.generate();
            KeySource.DataKey b = c.generate();
            check(a.equals(b) && calls[0] == 1, "a generated key is reused within its period");
            check(java.util.Arrays.equals(c.unwrap(a.keyId(), a.wrapped()), a.raw()) && calls[1] == 0, "a generated key unwraps from the cache");
            KeySource.DataKey other = counting.generate();
            c.unwrap(other.keyId(), other.wrapped());
            c.unwrap(other.keyId(), other.wrapped());
            check(calls[1] == 1, "an unwrapped key is cached: " + calls[1]);
            try {
                c.unwrap("unknown", other.wrapped());
                check(false, "an unknown key id unwraps");
            } catch (IOException e) {
                check(e.getMessage().contains("unknown"), "an unknown key id is refused");
            }
        }
    }

    /** Store directories the native engine must not bypass. */
    private static void refusals() throws IOException {
        Path root = Files.createTempDirectory("encrypted-store-refusals");
        try (MMapDirectory plain = new MMapDirectory(root)) {
            check(NativeStores.refusal(plain) == null, "a plain directory is served");
            Directory foreign = new FilterDirectory(plain) {};
            check(NativeStores.refusal(foreign) == null, "a wrapper from Lucene's or OpenSearch's class loader is served");
            check(NativeStoresAccess.refusal(foreign, c -> c != foreign.getClass()) != null, "a plugin's wrapper is refused");
            check(NativeStoresAccess.refusal(plain, c -> c != MMapDirectory.class) != null, "a plugin's FS directory is refused");
            check(NativeStores.STORE_TYPES.contains("cryptofs") == false, "another plugin's store type is refused");
        } finally {
            delete(root);
        }
    }

    private static void checkIndex(String where, Directory dir) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (CheckIndex ci = new CheckIndex(dir)) {
            ci.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
            ci.setInfoStream(new PrintStream(out, true, StandardCharsets.UTF_8));
            CheckIndex.Status status = ci.checkIndex();
            check(status.clean, where + ": CheckIndex\n" + out.toString(StandardCharsets.UTF_8));
        }
    }

    private static void delete(Path root) throws IOException {
        try (Stream<Path> s = Files.walk(root)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    private static void check(boolean ok, String what) {
        checks++;
        if (ok == false) {
            failures++;
            if (failures <= 40) {
                System.out.println("FAIL: " + what);
            }
        }
    }
}
