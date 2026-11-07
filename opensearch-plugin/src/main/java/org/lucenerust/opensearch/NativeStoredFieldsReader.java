/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.StoredFieldDataInput;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.ByteArrayDataInput;

import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * A searcher's reader whose segments answer {@link StoredFields#document} natively:
 * the fetch phase's {@code _source}, {@code _id} and stored fields, and the get API's, read from the
 * Rust reader of the same segments ({@code ffi_jvm_reader_document}) and replayed into OpenSearch's
 * {@link StoredFieldVisitor} in stored order, with its {@code needsField} answers ({@code NO}
 * skips a field, {@code STOP} ends the document) -- what Lucene's {@code
 * Lucene90CompressingStoredFieldsReader.document} does.
 *
 * <p>Installed as the index's reader wrapper ({@code IndexModule.setReaderWrapper}), so it wraps
 * each searcher OpenSearch hands out, per request; it changes nothing else: every other read goes to
 * the wrapped reader, and both cache helpers are the wrapped reader's (OpenSearch requires it). The
 * native handle is the query phase's own ({@link NativeReaders#peek}, keyed by that cache key), never
 * opened here: a reader no search has opened natively -- the realtime get's internal reader -- is
 * read by Lucene. A reader the native side cannot open, or a document it
 * fails to read, is read by Lucene, as is every document while {@code index.lucene_rust.fetch.enabled}
 * is off.
 */
final class NativeStoredFieldsReader extends FilterDirectoryReader {
    private static final Logger logger = LogManager.getLogger(NativeStoredFieldsReader.class);

    static final byte STRING = 0;
    static final byte BINARY = 1;
    static final byte INT = 2;
    static final byte LONG = 3;
    static final byte FLOAT = 4;
    static final byte DOUBLE = 5;
    /** A derived {@code _source} whose value the blob leaves out: read with {@link SourceLoader} when wanted. */
    static final byte DERIVED_SOURCE = 6;

    /** A derived {@code _source}'s value, read only when the visitor asks for it. */
    @FunctionalInterface
    interface SourceLoader {
        /** Hands {@code fi}'s value to {@code visitor}. */
        void load(FieldInfo fi, StoredFieldVisitor visitor) throws IOException;
    }

    private static final SourceLoader NO_DERIVED_SOURCE = (fi, visitor) -> {
        throw new IOException("a derived _source from a reader that has none");
    };

    /** What the leaves share: the top reader, once built, and its native handle. */
    static final class Shared {
        final NativeReaders readers;
        final SearchStats stats;
        /** Whether documents are read natively ({@code index.lucene_rust.fetch.enabled}). */
        final BooleanSupplier enabled;
        volatile DirectoryReader top;
        private volatile long handle = -1;

        Shared(NativeReaders readers, SearchStats stats, BooleanSupplier enabled) {
            this.readers = readers;
            this.stats = stats;
            this.enabled = enabled;
        }

        /** The query phase's native handle for this reader, looked up on first use; 0 when there is none. */
        long handle() {
            long h = handle;
            if (h < 0) {
                h = readers.peek(top);
                handle = h;
            }
            return h;
        }
    }

    private final Shared shared;

    static DirectoryReader wrap(DirectoryReader in, NativeReaders readers, SearchStats stats, BooleanSupplier enabled)
        throws IOException {
        Shared shared = new Shared(readers, stats, enabled);
        NativeStoredFieldsReader r = new NativeStoredFieldsReader(in, shared);
        shared.top = r;
        return r;
    }

    private NativeStoredFieldsReader(DirectoryReader in, Shared shared) throws IOException {
        super(in, new SubReaderWrapper() {
            @Override
            protected LeafReader[] wrap(List<? extends LeafReader> readers) {
                LeafReader[] out = new LeafReader[readers.size()];
                for (int i = 0; i < out.length; i++) {
                    out[i] = new Leaf(readers.get(i), shared, i);
                }
                return out;
            }

            @Override
            public LeafReader wrap(LeafReader reader) {
                throw new UnsupportedOperationException("leaves are wrapped with their ordinals");
            }
        });
        this.shared = shared;
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
        return wrap(in, shared.readers, shared.stats, shared.enabled);
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return in.getReaderCacheHelper();
    }

    /**
     * One segment: its stored fields from the native reader, everything else from Lucene. A {@link
     * SequentialStoredFieldsLeafReader}, as OpenSearch's own leaf wrappers are, so the fetch phase's
     * sequential reader (and any wrapper around this one asking for it) reads natively too.
     */
    static final class Leaf extends SequentialStoredFieldsLeafReader {
        private final Shared shared;
        private final int ord;
        private final boolean live;

        Leaf(LeafReader in, Shared shared, int ord) {
            super(in);
            this.shared = shared;
            this.ord = ord;
            this.live = LiveLeafReader.unwrap(in) != null;
        }

        @Override
        public StoredFields storedFields() throws IOException {
            StoredFields lucene = in.storedFields();
            return new StoredFields() {
                @Override
                public void prefetch(int docID) throws IOException {
                    lucene.prefetch(docID);
                }

                @Override
                public void document(int docID, StoredFieldVisitor visitor) throws IOException {
                    read(docID, visitor, lucene::document, false);
                }
            };
        }

        @Override
        protected StoredFieldsReader doGetSequentialStoredFieldsReader(StoredFieldsReader reader) {
            return new NativeFieldsReader(reader);
        }

        /** The sequential reader: native documents, Lucene's reader for the rest of its contract. */
        private final class NativeFieldsReader extends StoredFieldsReader {
            private final StoredFieldsReader lucene;

            NativeFieldsReader(StoredFieldsReader lucene) {
                this.lucene = lucene;
            }

            @Override
            public void document(int docID, StoredFieldVisitor visitor) throws IOException {
                read(docID, visitor, lucene::document, true);
            }

            @Override
            public StoredFieldsReader clone() {
                return new NativeFieldsReader(lucene.clone());
            }

            @Override
            public void checkIntegrity() throws IOException {
                lucene.checkIntegrity();
            }

            @Override
            public void close() throws IOException {
                lucene.close();
            }
        }

        /** Lucene's own read of a document, the fallback. */
        @FunctionalInterface
        private interface LuceneDocument {
            void document(int docID, StoredFieldVisitor visitor) throws IOException;
        }

        /** Document {@code docID}, natively when enabled and possible, else by Lucene; timed either way. */
        private void read(int docID, StoredFieldVisitor visitor, LuceneDocument lucene, boolean sequential) throws IOException {
            long start = System.nanoTime();
            if (shared.enabled.getAsBoolean() && nativeDocument(docID, visitor, sequential, lucene)) {
                shared.stats.fetchTime(true, System.nanoTime() - start);
                return;
            }
            lucene.document(docID, visitor);
            shared.stats.fetchTime(false, System.nanoTime() - start);
        }

        /** Announces that {@code docs} (segment-local, sorted) are about to be read natively. */
        void prefetchNative(int[] docs) {
            if (live || docs.length == 0 || shared.enabled.getAsBoolean() == false) {
                return;
            }
            long h = shared.handle();
            if (h == 0) {
                return;
            }
            int rc = NativeBridge.prefetchDocuments(h, ord, docs);
            if (rc != NativeBridge.OK) {
                // Only a hint: the reads that follow fetch what they need and report their own errors.
                logger.debug("lucene-rust: native stored-fields prefetch failed ({}): {}", rc, NativeBridge.lastError());
            }
        }

        /** Document {@code docID} replayed into {@code visitor} from the native reader; false when it cannot be. */
        private boolean nativeDocument(int docID, StoredFieldVisitor visitor, boolean sequential, LuceneDocument lucene)
            throws IOException {
            if (live) {
                // The writer's buffered documents: the leaf reads them from its view itself.
                return false;
            }
            long h = shared.handle();
            if (h == 0) {
                return false;
            }
            byte[][] out = new byte[1][];
            int rc = NativeBridge.document(h, ord, docID, out);
            if (rc != NativeBridge.OK) {
                shared.stats.nativeError();
                logger.warn("lucene-rust: native stored fields failed ({}), reading with Lucene: {}", rc, NativeBridge.lastError());
                return false;
            }
            replay(getFieldInfos(), out[0], visitor, (fi, v) -> derivedSource(h, docID, fi, v, lucene));
            shared.stats.nativeFetch(sequential);
            return true;
        }

        /**
         * The {@code _source} of {@code docID} with its vectors spliced back, natively, else from
         * Lucene's codec (which splices them too): the document's other fields are already replayed.
         */
        private void derivedSource(long h, int docID, FieldInfo fi, StoredFieldVisitor visitor, LuceneDocument lucene)
            throws IOException {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.derivedSource(h, ord, docID, out);
            if (rc == NativeBridge.OK) {
                visitor.binaryField(fi, new StoredFieldDataInput(new ByteArrayDataInput(out[0]), out[0].length));
                return;
            }
            shared.stats.nativeError();
            logger.warn("lucene-rust: native derived _source failed ({}), reading with Lucene: {}", rc, NativeBridge.lastError());
            lucene.document(docID, new StoredFieldVisitor() {
                @Override
                public Status needsField(FieldInfo field) {
                    return field.number == fi.number ? Status.YES : Status.NO;
                }

                @Override
                public void binaryField(FieldInfo field, StoredFieldDataInput value) throws IOException {
                    visitor.binaryField(fi, value);
                }
            });
        }


        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    /**
     * The fetch phase's documents announced before it reads them one at a time
     * ({@code SearchOperationListener.onPreFetchPhase}): the global {@code docIds} grouped by the
     * leaf of {@code reader} that holds them, each group handed to that leaf's native reader in one
     * call, so a segment read by range fetches their stored fields together. Leaves not read
     * natively are skipped.
     */
    static void prefetch(IndexReader reader, int[] docIds, int from, int size) {
        if (size <= 0) {
            return;
        }
        int[] docs = Arrays.copyOfRange(docIds, from, from + size);
        Arrays.sort(docs);
        List<LeafReaderContext> leaves = reader.leaves();
        int at = 0;
        while (at < docs.length) {
            int index = ReaderUtil.subIndex(docs[at], leaves);
            LeafReaderContext ctx = leaves.get(index);
            int end = ctx.docBase + ctx.reader().maxDoc();
            int stop = at;
            while (stop < docs.length && docs[stop] < end) {
                stop++;
            }
            Leaf leaf = unwrapLeaf(ctx.reader());
            if (leaf != null) {
                int[] local = new int[stop - at];
                for (int i = 0; i < local.length; i++) {
                    local[i] = docs[at + i] - ctx.docBase;
                }
                leaf.prefetchNative(local);
            }
            at = Math.max(stop, at + 1);
        }
    }

    /** This reader's leaf under {@code reader}'s filter wrappers, or null when it has none. */
    private static Leaf unwrapLeaf(LeafReader reader) {
        LeafReader r = reader;
        while (true) {
            if (r instanceof Leaf leaf) {
                return leaf;
            }
            if (r instanceof FilterLeafReader f) {
                r = f.getDelegate();
            } else {
                return null;
            }
        }
    }

    /**
     * A native document's fields into {@code visitor} as {@code
     * Lucene90CompressingStoredFieldsReader.document} visits them: each field's {@code FieldInfo}
     * (null for a number {@code infos} does not know, asked all the same) to {@code needsField} before
     * its value, {@code NO} skipping the value and {@code STOP} ending the document, a binary value
     * handed over as a {@link StoredFieldDataInput}. A {@code DERIVED_SOURCE} field's value comes from
     * {@code source}, asked only when the visitor wants it.
     */
    static void replay(FieldInfos infos, byte[] blob, StoredFieldVisitor visitor) throws IOException {
        replay(infos, blob, visitor, NO_DERIVED_SOURCE);
    }

    static void replay(FieldInfos infos, byte[] blob, StoredFieldVisitor visitor, SourceLoader source) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int n = b.getInt();
        for (int i = 0; i < n; i++) {
            FieldInfo fi = infos.fieldInfo(b.getInt());
            byte type = b.get();
            if (type == DERIVED_SOURCE) {
                StoredFieldVisitor.Status status = visitor.needsField(fi);
                if (status == StoredFieldVisitor.Status.STOP) {
                    return;
                }
                if (status == StoredFieldVisitor.Status.YES) {
                    source.load(fi, visitor);
                }
                continue;
            }
            int len = type == STRING || type == BINARY ? b.getInt() : 0;
            StoredFieldVisitor.Status status = visitor.needsField(fi);
            if (status == StoredFieldVisitor.Status.STOP) {
                return;
            }
            boolean yes = status == StoredFieldVisitor.Status.YES;
            switch (type) {
                case STRING -> {
                    if (yes) {
                        visitor.stringField(fi, new String(blob, b.position(), len, StandardCharsets.UTF_8));
                    }
                    b.position(b.position() + len);
                }
                case BINARY -> {
                    if (yes) {
                        // Two arguments: `ByteArrayDataInput.length()` is its
                        // limit (offset + len), not len.
                        visitor.binaryField(fi, new StoredFieldDataInput(new ByteArrayDataInput(blob, b.position(), len), len));
                    }
                    b.position(b.position() + len);
                }
                case INT -> {
                    int v = b.getInt();
                    if (yes) {
                        visitor.intField(fi, v);
                    }
                }
                case LONG -> {
                    long v = b.getLong();
                    if (yes) {
                        visitor.longField(fi, v);
                    }
                }
                case FLOAT -> {
                    float v = Float.intBitsToFloat(b.getInt());
                    if (yes) {
                        visitor.floatField(fi, v);
                    }
                }
                case DOUBLE -> {
                    double v = Double.longBitsToDouble(b.getLong());
                    if (yes) {
                        visitor.doubleField(fi, v);
                    }
                }
                default -> throw new IOException("native stored field of unknown type " + type);
            }
        }
    }
}
