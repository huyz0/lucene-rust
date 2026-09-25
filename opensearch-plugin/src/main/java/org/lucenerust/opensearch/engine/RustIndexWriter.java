/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexCommit;
import org.apache.lucene.index.IndexDeletionPolicy;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LiveIndexWriterConfig;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.common.lease.Releasable;
import org.opensearch.index.engine.DocumentIndexWriter;
import org.opensearch.index.mapper.ParseContext;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * OpenSearch's {@link DocumentIndexWriter} over the Rust writer ({@code engine_writer.rs}).
 *
 * <p>Documents are inverted in the calling thread ({@link DocumentEncoder}), in parallel; only the
 * hand-off to Rust is serialized, on {@link #lock}. A commit takes the same lock, so the commit
 * data it evaluates -- the local checkpoint and the maximum sequence number -- describes exactly
 * the operations the commit contains: nothing can be added between the evaluation and the commit.
 *
 * <p><b>Refresh is a commit.</b> Java readers open commits ({@link #acquireLatest}), so making new
 * documents visible publishes a {@code segments_N} with the live commit data. Every commit is
 * therefore a complete recovery point, and the writer's merges -- which run over committed
 * segments -- happen as documents arrive rather than only at flush.
 *
 * <p><b>Commit points are OpenSearch's to drop.</b> After every commit the deletion policy the
 * engine passes (its {@code CombinedDeletionPolicy}) sees the full list, exactly as Lucene's {@code
 * IndexFileDeleter} would show it, and the commits it deletes are dropped on the Rust side.
 *
 * <p><b>Failures.</b> A document the writer refuses throws {@link IllegalArgumentException} and
 * changes nothing. Any other native failure, a caught panic included, is this writer's tragic
 * exception: it is thrown once, and every later call throws {@link AlreadyClosedException} carrying
 * it, which is how the engine knows to fail the shard.
 */
final class RustIndexWriter implements DocumentIndexWriter {
    private final Directory directory;
    private final long handle;
    private final DocumentEncoder encoder;
    private final IndexWriterConfig config;
    private final Object lock = new Object();
    private final Map<Long, RustIndexCommit> commits = new HashMap<>();
    private final AtomicLong docsSinceCommit = new AtomicLong();

    private volatile Throwable tragic;
    private volatile boolean closed;
    private boolean nativeClosed;
    private volatile Iterable<Map.Entry<String, String>> liveCommitData;
    private IndexDeletionPolicy deletionPolicy;
    private LongSupplier minRetainedSeqNo;
    private boolean policyInitialized;
    private volatile long lastCommitMaxDoc;
    private volatile long latestGeneration;

    private RustIndexWriter(
        Directory directory,
        long handle,
        Analyzer analyzer,
        Similarity similarity,
        int indexCreatedVersionMajor,
        String softDeletesField,
        DocumentEncoder.FormatCheck formats
    ) {
        this.directory = directory;
        this.handle = handle;
        FieldRegistry registry = new FieldRegistry(this::register);
        this.encoder = new DocumentEncoder(registry, analyzer, similarity, indexCreatedVersionMajor, softDeletesField, formats);
        this.config = new IndexWriterConfig(analyzer).setSimilarity(similarity).setSoftDeletesField(softDeletesField);
    }

    /**
     * Opens the writer over {@code indexPath} (the store's directory, which must hold a commit), and
     * hands every existing commit to {@code deletionPolicy.onInit}, as {@code IndexWriter}'s
     * constructor does.
     */
    static RustIndexWriter open(
        Directory directory,
        Path indexPath,
        double ramBufferMb,
        boolean faultInjection,
        Analyzer analyzer,
        Similarity similarity,
        int indexCreatedVersionMajor,
        String softDeletesField,
        DocumentEncoder.FormatCheck formats,
        IndexDeletionPolicy deletionPolicy,
        LongSupplier minRetainedSeqNo
    ) throws IOException {
        long[] out = new long[1];
        int status = NativeBridge.writerOpen(
            indexPath.toString().getBytes(StandardCharsets.UTF_8),
            ramBufferMb,
            faultInjection,
            out
        );
        if (status != NativeBridge.OK) {
            throw new IOException("cannot open the Rust writer at [" + indexPath + "]: " + NativeBridge.lastError());
        }
        RustIndexWriter w = new RustIndexWriter(
            directory,
            out[0],
            analyzer,
            similarity,
            indexCreatedVersionMajor,
            softDeletesField,
            formats
        );
        boolean success = false;
        try {
            w.deletionPolicy = deletionPolicy;
            w.minRetainedSeqNo = minRetainedSeqNo;
            synchronized (w.lock) {
                w.liveCommitData = SegmentInfos.readLatestCommit(directory).getUserData().entrySet();
                w.onNewCommits();
            }
            success = true;
            return w;
        } finally {
            if (success == false) {
                NativeBridge.writerClose(out[0]);
            }
        }
    }

    // ---- status handling ----

    private void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this IndexWriter is closed", tragic);
        }
    }

    /** Maps a native status to Java: refusals to {@link IllegalArgumentException}, the rest tragic. */
    private void check(int status, String what) throws IOException {
        if (status == NativeBridge.OK) {
            return;
        }
        String message = NativeBridge.lastError();
        if (status == NativeBridge.INVALID_ARGUMENT || status == NativeBridge.DECODE) {
            throw new IllegalArgumentException(message);
        }
        IOException e = new IOException(what + " failed in the Rust writer (status " + status + "): " + message);
        if (tragic == null) {
            tragic = e;
        }
        closed = true;
        throw e;
    }

    private int register(FieldSchema schema) throws IOException {
        int[] out = new int[1];
        synchronized (lock) {
            ensureOpen();
            check(NativeBridge.writerRegisterField(handle, schema.encode(), out), "registering field [" + schema.name + "]");
        }
        return out[0];
    }

    private void apply(Blob op, int docs) throws IOException {
        synchronized (lock) {
            ensureOpen();
            check(NativeBridge.writerApply(handle, op.array(), op.length()), "indexing");
            docsSinceCommit.addAndGet(docs);
        }
    }

    // ---- indexing ----

    @Override
    public long addDocument(ParseContext.Document doc, Term uid) throws IOException {
        ensureOpen();
        apply(encoder.add(List.of(doc)), 1);
        return 0;
    }

    @Override
    public long addDocuments(List<ParseContext.Document> docs, Term uid) throws IOException {
        ensureOpen();
        apply(encoder.add(docs), docs.size());
        return 0;
    }

    /** An add of documents that are not {@code ParseContext.Document}s, e.g. a no-op tombstone. */
    void addDocument(Iterable<? extends IndexableField> doc) throws IOException {
        ensureOpen();
        apply(encoder.add(List.of(doc)), 1);
    }

    @Override
    public void softUpdateDocuments(
        Term uid,
        List<ParseContext.Document> docs,
        long version,
        long seqNo,
        long primaryTerm,
        Field... softDeletesField
    ) throws IOException {
        ensureOpen();
        apply(encoder.softUpdate(uid, docs, onlySoftDeletesField(softDeletesField)), docs.size());
    }

    @Override
    public void softUpdateDocument(Term uid, ParseContext.Document doc, long version, long seqNo, long primaryTerm, Field... softDeletesField)
        throws IOException {
        ensureOpen();
        apply(encoder.softUpdate(uid, List.of(doc), onlySoftDeletesField(softDeletesField)), 1);
    }

    @Override
    public void deleteDocument(
        Term uid,
        boolean isStaleOperation,
        ParseContext.Document doc,
        long version,
        long seqNo,
        long primaryTerm,
        Field... softDeletesField
    ) throws IOException {
        // LuceneIndexWriter: a stale delete's tombstone is only added; a live one replaces.
        if (isStaleOperation) {
            addDocument(doc, uid);
        } else {
            softUpdateDocument(uid, doc, version, seqNo, primaryTerm, softDeletesField);
        }
    }

    private static Field onlySoftDeletesField(Field... fields) {
        if (fields.length != 1) {
            throw new IllegalArgumentException("expected exactly one soft-deletes field, got " + fields.length);
        }
        return fields[0];
    }

    // ---- commits ----

    @Override
    public void setLiveCommitData(Iterable<Map.Entry<String, String>> commitUserData) {
        this.liveCommitData = commitUserData;
    }

    @Override
    public Iterable<Map.Entry<String, String>> getLiveCommitData() {
        return liveCommitData;
    }

    @Override
    public long commit() throws IOException {
        synchronized (lock) {
            ensureOpen();
            return commitLocked();
        }
    }

    private long commitLocked() throws IOException {
        Blob data = new Blob(256);
        int countAt = data.reserveI32();
        int n = 0;
        // Evaluated here, under the lock: see the class comment.
        for (Map.Entry<String, String> e : liveCommitData) {
            data.string(e.getKey()).string(e.getValue());
            n++;
        }
        data.patchI32(countAt, n);
        applyRetention();
        long[] out = new long[1];
        check(NativeBridge.writerCommit(handle, data.toArray(), out), "commit");
        docsSinceCommit.set(0);
        onNewCommits();
        return out[0];
    }

    private void applyRetention() throws IOException {
        if (minRetainedSeqNo != null) {
            check(NativeBridge.writerSetRetention(handle, true, minRetainedSeqNo.getAsLong()), "setting retention");
        }
    }

    /**
     * {@code IndexFileDeleter}'s half of a commit: the deletion policy sees every commit point, and
     * the ones it deletes are dropped. Called under {@link #lock}.
     */
    private void onNewCommits() throws IOException {
        long[] gens = new long[16];
        int[] len = new int[1];
        int status = NativeBridge.writerCommitGenerations(handle, gens, len);
        if (status == NativeBridge.BUFFER_TOO_SMALL) {
            gens = new long[len[0]];
            status = NativeBridge.writerCommitGenerations(handle, gens, len);
        }
        check(status, "listing commits");
        List<IndexCommit> list = new ArrayList<>(len[0]);
        Map<Long, RustIndexCommit> live = new HashMap<>();
        for (int i = 0; i < len[0]; i++) {
            long gen = gens[i];
            RustIndexCommit c = commits.get(gen);
            if (c == null) {
                c = new RustIndexCommit(directory, gen);
            }
            live.put(gen, c);
            list.add(c);
        }
        commits.clear();
        commits.putAll(live);
        RustIndexCommit newest = (RustIndexCommit) list.get(list.size() - 1);
        latestGeneration = newest.getGeneration();
        lastCommitMaxDoc = newest.segmentInfos().totalMaxDoc();
        if (deletionPolicy == null) {
            return;
        }
        if (policyInitialized == false) {
            deletionPolicy.onInit(list);
            policyInitialized = true;
        } else {
            deletionPolicy.onCommit(list);
        }
        List<Long> doomed = new ArrayList<>();
        for (IndexCommit c : list) {
            if (c.isDeleted()) {
                doomed.add(c.getGeneration());
            }
        }
        if (doomed.isEmpty() == false) {
            long[] arr = doomed.stream().mapToLong(Long::longValue).toArray();
            check(NativeBridge.writerDeleteCommits(handle, arr), "deleting commits");
            doomed.forEach(commits::remove);
        }
    }

    /**
     * Makes every buffered change visible: commits it if there is any, then pins the newest commit's
     * files for a reader. Returns {@code {generation, hold}}; release the hold with {@link
     * #releaseHold} once the reader is closed.
     */
    long[] acquireLatest() throws IOException {
        synchronized (lock) {
            ensureOpen();
            if (stat(NativeBridge.STAT_UNCOMMITTED) != 0) {
                commitLocked();
            }
            long gen = latestGeneration;
            long[] hold = new long[1];
            check(NativeBridge.writerHoldCommit(handle, gen, hold), "holding commit " + gen);
            return new long[] { gen, hold[0] };
        }
    }

    long latestGeneration() {
        return latestGeneration;
    }

    void releaseHold(long hold) {
        synchronized (lock) {
            if (nativeClosed == false) {
                NativeBridge.writerReleaseHold(handle, hold);
            }
        }
    }

    // ---- merges ----

    @Override
    public void forceMergeDeletes(boolean doWait) throws IOException {
        runForceMerge(1, true);
    }

    @Override
    public void maybeMerge() throws IOException {
        // Merges run with every commit.
    }

    @Override
    public void forceMerge(int maxNumSegments, boolean doWait) throws IOException {
        runForceMerge(maxNumSegments, false);
    }

    private void runForceMerge(int maxNumSegments, boolean onlyDeletes) throws IOException {
        synchronized (lock) {
            ensureOpen();
            commitLocked();
            applyRetention();
            long[] out = new long[1];
            check(NativeBridge.writerForceMerge(handle, maxNumSegments, onlyDeletes, out), "force merge");
            onNewCommits();
        }
    }

    @Override
    public boolean hasPendingMerges() {
        return false;
    }

    // ---- state ----

    private long stat(int index) throws IOException {
        long[] out = new long[NativeBridge.STAT_COUNT];
        check(NativeBridge.writerStats(handle, out), "reading stats");
        return out[index];
    }

    @Override
    public long ramBytesUsed() {
        if (closed) {
            return 0;
        }
        long[] out = new long[NativeBridge.STAT_COUNT];
        return NativeBridge.writerStats(handle, out) == NativeBridge.OK ? out[NativeBridge.STAT_RAM_BYTES] : 0;
    }

    @Override
    public long getFlushingBytes() {
        return 0;
    }

    @Override
    public long getPendingNumDocs() {
        return lastCommitMaxDoc + docsSinceCommit.get();
    }

    @Override
    public boolean hasUncommittedChanges() {
        ensureOpen();
        try {
            return stat(NativeBridge.STAT_UNCOMMITTED) != 0;
        } catch (IOException e) {
            throw new AlreadyClosedException("this IndexWriter is closed", e);
        }
    }

    @Override
    public Throwable getTragicException() {
        return tragic;
    }

    @Override
    public LiveIndexWriterConfig getConfig() {
        return config;
    }

    @Override
    public void deleteUnusedFiles() throws IOException {
        synchronized (lock) {
            ensureOpen();
            onNewCommits();
        }
    }

    @Override
    public void rollback() throws IOException {
        close();
    }

    @Override
    public void close() throws IOException {
        synchronized (lock) {
            closed = true;
            if (nativeClosed == false) {
                nativeClosed = true;
                NativeBridge.writerClose(handle);
            }
        }
    }

    /** Test hook: a panic inside the writer's lock, if the index enabled fault injection. */
    void injectPanic() throws IOException {
        apply(new Blob(1).u8(DocumentEncoder.OP_PANIC), 0);
    }

    @Override
    public IndexWriter getAccumulatingIndexWriter() {
        throw new UnsupportedOperationException("the Rust engine has no Lucene IndexWriter");
    }

    @Override
    public boolean hasNewIndexingOrUpdates() {
        return false;
    }

    @Override
    public boolean isWriteLockedByCurrentThread() {
        return true;
    }

    @Override
    public void beforeRefresh() {}

    @Override
    public void afterRefresh(boolean didRefresh) {}

    @Override
    public Releasable obtainWriteLockOnAllMap() {
        return () -> {};
    }

    @Override
    public boolean validateImmutableFieldNotUpdated(ParseContext.Document previousDocument, BytesRef currentUID) {
        return false;
    }

    /** A commit point as {@code IndexDeletionPolicy} sees it; its {@code segments_N} is read lazily. */
    static final class RustIndexCommit extends IndexCommit {
        private final Directory directory;
        private final long generation;
        private final String segmentsFileName;
        private SegmentInfos infos;
        private boolean deleted;

        RustIndexCommit(Directory directory, long generation) {
            this.directory = directory;
            this.generation = generation;
            this.segmentsFileName = org.apache.lucene.index.IndexFileNames.fileNameFromGeneration(
                org.apache.lucene.index.IndexFileNames.SEGMENTS,
                "",
                generation
            );
        }

        synchronized SegmentInfos segmentInfos() throws IOException {
            if (infos == null) {
                infos = SegmentInfos.readCommit(directory, segmentsFileName);
            }
            return infos;
        }

        @Override
        public String getSegmentsFileName() {
            return segmentsFileName;
        }

        @Override
        public Collection<String> getFileNames() throws IOException {
            return Collections.unmodifiableCollection(segmentInfos().files(true));
        }

        @Override
        public Directory getDirectory() {
            return directory;
        }

        @Override
        public void delete() {
            deleted = true;
        }

        @Override
        public boolean isDeleted() {
            return deleted;
        }

        @Override
        public int getSegmentCount() {
            try {
                return segmentInfos().size();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public long getGeneration() {
            return generation;
        }

        @Override
        public Map<String, String> getUserData() throws IOException {
            return segmentInfos().getUserData();
        }

        @Override
        public String toString() {
            return "RustIndexCommit(" + segmentsFileName + ")";
        }
    }
}
