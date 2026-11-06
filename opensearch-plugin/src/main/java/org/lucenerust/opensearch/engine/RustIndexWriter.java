/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.common.lease.Releasable;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.index.engine.DocumentIndexWriter;
import org.opensearch.index.mapper.ParseContext;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.LongSupplier;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * OpenSearch's {@link DocumentIndexWriter} over the Rust writer ({@code engine_writer.rs}).
 *
 * <p>Documents are encoded in the calling thread ({@link DocumentEncoder}) and handed to the Rust
 * writer, which buffers each indexing thread's documents separately; adds run in parallel. A commit
 * does not stop them: the Rust writer flushes the buffers holding the operations from before it, and
 * only then reads the commit data through an upcall ({@link #commitDataUpcall}), as {@code
 * IndexWriter.commit} iterates its live commit data after {@code flushAllThreads} -- so the maximum
 * sequence number it records covers every operation the commit holds.
 *
 * <p><b>Refresh is not a commit.</b> A refresh flushes the buffers into segments and opens a Java
 * reader on the writer's live segment list ({@link #acquireNrt}), as {@code IndexWriter.getReader}
 * does; nothing is synced. Only {@link #commit} (an OpenSearch flush) writes a {@code segments_N}
 * with the live commit data, so every commit is a complete recovery point and the translog covers
 * the rest. With live search on, a refresh is a commit ({@link #acquireLatest}): live views are
 * keyed by commit generation. The writer's merges run on its merge threads ({@code
 * index.merge.scheduler.*}); a merged segment is in the next reader and the next commit.
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
    private static final Logger LOGGER = LogManager.getLogger(RustIndexWriter.class);

    private final Directory directory;
    private final long handle;
    private final FieldRegistry registry;
    private final DocumentEncoder encoder;
    private final IndexWriterConfig config;
    private final Object lock = new Object();
    /**
     * Indexing threads hold the read side while they call into the native writer, which takes
     * adds from many threads at once; {@link #close} takes the write side, so the handle is never
     * closed under an add.
     */
    private final ReentrantReadWriteLock indexing = new ReentrantReadWriteLock();
    /** Guards {@link #accounted}. */
    private final Object accounting = new Object();
    private final Map<Long, RustIndexCommit> commits = new HashMap<>();
    private final AtomicLong docsSinceCommit = new AtomicLong();

    private volatile Throwable tragic;
    private volatile boolean closed;
    private boolean nativeClosed;
    private volatile Iterable<Map.Entry<String, String>> liveCommitData;
    private IndexDeletionPolicy deletionPolicy;
    private LongSupplier minRetainedSeqNo;
    private CircuitBreaker breaker;
    /** Bytes of the Rust writer's buffer currently accounted to {@link #breaker}; under {@link #accounting}. */
    private long accounted;
    private boolean faultInjection;
    /** Whether searches read the buffered documents ({@code index.lucene_rust.realtime_search}). */
    private boolean liveSearch;
    /** The field whose value is a document's partition ({@code index.lucene_rust.partition_field}), or null. */
    private String partitionField;
    /** The newest commit generation whose readers' buffers {@link #releaseLive} let go; under {@link #liveLock}. */
    private long liveReleased = -1;
    private final Object liveLock = new Object();
    /** The leaf's field infos and the registry size they were built at; under {@link #liveLock}. */
    private org.apache.lucene.index.FieldInfos liveInfos;
    private int liveInfosSize = -1;
    private boolean policyInitialized;
    /** The merge policy settings last sent to the writer; under {@link #lock}. */
    private double[] mergePolicySettings;
    /** The writer options last sent ({@link #setOptions}); under {@link #lock}. */
    private double[] writerOptions;
    /**
     * The merge-on-flush wait last set, or -1 while no {@code MergeOnFlushMergePolicy} wraps the
     * merge policy; under {@link #lock}. The wrapper copies the policy's settings when it is set, so
     * a change to them sets it again.
     */
    private long mergeOnFlushPolicyWait = -1;
    /** The BP reordering settings last sent to the writer; under {@link #lock}. */
    private BpReorderSettings.Values bpReorder;
    private volatile long lastCommitMaxDoc;
    private volatile long latestGeneration;

    private RustIndexWriter(
        Directory directory,
        long handle,
        Analyzer analyzer,
        Similarity similarity,
        int indexCreatedVersionMajor,
        String softDeletesField,
        DocumentEncoder.FormatCheck formats,
        org.apache.lucene.codecs.Codec codec
    ) {
        this.directory = directory;
        this.handle = handle;
        this.registry = new FieldRegistry(this::register);
        this.encoder = new DocumentEncoder(registry, analyzer, similarity, indexCreatedVersionMajor, softDeletesField, formats);
        this.config = new IndexWriterConfig(analyzer).setSimilarity(similarity).setSoftDeletesField(softDeletesField);
        if (codec != null) {
            config.setCodec(codec);
        }
    }

    /**
     * Opens the writer over {@code indexPath} (the store's directory, which must hold a commit), and
     * hands every existing commit to {@code deletionPolicy.onInit}, as {@code IndexWriter}'s
     * constructor does. With a {@code partitionField}, each document goes to the segments of its
     * partition ({@link #partition}).
     */
    static RustIndexWriter open(
        Directory directory,
        Path indexPath,
        double ramBufferMb,
        boolean faultInjection,
        boolean sparse,
        boolean liveSearch,
        boolean adaptiveDocValues,
        String partitionField,
        int maxDocs,
        Analyzer analyzer,
        Similarity similarity,
        int indexCreatedVersionMajor,
        String softDeletesField,
        DocumentEncoder.FormatCheck formats,
        IndexDeletionPolicy deletionPolicy,
        LongSupplier minRetainedSeqNo,
        CircuitBreaker breaker,
        org.apache.lucene.codecs.Codec codec
    ) throws IOException {
        installCommitDataUpcall();
        long[] out = new long[1];
        int status = NativeBridge.writerOpen(
            org.lucenerust.opensearch.store.RustEncryptedDirectory.nativePath(indexPath).getBytes(StandardCharsets.UTF_8),
            ramBufferMb,
            faultInjection,
            sparse,
            liveSearch,
            adaptiveDocValues,
            partitionField != null,
            maxDocs,
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
            formats,
            codec
        );
        boolean success = false;
        try {
            w.deletionPolicy = deletionPolicy;
            w.minRetainedSeqNo = minRetainedSeqNo;
            w.breaker = breaker;
            w.faultInjection = faultInjection;
            w.liveSearch = liveSearch;
            w.partitionField = partitionField;
            // IndexWriterConfig.setSoftDeletesField: the writer knows the field before any document
            // carries it, so the first soft update -- which may be the first use -- has a field to
            // land on.
            FieldSchema soft = new FieldSchema(softDeletesField);
            soft.docValuesType = org.apache.lucene.index.DocValuesType.NUMERIC;
            soft.softDeletes = true;
            w.registry.resolve(soft);
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

    /** Whether the operation this thread is applying may be refused by the breaker; see {@link #refusableIf}. */
    private static final ThreadLocal<Boolean> REFUSABLE = ThreadLocal.withInitial(() -> false);

    /**
     * Marks the operations this thread applies until the returned scope closes as ones the breaker
     * may refuse, when {@code refusable}. Only an index operation on the primary can be refused: a
     * refusal there becomes a document failure (a 429) recorded as a no-op, while a failure of that
     * no-op, of a delete, or of any operation on a replica or in recovery fails the shard. Those
     * are accounted without breaking, as Lucene's {@code IndexWriter} never refuses a document for
     * memory; the writer's stall control (indexing waits once buffered and flushing bytes pass
     * twice the RAM buffer) still bounds them.
     */
    public static Releasable refusableIf(boolean refusable) {
        if (refusable == false) {
            return () -> {};
        }
        REFUSABLE.set(true);
        return () -> REFUSABLE.set(false);
    }

    /**
     * The Rust writer's buffer is native memory the JVM cannot see, so it is accounted to the
     * {@code accounting} circuit breaker: an operation first reserves an estimate -- which trips the
     * breaker, refusing the document, when memory is short and the operation is refusable (see
     * {@link #refusableIf}) -- and the reservation is then trued up to what the writer reports, and
     * released as commits empty the buffer.
     */
    private void apply(Blob op, int docs) throws IOException {
        long estimate = BUFFER_BYTES_PER_BLOB_BYTE * op.length();
        if (breaker != null) {
            if (REFUSABLE.get()) {
                breaker.addEstimateBytesAndMaybeBreak(estimate, "lucene_rust_writer");
            } else {
                breaker.addWithoutBreaking(estimate);
            }
        }
        // A blob with fields left to Rust to analyze is decoded (analyzed) first, outside the
        // lock, so indexing threads analyze in parallel as they do in Java.
        long prepared = 0;
        if (op.analyzesNatively) {
            long[] id = new long[1];
            int rc = NativeBridge.writerPrepare(op.array(), op.length(), id);
            if (rc != NativeBridge.OK) {
                if (breaker != null) {
                    breaker.addWithoutBreaking(-estimate);
                }
                synchronized (lock) {
                    check(rc, "analyzing");
                }
            }
            prepared = id[0];
        }
        // No writer-wide lock: the native writer buffers each thread's documents separately, as
        // Lucene's per-thread buffers do.
        indexing.readLock().lock();
        try {
            synchronized (accounting) {
                accounted += estimate;
            }
            ensureOpen();
            if (prepared != 0) {
                long id = prepared;
                prepared = 0;
                check(NativeBridge.writerApplyPrepared(handle, id), "indexing");
            } else {
                check(NativeBridge.writerApply(handle, op.array(), op.length()), "indexing");
            }
            docsSinceCommit.addAndGet(docs);
        } finally {
            if (prepared != 0) {
                NativeBridge.preparedFree(prepared);
            }
            reconcileBreaker();
            indexing.readLock().unlock();
        }
    }

    /** A conservative ratio of buffered bytes to blob bytes, for the pre-operation reservation. */
    private static final long BUFFER_BYTES_PER_BLOB_BYTE = 4;

    /** Trues the breaker up to the writer's own count. */
    private void reconcileBreaker() {
        if (breaker == null) {
            return;
        }
        long[] out = new long[1];
        long ram = closed == false && NativeBridge.writerStats(handle, out) == NativeBridge.OK ? out[NativeBridge.STAT_RAM_BYTES] : 0;
        synchronized (accounting) {
            breaker.addWithoutBreaking(ram - accounted);
            accounted = ram;
        }
    }

    /** Bytes currently accounted to the circuit breaker, for tests. */
    long accountedBytes() {
        synchronized (accounting) {
            return accounted;
        }
    }

    /** The fault-injection trigger: a document carrying this field, on an index that allows it. */
    static final String PANIC_FIELD = "__lucene_rust_panic";

    private void maybeInjectFault(Iterable<? extends IndexableField> doc) throws IOException {
        if (faultInjection) {
            for (IndexableField f : doc) {
                if (f.name().equals(PANIC_FIELD)) {
                    injectPanic();
                }
            }
        }
    }

    // ---- indexing ----

    @Override
    public long addDocument(ParseContext.Document doc, Term uid) throws IOException {
        ensureOpen();
        maybeInjectFault(doc);
        List<ParseContext.Document> docs = List.of(doc);
        apply(encoder.add(partition(docs), docs), 1);
        return 0;
    }

    @Override
    public long addDocuments(List<ParseContext.Document> docs, Term uid) throws IOException {
        ensureOpen();
        apply(encoder.add(partition(docs), docs), docs.size());
        return 0;
    }

    /** An add of documents that are not {@code ParseContext.Document}s, e.g. a no-op tombstone. */
    void addDocument(Iterable<? extends IndexableField> doc) throws IOException {
        ensureOpen();
        apply(encoder.add(List.of(doc)), 1);
    }

    /**
     * The partition of a block of documents: the value of the partition field on its root document,
     * which OpenSearch puts last (nested documents before their parent). A keyword's value is its
     * doc-values bytes (UTF-8, or {@code 0x} and hex for bytes that are not, such as an ip's); a
     * numeric's or date's is its long; with several values, the smallest. Null -- the shard's default
     * partition -- without a partition field or a value. A value the field holds only as analyzed
     * text is refused: a partition is one exact value.
     */
    String partition(List<? extends Iterable<? extends IndexableField>> docs) {
        if (partitionField == null || docs.isEmpty()) {
            return null;
        }
        BytesRef bytes = null;
        boolean numeric = false;
        long number = Long.MAX_VALUE;
        BytesRef indexed = null;
        boolean text = false;
        for (IndexableField f : docs.get(docs.size() - 1)) {
            if (f.name().equals(partitionField) == false) {
                continue;
            }
            switch (f.fieldType().docValuesType()) {
                case SORTED, SORTED_SET -> {
                    BytesRef v = f.binaryValue();
                    if (bytes == null || v.compareTo(bytes) < 0) {
                        bytes = v;
                    }
                }
                case NUMERIC, SORTED_NUMERIC -> {
                    numeric = true;
                    number = Math.min(number, f.numericValue().longValue());
                }
                default -> {
                    if (f.fieldType().indexOptions() == org.apache.lucene.index.IndexOptions.NONE) {
                        continue;
                    }
                    if (f.fieldType().tokenized()) {
                        text = true;
                    } else if (f.binaryValue() != null && (indexed == null || f.binaryValue().compareTo(indexed) < 0)) {
                        // A keyword without doc values: its indexed term.
                        indexed = f.binaryValue();
                    }
                }
            }
        }
        if (bytes != null) {
            return partitionString(bytes);
        }
        if (numeric) {
            return Long.toString(number);
        }
        if (indexed != null) {
            return partitionString(indexed);
        }
        if (text) {
            throw new IllegalArgumentException(
                "the partition field [" + partitionField + "] must be a keyword, numeric, date or ip field, not text"
            );
        }
        return null;
    }

    /** A partition value's bytes as the string the segment records: UTF-8, or {@code 0x} and hex. */
    private static String partitionString(BytesRef v) {
        try {
            return StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(v.bytes, v.offset, v.length))
                .toString();
        } catch (java.nio.charset.CharacterCodingException e) {
            return "0x" + java.util.HexFormat.of().formatHex(v.bytes, v.offset, v.offset + v.length);
        }
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
        apply(encoder.softUpdate(partition(docs), uid, docs, onlySoftDeletesField(softDeletesField)), docs.size());
    }

    @Override
    public void softUpdateDocument(Term uid, ParseContext.Document doc, long version, long seqNo, long primaryTerm, Field... softDeletesField)
        throws IOException {
        ensureOpen();
        maybeInjectFault(doc);
        List<ParseContext.Document> docs = List.of(doc);
        apply(encoder.softUpdate(partition(docs), uid, docs, onlySoftDeletesField(softDeletesField)), 1);
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

    /**
     * Live commit data read in two steps, as {@code InternalEngine.commitIndexWriter} reads it
     * around {@code IndexWriter.commit}: {@link #beforeFlush} as a commit starts (the local
     * checkpoint), and the iterable it returns once the commit's flush is done (the maximum sequence
     * number). Iterating it directly does both at once.
     */
    @FunctionalInterface
    interface StagedCommitData extends Iterable<Map.Entry<String, String>> {
        Iterable<Map.Entry<String, String>> beforeFlush();

        @Override
        default java.util.Iterator<Map.Entry<String, String>> iterator() {
            return beforeFlush().iterator();
        }
    }

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

    /** Commits while adds go on (see the class comment). Called under {@link #lock}. */
    private long commitLocked() throws IOException {
        applyRetention();
        // Adds counted from here on may or may not be in the commit; they stay counted until the
        // next one, so the pending count errs high, never low.
        long counted = docsSinceCommit.get();
        Iterable<Map.Entry<String, String>> data = liveCommitData;
        committingData = data instanceof StagedCommitData staged ? staged.beforeFlush() : data;
        long[] out = new long[1];
        int status;
        commitDataFailure = null;
        COMMITTING.set(this);
        try {
            status = NativeBridge.writerCommit(handle, handle, out);
        } finally {
            COMMITTING.remove();
            committingData = null;
        }
        Throwable failure = commitDataFailure;
        if (failure != null) {
            commitDataFailure = null;
            if (failure instanceof IOException e) {
                throw e;
            }
            if (failure instanceof RuntimeException e) {
                throw e;
            }
            throw new IOException("evaluating the commit data failed", failure);
        }
        check(status, "commit");
        docsSinceCommit.addAndGet(-counted);
        reconcileBreaker();
        onNewCommits();
        return out[0];
    }

    /** The writer committing on this thread, for {@link #commitDataUpcall}. */
    private static final ThreadLocal<RustIndexWriter> COMMITTING = new ThreadLocal<>();
    /** The running commit's data, for {@link #commitDataUpcall} to read; under {@link #lock}. */
    private Iterable<Map.Entry<String, String>> committingData;
    /** What evaluating the live commit data threw during the running commit; under {@link #lock}. */
    private Throwable commitDataFailure;
    private static boolean upcallInstalled;

    /** Registers {@link #commitDataUpcall} with the native library, once per JVM. */
    @SuppressWarnings("restricted") // an upcall stub is the point
    private static synchronized void installCommitDataUpcall() {
        if (upcallInstalled) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    RustIndexWriter.class,
                    "commitDataUpcall",
                    MethodType.methodType(int.class, long.class, MemorySegment.class, long.class, MemorySegment.class)
                );
            FunctionDescriptor fd = FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS);
            MemorySegment stub = Linker.nativeLinker().upcallStub(target, fd, Arena.global());
            if (NativeBridge.setCommitDataUpcall(stub) != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the commit-data upcall failed: " + NativeBridge.lastError());
            }
            upcallInstalled = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * The native writer's request for a commit's user data ({@code CommitDataUpcall} in {@code
     * engine_writer.rs}), made on the committing thread once the commit's buffers are flushed: the live
     * commit data, evaluated now. It never throws, since an exception crossing an upcall ends the JVM;
     * what the evaluation throws is kept for {@link #commitLocked} to rethrow.
     */
    @SuppressWarnings("restricted") // the segments are the native caller's, sized by its contract
    private static int commitDataUpcall(long context, MemorySegment out, long cap, MemorySegment outLen) {
        RustIndexWriter w = COMMITTING.get();
        if (w == null || w.handle != context) {
            return NativeBridge.COMMIT_DATA_FAILED;
        }
        try {
            Blob data = new Blob(256);
            int countAt = data.reserveI32();
            int n = 0;
            for (Map.Entry<String, String> e : w.committingData) {
                data.string(e.getKey()).string(e.getValue());
                n++;
            }
            data.patchI32(countAt, n);
            outLen.reinterpret(Long.BYTES).set(JAVA_LONG, 0, data.length());
            if (data.length() > cap) {
                return NativeBridge.COMMIT_DATA_BUFFER_TOO_SMALL;
            }
            MemorySegment.copy(data.array(), 0, out.reinterpret(cap), JAVA_BYTE, 0, data.length());
            return NativeBridge.COMMIT_DATA_OK;
        } catch (Throwable t) {
            w.commitDataFailure = t;
            return NativeBridge.COMMIT_DATA_FAILED;
        }
    }

    /**
     * The merge policy's settings, as {@code IndexWriterConfig.setMergePolicy} hands the shard's
     * {@code TieredMergePolicy} to {@code IndexWriter}; sent again only when one changed.
     */
    void setMergePolicy(TieredMergePolicy policy) throws IOException {
        double[] settings = {
            policy.getMaxMergeAtOnce(),
            policy.getSegmentsPerTier(),
            policy.getMaxMergedSegmentMB(),
            policy.getFloorSegmentMB(),
            policy.getDeletesPctAllowed(),
            policy.getForceMergeDeletesPctAllowed() };
        synchronized (lock) {
            if (Arrays.equals(settings, mergePolicySettings)) {
                return;
            }
            check(
                NativeBridge.writerSetMergePolicy(
                    handle,
                    policy.getMaxMergeAtOnce(),
                    policy.getSegmentsPerTier(),
                    megabytes(policy.getMaxMergedSegmentMB()),
                    megabytes(policy.getFloorSegmentMB()),
                    policy.getDeletesPctAllowed(),
                    policy.getForceMergeDeletesPctAllowed()
                ),
                "setting the merge policy"
            );
            mergePolicySettings = settings;
            if (mergeOnFlushPolicyWait >= 0) {
                check(NativeBridge.writerSetMergeOnFlush(handle, mergeOnFlushPolicyWait, true), "setting merge-on-flush");
            }
        }
    }

    /**
     * {@code IndexWriterConfig.setUseCompoundFile}, the merge policy's {@code setNoCFSRatio} and
     * {@code setMaxCFSSegmentSizeMB}, {@code setCheckPendingFlushUpdate}, and OpenSearch's
     * {@code ShuffleForcedMergePolicy}; sent again only when one changed.
     */
    void setOptions(boolean useCompoundFile, double noCfsRatio, double maxCfsSegmentMb, boolean checkPendingFlush, boolean shuffle)
        throws IOException {
        double[] options = { useCompoundFile ? 1 : 0, noCfsRatio, maxCfsSegmentMb, checkPendingFlush ? 1 : 0, shuffle ? 1 : 0 };
        synchronized (lock) {
            if (Arrays.equals(options, writerOptions)) {
                return;
            }
            check(
                NativeBridge.writerSetOptions(handle, useCompoundFile, noCfsRatio, megabytes(maxCfsSegmentMb), checkPendingFlush, shuffle),
                "setting the writer options"
            );
            writerOptions = options;
        }
    }

    /**
     * {@code BPReorderingMergePolicy} around the merge policy, as {@code index.lucene_rust.bp_reorder.*}
     * describes it; sent again only when a value changed.
     */
    void setBpReorder(BpReorderSettings.Values values) throws IOException {
        synchronized (lock) {
            if (values.equals(bpReorder) || (bpReorder == null && values.enabled() == false)) {
                return;
            }
            check(
                NativeBridge.writerSetBpReorder(
                    handle,
                    values.enabled(),
                    values.minDocFreq(),
                    values.maxDocFreq(),
                    values.minPartitionSize(),
                    values.maxIters(),
                    values.ramBudgetMb(),
                    values.minMergeDocs(),
                    values.minMergeRatio(),
                    values.fields()
                ),
                "setting bp_reorder"
            );
            bpReorder = values;
        }
    }

    /** {@code TieredMergePolicy}'s own MB-to-bytes conversion, capped as its setters cap it. */
    private static long megabytes(double mb) {
        return mb > Long.MAX_VALUE / 1024 / 1024 ? Long.MAX_VALUE : (long) (mb * 1024 * 1024);
    }

    /**
     * {@code IndexWriterConfig.setMaxFullFlushMergeWaitMillis}, with the merge policy wrapped in
     * {@code MergeOnFlushMergePolicy} when {@code mergeOnFlushPolicy}.
     */
    void setMergeOnFlush(long waitMillis, boolean mergeOnFlushPolicy) throws IOException {
        synchronized (lock) {
            check(NativeBridge.writerSetMergeOnFlush(handle, waitMillis, mergeOnFlushPolicy), "setting merge-on-flush");
            mergeOnFlushPolicyWait = mergeOnFlushPolicy ? waitMillis : -1;
        }
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
     * Makes every buffered change visible through a commit: commits it if there is any, then pins
     * the newest commit's files for a reader. Returns {@code {generation, hold}}; release the hold
     * with {@link #releaseHold} once the reader is closed. Live search keys the buffers it reads by
     * that commit's generation, so a writer with live search refreshes this way.
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

    /**
     * Flushes what is buffered and pins the writer's live segment list for a reader: returns its
     * {@code segments_N} bytes; {@code out[0]} is the hold to {@link #releaseHold} when the reader
     * closes, {@code out[1]} the last commit's generation. Adds go on meanwhile. The flushed buffers
     * leave the circuit breaker.
     */
    byte[] acquireNrt(long[] out) throws IOException {
        indexing.readLock().lock();
        try {
            ensureOpen();
            byte[][] bytes = new byte[1][];
            check(NativeBridge.writerNrtSnapshot(handle, out, bytes), "opening a near-real-time view");
            return bytes[0];
        } finally {
            reconcileBreaker();
            indexing.readLock().unlock();
        }
    }

    long latestGeneration() {
        return latestGeneration;
    }

    // ---- live search ----

    /** Whether searches read the buffered documents. */
    boolean liveSearch() {
        return liveSearch;
    }

    /**
     * A view of the documents buffered after commit {@code generation}: its handle and blob ({@link
     * NativeBridge#liveViewOpen}), or null when the buffers cannot be read for that commit (they were
     * let go, a delete by query reached them, the writer is closed); the reader is then searched
     * alone.
     */
    LiveViewHandle openLiveView(long generation) {
        if (liveSearch == false || indexing.readLock().tryLock() == false) {
            return null;
        }
        try {
            if (closed || nativeClosed) {
                return null;
            }
            long[] view = new long[1];
            byte[][] blob = new byte[1][];
            if (NativeBridge.liveViewOpen(handle, generation, view, blob) != NativeBridge.OK) {
                if (LOGGER.isDebugEnabled()) {
                    LOGGER.debug("lucene-rust: no view of the buffers for commit {}: {}", generation, NativeBridge.lastError());
                }
                return null;
            }
            return new LiveViewHandle(view[0], blob[0]);
        } finally {
            indexing.readLock().unlock();
        }
    }

    /** An open view: {@link NativeBridge#liveViewClose} it once nothing reads it. */
    record LiveViewHandle(long id, byte[] blob) {}

    /**
     * Lets go of the buffered documents a reader of commit {@code generation} already has, once no
     * view reads them: every new search reads that commit or a newer one.
     */
    void releaseLive(long generation) {
        synchronized (liveLock) {
            if (liveSearch == false || generation <= liveReleased) {
                return;
            }
            indexing.readLock().lock();
            try {
                if (nativeClosed == false && NativeBridge.writerReleaseLive(handle, generation) == NativeBridge.OK) {
                    liveReleased = generation;
                }
            } finally {
                indexing.readLock().unlock();
            }
        }
    }

    /**
     * The registered fields as Lucene field infos, for the leaf of buffered documents: the schema
     * every segment writes, so the leaf agrees with the segments' field infos. Null when a field
     * carries vectors, whose encoding the registry does not keep.
     */
    org.apache.lucene.index.FieldInfos liveFieldInfos() {
        synchronized (liveLock) {
            int size = registry.size();
            if (size != liveInfosSize) {
                liveInfos = buildLiveFieldInfos();
                liveInfosSize = size;
            }
            return liveInfos;
        }
    }

    private org.apache.lucene.index.FieldInfos buildLiveFieldInfos() {
        List<org.apache.lucene.index.FieldInfo> list = new ArrayList<>();
        for (FieldRegistry.Entry e : registry.entries()) {
            FieldSchema f = e.schema();
            if (f.vectorDimension != 0) {
                return null;
            }
            list.add(
                new org.apache.lucene.index.FieldInfo(
                    f.name,
                    e.number(),
                    f.storeTermVectors,
                    f.omitNorms,
                    false,
                    f.indexOptions,
                    f.docValuesType,
                    f.skipIndex,
                    -1,
                    new HashMap<>(f.attributes),
                    f.pointDimensionCount,
                    f.pointIndexDimensionCount,
                    f.pointNumBytes,
                    0,
                    org.apache.lucene.index.VectorEncoding.FLOAT32,
                    org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
                    f.softDeletes,
                    false
                )
            );
        }
        return new org.apache.lucene.index.FieldInfos(list.toArray(new org.apache.lucene.index.FieldInfo[0]));
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
        // The writer's merge threads take every merge its policy finds after each commit.
    }

    /** {@code index.merge.scheduler.*}, from the shard's {@code MergeSchedulerConfig}. */
    void setMergeScheduler(int maxMergeCount, int maxThreadCount, boolean autoThrottle, double maxForceMergeMbPerSec)
        throws IOException {
        if (closed) {
            return;
        }
        int status = NativeBridge.writerSetMergeScheduler(handle, maxMergeCount, maxThreadCount, autoThrottle, maxForceMergeMbPerSec);
        if (status == NativeBridge.INVALID_ARGUMENT) {
            throw new IllegalArgumentException(NativeBridge.lastError());
        }
        check(status, "setting the merge scheduler");
    }

    @Override
    public void forceMerge(int maxNumSegments, boolean doWait) throws IOException {
        runForceMerge(maxNumSegments, false);
    }

    private void runForceMerge(int maxNumSegments, boolean onlyDeletes) throws IOException {
        synchronized (lock) {
            ensureOpen();
            // `IndexWriter.forceMerge` flushes what is buffered first, and writes no commit when
            // nothing is: the merge is the only new commit then.
            if (stat(NativeBridge.STAT_UNCOMMITTED) != 0) {
                commitLocked();
            }
            applyRetention();
            long[] out = new long[1];
            // The merge commits with the last commit's user data, so nothing may be added meanwhile.
            indexing.writeLock().lock();
            try {
                check(NativeBridge.writerForceMerge(handle, maxNumSegments, onlyDeletes, out), "force merge");
            } finally {
                indexing.writeLock().unlock();
            }
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
        long[] out = new long[1];
        return NativeBridge.writerStats(handle, out) == NativeBridge.OK ? out[NativeBridge.STAT_RAM_BYTES] : 0;
    }

    @Override
    public long getFlushingBytes() {
        if (closed) {
            return 0;
        }
        long[] out = new long[NativeBridge.STAT_FLUSHING_BYTES + 1];
        return NativeBridge.writerStats(handle, out) == NativeBridge.OK ? out[NativeBridge.STAT_FLUSHING_BYTES] : 0;
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
            indexing.writeLock().lock();
            try {
                if (nativeClosed == false) {
                    nativeClosed = true;
                    NativeBridge.writerClose(handle);
                }
            } finally {
                indexing.writeLock().unlock();
            }
            synchronized (accounting) {
                if (breaker != null) {
                    breaker.addWithoutBreaking(-accounted);
                }
                accounted = 0;
            }
        }
    }

    /** Test hook: a panic inside the writer's lock, if the index enabled fault injection. */
    void injectPanic() throws IOException {
        synchronized (lock) {
            ensureOpen();
            check(NativeBridge.writerApply(handle, new byte[] { (byte) DocumentEncoder.OP_PANIC }, 1), "indexing");
        }
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
