/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

/**
 * The JNI surface of {@code liblucene_ffi} ({@code crates/lucene-ffi/src/jni_bridge.rs}).
 *
 * <p>Every call returns a status code ({@link #OK} on success) and never throws: results come
 * back through caller-allocated arrays, and a failure's message through {@link #lastError()}, which
 * is thread-local on the Rust side and so must be read on the thread that saw the failure. A Rust
 * panic is caught before it reaches the JVM and reported as {@link #PANIC}.
 */
public final class NativeBridge {
    /** The contract version this jar was built against; {@code JVM_ABI_VERSION} in {@code jvm_reader.rs}. */
    public static final int EXPECTED_ABI_VERSION = 5;

    public static final int OK = 0;
    public static final int INVALID_HANDLE = 3;
    public static final int IO = 4;
    public static final int DECODE = 5;
    public static final int BUFFER_TOO_SMALL = 8;
    public static final int PANIC = 9;
    public static final int INVALID_ARGUMENT = 10;

    /** Query blob tags ({@code decode_query} in {@code jvm_reader.rs}). */
    public static final byte QUERY_TERM = 0;
    public static final byte QUERY_BOOLEAN = 1;

    private NativeBridge() {}

    public static native int abiVersion();

    public static native String lastError();

    /**
     * Opens a reader over the segments listed in {@code segmentInfos} (bytes written by {@code
     * SegmentInfos.write(IndexOutput)} at {@code generation}), checking that segment {@code i} has
     * {@code maxDocs[i]} documents and masking it with {@code liveDocs[i]} ({@code
     * FixedBitSet.getBits()} words, exactly {@code ceil(maxDoc / 64)} of them; null for no
     * deletions; a null array for none anywhere). {@code previous} is an open handle whose unchanged
     * segments are reused, or 0. The new handle is written to {@code outHandle[0]}; it is immutable,
     * so searches on it never wait on each other or on opens and closes.
     */
    public static native int openReader(
        byte[] indexPathUtf8,
        byte[] segmentInfos,
        long generation,
        long previous,
        int[] maxDocs,
        long[][] liveDocs,
        long[] outHandle
    );

    /**
     * Runs a query blob. {@code outCounts[0]} receives the number of hits written; {@code
     * outCounts[1]} the total hits, exact below {@code countLimit} and otherwise a lower bound of at
     * least {@code countLimit}, with {@code outCounts[2]} 1 in that case -- Lucene's {@code
     * totalHitsThreshold}. A {@code countLimit} of 0 counts nothing ({@code outCounts[1]} is -1);
     * {@link Long#MAX_VALUE} counts exactly.
     */
    public static native int search(
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    );

    public static native int closeReader(long handle);

    // ---- The engine writer (engine_writer.rs, M5). ----

    /** Indices into {@link #writerStats}' output. */
    public static final int STAT_RAM_BYTES = 0;
    public static final int STAT_PENDING_DOCS = 1;
    public static final int STAT_UNCOMMITTED = 2;
    public static final int STAT_GENERATION = 3;
    public static final int STAT_SEGMENTS = 4;
    public static final int STAT_COUNT = 5;

    /**
     * Opens a shard's writer over the index at {@code path}, which must already hold a commit. Commit
     * points are dropped only through {@link #writerDeleteCommits}. {@code faultInjection} allows the
     * test-only panic operation.
     */
    public static native int writerOpen(byte[] indexPathUtf8, double ramBufferMb, boolean faultInjection, long[] outHandle);

    /** Registers a field spec (see {@code decode_field}); its global number goes to {@code outNumber[0]}. */
    public static native int writerRegisterField(long handle, byte[] spec, int[] outNumber);

    /**
     * Applies the operation in {@code op[0..len)} (see {@code decode_op}). {@link #INVALID_ARGUMENT} and
     * {@link #DECODE} refuse the document and change nothing; any other failure is tragic.
     */
    public static native int writerApply(long handle, byte[] op, int len);

    /** Commits with {@code userData} (see {@code decode_user_data}), runs merges, writes the newest generation. */
    public static native int writerCommit(long handle, byte[] userData, long[] outGeneration);

    /** The generations of every live commit point, oldest first; their number goes to {@code outLen[0]}. */
    public static native int writerCommitGenerations(long handle, long[] out, int[] outLen);

    /** Drops the named commit points (never the newest). */
    public static native int writerDeleteCommits(long handle, long[] generations);

    /** Pins a commit's segment files for a reader; the hold id goes to {@code outHold[0]}. */
    public static native int writerHoldCommit(long handle, long generation, long[] outHold);

    public static native int writerReleaseHold(long handle, long hold);

    /** Soft-deleted documents with {@code _seq_no} below {@code minRetainedSeqNo} are dropped by merges. */
    public static native int writerSetRetention(long handle, boolean enabled, long minRetainedSeqNo);

    /** {@code forceMerge(maxSegments)}, or {@code forceMergeDeletes()}; writes the newest generation. */
    public static native int writerForceMerge(long handle, int maxSegments, boolean onlyDeletes, long[] outGeneration);

    /** Fills {@code out} with the {@code STAT_*} counters. */
    public static native int writerStats(long handle, long[] out);

    /** Closes without committing (Lucene's {@code rollback()}). */
    public static native int writerClose(long handle);
}
