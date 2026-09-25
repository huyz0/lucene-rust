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
    public static final int EXPECTED_ABI_VERSION = 4;

    public static final int OK = 0;
    public static final int INVALID_HANDLE = 3;
    public static final int PANIC = 9;

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
}
