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
    public static final int EXPECTED_ABI_VERSION = 1;

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
     * {@code maxDocs[i]} documents. {@code previous} is an open handle whose unchanged segments are
     * reused, or 0. The new handle is written to {@code outHandle[0]}.
     */
    public static native int openReader(
        byte[] indexPathUtf8,
        byte[] segmentInfos,
        long generation,
        long previous,
        int[] maxDocs,
        long[] outHandle
    );

    /** Replaces segment {@code segment}'s live docs; {@code words} null means no deletions. */
    public static native int setLiveDocs(long handle, int segment, long[] words);

    /**
     * Runs a query blob. {@code outCounts[0]} receives the number of hits written, {@code
     * outCounts[1]} the exact total when {@code countTotal}, else -1.
     */
    public static native int search(
        long handle,
        byte[] query,
        int topN,
        boolean countTotal,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    );

    public static native int closeReader(long handle);
}
