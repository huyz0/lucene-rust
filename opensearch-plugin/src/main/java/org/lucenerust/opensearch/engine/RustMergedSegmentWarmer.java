/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.StringHelper;
import org.lucenerust.opensearch.NativeBridge;

/**
 * {@code IndexWriterConfig.setMergedSegmentWarmer} for the Rust writer: once a merge has written its
 * segment, and before the merge is published, the native writer calls back (the warm upcall) and the
 * engine's {@link IndexWriter.IndexReaderWarmer} gets the merged segment's {@link SegmentReader}, as
 * {@code IndexWriter.mergeMiddle} hands it -- which is how OpenSearch pre-copies merged segments to
 * segment-replication replicas.
 */
final class RustMergedSegmentWarmer {
    private static final Logger logger = LogManager.getLogger(RustMergedSegmentWarmer.class);

    private final Directory directory;
    private final int indexCreatedVersionMajor;
    private final IndexWriter.IndexReaderWarmer warmer;

    private RustMergedSegmentWarmer(Directory directory, int indexCreatedVersionMajor, IndexWriter.IndexReaderWarmer warmer) {
        this.directory = directory;
        this.indexCreatedVersionMajor = indexCreatedVersionMajor;
        this.warmer = warmer;
    }

    /** The warmer for each native writer that has one, by writer handle. */
    private static final Map<Long, RustMergedSegmentWarmer> WARMERS = new ConcurrentHashMap<>();
    private static boolean upcallInstalled;

    /** Has {@code warmer} warm every segment the writer {@code handle} names merges into {@code directory}. */
    static void install(long handle, Directory directory, int indexCreatedVersionMajor, IndexWriter.IndexReaderWarmer warmer) {
        installUpcall();
        WARMERS.put(handle, new RustMergedSegmentWarmer(directory, indexCreatedVersionMajor, warmer));
        if (NativeBridge.writerSetMergedSegmentWarmer(handle, true) != NativeBridge.OK) {
            WARMERS.remove(handle);
            throw new IllegalStateException("lucene-rust: setting the merged-segment warmer failed: " + NativeBridge.lastError());
        }
    }

    /** Forgets the warmer of the writer {@code handle} names, once it is closed. */
    static void remove(long handle) {
        WARMERS.remove(handle);
    }

    /** Registers {@link #warmUpcall} with the native library, once per JVM. */
    @SuppressWarnings("restricted") // an upcall stub is the point
    private static synchronized void installUpcall() {
        if (upcallInstalled) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    RustMergedSegmentWarmer.class,
                    "warmUpcall",
                    MethodType.methodType(int.class, long.class, MemorySegment.class, long.class, MemorySegment.class)
                );
            FunctionDescriptor fd = FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS);
            MemorySegment stub = Linker.nativeLinker().upcallStub(target, fd, Arena.global());
            if (NativeBridge.setWarmUpcall(stub) != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the warm upcall failed: " + NativeBridge.lastError());
            }
            upcallInstalled = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * The native writer's request to warm a merged segment ({@code WarmUpcall} in {@code
     * engine_writer.rs}), on the merge thread. It never throws, since an exception crossing an upcall
     * ends the JVM; a failure fails the merge, as a throwing warmer fails Lucene's.
     */
    @SuppressWarnings("restricted") // the segments are the native caller's, sized by its contract
    private static int warmUpcall(long handle, MemorySegment name, long nameLen, MemorySegment id) {
        RustMergedSegmentWarmer w = WARMERS.get(handle);
        if (w == null) {
            return NativeBridge.WARM_OK;
        }
        String segment = null;
        try {
            segment = new String(name.reinterpret(nameLen).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
            byte[] segmentId = id.reinterpret(StringHelper.ID_LENGTH).toArray(JAVA_BYTE);
            w.warm(segment, segmentId);
            return NativeBridge.WARM_OK;
        } catch (Throwable t) {
            try {
                logger.warn("warming merged segment [" + segment + "] failed", t);
            } catch (Throwable ignored) {
                // Nothing may cross the upcall.
            }
            return NativeBridge.WARM_FAILED;
        }
    }

    private void warm(String segment, byte[] segmentId) throws Exception {
        SegmentInfo info = Codec.getDefault().segmentInfoFormat().read(directory, segment, segmentId, IOContext.DEFAULT);
        SegmentCommitInfo commitInfo = new SegmentCommitInfo(info, 0, 0, -1, -1, -1, StringHelper.randomId());
        try (LeafReader reader = SEGMENT_READER.newInstance(commitInfo, indexCreatedVersionMajor, IOContext.DEFAULT)) {
            warmer.warm(reader);
        }
    }

    /** {@code SegmentReader(SegmentCommitInfo, int, IOContext)}, as {@code IndexWriter} opens a merged segment. */
    private static final Constructor<SegmentReader> SEGMENT_READER;

    static {
        try {
            SEGMENT_READER = SegmentReader.class.getDeclaredConstructor(SegmentCommitInfo.class, int.class, IOContext.class);
            SEGMENT_READER.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }
}
