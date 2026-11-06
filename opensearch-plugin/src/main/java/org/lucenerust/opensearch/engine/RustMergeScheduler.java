/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.common.logging.Loggers;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ConcurrentCollections;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.MergeSchedulerConfig;
import org.opensearch.index.merge.MergeStats;
import org.opensearch.index.merge.MergeStatsTracker;
import org.opensearch.index.merge.MergedSegmentTransferTracker;
import org.opensearch.index.merge.OnGoingMerge;

/**
 * {@code OpenSearchConcurrentMergeScheduler} for the Rust writer, whose own merge scheduler runs the
 * merges: the native writer reports each merge before and after it runs (the merge upcall), and this
 * class keeps the same merge stats and ongoing-merge set and calls the same {@link #beforeMerge},
 * {@link #afterMerge} and {@link #handleMergeException} hooks the engine overrides. As with
 * OpenSearch's scheduler, the native scheduler never stalls indexing threads; the engine throttles
 * indexing when merges fall behind.
 */
public class RustMergeScheduler implements Closeable {

    protected final Logger logger;
    private final MergeSchedulerConfig config;
    private final MergedSegmentTransferTracker mergedSegmentTransferTracker;
    private final MergeStatsTracker mergeStatsTracker = new MergeStatsTracker();
    private final Set<OnGoingMerge> onGoingMerges = ConcurrentCollections.newConcurrentSet();
    private final Set<OnGoingMerge> readOnlyOnGoingMerges = Collections.unmodifiableSet(onGoingMerges);
    /** The running merges by native merge id. */
    private final Map<Long, NativeMerge> running = new ConcurrentHashMap<>();
    private volatile RustIndexWriter writer;

    protected RustMergeScheduler(ShardId shardId, IndexSettings indexSettings, MergedSegmentTransferTracker mergedSegmentTransferTracker) {
        this.config = indexSettings.getMergeSchedulerConfig();
        this.mergedSegmentTransferTracker = mergedSegmentTransferTracker;
        this.logger = Loggers.getLogger(getClass(), shardId);
    }

    /** Starts hearing {@code writer}'s merges. */
    void attach(RustIndexWriter writer) {
        RustMergeScheduler previous = SCHEDULERS.put(writer.handle(), this);
        assert previous == null || previous == this : "two schedulers for one writer";
        this.writer = writer;
    }

    public Set<OnGoingMerge> onGoingMerges() {
        return readOnlyOnGoingMerges;
    }

    /** {@code index.merge.scheduler.max_merge_count}: merges in flight before indexing is throttled. */
    public int getMaxMergeCount() {
        return config.getMaxMergeCount();
    }

    /** A callback before a merge starts. */
    protected void beforeMerge(OnGoingMerge merge) {}

    /** A callback after a merge ended, successfully or not. */
    protected void afterMerge(OnGoingMerge merge) {}

    /** A callback after a merge failed, on the thread that ran it. */
    protected void handleMergeException(Throwable exc) {}

    public MergeStats stats() {
        RustIndexWriter w = writer;
        double rate = Double.POSITIVE_INFINITY;
        if (config.isAutoThrottle() && w != null) {
            rate = w.mergeIoRateMbPerSec();
        }
        final MergeStats mergeStats = mergeStatsTracker.toMergeStats(rate);
        mergeStats.add(mergedSegmentTransferTracker.stats());
        return mergeStats;
    }

    /** The scheduler's settings are pushed to the native writer by {@code RustEngineSupport.refreshMergeScheduler}. */
    public void refreshConfig() {}

    @Override
    public void close() {
        RustIndexWriter w = writer;
        if (w != null) {
            SCHEDULERS.remove(w.handle(), this);
        }
    }

    private void onBefore(long id, long docs, long bytes, String names) {
        NativeMerge merge = new NativeMerge(segments(names), docs, bytes);
        mergeStatsTracker.beforeMerge(docs, bytes);
        OnGoingMerge onGoingMerge = new OnGoingMerge(merge);
        merge.onGoing = onGoingMerge;
        running.put(id, merge);
        onGoingMerges.add(onGoingMerge);
        if (logger.isTraceEnabled()) {
            logger.trace(
                "merge starting..., merging [{}] segments, [{}] docs, [{}] size",
                merge.segments.size(),
                docs,
                new ByteSizeValue(bytes)
            );
        }
        beforeMerge(onGoingMerge);
    }

    private void onAfter(long id, long docs, long bytes, long tookMS, long stoppedMS, long throttledMS, int failureKind, String failure) {
        NativeMerge merge = running.remove(id);
        if (merge == null) {
            // Started before this scheduler was attached.
            return;
        }
        try {
            onGoingMerges.remove(merge.onGoing);
            afterMerge(merge.onGoing);
        } finally {
            mergeStatsTracker.afterMerge(tookMS, docs, bytes);
            mergeStatsTracker.incStoppedTime(stoppedMS);
            mergeStatsTracker.incThrottledTime(throttledMS);
            if (tookMS > 20000) {
                logger.debug(
                    "merge done: took [{}], [{} MB], [{} docs], [{} stopped], [{} throttled]",
                    TimeValue.timeValueMillis(tookMS),
                    bytes / 1024f / 1024f,
                    docs,
                    TimeValue.timeValueMillis(stoppedMS),
                    TimeValue.timeValueMillis(throttledMS)
                );
            }
        }
        if (failureKind != NativeBridge.MERGE_OK) {
            handleMergeException(failureKind == NativeBridge.MERGE_FAILED_IO ? new IOException(failure) : new RuntimeException(failure));
        }
    }

    // ---- the merge upcall ----

    /** The scheduler hearing each open writer's merges, by writer handle. */
    private static final Map<Long, RustMergeScheduler> SCHEDULERS = new ConcurrentHashMap<>();
    private static boolean upcallInstalled;

    /** Stops reporting the merges of the writer {@code handle} names, once it is closed. */
    static void detach(long handle) {
        SCHEDULERS.remove(handle);
    }

    /** Registers {@link #mergeUpcall} with the native library, once per JVM, before any writer opens. */
    @SuppressWarnings("restricted") // an upcall stub is the point
    static synchronized void installUpcall() {
        if (upcallInstalled) {
            return;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                .findStatic(
                    RustMergeScheduler.class,
                    "mergeUpcall",
                    MethodType.methodType(
                        void.class,
                        long.class,
                        int.class,
                        long.class,
                        long.class,
                        long.class,
                        int.class,
                        long.class,
                        long.class,
                        long.class,
                        int.class,
                        MemorySegment.class,
                        long.class,
                        MemorySegment.class,
                        long.class
                    )
                );
            FunctionDescriptor fd = FunctionDescriptor.ofVoid(
                JAVA_LONG,
                JAVA_INT,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_INT,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_INT,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG
            );
            MemorySegment stub = Linker.nativeLinker().upcallStub(target, fd, Arena.global());
            if (NativeBridge.setMergeUpcall(stub) != NativeBridge.OK) {
                throw new IllegalStateException("lucene-rust: registering the merge upcall failed: " + NativeBridge.lastError());
            }
            upcallInstalled = true;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * The native writer's report of one merge ({@code MergeUpcall} in {@code engine_writer.rs}), on the
     * thread running it. It never throws, since an exception crossing an upcall ends the JVM.
     */
    @SuppressWarnings("restricted") // the segments are the native caller's, sized by its contract
    private static void mergeUpcall(
        long handle,
        int event,
        long id,
        long docs,
        long bytes,
        int maxNumSegments,
        long tookMS,
        long stoppedMS,
        long throttledMS,
        int failureKind,
        MemorySegment names,
        long namesLen,
        MemorySegment failure,
        long failureLen
    ) {
        RustMergeScheduler scheduler = SCHEDULERS.get(handle);
        if (scheduler == null) {
            return;
        }
        try {
            if (event == NativeBridge.MERGE_EVENT_BEFORE) {
                scheduler.onBefore(id, docs, bytes, utf8(names, namesLen));
            } else {
                String message = failureLen == 0 ? null : utf8(failure, failureLen);
                scheduler.onAfter(id, docs, bytes, tookMS, stoppedMS, throttledMS, failureKind, message);
            }
        } catch (Throwable t) {
            try {
                scheduler.logger.warn("merge callback failed", t);
            } catch (Throwable ignored) {
                // Nothing may cross the upcall.
            }
        }
    }

    @SuppressWarnings("restricted") // sized by the upcall's contract
    private static String utf8(MemorySegment s, long len) {
        if (len == 0) {
            return "";
        }
        return new String(s.reinterpret(len).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
    }

    // ---- the merges, as Lucene merges ----

    /** Carries segment names only: the native writer owns the real segments. */
    private static final Directory NAMES_ONLY = new ByteBuffersDirectory();

    /** The merged segments, named, for {@code OnGoingMerge.getMergedSegments}. */
    private static List<SegmentCommitInfo> segments(String names) {
        List<SegmentCommitInfo> out = new ArrayList<>();
        for (String name : names.split("\n")) {
            if (name.isEmpty() == false) {
                out.add(named(name));
            }
        }
        if (out.isEmpty()) {
            out.add(named("_na_"));
        }
        return out;
    }

    private static SegmentCommitInfo named(String name) {
        SegmentInfo info = new SegmentInfo(
            NAMES_ONLY,
            Version.LATEST,
            Version.LATEST,
            name,
            0,
            false,
            false,
            null,
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null
        );
        return new SegmentCommitInfo(info, 0, 0, -1, -1, -1, StringHelper.randomId());
    }

    /** A native merge as a {@code OneMerge}: its sources, documents and bytes. */
    private static final class NativeMerge extends MergePolicy.OneMerge {
        private final long docs;
        private final long bytes;
        OnGoingMerge onGoing;

        NativeMerge(List<SegmentCommitInfo> segments, long docs, long bytes) {
            super(segments);
            this.docs = docs;
            this.bytes = bytes;
        }

        @Override
        public long totalBytesSize() {
            return bytes;
        }

        @Override
        public int totalNumDocs() {
            return (int) Math.min(Integer.MAX_VALUE, docs);
        }
    }
}
