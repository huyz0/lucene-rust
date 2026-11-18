/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.NativeLive;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Realtime search over the accelerated writer's buffers ({@code index.search.realtime.*}): opens
 * and releases the views of the documents a reader of the segment list lacks, keeps the reader
 * searches share under a staleness bound, and builds the field infos of the leaf of buffered
 * documents. The writer owns the native handle and its locks; this holds the live-search state and
 * calls back for them.
 */
final class LiveSearchState {
    private static final Logger LOGGER = LogManager.getLogger(LiveSearchState.class);

    private final AcceleratedIndexWriter writer;
    /** Whether searches read the buffered documents ({@code index.search.realtime.enabled}). */
    private volatile boolean enabled;
    /** How old a view of the buffers a search may read ({@code index.search.realtime.max_staleness}). */
    private volatile long stalenessNanos;
    /** The reader over the view searches share under {@link #stalenessNanos}, kept until a refresh. */
    private final SharedLiveReader sharedLive = new SharedLiveReader();
    /** The newest segment list version whose readers' buffers {@link #release} let go; under {@link #liveLock}. */
    private long liveReleased = -1;
    private final Object liveLock = new Object();
    /** The leaf's field infos and the registry size they were built at; under {@link #liveLock}. */
    private org.apache.lucene.index.FieldInfos liveInfos;
    private int liveInfosSize = -1;

    LiveSearchState(AcceleratedIndexWriter writer) {
        this.writer = writer;
    }

    void enable(boolean enabled) {
        this.enabled = enabled;
    }

    /** Whether searches read the buffered documents. */
    boolean enabled() {
        return enabled;
    }

    /** How long, at most, a write takes to show in a search: zero when every search sees every write before it. */
    long stalenessNanos() {
        return stalenessNanos;
    }

    /** Lets searches read a view of the buffers up to {@code max} old ({@link NativeLive#writerSetLiveStaleness}). */
    void setStaleness(org.opensearch.common.unit.TimeValue max) throws IOException {
        if (enabled == false || max.millis() == 0) {
            return;
        }
        writer.locked(() -> NativeLive.writerSetLiveStaleness(writer.handle(), max.millis()), "setting the realtime staleness");
        stalenessNanos = max.nanos();
    }

    /**
     * Makes every later search read a view of the buffers opened from now on; false when it could not
     * (the writer busy with a force merge, or closed), to be tried again.
     */
    boolean refreshShared() throws IOException {
        if (writer.indexingLock().readLock().tryLock() == false) {
            return false;
        }
        try {
            if (writer.closedOrNative()) {
                return false;
            }
            writer.check(NativeLive.writerLiveRefreshShared(writer.handle()), "refreshing the shared realtime view");
            return true;
        } finally {
            writer.indexingLock().readLock().unlock();
        }
    }

    /**
     * A view of the documents a reader of the segment list at {@code version} lacks: its handle and blob ({@link
     * NativeLive#liveViewOpen}), or null when the buffers cannot be read for that list (they were
     * let go, a delete by query reached them, the writer is closed); the reader is then searched
     * alone.
     */
    ViewHandle openView(long version) {
        if (enabled == false || writer.indexingLock().readLock().tryLock() == false) {
            return null;
        }
        try {
            if (writer.closedOrNative()) {
                return null;
            }
            long[] view = new long[1];
            byte[][] blob = new byte[1][];
            if (NativeLive.liveViewOpen(writer.handle(), version, view, blob) != NativeBridge.OK) {
                if (LOGGER.isDebugEnabled()) {
                    LOGGER.debug("accelerated-search: no view of the buffers for segment list version {}: {}", version, NativeBridge.lastError());
                }
                return null;
            }
            return new ViewHandle(view[0], blob[0]);
        } finally {
            writer.indexingLock().readLock().unlock();
        }
    }

    /** An open view: {@link NativeLive#liveViewClose} it once nothing reads it. */
    record ViewHandle(long id, byte[] blob) {}

    /** The reader over the view searches share, kept until the next refresh. */
    SharedLiveReader sharedLive() {
        return sharedLive;
    }

    /**
     * Lets the kept reader over the shared view go, on a refresh or, with {@code close}, for good
     * when the writer closes.
     */
    void releaseShared(boolean close) {
        OpenSearchDirectoryReader old = sharedLive.clear(close);
        if (old != null) {
            try {
                old.decRef();
            } catch (IOException e) {
                LOGGER.warn("accelerated-search: closing a realtime search's view failed", e);
            }
        }
    }

    /**
     * Lets go of the buffered documents a reader of the segment list at {@code version} already has,
     * once no view reads them: every new search reads that list or a newer one.
     */
    void release(long version) {
        synchronized (liveLock) {
            if (enabled == false || version <= liveReleased) {
                return;
            }
            writer.indexingLock().readLock().lock();
            try {
                if (writer.nativeClosed() == false && NativeLive.writerReleaseLive(writer.handle(), version) == NativeBridge.OK) {
                    liveReleased = version;
                }
            } finally {
                writer.indexingLock().readLock().unlock();
            }
        }
    }

    /**
     * The registered fields as Lucene field infos, for the leaf of buffered documents: the schema
     * every segment writes, so the leaf agrees with the segments' field infos.
     */
    org.apache.lucene.index.FieldInfos fieldInfos() {
        synchronized (liveLock) {
            int size = writer.registry().size();
            if (size != liveInfosSize) {
                liveInfos = buildFieldInfos();
                liveInfosSize = size;
            }
            return liveInfos;
        }
    }

    private org.apache.lucene.index.FieldInfos buildFieldInfos() {
        List<org.apache.lucene.index.FieldInfo> list = new ArrayList<>();
        for (FieldRegistry.Entry e : writer.registry().entries()) {
            FieldSchema f = e.schema();
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
                    f.vectorDimension,
                    f.vectorEncoding,
                    f.vectorSimilarity,
                    f.softDeletes,
                    false
                )
            );
        }
        return new org.apache.lucene.index.FieldInfos(list.toArray(new org.apache.lucene.index.FieldInfo[0]));
    }

}
