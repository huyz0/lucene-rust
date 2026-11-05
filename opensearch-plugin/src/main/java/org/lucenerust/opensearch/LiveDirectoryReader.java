/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A searcher's reader with a view of the writer's buffers: the leaves of {@code base} (one of the
 * engine's readers, kept open as long as this one), then a {@link LiveLeafReader} of the view.
 * Its own cache key, so nothing cached for {@code base} answers for it; the native reader is
 * {@code base}'s ({@link #baseOf}), whose segments are the same. Closing it closes the view.
 */
public final class LiveDirectoryReader extends FilterDirectoryReader {
    private final OpenSearchDirectoryReader base;
    private final LiveLeafReader live;
    private final IndexReader.CacheKey key = CacheKeys.newKey();
    private final List<ClosedListener> listeners = new CopyOnWriteArrayList<>();
    private final CacheHelper cacheHelper = new CacheHelper() {
        @Override
        public CacheKey getKey() {
            return key;
        }

        @Override
        public void addClosedListener(ClosedListener listener) {
            listeners.add(listener);
        }
    };

    private LiveDirectoryReader(OpenSearchDirectoryReader base, LiveLeafReader live) throws IOException {
        super(base.getDelegate(), new Appending(live));
        this.base = base;
        this.live = live;
    }

    /**
     * {@code base} and {@code view}'s documents, wrapped for OpenSearch as {@code base} is. Takes a
     * reference on {@code base} and ownership of {@code view}: both go when the result closes.
     */
    public static OpenSearchDirectoryReader wrap(OpenSearchDirectoryReader base, LiveView view, FieldInfos infos)
        throws IOException {
        base.incRef();
        try {
            return OpenSearchDirectoryReader.wrap(new LiveDirectoryReader(base, new LiveLeafReader(view, infos)), base.shardId());
        } catch (IOException | RuntimeException e) {
            try {
                view.close();
            } finally {
                base.decRef();
            }
            throw e;
        }
    }

    /** The engine reader under {@code reader}'s view of the buffers, or null when it has none. */
    public static DirectoryReader baseOf(IndexReader reader) {
        LiveDirectoryReader live = find(reader);
        return live == null ? null : live.base;
    }

    /** The {@link LiveDirectoryReader} under {@code reader}'s wrappers, or null. */
    public static LiveDirectoryReader find(IndexReader reader) {
        while (reader instanceof FilterDirectoryReader f) {
            if (f instanceof LiveDirectoryReader live) {
                return live;
            }
            reader = f.getDelegate();
        }
        return null;
    }

    /** The view of the buffers. */
    public LiveView view() {
        return live.view();
    }

    /** The doc ID of the view's first document: every document of {@code base} comes before. */
    public int liveDocBase() {
        return base.maxDoc();
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) {
        throw new UnsupportedOperationException("a view of the buffers is not reopened");
    }

    @Override
    protected void doClose() throws IOException {
        try {
            for (ClosedListener l : listeners) {
                l.onClose(key);
            }
            live.close();
        } finally {
            try {
                live.view().close();
            } finally {
                base.decRef();
            }
        }
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return cacheHelper;
    }

    /** {@code in}'s leaves, then the leaf of buffered documents. */
    private static final class Appending extends SubReaderWrapper {
        private final LiveLeafReader live;

        Appending(LiveLeafReader live) {
            this.live = live;
        }

        @Override
        protected LeafReader[] wrap(List<? extends LeafReader> readers) {
            LeafReader[] out = readers.toArray(new LeafReader[readers.size() + 1]);
            out[readers.size()] = live;
            return out;
        }

        @Override
        public LeafReader wrap(LeafReader reader) {
            return reader;
        }
    }
}
