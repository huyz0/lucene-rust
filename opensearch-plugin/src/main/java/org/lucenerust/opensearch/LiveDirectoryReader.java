/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.BaseCompositeReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;
import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A searcher's reader with a view of the writer's buffers: the leaves of {@code base} (one of the
 * engine's readers, kept open as long as this one), then a {@link LiveLeafReader} per chunk of the
 * view's buffers.
 * Its own cache key, so nothing cached for {@code base} answers for it; the native reader is
 * {@code base}'s ({@link #baseOf}), whose segments are the same. Closing it closes the view.
 *
 * <p>Lucene sees only {@code base}'s documents, as of its commit, until a search the native engine
 * cannot run calls {@link #readByLucene}: then every leaf of buffered documents holds a segment of
 * them, and {@code base}'s leaves leave out what was deleted since the commit, so Lucene answers as
 * the native engine would.
 */
public final class LiveDirectoryReader extends FilterDirectoryReader {
    private final OpenSearchDirectoryReader base;
    private final LiveView view;
    private final LiveLeafReader[] live;
    private final Committed[] committed;
    /** Whether {@link #readByLucene} gave Lucene the buffered documents. */
    private volatile boolean lucene;
    private volatile IndexReader.CacheKey key = CacheKeys.newKey();
    /** Each listener, with the key it was added under: it hears of that key when the reader closes. */
    private final List<java.util.Map.Entry<IndexReader.CacheKey, ClosedListener>> listeners = new CopyOnWriteArrayList<>();
    private final CacheHelper cacheHelper = new CacheHelper() {
        @Override
        public CacheKey getKey() {
            return key;
        }

        @Override
        public void addClosedListener(ClosedListener listener) {
            listeners.add(java.util.Map.entry(key, listener));
        }
    };

    private LiveDirectoryReader(OpenSearchDirectoryReader base, LiveView view, LiveLeafReader[] live) throws IOException {
        this(base, view, live, new Appending(live));
    }

    private LiveDirectoryReader(OpenSearchDirectoryReader base, LiveView view, LiveLeafReader[] live, Appending appending)
        throws IOException {
        super(base.getDelegate(), appending);
        this.base = base;
        this.view = view;
        this.live = live;
        this.committed = appending.committed.toArray(new Committed[0]);
    }

    /**
     * Gives Lucene the view's documents, once: the buffers as segments of the native reader at
     * {@code nativeHandle} ({@code base}'s), waiting up to {@code waitMillis} for a buffer being
     * flushed; {@code top} is the searcher's reader, which wraps this one. False when they cannot be
     * read: Lucene then goes on seeing {@code base} alone. What was read before it across the
     * leaves (the global ordinals aggregators built) is of the old documents: the reader takes a new
     * cache key, so none of it answers after it.
     */
    public synchronized boolean readByLucene(long nativeHandle, long waitMillis, IndexReader top) throws IOException {
        if (lucene) {
            return true;
        }
        byte[][] out = new byte[1][];
        int rc = NativeBridge.liveViewLucene(nativeHandle, view.id(), waitMillis, out);
        if (rc != NativeBridge.OK) {
            return false;
        }
        int[] baseMaxDocs = new int[committed.length];
        for (int i = 0; i < committed.length; i++) {
            baseMaxDocs[i] = committed[i].maxDoc();
        }
        int[] docs = new int[live.length];
        for (int i = 0; i < live.length; i++) {
            docs[i] = live[i].maxDoc();
        }
        LiveSegments segments = LiveSegments.open(
            out[0],
            baseMaxDocs,
            docs,
            ((StandardDirectoryReader) FilterDirectoryReader.unwrap(base)).directory()
        );
        for (int i = 0; i < committed.length; i++) {
            if (segments.baseLive[i] != null) {
                committed[i].deleteSinceCommit(segments.baseLive[i]);
            }
        }
        for (int i = 0; i < live.length; i++) {
            live[i].useLucene(segments.leaves[i], view.chunkLive(i), segments.owners[i]);
        }
        IndexReader.CacheKey renewed = CacheKeys.newKey();
        key = renewed;
        lucene = true;
        // numDocs() is counted once per composite reader: every one from the searcher's down to
        // this counts again.
        for (IndexReader r = top; r instanceof FilterDirectoryReader f; r = f.getDelegate()) {
            recount(f);
            if (f == this) {
                break;
            }
        }
        return true;
    }

    /** Whether {@link #readByLucene} gave Lucene the view's documents. */
    public boolean readsLucene() {
        return lucene;
    }

    private static final Field NUM_DOCS = Reflect.declared(BaseCompositeReader.class, "numDocs");

    /** Has {@code reader} count its live documents again on its next {@code numDocs()}. */
    private static void recount(IndexReader reader) {
        if (NUM_DOCS == null) {
            return;
        }
        try {
            if (NUM_DOCS.get(reader) instanceof AtomicInteger n) {
                n.set(-1);
            }
        } catch (IllegalAccessException e) {
            // Left as counted.
        }
    }

    /**
     * {@code base} and {@code view}'s documents, wrapped for OpenSearch as {@code base} is. Takes a
     * reference on {@code base} and ownership of {@code view}: both go when the result closes.
     */
    public static OpenSearchDirectoryReader wrap(OpenSearchDirectoryReader base, LiveView view, FieldInfos infos)
        throws IOException {
        base.incRef();
        try {
            LiveLeafReader[] leaves = new LiveLeafReader[view.chunks()];
            for (int c = 0; c < leaves.length; c++) {
                leaves[c] = new LiveLeafReader(view, view.chunkBuffer(c), view.chunkStart(c), view.chunkMaxDoc(c), infos);
            }
            return OpenSearchDirectoryReader.wrap(new LiveDirectoryReader(base, view, leaves), base.shardId());
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
        return view;
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
            for (java.util.Map.Entry<IndexReader.CacheKey, ClosedListener> l : listeners) {
                l.getValue().onClose(l.getKey());
            }
            IOUtils.close(live);
        } finally {
            try {
                view.close();
            } finally {
                base.decRef();
            }
        }
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return cacheHelper;
    }

    /** {@code in}'s leaves, each as {@link Committed}, then the leaves of buffered documents. */
    private static final class Appending extends SubReaderWrapper {
        private final LiveLeafReader[] live;
        private final List<Committed> committed = new java.util.ArrayList<>();

        Appending(LiveLeafReader[] live) {
            this.live = live;
        }

        @Override
        protected LeafReader[] wrap(List<? extends LeafReader> readers) {
            LeafReader[] out = new LeafReader[readers.size() + live.length];
            for (int i = 0; i < readers.size(); i++) {
                out[i] = wrap(readers.get(i));
            }
            System.arraycopy(live, 0, out, readers.size(), live.length);
            return out;
        }

        @Override
        public LeafReader wrap(LeafReader reader) {
            Committed c = new Committed(reader);
            committed.add(c);
            return c;
        }
    }

    /**
     * A leaf of the reader's commit, as it is, until {@link #readByLucene}: then without the
     * documents deleted since the commit, which the buffers may hold again (an update's new
     * version).
     */
    static final class Committed extends SequentialStoredFieldsLeafReader {
        private volatile Bits live;
        private volatile int numDocs = -1;
        private volatile CacheHelper readerHelper;

        Committed(LeafReader in) {
            super(in);
        }

        @Override
        protected StoredFieldsReader doGetSequentialStoredFieldsReader(StoredFieldsReader reader) {
            return reader;
        }

        void deleteSinceCommit(FixedBitSet liveNow) {
            if (liveNow.length() != maxDoc()) {
                throw new IllegalArgumentException("live docs of " + liveNow.length() + " for a segment of " + maxDoc());
            }
            numDocs = liveNow.cardinality();
            live = liveNow;
            IndexReader.CacheKey k = CacheKeys.newKey();
            readerHelper = new CacheHelper() {
                @Override
                public CacheKey getKey() {
                    return k;
                }

                @Override
                public void addClosedListener(ClosedListener listener) {
                    // Closed as the leaf under it is.
                    CacheHelper under = in.getReaderCacheHelper();
                    if (under != null) {
                        under.addClosedListener(closed -> listener.onClose(k));
                    }
                }
            };
        }

        @Override
        public Bits getLiveDocs() {
            Bits l = live;
            return l != null ? l : in.getLiveDocs();
        }

        @Override
        public int numDocs() {
            int n = numDocs;
            return n >= 0 ? n : in.numDocs();
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            CacheHelper h = readerHelper;
            return h != null ? h : in.getReaderCacheHelper();
        }
    }
}
