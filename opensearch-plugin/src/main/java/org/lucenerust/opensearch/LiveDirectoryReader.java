/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexCommit;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * SPIKE (real-time study, not for merge): the refreshed segments of {@code base} plus one live leaf.
 * Not reopenable; it holds {@code base} open for its own life.
 */
public final class LiveDirectoryReader extends DirectoryReader {
    private final DirectoryReader base;

    private final CacheKey cacheKey = newKey();

    private static CacheKey newKey() {
        try {
            var c = CacheKey.class.getDeclaredConstructor();
            c.setAccessible(true);
            return c.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
    private final java.util.List<ClosedListener> closedListeners = new java.util.concurrent.CopyOnWriteArrayList<>();
    private final CacheHelper cacheHelper = new CacheHelper() {
        @Override
        public CacheKey getKey() {
            return cacheKey;
        }

        @Override
        public void addClosedListener(ClosedListener listener) {
            closedListeners.add(listener);
        }
    };

    private void fireClosed() throws java.io.IOException {
        for (ClosedListener l : closedListeners) {
            l.onClose(cacheKey);
        }
    }

    private LiveDirectoryReader(Directory dir, LeafReader[] leaves, DirectoryReader base) throws IOException {
        super(dir, leaves, null);
        this.base = base;
    }

    /** {@code base}'s leaves plus {@code live}; {@code base} is incRef'd and released on close. */
    public static LiveDirectoryReader of(DirectoryReader base, LeafReader live) throws IOException {
        List<LeafReader> leaves = new ArrayList<>();
        for (LeafReaderContext ctx : base.leaves()) {
            leaves.add(ctx.reader());
        }
        leaves.add(live);
        base.incRef();
        return new LiveDirectoryReader(base.directory(), leaves.toArray(new LeafReader[0]), base);
    }

    @Override
    protected DirectoryReader doOpenIfChanged() {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(ExecutorService executor) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexCommit commit) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexCommit commit, ExecutorService executor) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexWriter writer, boolean applyAllDeletes) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexWriter writer, boolean applyAllDeletes, ExecutorService executor) {
        return null;
    }

    @Override
    public long getVersion() {
        return base.getVersion();
    }

    @Override
    public boolean isCurrent() {
        return false;
    }

    @Override
    public IndexCommit getIndexCommit() throws IOException {
        return base.getIndexCommit();
    }

    @Override
    protected void doClose() throws IOException {
        try {
            fireClosed();
        } finally {
            base.decRef();
        }
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return cacheHelper;
    }
}
