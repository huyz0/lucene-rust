/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;

/**
 * The reader over the view of the buffers searches share under a staleness bound ({@code
 * index.lucene_rust.realtime_max_staleness}): the native engine gives every search of a shared
 * view the same handle, so a search that gets the handle the kept reader reads, over the same
 * commit's reader, takes that reader instead of wrapping its own.
 */
final class SharedLiveReader {
    private long view;
    private OpenSearchDirectoryReader base;
    /** The kept reader, with the reference this holds on it; null when none is kept. */
    private OpenSearchDirectoryReader reader;
    /** Set once the writer closed: nothing is kept after. */
    private boolean closed;

    /**
     * The kept reader of view handle {@code view} over {@code base}, with a reference taken for the
     * caller, or null.
     */
    synchronized OpenSearchDirectoryReader take(long view, OpenSearchDirectoryReader base) {
        if (reader != null && this.view == view && this.base == base && reader.tryIncRef()) {
            return reader;
        }
        return null;
    }

    /**
     * Keeps {@code reader}, view handle {@code view} over {@code base}, taking a reference on it,
     * unless the writer closed. Returns the reader it kept before, whose reference the caller drops,
     * or null.
     */
    synchronized OpenSearchDirectoryReader keep(long view, OpenSearchDirectoryReader base, OpenSearchDirectoryReader reader) {
        if (closed) {
            return null;
        }
        reader.incRef();
        OpenSearchDirectoryReader old = this.reader;
        this.view = view;
        this.base = base;
        this.reader = reader;
        return old;
    }

    /**
     * Forgets the kept reader, and keeps none from now on when {@code close}. Returns it, whose
     * reference the caller drops, or null.
     */
    synchronized OpenSearchDirectoryReader clear(boolean close) {
        closed |= close;
        OpenSearchDirectoryReader old = reader;
        reader = null;
        base = null;
        return old;
    }
}
