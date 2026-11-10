/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.FieldInfos;
import org.lucenerust.opensearch.LiveDirectoryReader;
import org.lucenerust.opensearch.LiveView;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineException;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * External searchers that also read the writer's buffered documents ({@code
 * index.lucene_rust.realtime_search}): each searcher supplier -- one search request, its query and
 * fetch phases, or a point in time -- opens one view of the buffers beside its reader as it is
 * made, so every phase numbers the buffered documents alike, and closes it with the supplier.
 */
final class LiveSearchers {
    private final RustIndexWriter writer;
    /** The reader with the view, once the first searcher opened it; null when there is none. */
    private OpenSearchDirectoryReader live;
    private boolean opened;
    /** Where the view is closed, off the search's thread; null closes it there. */
    private final Executor closer;

    private LiveSearchers(RustIndexWriter writer, Executor closer) {
        this.writer = writer;
        this.closer = closer;
    }

    /**
     * {@code base}'s supplier for {@code scope}, its external searchers reading the buffers too when
     * {@code writer} keeps them.
     */
    static Engine.SearcherSupplier supplier(
        Object writer,
        Function<Engine.Searcher, Engine.Searcher> wrapper,
        Engine.SearcherScope scope,
        BiFunction<Function<Engine.Searcher, Engine.Searcher>, Engine.SearcherScope, Engine.SearcherSupplier> base,
        Executor closer
    ) throws EngineException {
        if (scope != Engine.SearcherScope.EXTERNAL || !(writer instanceof RustIndexWriter w) || w.liveSearch() == false) {
            return base.apply(wrapper, scope);
        }
        LiveSearchers live = new LiveSearchers(w, closer);
        // The view goes on in acquireSearcherInternal, not through a wrapper: a supplier applies
        // none to a can-match searcher, whose bounds must count the buffered documents too.
        Engine.SearcherSupplier inner = base.apply(Function.identity(), scope);
        // Opened now, not on the first search: a point in time reads the documents written before
        // it was opened, as its reader does, not those before its first search.
        try (Engine.Searcher s = inner.acquireSearcher("live_view")) {
            live.open(s);
        } catch (RuntimeException e) {
            try {
                inner.close();
            } finally {
                live.close();
            }
            throw e;
        }
        return new Engine.SearcherSupplier(wrapper) {
            @Override
            protected Engine.Searcher acquireSearcherInternal(String source) {
                Engine.Searcher s = inner.acquireSearcher(source);
                try {
                    return live.wrap(s);
                } catch (RuntimeException e) {
                    org.opensearch.common.lease.Releasables.closeWhileHandlingException(s);
                    throw e;
                }
            }

            @Override
            protected void doClose() {
                try {
                    inner.close();
                } finally {
                    live.close();
                }
            }
        };
    }

    /** Opens the view of the buffers {@code s}'s reader's commit lacks, once. */
    private synchronized void open(Engine.Searcher s) {
        if (opened == false) {
            opened = true;
            live = open((OpenSearchDirectoryReader) s.getDirectoryReader());
        }
    }

    /** {@code s} over its reader and the view of the buffers its reader's commit lacks. */
    private synchronized Engine.Searcher wrap(Engine.Searcher s) {
        open(s);
        OpenSearchDirectoryReader reader = live;
        if (reader == null) {
            return s;
        }
        reader.incRef();
        return new Engine.Searcher(s.source(), reader, s.getSimilarity(), s.getQueryCache(), s.getQueryCachingPolicy(), () -> {
            try {
                reader.decRef();
            } finally {
                s.close();
            }
        });
    }

    private OpenSearchDirectoryReader open(OpenSearchDirectoryReader reader) {
        long generation = RustReaderManager.generation(reader);
        // A search on this commit asks: the buffers older commits lacked can go.
        writer.releaseLive(generation);
        RustIndexWriter.LiveViewHandle handle = writer.openLiveView(generation);
        if (handle == null) {
            return null;
        }
        LiveView view = LiveView.of(handle.id(), handle.blob());
        // After the view: every field of its documents is registered by now.
        FieldInfos infos = writer.liveFieldInfos();
        if (view.maxDoc() == 0 || infos == null) {
            view.close();
            return null;
        }
        try {
            return LiveDirectoryReader.wrap(reader, view, infos);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Drops the supplier's reference on the view's reader. The last one closes the view, and with
     * it, often, a commit's reader no newer search holds: that runs on {@code closer}, after the
     * search answered, not before.
     */
    private void close() {
        OpenSearchDirectoryReader reader;
        synchronized (this) {
            reader = live;
            live = null;
            opened = true;
        }
        if (reader == null) {
            return;
        }
        if (closer != null) {
            try {
                closer.execute(() -> decRef(reader));
                return;
            } catch (RejectedExecutionException e) {
                // The node is shutting down: closed here.
            }
        }
        decRef(reader);
    }

    private static void decRef(OpenSearchDirectoryReader reader) {
        try {
            reader.decRef();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
