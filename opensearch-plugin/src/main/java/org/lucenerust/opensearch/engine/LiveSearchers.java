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
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * External searchers that also read the writer's buffered documents ({@code
 * index.lucene_rust.realtime_search}): each searcher supplier -- one search request, its query and
 * fetch phases -- opens one view of the buffers beside its reader, so every phase numbers the
 * buffered documents alike, and closes it with the supplier.
 */
final class LiveSearchers {
    private final RustIndexWriter writer;
    /** The reader with the view, once the first searcher opened it; null when there is none. */
    private OpenSearchDirectoryReader live;
    private boolean opened;

    private LiveSearchers(RustIndexWriter writer) {
        this.writer = writer;
    }

    /**
     * {@code base}'s supplier for {@code scope}, its external searchers reading the buffers too when
     * {@code writer} keeps them.
     */
    static Engine.SearcherSupplier supplier(
        Object writer,
        Function<Engine.Searcher, Engine.Searcher> wrapper,
        Engine.SearcherScope scope,
        BiFunction<Function<Engine.Searcher, Engine.Searcher>, Engine.SearcherScope, Engine.SearcherSupplier> base
    ) throws EngineException {
        if (scope != Engine.SearcherScope.EXTERNAL || !(writer instanceof RustIndexWriter w) || w.liveSearch() == false) {
            return base.apply(wrapper, scope);
        }
        LiveSearchers live = new LiveSearchers(w);
        Engine.SearcherSupplier inner = base.apply(s -> wrapper.apply(live.wrap(s)), scope);
        return new Engine.SearcherSupplier(Function.identity()) {
            @Override
            protected Engine.Searcher acquireSearcherInternal(String source) {
                return inner.acquireSearcher(source);
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

    /** {@code s} over its reader and the view of the buffers its reader's commit lacks. */
    private synchronized Engine.Searcher wrap(Engine.Searcher s) {
        if (opened == false) {
            opened = true;
            live = open((OpenSearchDirectoryReader) s.getDirectoryReader());
        }
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
        if (view.maxDoc() == 0 || infos == null || writer.updatedSince(generation)) {
            view.close();
            return null;
        }
        try {
            return LiveDirectoryReader.wrap(reader, view, infos);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private synchronized void close() {
        if (live != null) {
            try {
                live.decRef();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            live = null;
        }
        opened = true;
    }
}
