/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DirectoryReader;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineException;
import org.opensearch.index.engine.NRTReplicationEngine;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.function.Function;

/**
 * A segment-replication replica of an index with doc-values-only {@code _seq_no}: its searchers
 * show {@code _seq_no} as points ({@link SeqNoPointsReader}), as the primary's do.
 */
final class SeqNoReplicaEngine extends NRTReplicationEngine {
    SeqNoReplicaEngine(EngineConfig config) {
        super(config);
    }

    @Override
    public SearcherSupplier acquireSearcherSupplier(Function<Searcher, Searcher> wrapper, SearcherScope scope) throws EngineException {
        return super.acquireSearcherSupplier(searcher -> wrapper.apply(withPoints(searcher)), scope);
    }

    private Searcher withPoints(Searcher searcher) {
        DirectoryReader in = searcher.getDirectoryReader();
        try {
            DirectoryReader wrapped = SeqNoPointsReader.wrapIfNeeded(in);
            if (wrapped == in) {
                return searcher;
            }
            OpenSearchDirectoryReader reader = OpenSearchDirectoryReader.wrap(wrapped, shardId);
            return new Searcher(
                searcher.source(),
                reader,
                searcher.getSimilarity(),
                searcher.getQueryCache(),
                searcher.getQueryCachingPolicy(),
                () -> IOUtils.close(searcher)
            );
        } catch (IOException e) {
            IOUtils.closeWhileHandlingException(searcher);
            throw new UncheckedIOException(e);
        }
    }
}
