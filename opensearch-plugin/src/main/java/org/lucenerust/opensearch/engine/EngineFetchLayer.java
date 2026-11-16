/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DirectoryReader;
import org.opensearch.common.lease.Releasables;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Stored fields read natively under a reader wrapper that is not this plugin's.
 *
 * <p>An index takes one reader wrapper ({@code IndexModule.setReaderWrapper} is set-once), and the
 * security plugin claims it for field- and document-level security. The plugin's own layer ({@code
 * NativeStoredFieldsReader}) then goes where the engine builds its searchers, between the engine's
 * reader and the shard's wrapper: it replaces what the segments' stored fields read, and the
 * wrapper above it filters, masks and audits what it reads exactly as it does on Lucene's own
 * stored fields reader. An index whose reader wrapper is this plugin's ({@link #exempt}) is not
 * layered twice.
 */
public final class EngineFetchLayer {
    /** Wraps an engine reader of an index with the layer. */
    @FunctionalInterface
    public interface Layer {
        DirectoryReader wrap(IndexSettings settings, DirectoryReader reader) throws IOException;
    }

    private static volatile Layer layer;
    private static final Set<String> EXEMPT = ConcurrentHashMap.newKeySet();

    private EngineFetchLayer() {}

    /** The layer the engines of this node put under the shard's reader wrapper. */
    public static void install(Layer l) {
        layer = l;
    }

    /** The index with this UUID wraps its readers with the layer itself: the engine leaves it alone. */
    public static void exempt(String indexUuid) {
        EXEMPT.add(indexUuid);
    }

    /** {@code shard}, the searcher wrapper the shard handed the engine, applied over the layer. */
    static Function<Engine.Searcher, Engine.Searcher> under(Function<Engine.Searcher, Engine.Searcher> shard, IndexSettings settings) {
        Layer l = layer;
        if (l == null || shard == null || EXEMPT.contains(settings.getIndex().getUUID())) {
            return shard;
        }
        return searcher -> {
            Engine.Searcher layered;
            try {
                layered = IndexShard.wrapSearcher(searcher, reader -> l.wrap(settings, reader));
            } catch (IOException | RuntimeException e) {
                Releasables.closeWhileHandlingException(searcher);
                throw e instanceof RuntimeException r ? r : new org.opensearch.OpenSearchException("failed to layer the searcher", e);
            }
            return shard.apply(layered);
        };
    }
}
