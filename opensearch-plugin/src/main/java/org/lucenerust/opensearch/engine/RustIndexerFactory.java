/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineBackedIndexer;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.exec.EngineBackedIndexerFactory;
import org.opensearch.index.engine.exec.Indexer;
import org.opensearch.index.engine.exec.IndexerFactory;
import org.opensearch.index.shard.IndexEventListener;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.common.settings.Settings;

import java.lang.reflect.Field;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * What lets a {@link RustEngine} be a segment-replication primary on OpenSearch 3.8.
 *
 * <p>A primary's {@code CopyState} asks the shard's {@code Indexer} for its last refreshed
 * checkpoint, and {@code EngineBackedIndexer} answers only for an {@code InternalEngine} (whose
 * {@code lastRefreshedCheckpoint()} is final, so a plugin engine cannot be one). This factory
 * wraps a {@link RustEngine} in an {@code EngineBackedIndexer} that answers from the engine's
 * own checkpoint listener -- the same {@code LastRefreshedCheckpointListener} logic InternalEngine
 * uses -- and builds every other engine exactly as {@link EngineBackedIndexerFactory} does.
 *
 * <p>{@code IndexShard} creates every engine it runs -- the first, an engine reset, a replica's
 * promotion -- through its own {@code indexerFactory} field, which OpenSearch fills from {@code
 * IndicesService} and offers no hook for. {@link #listener()} replaces that field when the shard
 * is created, before any engine exists; it is the plugin's one reflective write into OpenSearch.
 * Shards it could not swap are remembered as absent from {@link #swapped}, and {@link
 * RustEngineSupport#checkSupported} refuses them as segment-replication primaries.
 */
public final class RustIndexerFactory extends EngineBackedIndexerFactory {
    private static final Logger logger = LogManager.getLogger(RustIndexerFactory.class);

    /** Shards whose {@code IndexShard} builds its engines through this factory. */
    private static final Set<ShardId> swapped = ConcurrentHashMap.newKeySet();

    RustIndexerFactory(EngineFactory engineFactory) {
        super(engineFactory);
    }

    @Override
    public Indexer createIndexer(EngineConfig engineConfig) {
        Engine engine = getEngineFactory().newReadWriteEngine(engineConfig);
        return engine instanceof RustEngine rust ? new RustEngineIndexer(rust) : new EngineBackedIndexer(engine);
    }

    /** Whether {@code shardId}'s shard builds its engines here. */
    static boolean installed(ShardId shardId) {
        return swapped.contains(shardId);
    }

    /** Installs this factory into every shard of an index whose engines are {@link RustEngineFactory}'s. */
    public static IndexEventListener listener() {
        return new IndexEventListener() {
            @Override
            public void afterIndexShardCreated(IndexShard shard) {
                install(shard);
            }

            @Override
            public void afterIndexShardClosed(ShardId shardId, IndexShard shard, Settings settings) {
                swapped.remove(shardId);
            }
        };
    }

    static void install(IndexShard shard) {
        try {
            Field field = IndexShard.class.getDeclaredField("indexerFactory");
            field.setAccessible(true);
            IndexerFactory current = (IndexerFactory) field.get(shard);
            if (current instanceof EngineBackedIndexerFactory f && f.getEngineFactory() instanceof RustEngineFactory) {
                if ((current instanceof RustIndexerFactory) == false) {
                    field.set(shard, new RustIndexerFactory(f.getEngineFactory()));
                }
                swapped.add(shard.shardId());
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.warn(
                "cannot install the Rust engine's indexer on " + shard.shardId() + "; it cannot be a segment-replication primary",
                e
            );
        }
    }

    /** An {@code EngineBackedIndexer} over a {@link RustEngine}, answering its refresh checkpoints. */
    static final class RustEngineIndexer extends EngineBackedIndexer {
        private final RustEngine engine;

        RustEngineIndexer(RustEngine engine) {
            super(engine);
            this.engine = engine;
        }

        @Override
        public long lastRefreshedCheckpoint() {
            return engine.lastRefreshedCheckpoint();
        }

        @Override
        public long currentOngoingRefreshCheckpoint() {
            return engine.currentOngoingRefreshCheckpoint();
        }
    }
}
