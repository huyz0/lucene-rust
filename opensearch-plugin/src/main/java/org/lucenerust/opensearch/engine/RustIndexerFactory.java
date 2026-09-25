/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.index.IndexModule;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineBackedIndexer;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.exec.EngineBackedIndexerFactory;
import org.opensearch.index.engine.exec.Indexer;
import org.opensearch.index.engine.exec.IndexerFactory;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicLong;

/**
 * What lets a {@link RustEngine} be a segment-replication primary on OpenSearch 3.8.
 *
 * <p>A primary's {@code CopyState} asks the shard's {@code Indexer} for its last refreshed
 * checkpoint, and {@code EngineBackedIndexer} answers only for an {@code InternalEngine} (whose
 * {@code lastRefreshedCheckpoint()} is final, and too much of which is final for a plugin engine to
 * be one). This factory wraps a {@link RustEngine} in an {@code EngineBackedIndexer} that answers
 * from the engine's own checkpoint listener -- the same {@code LastRefreshedCheckpointListener}
 * InternalEngine uses -- and builds every other engine as {@link EngineBackedIndexerFactory} does.
 *
 * <p>OpenSearch chooses the indexer factory in {@code IndicesService} and offers no hook for it.
 * The narrowest place to change it is the index's {@code IndexModule}: plugins see it in {@code
 * onIndexModule} before it builds the {@code IndexService}, which hands its factory to every shard,
 * and every writable engine a shard runs -- the first, a reset, a replica's promotion -- is built
 * through it. {@link #install} makes that one reflective write per index. The field is resolved
 * when this class loads, which the plugin forces at node start, so an OpenSearch version without
 * it fails the node rather than a shard at failover.
 */
public final class RustIndexerFactory extends EngineBackedIndexerFactory {
    private static final Field INDEX_MODULE_FACTORY;

    static {
        try {
            INDEX_MODULE_FACTORY = IndexModule.class.getDeclaredField("indexerFactory");
            INDEX_MODULE_FACTORY.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /** Set while this factory builds an engine: what {@link RustEngineSupport#checkSupported} checks. */
    private static final ThreadLocal<Boolean> BUILDING = ThreadLocal.withInitial(() -> false);

    /** Rust engines wrapped by this factory, for the plugin's stats. */
    public static final AtomicLong INDEXERS = new AtomicLong();

    RustIndexerFactory(EngineFactory engineFactory) {
        super(engineFactory);
    }

    @Override
    public Indexer createIndexer(EngineConfig engineConfig) {
        Engine engine;
        BUILDING.set(true);
        try {
            engine = getEngineFactory().newReadWriteEngine(engineConfig);
        } finally {
            BUILDING.set(false);
        }
        if (engine instanceof RustEngine rust) {
            INDEXERS.incrementAndGet();
            return new RustEngineIndexer(rust);
        }
        return new EngineBackedIndexer(engine);
    }

    /** Whether the engine being built on this thread is being built by this factory. */
    static boolean building() {
        return BUILDING.get();
    }

    /** Resolves the field; called at node start so a missing one fails there. */
    public static void verify() {}

    /** Installs this factory into an index whose engines are {@link RustEngineFactory}'s. */
    public static void install(IndexModule module) {
        try {
            IndexerFactory current = (IndexerFactory) INDEX_MODULE_FACTORY.get(module);
            if (current instanceof EngineBackedIndexerFactory f
                && f.getEngineFactory() instanceof RustEngineFactory
                && (current instanceof RustIndexerFactory) == false) {
                INDEX_MODULE_FACTORY.set(module, new RustIndexerFactory(f.getEngineFactory()));
            }
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot install the Rust engine's indexer", e);
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
