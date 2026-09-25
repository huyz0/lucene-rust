/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.InternalEngine;
import org.opensearch.index.engine.NRTReplicationEngine;

/**
 * The engine for an index with {@code index.lucene_rust.engine: true}: {@link RustEngine} for every
 * shard that indexes, and OpenSearch's own {@link NRTReplicationEngine} for a segment-replication
 * replica, exactly as {@code NRTReplicationEngineFactory} picks -- a replica only opens the
 * segments its primary wrote, which the plugin's query phase already searches natively.
 *
 * <p>On a node started with {@code lucene_rust.engine.node_enabled: false} the factory builds
 * OpenSearch's {@link InternalEngine} instead, which is how one cluster runs a Rust primary next
 * to a Java one for the same index.
 */
public final class RustEngineFactory implements EngineFactory {
    private final boolean nodeEnabled;

    public RustEngineFactory(boolean nodeEnabled) {
        this.nodeEnabled = nodeEnabled;
    }

    @Override
    public Engine newReadWriteEngine(EngineConfig config) {
        if (config.isReadOnlyReplica()) {
            return new NRTReplicationEngine(config);
        }
        return nodeEnabled ? new RustEngine(config) : new InternalEngine(config);
    }
}
