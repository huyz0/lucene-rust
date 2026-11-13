/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.InternalEngine;
import org.opensearch.index.engine.NRTReplicationEngine;

import java.util.concurrent.atomic.AtomicLong;

/**
 * The engine for an index with {@code index.engine.accelerated: true}: {@link RustEngine} for every
 * shard that indexes, and OpenSearch's own {@link NRTReplicationEngine} for a segment-replication
 * replica, exactly as {@code NRTReplicationEngineFactory} picks -- a replica only opens the
 * segments its primary wrote, which the plugin's query phase already searches natively.
 *
 * <p>On a node started with {@code indices.engine.accelerated.node_enabled: false} the factory builds
 * OpenSearch's {@link InternalEngine} instead, which is how one cluster runs a Rust primary next
 * to a Java one for the same index.
 */
public final class RustEngineFactory implements EngineFactory {
    /** Engines created on this node, by kind, for the stats endpoint. */
    public static final AtomicLong RUST = new AtomicLong();

    /** Text field values the Rust engine's writer analyzed in Rust, and in Java, since startup. */
    public static long[] analyzedValues() {
        return new long[] { DocumentEncoder.ANALYZED_NATIVELY.get(), DocumentEncoder.ANALYZED_IN_JAVA.get() };
    }
    public static final AtomicLong JAVA = new AtomicLong();
    public static final AtomicLong NRT_REPLICA = new AtomicLong();

    private final boolean nodeEnabled;
    private final boolean follower;

    static final String SEQ_NO_NEEDS_ENGINE = "[index.seq_no.index_options: doc_values_only] needs the accelerated engine:"
        + " set [index.engine.accelerated] or [indices.engine.accelerated.default] on this node";

    public RustEngineFactory(boolean nodeEnabled) {
        this(nodeEnabled, false);
    }

    /** With {@code follower}, the engines are {@link FollowerEngines}: a cross-cluster replication follower's. */
    public RustEngineFactory(boolean nodeEnabled, boolean follower) {
        this.nodeEnabled = nodeEnabled;
        this.follower = follower;
    }

    /** OpenSearch's engine for a follower index the accelerated engine does not serve. */
    public static EngineFactory standardFollower() {
        return config -> {
            if (config.isReadOnlyReplica()) {
                NRT_REPLICA.incrementAndGet();
                return new NRTReplicationEngine(config);
            }
            JAVA.incrementAndGet();
            return new FollowerEngines.Standard(config);
        };
    }

    @Override
    public Engine newReadWriteEngine(EngineConfig config) {
        boolean seqNoDocValuesOnly = SeqNoSettings.docValuesOnly(config.getIndexSettings().getSettings());
        if (config.isReadOnlyReplica()) {
            NRT_REPLICA.incrementAndGet();
            return seqNoDocValuesOnly ? new SeqNoReplicaEngine(config) : new NRTReplicationEngine(config);
        }
        // Another engine plugin's PrimaryOperationPolicy: an index that only inherits the node
        // default gets OpenSearch's engine, one that asked for the Rust engine its refusal.
        boolean asked = RustEngineSupport.ENGINE_ENABLED.exists(config.getIndexSettings().getSettings());
        if (nodeEnabled && (asked || RustEngineSupport.unsupportedPolicy(config) == null)) {
            RUST.incrementAndGet();
            return follower ? new FollowerEngines.Accelerated(config) : new RustEngine(config);
        }
        if (seqNoDocValuesOnly) {
            // OpenSearch's engine would run its history and retention ranges on points these segments lack.
            throw new IllegalArgumentException(SEQ_NO_NEEDS_ENGINE);
        }
        JAVA.incrementAndGet();
        return follower ? new FollowerEngines.Standard(config) : new InternalEngine(config);
    }
}
