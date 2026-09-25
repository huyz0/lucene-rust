/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.opensearch.index.engine;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.common.SetOnce;
import org.opensearch.common.util.concurrent.ReleasableLock;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.translog.Translog;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.store.Store;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The protected and package-private members of OpenSearch's engine classes that
 * InternalEngineTests reaches from inside {@code org.opensearch.index.engine}, for the derived
 * RustEngineTests, which lives in the plugin's package. Test code only: in the test JVM this class
 * and OpenSearch's share one class loader, so the split package is legal here and nowhere else.
 */
public final class EngineTestAccess {
    private EngineTestAccess() {}

    public static AtomicBoolean isClosed(Engine engine) {
        return engine.isClosed;
    }

    public static SetOnce<Exception> failedEngine(Engine engine) {
        return engine.failedEngine;
    }

    public static EngineConfig engineConfig(Engine engine) {
        return engine.engineConfig;
    }

    public static Store store(Engine engine) {
        return engine.store;
    }

    public static ReleasableLock writeLock(Engine engine) {
        return engine.writeLock;
    }

    public static long getMaxSeqNoFromSearcher(Engine engine, IndexSearcher searcher) throws IOException {
        return engine.getMaxSeqNoFromSearcher(searcher);
    }

    public static String id(Engine.Operation op) {
        return op.id();
    }

    // The rewrite turns every x.id() into id(x); these keep the public ones as they were.
    public static String id(ParsedDocument doc) {
        return doc.id();
    }

    public static int id(ShardId shardId) {
        return shardId.id();
    }

    public static String id(Translog.Index op) {
        return op.id();
    }

    public static String id(Translog.Delete op) {
        return op.id();
    }

    public static Engine.Index newIndex(Term uid, long primaryTerm, ParsedDocument doc, long version) {
        return new Engine.Index(uid, primaryTerm, doc, version);
    }

    public static org.apache.lucene.index.MergePolicy newPrunePostingsMergePolicy(
        org.apache.lucene.index.MergePolicy in,
        String idField
    ) {
        return new PrunePostingsMergePolicy(in, idField);
    }

    public static long appliedOperations(TranslogHandler handler) {
        return handler.appliedOperations();
    }
}
