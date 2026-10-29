/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.NoMergePolicy;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineBackedIndexer;
import org.opensearch.index.engine.DeletionStrategy;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.IndexingStrategy;
import org.opensearch.index.engine.InternalEngine;
import org.opensearch.index.engine.OperationStrategyPlanner;
import org.opensearch.index.engine.PrimaryOperationPolicy;
import org.opensearch.index.engine.exec.Indexer;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.seqno.SequenceNumbers;
import org.opensearch.index.store.Store;
import org.opensearch.index.translog.Translog;
import org.opensearch.indices.replication.common.ReplicationType;
import org.opensearch.test.IndexSettingsModule;

import java.io.IOException;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;

/**
 * {@link RustIndexerFactory}: a segment-replication primary's {@code Indexer} answers the refresh
 * checkpoint {@code CopyState} asks for, and a {@link RustEngine} built any other way refuses to
 * be one; and an engine plugin's {@link PrimaryOperationPolicy} is refused by the Rust engine, or
 * handed to OpenSearch's where the index only inherits the node default. Hand-written, beside the
 * derived {@code RustEngineTests}.
 */
public class RustIndexerFactoryTests extends RustEngineTestCase {

    private EngineConfig segrepConfig(Store store) throws IOException {
        IndexSettings segrep = IndexSettingsModule.newIndexSettings(
            "test",
            Settings.builder()
                .put(defaultSettings.getSettings())
                .put(IndexMetadata.SETTING_REPLICATION_TYPE, ReplicationType.SEGMENT)
                .build()
        );
        EngineConfig config = config(segrep, store, createTempDir(), NoMergePolicy.INSTANCE, null);
        store.createEmpty(segrep.getIndexVersionCreated().luceneVersion);
        String translogUuid = Translog.createEmptyTranslog(
            config.getTranslogConfig().getTranslogPath(),
            SequenceNumbers.NO_OPS_PERFORMED,
            shardId,
            primaryTerm.get()
        );
        store.associateIndexWithNewTranslog(translogUuid);
        return config;
    }

    public void testASegmentReplicationPrimaryIsBuiltThroughTheFactory() throws IOException {
        try (Store store = createStore(newFSDirectory(createTempDir()))) {
            EngineConfig config = segrepConfig(store);
            IllegalArgumentException refused = expectThrows(IllegalArgumentException.class, () -> new RustEngine(config));
            assertThat(refused.getMessage(), containsString("was not built through RustIndexerFactory"));

            long before = RustIndexerFactory.INDEXERS.get();
            Indexer indexer = new RustIndexerFactory(new RustEngineFactory(true)).createIndexer(config);
            try {
                assertThat(indexer, instanceOf(RustIndexerFactory.RustEngineIndexer.class));
                assertEquals(before + 1, RustIndexerFactory.INDEXERS.get());
                Engine engine = ((EngineBackedIndexer) indexer).getEngine();
                assertThat(engine, instanceOf(RustEngine.class));
                RustEngine rust = (RustEngine) engine;
                // CopyState's question, answered before and after a refresh.
                assertEquals(SequenceNumbers.NO_OPS_PERFORMED, indexer.lastRefreshedCheckpoint());
                for (int i = 0; i < 3; i++) {
                    ParsedDocument doc = testParsedDocument(Integer.toString(i), null, testDocument(), new BytesArray("{}"), null);
                    engine.index(indexForDoc(doc));
                }
                engine.refresh("test");
                assertEquals(2, indexer.lastRefreshedCheckpoint());
                assertEquals(rust.lastRefreshedCheckpoint(), indexer.lastRefreshedCheckpoint());
                assertEquals(rust.currentOngoingRefreshCheckpoint(), indexer.currentOngoingRefreshCheckpoint());
            } finally {
                indexer.close();
            }
        }
    }

    /** Another engine plugin's policy: planning as primary, like the default, but not the default. */
    private static final PrimaryOperationPolicy FOREIGN_POLICY = new PrimaryOperationPolicy() {
        @Override
        public boolean acceptsPreAssignedSeqNos() {
            return false;
        }

        @Override
        public IndexingStrategy planIndex(OperationStrategyPlanner<Engine.Index, IndexingStrategy> planner, Engine.Index index)
            throws IOException {
            return planner.planOperationAsPrimary(index);
        }

        @Override
        public DeletionStrategy planDelete(OperationStrategyPlanner<Engine.Delete, DeletionStrategy> planner, Engine.Delete delete)
            throws IOException {
            return planner.planOperationAsPrimary(delete);
        }

        @Override
        public String toString() {
            return "foreign";
        }
    };

    private EngineConfig policyConfig(Store store, Settings extra) throws IOException {
        IndexSettings settings = IndexSettingsModule.newIndexSettings(
            "test",
            Settings.builder().put(defaultSettings.getSettings()).put(extra).build()
        );
        EngineConfig config = config(settings, store, createTempDir(), NoMergePolicy.INSTANCE, null);
        store.createEmpty(settings.getIndexVersionCreated().luceneVersion);
        String translogUuid = Translog.createEmptyTranslog(
            config.getTranslogConfig().getTranslogPath(),
            SequenceNumbers.NO_OPS_PERFORMED,
            shardId,
            primaryTerm.get()
        );
        store.associateIndexWithNewTranslog(translogUuid);
        return config.toBuilder().primaryOperationPolicy(FOREIGN_POLICY).build();
    }

    public void testAnotherPluginsPrimaryOperationPolicyIsRefusedOrHandedToOpenSearch() throws IOException {
        // The index asked for the Rust engine: the engine refuses, naming the policy.
        try (Store store = createStore(newFSDirectory(createTempDir()))) {
            EngineConfig asked = policyConfig(store, Settings.builder().put(RustEngineSupport.ENGINE_ENABLED.getKey(), true).build());
            IllegalArgumentException refused = expectThrows(IllegalArgumentException.class, () -> new RustEngine(asked));
            assertThat(refused.getMessage(), containsString("default primary operation policy only"));
            assertThat(refused.getMessage(), containsString("[foreign]"));
            expectThrows(IllegalArgumentException.class, () -> new RustEngineFactory(true).newReadWriteEngine(asked));
        }
        // The index only inherits the node default: OpenSearch's engine serves it.
        try (Store store = createStore(newFSDirectory(createTempDir()))) {
            EngineConfig inherited = policyConfig(store, Settings.EMPTY);
            try (Engine engine = new RustEngineFactory(true).newReadWriteEngine(inherited)) {
                assertThat(engine, instanceOf(InternalEngine.class));
                assertFalse(engine instanceof RustEngine);
            }
        }
    }
}
