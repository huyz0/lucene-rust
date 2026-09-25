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
import org.opensearch.index.engine.EngineConfig;
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
 * be one. Hand-written, beside the derived {@code RustEngineTests}.
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
}
