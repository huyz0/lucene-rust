/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.UUIDs;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineTestAccess;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.mapper.SeqNoFieldMapper;
import org.opensearch.index.seqno.SequenceNumbers;
import org.opensearch.index.store.Store;
import org.opensearch.test.IndexSettingsModule;

import java.util.concurrent.atomic.AtomicLong;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;

/** The Rust engine with {@link SeqNoSettings}' leaner {@code _seq_no} layouts. */
public class SeqNoLayoutTests extends RustEngineTestCase {
    /** Index settings with {@link SeqNoSettings}' options on top of {@code defaultSettings}. */
    private IndexSettings seqNoSettings(boolean sentinels) {
        Settings.Builder settings = Settings.builder()
            .put(defaultSettings.getSettings())
            .put(IndexSettings.INDEX_SOFT_DELETES_RETENTION_OPERATIONS_SETTING.getKey(), 0)
            .put(SeqNoSettings.INDEX_OPTIONS.getKey(), SeqNoSettings.DOC_VALUES_ONLY)
            .put(SeqNoSettings.DISABLE_SEQUENCE_NUMBERS.getKey(), sentinels);
        IndexMetadata indexMetadata = IndexMetadata.builder(defaultSettings.getIndexMetadata()).settings(settings).build();
        return IndexSettingsModule.newIndexSettings(indexMetadata, SeqNoSettings.INDEX_OPTIONS, SeqNoSettings.DISABLE_SEQUENCE_NUMBERS);
    }

    /**
     * {@code doc_values_only}: {@code _seq_no} has doc values with a skip index and no points, and
     * history (snapshots, counts, recovery after a restart) reads the same operations.
     */
    public void testSeqNoDocValuesOnlyKeepsHistory() throws Exception {
        IndexSettings indexSettings = seqNoSettings(false);
        final AtomicLong globalCheckpoint = new AtomicLong(SequenceNumbers.NO_OPS_PERFORMED);
        try (Store store = createStore()) {
            int numDocs = scaledRandomIntBetween(10, 100);
            RustEngine engine = createEngine(config(indexSettings, store, createTempDir(), newMergePolicy(), null, null, globalCheckpoint::get));
            try {
                for (int i = 0; i < numDocs; i++) {
                    ParsedDocument doc = testParsedDocument(Integer.toString(i), null, testDocument(), B_1, null);
                    engine.index(indexForDoc(doc));
                    if (randomBoolean()) {
                        engine.delete(new Engine.Delete(EngineTestAccess.id(doc), newUid(EngineTestAccess.id(doc)), primaryTerm.get()));
                    }
                    if (rarely()) {
                        engine.flush();
                    }
                }
                engine.refresh("test");
                try (Engine.Searcher searcher = engine.acquireSearcher("test", Engine.SearcherScope.INTERNAL)) {
                    for (LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
                        // The segment has no points; the reader shows its doc values as points.
                        org.apache.lucene.index.SegmentReader segment = org.opensearch.common.lucene.Lucene.segmentReader(leaf.reader());
                        FieldInfo info = segment.getFieldInfos().fieldInfo(SeqNoFieldMapper.NAME);
                        assertEquals(0, info.getPointDimensionCount());
                        assertEquals(DocValuesSkipIndexType.RANGE, info.docValuesSkipIndexType());
                        assertNotNull(segment.getDocValuesSkipper(SeqNoFieldMapper.NAME));
                        assertEquals(segment.maxDoc(), leaf.reader().getPointValues(SeqNoFieldMapper.NAME).getDocCount());
                    }
                    // A stock `_seq_no` range finds the same documents through those points.
                    org.apache.lucene.search.IndexSearcher all = new org.apache.lucene.search.IndexSearcher(
                        org.opensearch.common.lucene.Lucene.wrapAllDocsLive(searcher.getDirectoryReader())
                    );
                    long max = engine.getProcessedLocalCheckpoint();
                    assertEquals(max + 1, all.count(org.apache.lucene.document.LongPoint.newRangeQuery(SeqNoFieldMapper.NAME, 0, max)));
                    assertEquals(1, all.count(org.apache.lucene.document.LongPoint.newExactQuery(SeqNoFieldMapper.NAME, max)));
                    assertEquals(2, all.count(org.apache.lucene.document.LongPoint.newSetQuery(SeqNoFieldMapper.NAME, 0, max, max + 5)));
                    // The range the engine restores its version map and checkpoint with on reopen.
                    long from = randomLongBetween(0, max);
                    assertEquals(max - from + 1, all.count(SeqNoQueries.range(all.getIndexReader(), from, Long.MAX_VALUE)));
                    assertTrue(SeqNoQueries.range(all.getIndexReader(), from, Long.MAX_VALUE).getClass().getName().contains("DocValues"));
                }
                long last = engine.getProcessedLocalCheckpoint();
                assertConsistentHistoryBetweenTranslogAndLuceneIndex(engine);
                assertEquals(last + 1, engine.countNumberOfHistoryOperations("test", 0, last));
                assertEquals(last, engine.countNumberOfHistoryOperations("test", 1, last));
                // Points-only searches on `_seq_no` would see nothing.
                try (Engine.Searcher searcher = engine.acquireSearcher("test", Engine.SearcherScope.INTERNAL)) {
                    assertTrue(SeqNoQueries.docValuesOnly(searcher.getIndexReader().getContext()));
                }
                globalCheckpoint.set(last);
                engine.translogManager().syncTranslog();
                engine.flush(true, true);
                engine.close();
                engine = new RustEngine(engine.config());
                assertEquals(last, engine.getProcessedLocalCheckpoint());
                assertConsistentHistoryBetweenTranslogAndLuceneIndex(engine);
            } finally {
                engine.close();
            }
        }
    }

    /**
     * {@code disable_sequence_numbers}: a merge writes {@code UNASSIGNED_SEQ_NO} and {@code
     * UNASSIGNED_PRIMARY_TERM} for documents below the retained minimum, keeps the rest, and a get
     * of such a document reports no sequence number.
     */
    public void testDisableSequenceNumbersWritesSentinelsBelowRetention() throws Exception {
        IndexSettings indexSettings = seqNoSettings(true);
        final AtomicLong globalCheckpoint = new AtomicLong(SequenceNumbers.NO_OPS_PERFORMED);
        try (
            Store store = createStore();
            RustEngine engine = createEngine(config(indexSettings, store, createTempDir(), newMergePolicy(), null, null, globalCheckpoint::get))
        ) {
            int numDocs = scaledRandomIntBetween(20, 100);
            long half = numDocs / 2;
            for (int i = 0; i < numDocs; i++) {
                ParsedDocument doc = testParsedDocument(Integer.toString(i), null, testDocument(), B_1, null);
                engine.index(indexForDoc(doc));
                if (i == half - 1) {
                    // A safe commit at the global checkpoint, so retention starts right after it.
                    globalCheckpoint.set(half - 1);
                    engine.translogManager().syncTranslog();
                    engine.flush(true, true);
                }
            }
            engine.translogManager().syncTranslog();
            engine.flush(true, true);
            engine.forceMerge(true, 1, false, false, false, UUIDs.randomBase64UUID());
            engine.refresh("test");
            long minRetained = engine.getMinRetainedSeqNo();
            assertEquals(half, minRetained);
            int sentinels = 0;
            try (Engine.Searcher searcher = engine.acquireSearcher("test", Engine.SearcherScope.INTERNAL)) {
                for (LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
                    NumericDocValues seq = leaf.reader().getNumericDocValues(SeqNoFieldMapper.NAME);
                    NumericDocValues term = leaf.reader().getNumericDocValues(SeqNoFieldMapper.PRIMARY_TERM_NAME);
                    for (int d = 0; d < leaf.reader().maxDoc(); d++) {
                        assertTrue(seq.advanceExact(d));
                        assertTrue(term.advanceExact(d));
                        if (seq.longValue() == SequenceNumbers.UNASSIGNED_SEQ_NO) {
                            assertEquals(SequenceNumbers.UNASSIGNED_PRIMARY_TERM, term.longValue());
                            sentinels++;
                        } else {
                            assertThat(seq.longValue(), greaterThanOrEqualTo(minRetained));
                            assertEquals(primaryTerm.get(), term.longValue());
                        }
                    }
                }
            }
            assertEquals(minRetained, sentinels);
            assertEquals(numDocs - minRetained, engine.countNumberOfHistoryOperations("test", minRetained, numDocs - 1));

            ParsedDocument old = testParsedDocument("0", null, testDocument(), B_1, null);
            try (Engine.GetResult get = engine.get(new Engine.Get(true, false, "0", newUid(old)), engine::acquireSearcher)) {
                assertTrue(get.exists());
                assertEquals(SequenceNumbers.UNASSIGNED_SEQ_NO, get.docIdAndVersion().seqNo);
                assertEquals(SequenceNumbers.UNASSIGNED_PRIMARY_TERM, get.docIdAndVersion().primaryTerm);
            }
            // An update of an old document is a newer operation.
            Engine.IndexResult update = engine.index(indexForDoc(old));
            assertEquals(Engine.Result.Type.SUCCESS, update.getResultType());
            assertFalse(update.isCreated());
        }
    }
}
