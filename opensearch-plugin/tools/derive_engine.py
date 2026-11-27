#!/usr/bin/env python3
"""Derives AcceleratedEngine and the engine-package helpers it needs from OpenSearch's own sources.

OpenSearch's InternalEngine owns every contract the Rust engine must keep -- sequence numbers,
the local checkpoint, the version map and optimistic concurrency control, the translog, commit
user data, history retention, safe commits. Its IndexWriter is not pluggable (the writer is
created in a private method and readers are opened on it), so the Rust engine is InternalEngine
itself with the writer swapped: the file is copied from the pinned OpenSearch release and patched
where it touches Lucene's IndexWriter, and nowhere else. Package-private helpers it uses are
copied alongside, unchanged apart from their package, because a plugin's classes live in another
class loader and cannot reach OpenSearch's package-private members.

Every patch below must apply exactly once, so a new OpenSearch release that moves the code fails
here rather than producing an engine that silently differs.

Usage: derive_engine.py <extracted opensearch-X-sources.jar> <output dir>
  deps/gradlew -q -p deps fetch -Pcoordinates=org.opensearch:opensearch:3.9.0:sources -PoutDir=$PWD
  unzip -d os-src opensearch-3.9.0-sources.jar
  opensearch-plugin/tools/derive_engine.py os-src \\
      opensearch-plugin/src/main/java/org/lucenerust/opensearch/engine
"""

import re
import sys
from pathlib import Path

PACKAGE = "org.lucenerust.opensearch.engine"

# Copied unchanged (package-private in OpenSearch, or holding package-private members).
HELPERS = [
    "CombinedDeletionPolicy",
    "CombinedDocValues",
    "CompletionStatsCache",
    "DeleteVersionValue",
    "DeletionStrategy",
    "DeletionStrategyPlanner",
    "DocumentCountTracker",
    "IndexVersionValue",
    "IndexingStrategy",
    "IndexingStrategyPlanner",
    "IndexingThrottler",
    "LastRefreshedCheckpointListener",
    "LiveVersionMap",
    "LuceneChangesSnapshot",
    "OperationStrategy",
    "OperationStrategyPlanner",
    "SeqNoGapFiller",
    "SoftDeletesPolicy",
    "VersionValue",
]

NOTICE = """/*
 * Derived from OpenSearch 3.9.0 by opensearch-plugin/tools/derive_engine.py -- do not edit;
 * change the script and re-run it. The original notice follows.
 */
"""


def relocate(text: str) -> str:
    text = text.replace(
        "package org.opensearch.index.engine;",
        f"package {PACKAGE};\n\nimport org.opensearch.index.engine.*;",
        1,
    )
    # OpenSearch builds these without -Werror on deprecations; this plugin builds with it.
    text, n = re.subn(
        r"\n((?:public |final |abstract )*(?:class|interface|enum) )",
        r'\n@SuppressWarnings({ "deprecation", "try", "unchecked", "rawtypes", "cast", "static", "fallthrough", "this-escape", "removal" }) // derived code, linted upstream\n\1',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("no top-level type declaration found")
    return text


class Patcher:
    def __init__(self, text: str):
        self.text = text

    def sub(self, old: str, new: str, count: int = 1) -> None:
        found = self.text.count(old)
        if found != count:
            sys.exit(f"patch expected {count} match(es), found {found}: {old[:120]!r}")
        self.text = self.text.replace(old, new)

    def regex(self, pattern: str, new: str) -> None:
        text, n = re.subn(pattern, new, self.text, count=1, flags=re.S)
        if n != 1:
            sys.exit(f"regex patch did not apply: {pattern[:120]!r}")
        self.text = text

    def drop_block(self, start: str) -> None:
        """Removes the member that starts at `start`, through its matching closing brace."""
        i = self.text.find(start)
        if i < 0 or self.text.find(start, i + 1) >= 0:
            sys.exit(f"block start not found exactly once: {start!r}")
        # Include the member's own annotation lines.
        line_start = self.text.rfind("\n", 0, i) + 1
        while True:
            prev = self.text.rfind("\n", 0, line_start - 1) + 1
            if self.text[prev:line_start].strip().startswith("@"):
                line_start = prev
            else:
                break
        depth = 0
        j = self.text.index("{", i)
        while True:
            c = self.text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        end = self.text.index("\n", j) + 1
        self.text = self.text[:line_start] + self.text[end:]


def derive_engine(src: str) -> str:
    p = Patcher(relocate(src))

    p.sub("public class InternalEngine extends Engine {", "public class AcceleratedEngine extends Engine {")
    p.sub("public InternalEngine(", "public AcceleratedEngine(", count=2)
    p.sub("    InternalEngine(\n        EngineConfig engineConfig,", "    AcceleratedEngine(\n        EngineConfig engineConfig,")
    p.sub('logger.trace("created new InternalEngine");', 'logger.trace("created new AcceleratedEngine");')

    # Refuse what the Rust writer cannot produce, before anything is opened.
    p.sub(
        "        super(engineConfig);\n        if (engineConfig.isAutoGeneratedIDsOptimizationEnabled() == false) {",
        "        super(engineConfig);\n        AcceleratedEngineSupport.checkSupported(engineConfig);\n"
        "        if (engineConfig.isAutoGeneratedIDsOptimizationEnabled() == false) {",
    )

    # The Rust writer's own scheduler merges, configured from the same settings; AcceleratedMergeScheduler
    # hears its merges, so EngineMergeScheduler's callbacks (stats, throttling, flush after merges,
    # merge failures) run unchanged.
    p.sub("    private final OpenSearchConcurrentMergeScheduler mergeScheduler;\n",
          "    private final AcceleratedMergeScheduler mergeScheduler;\n")
    p.sub(
        "        mergeScheduler.refreshConfig();\n",
        "        AcceleratedEngineSupport.refreshMergeScheduler(documentIndexWriter, engineConfig);\n",
        count=2,
    )
    p.sub("    private final class EngineMergeScheduler extends OpenSearchConcurrentMergeScheduler {",
          "    private final class EngineMergeScheduler extends AcceleratedMergeScheduler {")

    # The writer: the Rust one, opened with the engine's own deletion and retention policies.
    p.sub("    private final IndexWriterFactory nativeIndexWriterFactory;\n", "")
    p.sub("                this.nativeIndexWriterFactory = new NativeLuceneIndexWriterFactory();\n", "")
    p.regex(
        r"    private DocumentIndexWriter getDocumentIndexWriter\(\) throws IOException \{.*?\n    \}\n",
        "    private DocumentIndexWriter getDocumentIndexWriter(int maxDocs) throws IOException {\n"
        "        return AcceleratedEngineSupport.openWriter(\n"
        "            engineConfig,\n"
        "            store,\n"
        "            combinedDeletionPolicy,\n"
        "            softDeletesPolicy::getMinRetainedSeqNo,\n"
        "            softDeletesField.name(),\n"
        "            maxDocs\n"
        "        );\n"
        "    }\n",
    )
    # Lucene's limit is the engine's: a test lowering one lowers the other.
    p.sub("                writer = getDocumentIndexWriter();\n", "                writer = getDocumentIndexWriter(maxDocs);\n")
    p.regex(r"    /\*\*\n     \* We should only take care of reopening parent writer.*?\n    private IndexWriter createWriter\(\) throws IOException \{.*?\n    \}\n", "")
    p.regex(r"    // pkg-private for testing\n    IndexWriter createWriter\(Directory directory, IndexWriterConfig iwc\) throws IOException \{.*?\n    \}\n", "")

    # Every commit -- refresh or flush -- carries the commit data, read in commitIndexWriter's
    # two steps: the local checkpoint as the commit starts, before its flush, and the rest once
    # the flush is done (AcceleratedIndexWriter.StagedCommitData).
    p.sub(
        "                forceMergeUUID = commitData.get(FORCE_MERGE_UUID_KEY);\n                documentIndexWriter = writer;\n",
        "                forceMergeUUID = commitData.get(FORCE_MERGE_UUID_KEY);\n"
        "                writer.setLiveCommitData(liveCommitData(translogUUID));\n"
        "                scheduler.attach((AcceleratedIndexWriter) writer);\n"
        "                documentIndexWriter = writer;\n",
    )
    p.regex(
        r"            final long localCheckpoint = localCheckpointTracker\.getProcessedCheckpoint\(\);\n"
        r"            writer\.setLiveCommitData\(\(\) -> \{.*?\n            \}\);\n",
        "            writer.setLiveCommitData(liveCommitData(translogUUID));\n",
    )
    p.sub(
        "    @Override\n    public void onSettingsChanged(",
        "    /**\n"
        "     * InternalEngine's commit data, for every commit the writer makes: the local checkpoint is read\n"
        "     * as the commit starts, before its flush, and the maximum sequence number once the flush is\n"
        "     * done, as commitIndexWriter reads them around IndexWriter.commit.\n"
        "     */\n"
        "    private AcceleratedIndexWriter.StagedCommitData liveCommitData(String translogUUID) {\n"
        "        return () -> {\n"
        "            final long localCheckpoint = localCheckpointTracker.getProcessedCheckpoint();\n"
        "            return () -> {\n"
        "                final Map<String, String> commitData = new HashMap<>(7);\n"
        "                commitData.put(Translog.TRANSLOG_UUID_KEY, translogUUID);\n"
        "                commitData.put(SequenceNumbers.LOCAL_CHECKPOINT_KEY, Long.toString(localCheckpoint));\n"
        "                commitData.put(SequenceNumbers.MAX_SEQ_NO, Long.toString(localCheckpointTracker.getMaxSeqNo()));\n"
        "                commitData.put(MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID, Long.toString(maxUnsafeAutoIdTimestamp.get()));\n"
        "                commitData.put(HISTORY_UUID_KEY, historyUUID);\n"
        "                commitData.put(Engine.MIN_RETAINED_SEQNO, Long.toString(softDeletesPolicy.getMinRetainedSeqNo()));\n"
        "                final String currentForceMergeUUID = forceMergeUUID;\n"
        "                if (currentForceMergeUUID != null) {\n"
        "                    commitData.put(FORCE_MERGE_UUID_KEY, currentForceMergeUUID);\n"
        "                }\n"
        "                logger.trace(\"committing writer with commit data [{}]\", commitData);\n"
        "                return commitData.entrySet().iterator();\n"
        "            };\n"
        "        };\n"
        "    }\n\n"
        "    @Override\n    public void onSettingsChanged(",
    )

    # Readers: Java readers on the Rust writer's commits.
    p.sub("\n    private final OpenSearchReaderManager internalReaderManager;\n",
          "\n    private final AcceleratedReaderManager internalReaderManager;\n")
    p.sub("        OpenSearchReaderManager internalReaderManager = null;\n",
          "        AcceleratedReaderManager internalReaderManager = null;\n", count=2)
    p.sub("        private final OpenSearchReaderManager internalReaderManager;\n",
          "        private final AcceleratedReaderManager internalReaderManager;\n")
    p.sub("            OpenSearchReaderManager internalReaderManager,\n", "            AcceleratedReaderManager internalReaderManager,\n")
    p.regex(
        r"                // We always open reader on parent IndexWriter\.\n.*?"
        r"internalReaderManager = new OpenSearchReaderManager\(directoryReader\);\n",
        "                internalReaderManager = new AcceleratedReaderManager(\n"
        "                    (AcceleratedIndexWriter) documentIndexWriter,\n"
        "                    store.directory(),\n"
        "                    shardId,\n"
        "                    engineConfig.getLeafSorter()\n"
        "                );\n",
    )
    p.sub("((StandardDirectoryReader) reader.getDelegate()).getSegmentInfos()",
          "AcceleratedReaderManager.standard(reader).getSegmentInfos()")
    # A reader is on the newest commit; buffered documents are what a refresh would add.
    p.sub(
        "        return documentIndexWriter.hasNewIndexingOrUpdates() || super.refreshNeeded();",
        "        return documentIndexWriter.hasUncommittedChanges() || super.refreshNeeded();",
    )

    # Only a primary's index operation may be refused by the writer's breaker (a document failure);
    # anywhere else a refusal would fail the shard.
    p.sub(
        "        index.parsedDoc().version().setLongValue(plan.version);\n        try {\n",
        "        index.parsedDoc().version().setLongValue(plan.version);\n"
        "        try (Releasable refusable = AcceleratedIndexWriter.refusableIf(index.origin() == Operation.Origin.PRIMARY)) {\n",
    )

    # A no-op's tombstone goes to the writer like any other document.
    p.sub("                        documentIndexWriter.getAccumulatingIndexWriter().addDocument(doc);",
          "                        ((AcceleratedIndexWriter) documentIndexWriter).addDocument(doc);")

    # Force merge: no Lucene merge policy to flag an upgrade on.
    p.regex(
        r"        assert documentIndexWriter\.getConfig\(\)\.getMergePolicy\(\) instanceof OpenSearchMergePolicy.*?"
        r"OpenSearchMergePolicy mp = \(OpenSearchMergePolicy\) documentIndexWriter\.getConfig\(\)\.getMergePolicy\(\);\n",
        "",
    )
    p.sub("                mp.setUpgradeInProgress(true, upgradeOnlyAncientSegments);\n", "")
    p.regex(
        r"            try \{\n                // reset it just to make sure we reset it in a case of an error\n"
        r"                mp\.setUpgradeInProgress\(false, false\);\n            \} finally \{\n"
        r"                optimizeLock\.unlock\(\);\n            \}\n",
        "            optimizeLock.unlock();\n",
    )

    # Package-private members of OpenSearch classes, reached through EngineAccess.
    p.sub("compareOpToVersionMapOnSeqNo(op.id(), ", "compareOpToVersionMapOnSeqNo(EngineAccess.id(op), ")
    p.sub(' + " id=" + op.id();', ' + " id=" + EngineAccess.id(op);')
    p.sub("getSegmentInfo(lastCommittedSegmentInfos, verbose)", "EngineAccess.segmentInfo(this, lastCommittedSegmentInfos, verbose)")
    p.sub("        stats.updateMaxUnsafeAutoIdTimestamp(maxUnsafeAutoIdTimestamp.get());",
          "        EngineAccess.updateMaxUnsafeAutoIdTimestamp(stats, maxUnsafeAutoIdTimestamp.get());")

    # PrimaryOperationPolicy (3.9.0) plans with OpenSearch's own IndexingStrategy and
    # DeletionStrategy, whose factories are package-private, so the copies here cannot be handed to
    # it. AcceleratedEngineSupport.checkSupported refuses any policy but the default one, which plans as
    # primary; the engine does that directly.
    p.sub("        return primaryOperationPolicy.planIndex(indexingStrategyPlanner, index);",
          "        return indexingStrategyPlanner.planOperationAsPrimary(index);")
    p.sub("        return primaryOperationPolicy.planDelete(deletionStrategyPlanner, delete);",
          "        return deletionStrategyPlanner.planOperationAsPrimary(delete);")

    # Live search: a searcher supplier that also reads the writer's buffers, a searcher on the
    # reader alone for the requests that are not searches, and the engine's fetch layer under both.
    p.sub(
        "    @Override\n    protected final ReferenceManager<OpenSearchDirectoryReader> getReferenceManager(SearcherScope scope) {",
        "    /**\n"
        "     * A searcher taken on its own (segment stats, a point in time's segment list, a get) reads the\n"
        "     * reader alone, as a refresh left it: the buffered documents are for search requests, which take\n"
        "     * a {@link #acquireSearcherSupplier supplier}. A search's can-match check takes one on its own\n"
        "     * too, and rewrites the query by the reader's bounds: it reads the buffers as the search does, or\n"
        "     * a range over buffered documents alone would be rewritten to match nothing.\n"
        "     */\n"
        "    @Override\n"
        "    public Searcher acquireSearcher(String source, SearcherScope scope, java.util.function.Function<Searcher, Searcher> wrapper)\n"
        "        throws EngineException {\n"
        "        SearcherSupplier supplier = CAN_MATCH_SEARCH_SOURCE.equals(source)\n"
        "            ? acquireSearcherSupplier(wrapper, scope)\n"
        "            : super.acquireSearcherSupplier(EngineFetchLayer.under(wrapper, engineConfig.getIndexSettings()), scope);\n"
        "        Searcher searcher = null;\n"
        "        try {\n"
        "            searcher = supplier.acquireSearcher(source);\n"
        "            Searcher acquired = searcher;\n"
        "            return new Searcher(\n"
        "                source,\n"
        "                acquired.getDirectoryReader(),\n"
        "                acquired.getSimilarity(),\n"
        "                acquired.getQueryCache(),\n"
        "                acquired.getQueryCachingPolicy(),\n"
        "                () -> org.opensearch.common.lease.Releasables.close(acquired, supplier)\n"
        "            );\n"
        "        } catch (RuntimeException e) {\n"
        "            org.opensearch.common.lease.Releasables.closeWhileHandlingException(searcher, supplier);\n"
        "            throw e;\n"
        "        }\n"
        "    }\n\n"
        "    /** External searchers also read the writer's buffered documents when the index asks for it (LiveSearchers). */\n"
        "    @Override\n"
        "    public SearcherSupplier acquireSearcherSupplier(java.util.function.Function<Searcher, Searcher> wrapper, SearcherScope scope) throws EngineException {\n"
        "        return LiveSearchers.supplier(\n"
        "            documentIndexWriter,\n"
        "            EngineFetchLayer.under(wrapper, engineConfig.getIndexSettings()),\n"
        "            scope,\n"
        "            super::acquireSearcherSupplier,\n"
        "            engineConfig.getThreadPool() == null ? null : engineConfig.getThreadPool().generic()\n"
        "        );\n"
        "    }\n\n"
        "    @Override\n    protected final ReferenceManager<OpenSearchDirectoryReader> getReferenceManager(SearcherScope scope) {",
    )

    # Realtime: `wait_for` writes released without a refresh, the writer's external refresh listener,
    # and the reader manager told how many searches a refresh leaves.
    p.sub(
        "    private final LastRefreshedCheckpointListener lastRefreshedCheckpointListener;\n",
        "    private final LastRefreshedCheckpointListener lastRefreshedCheckpointListener;\n"
        "    /** Releases {@code wait_for} writes without a refresh when searches read the buffers; null otherwise. */\n"
        "    private volatile LiveWaitFor liveWaitFor;\n",
    )
    p.sub(
        "            internalReaderManager.addListener(documentIndexWriter);\n",
        "            internalReaderManager.addListener(documentIndexWriter);\n"
        "            if (documentIndexWriter instanceof AcceleratedIndexWriter rustWriter) {\n"
        "                this.externalReaderManager.addListener(rustWriter.externalRefreshListener());\n"
        "            }\n",
    )
    p.sub(
        "            this.primaryOperationPolicy = engineConfig.getPrimaryOperationPolicy();\n            success = true;\n",
        "            this.primaryOperationPolicy = engineConfig.getPrimaryOperationPolicy();\n"
        "            this.liveWaitFor = LiveWaitFor.start(engineConfig, documentIndexWriter);\n"
        "            success = true;\n",
    )
    p.sub(
        "                return newReader; // steal the reference\n",
        "                internalReaderManager.searchesLeft(referenceToRefresh);\n"
        "                return newReader; // steal the reference\n",
    )
    p.sub(
        '        refresh("write indexing buffer", SearcherScope.INTERNAL, false);\n',
        "        // Realtime searches see the buffered documents already, so moving them to the new segments\n"
        "        // changes no result, and lets the writer drop the buffers the new segments hold.\n"
        "        boolean live = documentIndexWriter instanceof AcceleratedIndexWriter w && w.liveSearch();\n"
        '        refresh("write indexing buffer", live ? SearcherScope.EXTERNAL : SearcherScope.INTERNAL, false);\n',
    )
    p.sub(
        "                if (onlyExpungeDeletes) {\n                    assert upgrade == false;\n",
        "                if (upgrade) {\n"
        "                    // OpenSearchMergePolicy's upgrade pass: each old segment rewritten alone before\n"
        "                    // the forced merge the request asks for.\n"
        "                    int upgraded = ((AcceleratedIndexWriter) documentIndexWriter).upgradeSegments(upgradeOnlyAncientSegments);\n"
        '                    logger.debug("rewrote {} segments for the upgrade", upgraded);\n'
        "                }\n"
        "                if (onlyExpungeDeletes) {\n                    assert upgrade == false;\n",
    )
    p.sub(
        "                if (flush) {\n                    flush(false, true);\n",
        "                // The writer commits what a forced merge or upgrade publishes, so the flush below finds\n"
        "                // nothing to commit: the committed segment list the segments API reports is read here.\n"
        "                refreshLastCommittedSegmentInfos();\n"
        "                if (flush) {\n                    flush(false, true);\n",
    )
    p.sub(
        "                this.versionMap.clear();\n",
        "                if (liveWaitFor != null) {\n"
        "                    liveWaitFor.close();\n"
        "                }\n"
        "                this.versionMap.clear();\n",
    )

    # `_seq_no` may be indexed without points (SeqNoSettings): the restore range reads it either way.
    p.sub(
        "            LongPoint.newRangeQuery(SeqNoFieldMapper.NAME, getPersistedLocalCheckpoint() + 1, Long.MAX_VALUE),",
        "            SeqNoQueries.range(directoryReader, getPersistedLocalCheckpoint() + 1, Long.MAX_VALUE),",
    )
    p.sub("import org.apache.lucene.document.LongPoint;\n", "")

    # Documents of an index with `index.engine.accelerated_source_parsing` are read in the library where
    # the mapping allows (SourceParsing); every other document, and the rest of Engine#prepareIndex, is OpenSearch's.
    p.sub(
        "    @Override\n    public IndexResult index(Index index) throws IOException {",
        "    private final SourceParsing sourceParsing = new SourceParsing();\n\n"
        "    @Override\n"
        "    public Index prepareIndex(org.opensearch.index.mapper.DocumentMapperForType m, org.opensearch.index.mapper.SourceToParse s, long seqNo, long term, long version, org.opensearch.index.VersionType vt, Operation.Origin origin, long autoIdTs, boolean retry, long ifSeqNo, long ifTerm) {\n"
        "        Index p = sourceParsing.prepare(engineConfig, documentIndexWriter, m, s, seqNo, term, version, vt, origin, autoIdTs, retry, ifSeqNo, ifTerm);\n"
        "        return p != null ? p : super.prepareIndex(m, s, seqNo, term, version, vt, origin, autoIdTs, retry, ifSeqNo, ifTerm);\n"
        "    }\n\n"
        "    @Override\n    public IndexResult index(Index index) throws IOException {",
    )

    # Package-private in Engine: not overridable from this package.
    p.drop_block("    final boolean assertSearcherIsWarmedUp(String source, SearcherScope scope) {")
    return NOTICE + p.text


def derive_changes_snapshot(text: str) -> str:
    """History ranges read `_seq_no` whichever way it is indexed (SeqNoSettings)."""
    p = Patcher(text)
    p.sub("import org.apache.lucene.document.LongPoint;\n", "import org.apache.lucene.index.IndexReader;\n")
    p.sub(
        "    private static Query operationsRangeQuery(long fromSeqNo, long toSeqNo) {\n"
        "        return new BooleanQuery.Builder().add(LongPoint.newRangeQuery(SeqNoFieldMapper.NAME, fromSeqNo, toSeqNo), BooleanClause.Occur.MUST)",
        "    private static Query operationsRangeQuery(IndexReader reader, long fromSeqNo, long toSeqNo) {\n"
        "        return new BooleanQuery.Builder().add(SeqNoQueries.range(reader, fromSeqNo, toSeqNo), BooleanClause.Occur.MUST)",
    )
    p.sub(
        "        return indexSearcher.count(operationsRangeQuery(fromSeqNo, toSeqNo));",
        "        return indexSearcher.count(operationsRangeQuery(indexSearcher.getIndexReader(), fromSeqNo, toSeqNo));",
    )
    p.sub(
        "        final Query rangeQuery = operationsRangeQuery(Math.max(fromSeqNo, lastSeenSeqNo), toSeqNo);",
        "        final Query rangeQuery = operationsRangeQuery(indexSearcher.getIndexReader(), Math.max(fromSeqNo, lastSeenSeqNo), toSeqNo);",
    )
    return p.text


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = Path(sys.argv[1]) / "org/opensearch/index/engine"
    out = Path(sys.argv[2])
    engine = derive_engine((src / "InternalEngine.java").read_text())
    (out / "AcceleratedEngine.java").write_text(engine)
    for name in HELPERS:
        text = relocate((src / f"{name}.java").read_text())
        if name == "LuceneChangesSnapshot":
            text = derive_changes_snapshot(text)
        if name == "IndexingThrottler":
            # Engine.NoOpLock is protected; EngineAccess.NoOpLock is the same class.
            h = Patcher(text)
            h.sub("new Engine.NoOpLock()", "new EngineAccess.NoOpLock()")
            text = h.text
        (out / f"{name}.java").write_text(NOTICE + text)


if __name__ == "__main__":
    main()
