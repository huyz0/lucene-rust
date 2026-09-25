#!/usr/bin/env python3
"""Derives RustEngine and the engine-package helpers it needs from OpenSearch's own sources.

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
  curl -O https://repo1.maven.org/maven2/org/opensearch/opensearch/3.8.0/opensearch-3.8.0-sources.jar
  unzip -d os-src opensearch-3.8.0-sources.jar
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
 * Derived from OpenSearch 3.8.0 by opensearch-plugin/tools/derive_engine.py -- do not edit;
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

    p.sub("public class InternalEngine extends Engine {", "public class RustEngine extends Engine {")
    p.sub("public InternalEngine(", "public RustEngine(", count=2)
    p.sub("    InternalEngine(\n        EngineConfig engineConfig,", "    RustEngine(\n        EngineConfig engineConfig,")
    p.sub('logger.trace("created new InternalEngine");', 'logger.trace("created new RustEngine");')

    # Refuse what the Rust writer cannot produce, before anything is opened.
    p.sub(
        "        super(engineConfig);\n        if (engineConfig.isAutoGeneratedIDsOptimizationEnabled() == false) {",
        "        super(engineConfig);\n        RustEngineSupport.checkSupported(engineConfig);\n"
        "        if (engineConfig.isAutoGeneratedIDsOptimizationEnabled() == false) {",
    )

    # No Lucene merge scheduler: the Rust writer merges as it commits.
    p.sub("    private final OpenSearchConcurrentMergeScheduler mergeScheduler;\n",
          "    private final MergeStats mergeStats = new MergeStats();\n")
    p.sub("        EngineMergeScheduler scheduler = null;\n", "")
    p.regex(r"            mergeScheduler = scheduler = new EngineMergeScheduler\(.*?\);\n", "")
    p.sub("translogManagerRef, internalReaderManager, externalReaderManager, scheduler);",
          "translogManagerRef, internalReaderManager, externalReaderManager);")
    p.sub("        mergeScheduler.refreshConfig();\n", "", count=2)
    p.drop_block("    private final class EngineMergeScheduler extends OpenSearchConcurrentMergeScheduler {")
    p.sub("        return mergeScheduler.stats();", "        return mergeStats;")
    p.sub("        return mergeScheduler.onGoingMerges().size();", "        return 0;")
    p.regex(
        r"            // fill in the merges flag\n.*?\n            return Arrays.asList\(segmentsArr\);",
        "            return Arrays.asList(segmentsArr);",
    )

    # The writer: the Rust one, opened with the engine's own deletion and retention policies.
    p.sub("    private final IndexWriterFactory nativeIndexWriterFactory;\n", "")
    p.sub("                this.nativeIndexWriterFactory = new NativeLuceneIndexWriterFactory();\n", "")
    p.regex(
        r"    private DocumentIndexWriter getDocumentIndexWriter\(\) throws IOException \{.*?\n    \}\n",
        "    private DocumentIndexWriter getDocumentIndexWriter() throws IOException {\n"
        "        return RustEngineSupport.openWriter(\n"
        "            engineConfig,\n"
        "            store,\n"
        "            combinedDeletionPolicy,\n"
        "            softDeletesPolicy::getMinRetainedSeqNo,\n"
        "            softDeletesField.name()\n"
        "        );\n"
        "    }\n",
    )
    p.regex(r"    /\*\*\n     \* We should only take care of reopening parent writer.*?\n    private IndexWriter createWriter\(\) throws IOException \{.*?\n    \}\n", "")
    p.regex(r"    // pkg-private for testing\n    IndexWriter createWriter\(Directory directory, IndexWriterConfig iwc\) throws IOException \{.*?\n    \}\n", "")

    # Commit data is evaluated inside the writer's commit, under the lock that also serializes
    # indexing, so the local checkpoint is read there too (see RustIndexWriter), and every
    # commit -- refresh or flush -- carries it.
    p.sub(
        "                forceMergeUUID = commitData.get(FORCE_MERGE_UUID_KEY);\n                documentIndexWriter = writer;\n",
        "                forceMergeUUID = commitData.get(FORCE_MERGE_UUID_KEY);\n"
        "                writer.setLiveCommitData(liveCommitData(translogUUID));\n"
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
        "     * InternalEngine's commit data, every value read when the writer commits: the writer holds\n"
        "     * the lock that serializes indexing while it iterates, so the local checkpoint and the\n"
        "     * maximum sequence number describe exactly the operations in the commit.\n"
        "     */\n"
        "    private Iterable<Map.Entry<String, String>> liveCommitData(String translogUUID) {\n"
        "        return () -> {\n"
        "            final Map<String, String> commitData = new HashMap<>(7);\n"
        "            commitData.put(Translog.TRANSLOG_UUID_KEY, translogUUID);\n"
        "            commitData.put(SequenceNumbers.LOCAL_CHECKPOINT_KEY, Long.toString(localCheckpointTracker.getProcessedCheckpoint()));\n"
        "            commitData.put(SequenceNumbers.MAX_SEQ_NO, Long.toString(localCheckpointTracker.getMaxSeqNo()));\n"
        "            commitData.put(MAX_UNSAFE_AUTO_ID_TIMESTAMP_COMMIT_ID, Long.toString(maxUnsafeAutoIdTimestamp.get()));\n"
        "            commitData.put(HISTORY_UUID_KEY, historyUUID);\n"
        "            commitData.put(Engine.MIN_RETAINED_SEQNO, Long.toString(softDeletesPolicy.getMinRetainedSeqNo()));\n"
        "            final String currentForceMergeUUID = forceMergeUUID;\n"
        "            if (currentForceMergeUUID != null) {\n"
        "                commitData.put(FORCE_MERGE_UUID_KEY, currentForceMergeUUID);\n"
        "            }\n"
        "            logger.trace(\"committing writer with commit data [{}]\", commitData);\n"
        "            return commitData.entrySet().iterator();\n"
        "        };\n"
        "    }\n\n"
        "    @Override\n    public void onSettingsChanged(",
    )

    # Readers: Java readers on the Rust writer's commits.
    p.sub("\n    private final OpenSearchReaderManager internalReaderManager;\n",
          "\n    private final RustReaderManager internalReaderManager;\n")
    p.sub("        OpenSearchReaderManager internalReaderManager = null;\n",
          "        RustReaderManager internalReaderManager = null;\n", count=2)
    p.sub("        private final OpenSearchReaderManager internalReaderManager;\n",
          "        private final RustReaderManager internalReaderManager;\n")
    p.sub("            OpenSearchReaderManager internalReaderManager,\n", "            RustReaderManager internalReaderManager,\n")
    p.regex(
        r"                // We always open reader on parent IndexWriter\.\n.*?"
        r"internalReaderManager = new OpenSearchReaderManager\(directoryReader\);\n",
        "                internalReaderManager = new RustReaderManager(\n"
        "                    (RustIndexWriter) documentIndexWriter,\n"
        "                    store.directory(),\n"
        "                    shardId,\n"
        "                    engineConfig.getLeafSorter()\n"
        "                );\n",
    )
    p.sub("((StandardDirectoryReader) reader.getDelegate()).getSegmentInfos()",
          "RustReaderManager.standard(reader).getSegmentInfos()")
    # A reader is on the newest commit; buffered documents are what a refresh would add.
    p.sub(
        "        return documentIndexWriter.hasNewIndexingOrUpdates() || super.refreshNeeded();",
        "        return documentIndexWriter.hasUncommittedChanges() || super.refreshNeeded();",
    )

    # A no-op's tombstone goes to the writer like any other document.
    p.sub("                        documentIndexWriter.getAccumulatingIndexWriter().addDocument(doc);",
          "                        ((RustIndexWriter) documentIndexWriter).addDocument(doc);")

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

    # Package-private in Engine: not overridable from this package.
    p.drop_block("    final boolean assertSearcherIsWarmedUp(String source, SearcherScope scope) {")
    return NOTICE + p.text


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = Path(sys.argv[1]) / "org/opensearch/index/engine"
    out = Path(sys.argv[2])
    engine = derive_engine((src / "InternalEngine.java").read_text())
    (out / "RustEngine.java").write_text(engine)
    for name in HELPERS:
        text = relocate((src / f"{name}.java").read_text())
        if name == "IndexingThrottler":
            # Engine.NoOpLock is protected; EngineAccess.NoOpLock is the same class.
            h = Patcher(text)
            h.sub("new Engine.NoOpLock()", "new EngineAccess.NoOpLock()")
            text = h.text
        (out / f"{name}.java").write_text(NOTICE + text)


if __name__ == "__main__":
    main()
