# The Rust engine in OpenSearch (M5)

An OpenSearch 3.8.0 shard whose `IndexWriter` is the Rust one. Searches still
go through M2's path (see [`opensearch-native-queries.md`](opensearch-native-queries.md)):
native per query where it can be, Lucene otherwise. This page covers what the
engine is, how to turn it on, what it refuses and where it behaves differently
from OpenSearch's own engine.

## What it is

`RustEngine` **is** `InternalEngine`, with the writer swapped out.
`opensearch-plugin/tools/derive_engine.py` copies `InternalEngine` from the
pinned OpenSearch sources and patches only the places that touch Lucene's
`IndexWriter`. Every patch must apply exactly once, so a new OpenSearch
release fails the derivation instead of silently producing a different engine.
The package-private helpers it needs are copied unchanged.

So sequence numbers, the local checkpoint, the version map and optimistic
concurrency control, the translog, commit user data, history retention and
safe commits all come from OpenSearch's own code, not from a reimplementation.

The writer side works like this:

| Concern | How |
|---|---|
| Documents | Inverted in Java by `DocumentEncoder`, a port of `IndexingChain` / `FreqProxTermsWriterPerField` / `FieldInvertState`, including norms through the index's `Similarity`. The inverted fields cross JNI as one blob per operation, and Rust writes them as explicit documents (`lucene-index/src/index_writer/explicit.rs`). |
| Updates and deletes | `softUpdateDocuments`: the new document (or delete tombstone) is added, and every earlier document with the `_id` gets `__soft_deletes`, atomically. |
| Refresh | A Rust **commit**, carrying the live commit data. That data is evaluated under the writer's lock, as `IndexWriter` evaluates it inside `commit`. The engine's readers are Java `StandardDirectoryReader`s opened on those commits and reuse every unchanged segment reader. `SoftDeletesReader` applies soft deletes the way `IndexWriter`'s NRT readers do, and keeps fully soft-deleted segments for the history they hold. |
| File lifetime | OpenSearch's `CombinedDeletionPolicy` runs in Java and names the commits to drop, and Rust deletes them. A reader pins its commit's files until it closes (`hold_commit` / `release_files`). |
| Merges | The Rust writer's `TieredMergePolicy`, run at commit, with `SoftDeletesRetentionMergePolicy`, `PrunePostingsMergePolicy` and `RecoverySourcePruneMergePolicy` ported. The minimum retained sequence number comes from OpenSearch's `SoftDeletesPolicy` at every commit and force merge. |
| Memory | The buffered documents' RAM is accounted to the `lucene_rust_writer` circuit breaker. It defaults to 20% of the heap, is set with `breaker.lucene_rust_writer.limit`, and counts toward the parent breaker. A trip is a 429, not an OOM. |
| Failure | A Rust panic is caught at the FFI boundary. The writer records it as tragic, the engine fails that one shard, and OpenSearch recovers the shard from its translog. |
| `maxDocs` | Enforced by the Rust writer (`IndexWriter.MAX_DOCS`, or the engine's lower limit), with Lucene's message. |

## Turning it on

| Setting | Scope | Meaning |
|---|---|---|
| `index.lucene_rust.engine` | index, final | `true` serves the index's primaries and document-replication replicas with the Rust engine. If the index is configured in a way the Rust writer refuses (below), creating the index fails with the reason. |
| `lucene_rust.engine.default` | node | Sets the default of the index setting above, so every index without its own setting (system indices included) uses the Rust engine **where it can**. An index that only inherits this default is served by OpenSearch's engine instead if it needs something the Rust writer refuses: a refused index configuration, or a creation mapping with a field whose documents it refuses. |
| `lucene_rust.engine.node_enabled` | node | `false` makes the plugin build OpenSearch's own engine for Rust-engine indices on this node, for a mixed cluster during a rollout or rollback. Rust-written segments are ordinary Lucene 10.5 segments, so either engine takes over a shard from the other. |
| `index.lucene_rust.engine.fault_injection` | index, final | Test only: a document with a `__lucene_rust_panic` field makes the writer panic. |

`GET /_plugins/lucene_rust/stats` counts the engines built, as
`engines.rust`, `engines.java` and `engines.nrt_replica`.

## What it refuses

When an index is created:

- a codec other than `default` or `lucene_default`
- index sorting
- context-aware segments
- segment replication, or remote-store nodes

For each document, as a 400 on that document (the shard carries on). A
mapping added after an index was created on the node default is not
re-checked, so its documents are refused this way too:

- term vectors
- vector fields
- completion fields (their postings format)
- payloads
- custom term frequencies
- any per-field postings or doc-values format other than Lucene104 / Lucene90

**Segment replication.** On OpenSearch 3.8, a plugin engine cannot be a
segment-replication primary. `CopyState` reads the primary's last refreshed
checkpoint through `EngineBackedIndexer.lastRefreshedCheckpoint()`, which only
answers for an `InternalEngine`, and `InternalEngine.lastRefreshedCheckpoint()`
is final. Document replication is fully supported: primaries and replicas both
run the Rust engine. Segment-replication *replicas* run OpenSearch's
`NRTReplicationEngine` behind M2's native reads.

## Where it behaves differently

- **Refresh is a commit.** Every refreshed operation is already in the last
  Lucene commit. A restart therefore replays only what was never refreshed.
  The safe commit also advances more often, so history retention (which is
  bounded by the safe commit) releases history sooner. On update-heavy indices
  this means merges reclaim deleted documents on a different schedule than
  Java's, and BM25 statistics, so scores, can differ until a force merge.
  Scores on insert-only indices match exactly.
- **Segment stats count only hard deletes.** `_segments` and `_cat/segments`
  read document counts off the unwrapped `SegmentReader`, which here does not
  have the soft deletes applied, as on `NRTReplicationEngine` replicas.
  Index-level `docs.count` and `docs.deleted` are exact.
- **Merges.** The Rust writer's merge policy uses its defaults, and
  `index.merge.policy.*` and `index.merge_on_flush` are not honoured. Merges
  run inside commits, not on a merge scheduler, so `_stats` reports no merge
  activity, and a big merge does not trigger an early flush. That flush is not
  needed, because the merged segment is already committed.
- **Segments are never compound.** The Rust writer writes non-compound
  segments only (it reads compound ones). The result is more files per segment.
- **No infoStream.** Lucene's `IndexWriter` debug log has no counterpart.

## How it is verified

| What | Where |
|---|---|
| OpenSearch's own `InternalEngineTests`, derived to build `RustEngine` | `scripts/opensearch-engine-tests.sh`. 117 pass. The 34 skipped are each listed with a reason in `opensearch-plugin/tools/derive_engine_tests.py`: the refusals above, failure injection through a Java `IndexWriter` or in-memory `Directory`, and the differences above. |
| The Rust writer against Lucene's `IndexWriter` on the same operations | `EngineWriterDiffTest` (`gradle -p opensearch-plugin check`) |
| One node, the Rust engine against OpenSearch's, operation for operation: bulk results, OCC, get, search, scores, aggregations; then restart, SIGKILL, a panic, the breaker, the refusals | `scripts/verify-opensearch.sh --engine` (`opensearch-plugin/e2e/verify_engine.py`) |
| OpenSearch's REST YAML suites with every index on the Rust engine, against a stock node | `scripts/verify-opensearch.sh --engine --yaml` |
| Three nodes: document replication, peer recovery, failover, relocation between Rust and Java nodes both ways, segment replication with native replicas | `scripts/verify-opensearch-cluster.sh` (`opensearch-plugin/e2e/verify_cluster.py`) |
