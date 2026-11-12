# The accelerated engine in OpenSearch (M5)

An OpenSearch 3.9.0 shard whose `IndexWriter` is the Rust one. Searches still
go through M2's path (see [`accelerated-queries.md`](accelerated-queries.md)):
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
| Documents | Inverted in Java by `DocumentEncoder`, a port of `IndexingChain` / `FreqProxTermsWriterPerField` / `FieldInvertState`, including payloads, and norms through the index's `Similarity`. The inverted fields cross JNI as one blob per operation, and Rust writes them as explicit documents (`lucene-index/src/index_writer/explicit.rs`). |
| Updates and deletes | `softUpdateDocuments`: the new document (or delete tombstone) is added, and every earlier document with the `_id` gets `__soft_deletes`, atomically. |
| Refresh | A Rust **commit**, carrying the live commit data. Indexing continues while it runs. The writer reads the commit data through an upcall after it has flushed the buffered documents, as `IndexWriter.commit` reads `liveCommitData` after `flushAllThreads`, so the recorded `max_seq_no` covers every flushed operation. The engine's readers are Java `StandardDirectoryReader`s opened on those commits and reuse every unchanged segment reader. `SoftDeletesReader` applies soft deletes the way `IndexWriter`'s NRT readers do, and keeps fully soft-deleted segments for the history they hold. |
| File lifetime | OpenSearch's `CombinedDeletionPolicy` runs in Java and names the commits to drop, and Rust deletes them. A reader pins its commit's files until it closes (`hold_commit` / `release_files`). |
| Merges | The Rust writer's `TieredMergePolicy`, run on a `ConcurrentMergeScheduler` configured from `index.merge.scheduler.*`, with `SoftDeletesRetentionMergePolicy`, `PrunePostingsMergePolicy` and `RecoverySourcePruneMergePolicy` ported. The minimum retained sequence number comes from OpenSearch's `SoftDeletesPolicy` at every commit and force merge. |
| Memory | The buffered documents' RAM is accounted to the `indexing_buffers` circuit breaker. Its limit, `breaker.indexing_buffers.limit`, defaults to three times `indices.memory.index_buffer_size` (30% of the heap by default) plus `indices.store.page_cache.size`, because the page cache's resident bytes are charged to it too, without ever refusing anything. The writer stalls indexing once its buffered and flushing bytes pass twice its RAM buffer, as Lucene's `DocumentsWriterStallControl` does, so the limit sits above that bound. Only an index operation on the primary can trip it, and the trip is a 429 on that document. The no-op that records the refusal, deletes, and operations on replicas or in recovery are counted without tripping, because a failure there would fail the shard, and Lucene's `IndexWriter` never refuses a document for memory. The parent breaker adds these bytes in only with `indices.breaker.total.use_real_memory: false` (the default real-memory parent measures the heap, and these bytes are native). |
| Failure | A Rust panic is caught at the FFI boundary. The writer records it as tragic, the engine fails that one shard, and OpenSearch recovers the shard from its translog. |
| `maxDocs` | Enforced by the accelerated writer (`IndexWriter.MAX_DOCS`, or the engine's lower limit), with Lucene's message. |

## Turning it on

| Setting | Scope | Meaning |
|---|---|---|
| `index.engine.accelerated` | index, final | `true` serves the index's primaries and document-replication replicas with the accelerated engine. If the index is configured in a way the accelerated writer refuses (below), creating the index fails with the reason. |
| `indices.engine.accelerated.default` | node | Sets the default of the index setting above, so every index without its own setting (system indices included) uses the accelerated engine **where it can**. An index that only inherits this default is served by OpenSearch's engine instead if it needs something the accelerated writer refuses: a refused index configuration, or a creation mapping with a field whose documents it refuses. |
| `indices.engine.accelerated.node_enabled` | node | `false` makes the plugin build OpenSearch's own engine for accelerated-engine indices on this node, for a mixed cluster during a rollout or rollback. Rust-written segments are ordinary Lucene 10.5 segments, so either engine takes over a shard from the other. |
| `index.testing.fault_injection` | index, final | Test only: a document with a `__lucene_rust_panic` field makes the writer panic. |
| `index.segment_partition.field` | index, final | Names a field whose value is each document's partition, like a partitioned table's partition key: see [Partitions](#partitions). Empty (the default) partitions nothing. |

`GET /_plugins/lucene_rust/stats` counts the engines built, as
`engines.accelerated`, `engines.standard` and `engines.nrt_replica`.

## Partitions

With `index.segment_partition.field` set, every segment of a shard holds
the documents of one partition, and merges never mix partitions, so a
partition's documents stay together on disk however many partitions share
the shard. Use it instead of an index per partition, which multiplies
shards and their memory.

- A document's partition is the field's value on the document (on the root
  of a nested block): a keyword's or ip's value, or a numeric's or date's.
  Documents without a value, and delete tombstones, share the default
  partition.
- The writer keeps one indexing buffer per partition in flight, up to 1024
  of them, all under the shard's indexing buffer: when that fills, the
  largest is flushed, and a refresh flushes them all, one segment per
  partition. Past 1024 partitions in flight, a buffer of another partition
  is flushed to make room, so segments get small.
- Each segment records its partition the way OpenSearch's context-aware
  segments do (the `bucket` attribute on the segment and on `_id` and
  `_seq_no`), and merges are chosen per partition, as
  `CriteriaBasedMergePolicy` does. OpenSearch's own engine therefore reads
  and merges these segments alike, and stock Lucene reads them as any other.
- A document's partition is the smallest value when it has several. Bytes
  that are not UTF-8 (an ip's) are recorded as `0x` and hex. A value held
  only as analyzed text is refused.
- A document may move to another partition on update: the earlier version is
  soft deleted wherever it is.
- The default partition is recorded as `-2`, as OpenSearch records it, so a
  partition whose value is `-2` shares segments with it.
- An index served by OpenSearch's engine (a fallback, or
  `indices.engine.accelerated.node_enabled: false`) ignores the setting.
- OpenSearch's own context-aware setting (`index.context_aware.enabled`, with a
  `context_aware_grouping` script) is still refused.

## Index sorting

`index.sort.*` is served natively: every segment is flushed and merged in the
sort's order, with OpenSearch's parent field (`__nested_parent`) keeping each
nested block whole, so the segments are those OpenSearch's own writer would
write. Sorting by a keyword partition field (`index.sort.field: tenant`) puts
each partition's documents side by side: a `term` filter on the sort's first
field is then matched as that run of doc ids, checked against the term's
postings, and the other clauses skip straight to it.

## Derived source

`index.derived_source.enabled` (and `index.derived_source.translog.enabled`)
works as on OpenSearch's own engine: `_source` is not stored, and OpenSearch's
mappers rebuild it at fetch from doc values and stored fields, so searches,
gets, `top_hits`, updates and reindexing return what a stock node returns and
the shard is the same size. What the mappers read comes from the native
reader: a leaf of such an index keeps the last document it read, so the one
read per rebuilt stored field becomes one read per hit, and a hit's doc
values come back in one native call (`ffi_jvm_reader_doc_values_alloc`) for
every field asked for. The fetch phase's `docvalue_fields` are read the same
way in every index. An iterator walked as a scan (`nextDoc`, `advance`, or
more than 128 documents) hands over to the segment's own.
With `index.search.realtime.enabled`, a buffered hit's doc values come
from the writer's buffers (`ffi_live_view_doc_values_alloc`), so its derived
`_source` and `docvalue_fields` are those it has once refreshed.

## Leaner sequence numbers

Every document carries `_seq_no`, which stock OpenSearch indexes twice: as a
point (for the range queries recovery and history use) and as doc values. Two
opt-in index settings, both fixed at creation, trim it, after Elasticsearch's
`index.seq_no.index_options` (9.1) and `index.disable_sequence_numbers` (9.4):

| Setting | Meaning |
|---|---|
| `index.seq_no.index_options` | `points` (default) as stock. `doc_values_only` writes `_seq_no` as doc values with a skip index and no points. Recovery, history snapshots and every search on `_seq_no` (`range`, `term`, `terms`, sorts, aggregations) answer exactly as stock: the engine's own ranges read the doc values, a search's `_seq_no` clause runs on them natively, and the shard's readers show them to Lucene as points, so OpenSearch's point queries rewrite and match as before. |
| `index.disable_sequence_numbers` | Needs `doc_values_only`. Merges also write `UNASSIGNED_SEQ_NO` (-2) and `UNASSIGNED_PRIMARY_TERM` (0) for every document below the shard's retained minimum (the operations soft-deletes retention no longer keeps for replicas and recovery), so those values compress to nothing. Replication, recovery and history above the minimum are unchanged. |

With `disable_sequence_numbers`, a document whose history is gone reports no
`_seq_no` or `_primary_term` in a get or a search hit (`seq_no_primary_term`),
`if_seq_no` writes on it conflict, an `_update` of it runs without a
sequence-number check (as for an index without them), and a search on
`_seq_no` sees it as -2. Use it for append-mostly data such as logs; keep
`doc_values_only` alone where clients rely on optimistic concurrency.
The segments of both stay readable by stock Lucene. Both need the accelerated engine:
a node that would open such an index with OpenSearch's engine refuses it,
since that engine runs its history and retention ranges on `_seq_no` points.
Segment-replication replicas and realtime-buffered documents show `_seq_no` as
points too.
Numbers: [docs/benchmarks/lean-seq-no.md](benchmarks/lean-seq-no.md).

## What it refuses

When an index is created:

- a codec other than `default` or `lucene_default`
- index sorting on an index created before 3.2.0 (whose nested blocks have no
  parent field), or together with `index.search.realtime.enabled`
- context-aware segments
- remote-backed storage (remote-store indices or nodes)

For each document, as a 400 on that document (the shard carries on). The
node-default fallback reads the creation mapping and its dynamic templates
only: a refused field that arrives later -- a mapping update, or dynamic
mapping outside any template -- has its documents refused this way on an
index that never asked for the accelerated engine. Set `index.engine.accelerated:
false` on such indices, or add their fields to the creation mapping:

- term vectors
- vector fields
- completion fields (their postings format)
- custom term frequencies
- any per-field postings or doc-values format other than Lucene104 / Lucene90

**Replication.** Both modes work. With document replication, primaries and
replicas both run the accelerated engine. With segment replication, the primary runs
the accelerated engine and its replicas run OpenSearch's `NRTReplicationEngine`,
which copies the accelerated writer's segments and serves them through M2's native
reads. A promoted replica becomes a Rust primary on the segments it copied.

A plugin engine is not a segment-replication primary on OpenSearch 3.8 by
itself. The primary's `CopyState` asks the shard's `Indexer` for the last
refreshed checkpoint, and `EngineBackedIndexer` answers only for an
`InternalEngine`. `InternalEngine.lastRefreshedCheckpoint()` is final, and
enough of `InternalEngine` is final that a subclass cannot work either.
`RustIndexerFactory` closes the gap. `IndicesService` chooses
the indexer factory with no plugin hook, but the index's `IndexModule` holds
it until it builds the `IndexService`, which hands it to every shard. Plugins
see the module first (`onIndexModule`). The plugin replaces the module's
factory there, once per index and before any shard exists. The replacement
wraps a `RustEngine` in an `EngineBackedIndexer` that answers from the
engine's own checkpoint listener, and builds every other engine as before.

That is the plugin's one reflective write into OpenSearch's state. It is
per-index because the module is the last place the factory passes through
before it is copied into each shard. The field is resolved at node start, so
an OpenSearch version without it fails the node, not a shard at failover.

Every writable engine a shard runs is built through that factory: the first
one, one after a reset, and one on a replica's promotion. A `RustEngine` built
any other way refuses to start as a segment-replication primary, and says
why. `_plugins/lucene_rust/stats` counts the wrapped engines as
`engines.accelerated_indexer`.

**Without the hook.** The upstream fix is an overridable
`Engine#lastRefreshedCheckpoint()` (and `currentOngoingRefreshCheckpoint()`)
that `EngineBackedIndexer` asks, instead of testing for `InternalEngine`.
`RustEngine`'s public methods of the same signature then override it with no
change on our side. The plugin checks at startup whether `Engine` declares
them. If it does, it writes nothing into `IndexModule`, and `stats` reports
`engines.segrep_primary: native` (otherwise `indexer_factory`). The change is
on `huyz0/opensearch`, branch `rust-engine-refresh-checkpoint`, off 3.8.0,
with a unit test. With that patched `opensearch-3.8.0.jar` in the node image,
the cluster proof passes with the hook unused:
`OS_IMAGE=<patched image> scripts/verify-opensearch-cluster.sh --no-build`.

The temporary read-only engine during an engine reset is the one exception,
as on stock OpenSearch: a `CopyState` in that window fails there too.
Remote-backed storage stays refused, because its upload listener asks for an
`InternalEngine` by class, in a place no factory reaches.

## Where it behaves differently

- **Refresh is a commit.** Every refreshed operation is already in the last
  Lucene commit. A restart therefore replays only what was never refreshed.
  The safe commit also advances more often, so history retention (which is
  bounded by the safe commit) releases history sooner. On update-heavy indices
  this means merges reclaim deleted documents on a different schedule than
  Java's, and BM25 statistics, so scores, can differ until a force merge.
  Scores on insert-only indices match exactly. A restart also rolls back to
  the newest safe commit, and that may be a refresh: the derived
  `testRebuildLocalCheckpointTrackerAndVersionMap` records every commit the
  engine writes as a commit point -- including the engine's own refreshes,
  such as a replica's `unsafe_version_map` refresh -- through an internal
  refresh listener, where `InternalEngineTests` records only flushes
  (`COMMIT_MODEL` in `opensearch-plugin/tools/derive_engine_tests.py`).
- **Segment stats count only hard deletes.** `_segments` and `_cat/segments`
  read document counts off the unwrapped `SegmentReader`, which here does not
  have the soft deletes applied, as on `NRTReplicationEngine` replicas.
  Index-level `docs.count` and `docs.deleted` are exact.
- **Merges run in the accelerated writer.** Its merge threads take the
  `index.merge.policy.*` (tiered or `log_byte_size`, and `index.merge.enabled`),
  `index.merge.scheduler.*` and `index.merge_on_flush.*` settings, and report each merge to the engine as OpenSearch's merge
  scheduler does: `_stats` merge counts and times, the indexing throttle once
  more than `max_merge_count` merges are in flight, the flush after a big merge
  or once merges finish on an idle shard, and a failed merge failing the
  shard. On segment-replication indices each merged segment goes to
  OpenSearch's merged-segment warmer before the merge is published
  (`indices.replication.merges.warmer.enabled`). A finished merge becomes
  visible at the next refresh.
- **No infoStream.** Lucene's `IndexWriter` debug log has no counterpart.

## How it is verified

| What | Where |
|---|---|
| OpenSearch's own `InternalEngineTests`, derived to build `RustEngine` | `scripts/opensearch-engine-tests.sh`. 118 pass. The 33 skipped are each listed with a reason in `opensearch-plugin/tools/derive_engine_tests.py`: the refusals above, failure injection through a Java `IndexWriter` or in-memory `Directory`, and the differences above. |
| The Rust writer against Lucene's `IndexWriter` on the same operations | `EngineWriterDiffTest` (`gradle -p opensearch-plugin check`) |
| One node, the accelerated engine against OpenSearch's, operation for operation: bulk results, OCC, get, search, scores, aggregations; then restart, SIGKILL, a panic, the breaker, the refusals | `scripts/verify-opensearch.sh --engine` (`opensearch-plugin/e2e/verify_engine.py`) |
| OpenSearch's REST YAML suites with every index on the accelerated engine, against a stock node | `scripts/verify-opensearch.sh --engine --yaml` |
| Three nodes: document replication, peer recovery, failover, relocation between Rust and Java nodes both ways; segment replication from a Java primary to native-read replicas, and from a Rust primary through failover, promotion into the accelerated engine, a replica on the Java node and a force merge | `scripts/verify-opensearch-cluster.sh` (`opensearch-plugin/e2e/verify_cluster.py`) |
