# Operating the lucene-rust plugin

For whoever installs and runs the plugin on an OpenSearch 3.9.0 cluster:
install, configure, watch, upgrade and roll back. What the plugin does and does
not handle is [`feature-matrix.md`](feature-matrix.md); how the two halves work
inside is [`accelerated-queries.md`](accelerated-queries.md) (native
search) and [`opensearch-engine.md`](opensearch-engine.md) (the accelerated engine).

## Requirements

| | |
|---|---|
| OpenSearch | **3.9.0** exactly (a plugin must match its node's version). Lucene 10.5.1, as that release bundles |
| JDK | the one OpenSearch 3.9.0 bundles (25); the plugin uses the Foreign Function & Memory API |
| Platform | Linux x86_64 or aarch64, glibc ≥ 2.34 (the `opensearchproject/opensearch:3.9.0` image has 2.34). No Windows or macOS |
| Other plugins | none that registers a `QueryPhaseSearcher` (`neural-search` does; OpenSearch accepts one per node). The security plugin works, with native stored-field reads off (below) |
| Storage | a local file system. Remote-backed storage and searchable snapshots are not supported: native search falls back and the accelerated engine refuses them |

## Install

Build the zip (or take it from a release): `scripts/opensearch-dist.sh`, then
`gradle -p opensearch-plugin bundlePlugin`. It carries `liblucene_ffi.so` for
each platform it was built with, plus `LICENSE` and `NOTICE`.

On every node:

```
bin/opensearch-plugin install file:///path/to/lucene-rust-0.1.0.zip
echo '--enable-native-access=ALL-UNNAMED' > config/jvm.options.d/lucene-rust.options
```

then restart the node (a rolling restart for a running cluster: see
[Upgrading](#upgrading-onto-the-plugin)). At startup the node loads the library
and checks its ABI version; a missing library, a wrong platform or a version
mismatch stops the node with one line naming the file and the platform. That is
the only way the plugin can stop a node.

Installing changes nothing about existing indices' data. Native search starts
serving eligible searches on every index at once, and the accelerated engine
serves every index that does not set `index.engine.accelerated: false` from the
shard's next start on that node, existing indices included; it reads and
writes stock Lucene segments. To keep OpenSearch's engine for indices that do
not ask, start the node with `indices.engine.accelerated.default: false`.

## Vocabulary

The plugin's settings, stats and messages use four words, each for one thing:

| Word | Means | Where it appears |
|---|---|---|
| accelerated | the plugin's own execution path: its engine (writer, merges, realtime), query phase, fetch, analysis | `index.engine.accelerated`, `index.search.accelerated`, `index.fetch.accelerated`, stats `accelerated_*`, messages "the accelerated engine" |
| standard | OpenSearch's own engine, when a whole index or node uses it | stats `engines.standard` |
| fallback | OpenSearch's path for one query or fetch the accelerated path does not serve | stats `fallbacks.*`, `*.fallback` |
| extended | the k-NN engine with index types stock does not have (IVF, disk HNSW, RaBitQ/OSQ quantizers, partition clusters) | `method.engine: extended` only |

A `knn_vector` field on the stock `lucene` k-NN engine also runs accelerated;
only `extended` adds the extra index types. "Engine" means two things here:
`index.engine.accelerated` picks the index engine (how documents are
written), `method.engine` picks the k-NN engine. `extended` needs
`index.engine.accelerated: true`.

## Configure

| Setting | Scope | Default | Meaning |
|---|---|---|---|
| `index.search.accelerated` | index, dynamic | `true` | run this index's eligible searches natively. `false` sends every search to OpenSearch (counted as `disabled`) |
| `index.fetch.accelerated` | index, dynamic | `true` | read stored fields (`_source`, `_id`, `stored_fields`, get/mget) natively |
| `index.search.accelerated_shapes` | index, dynamic | `fast` | `all` also runs shapes measured slower on the accelerated path; every shape measures faster today, so the two agree |
| `index.search.approximation.exact` | index, dynamic | `false` | `true`: a `match_all` or `range` that OpenSearch would answer with its approximation (sorted by the range's field, `track_total_hits` not `true`) runs natively and returns the exact top hits. They can differ from OpenSearch's only in which documents of the last hit's value are kept when that value is tied across a BKD leaf (see [benchmarks/newest-n.md](benchmarks/newest-n.md)) |
| `index.search.aggregation_segment_cache.enabled` | index, dynamic | `false` | keep each committed segment's collected aggregation part in the node's aggregation segment cache, so a dashboard repeated after a refresh collects only the new segments. Same results; see [the benchmark](benchmarks/agg-segment-cache.md) |
| `indices.aggregation_segment_cache.size` | node | `256mb` | bytes the aggregation segment cache keeps parts in across the indices that opt in; least recently used parts go first. `0` turns it off. Its counters are under `aggregation_segment_cache` in `GET /_plugins/lucene_rust/stats` |
| `indices.fetch.accelerated_reader` | node | `true`, `false` with the security plugin | install the reader wrapper native stored-field reads need. The security plugin needs the index's one reader wrapper slot for field- and document-level security, so with it installed the default is off |
| `index.engine.accelerated` | index, final | the node default | `true`: the index's shards are written by the accelerated engine. Set at creation; creation fails with the reason if the index's configuration is one the engine refuses |
| `indices.engine.accelerated.default` | node | `true` | the default of the setting above, for every index without its own setting: new ones, ones created before the plugin (on OpenSearch 3.x; 2.x-created indices keep OpenSearch's engine), and system indices. An index that only inherits it stores no setting (a node without the plugin opens it unchanged) and falls back to OpenSearch's engine where the accelerated one would refuse it. `false` keeps OpenSearch's engine for every index that does not ask |
| `indices.engine.accelerated.node_enabled` | node | `true` | `false`: this node builds OpenSearch's engine even for accelerated-engine indices. Used for rolling back and for mixed clusters |
| `breaker.indexing_buffers.limit` | node | 3 × `indices.memory.index_buffer_size` + `indices.store.page_cache.size` | circuit breaker for the accelerated writer's buffered documents, which the page cache's resident bytes are also charged to (never refused). A trip is a 429 on the indexing request |
| `-Dlucene_rust.library.path=FILE` | JVM system property | the zip's `native/<os>-<arch>/` | load `liblucene_ffi.so` from elsewhere; for testing a custom build |

`index.testing.fault_injection` exists for tests only. Do not set
it: with it on, a document carrying a `__lucene_rust_panic` field makes the
writer panic.

### Language data packs

The plugin's `japanese`, `korean` and `chinese` tokenizers
(Lucene's `JapaneseTokenizer`, `KoreanTokenizer` and `HMMChineseTokenizer`,
run in Rust on the index and query side alike) need their dictionaries,
which the plugin zip leaves out to stay small. Opt a node in per language:

```
scripts/make-data-packs.sh /tmp/packs     # kuromoji/ nori/ smartcn/ morfologik/ stempel/
cp -r /tmp/packs/kuromoji "$OPENSEARCH_PATH_CONF/lucene-rust/packs/"
```

The plugin reads `<config>/lucene-rust/packs/<pack>/` first, then the
plugin's own `packs/` (a zip built with `gradle bundlePlugin -PdataPacks`
carries all of them, about 21 MB). Restart the node after adding a pack. A
tokenizer whose pack is missing is not registered (OpenSearch builds every
registered tokenizer for every index), so an index naming it is refused as
naming an unknown tokenizer; the node logs at startup which pack it lacks:
`[japanese] not registered, its data pack [kuromoji] is not installed`.
A library built with `cargo build --release -p lucene-ffi --features
embedded-data` has every pack compiled in and needs no directory.

### Memory

The plugin allocates native memory the JVM heap does not see. Leave headroom
beyond `-Xmx` for it:

- **Search**, per segment of 10,000 documents or more: the query cache (at most
  1,000 entries and 16 MB per segment, and `indices.queries.cache.size`
  across every segment of the node, least recently used sets evicted first,
  whichever segment holds them), and for sorted searches decoded sort columns
  (at most 32 MB per segment and 512 MB per process). Both are freed with the
  segment.
- **Each native search, aggregation and rescore** is charged to OpenSearch's
  `request` circuit breaker while it runs, as OpenSearch's own aggregators
  charge it: the library counts what the call allocates and asks the breaker
  for more as it grows (at least 1 MB, then an eighth of what it holds). A
  refused charge stops the call at its next checkpoint (every 1,024 documents
  an aggregation collects, and before a count array sized by a field's
  cardinality) and the search fails with the breaker's
  `circuit_breaking_exception`, a 429, rather than running again on Lucene.
  Native aggregation results count against `search.max_buckets` as
  OpenSearch's do (`too_many_buckets_exception`).
- **The node's bound**, `indices.memory.off_heap.limit` (dynamic): everything
  the library holds outside the heap -- indexing buffers, caches, running
  requests. `auto` (the default) is half of what the machine or container
  (the JDK reads the cgroup limit) leaves after the heap and the page cache
  budget, at least 256 MB; `0` turns it off. Past it, native searches fail
  with a 429 until memory is freed; indexing goes on under its own breaker.
  `GET /_plugins/lucene_rust/stats` reports it under `off_heap_memory`: bytes
  held and the most held, the requests each bound stopped, the query caches'
  bytes and evictions, and the largest request seen.
- **Indexing**, per Rust-engine shard: the buffered documents until the next
  refresh. The writer flushes at `indices.memory.index_buffer_size` and
  stalls indexing at twice that, as Lucene's does; the `indexing_buffers`
  breaker (three times the buffer by default) refuses documents beyond it.
  The real-memory parent breaker does not see them.
- **Reads**: segments are memory-mapped, as OpenSearch's `mmapfs`/`hybridfs`
  do, so they count as page cache, not process memory. A compound (`.cfs`)
  segment is read into memory instead, while a native reader holds it.
  OpenSearch's small, freshly flushed segments are usually compound.

### Heap size

The heap keeps OpenSearch's own work: bulk and search requests while they
are handled (parsed `_source`, `IndexRequest`s, responses), translog writes,
cluster state and mappings, aggregation reduction, and the request and
fielddata caches. The accelerated engine takes Lucene's indexing buffers out of it,
so a plugin node needs less heap than a stock node for the same indexing
load. On the 1M-document benchmark (`scripts/bench-opensearch.sh`, one shard,
four bulk clients), the most the heap held after a young collection during
the load was 180 to 260 MB on the plugin node against 360 to 730 MB on a
stock one, and both held about 60 MB once the load ended.

The JVM's resident memory follows `-Xms`/`-Xmx`, not what the heap holds:
G1 touches the whole committed heap as it moves its young generation through
it. Sizing the heap to the node's need is what lowers a plugin node's
memory; the native memory listed above comes on top of it.

`indices.memory.index_buffer_size` is a share of the heap (10% by default),
and the Rust writers flush at it. When you shrink the heap, set the buffer as
a size (for example `200mb`) to keep the same flush size.

## Watch

### Stats

```
GET /_plugins/lucene_rust/stats
```

This is per node. Its counters are totals since the node started:

| Field | Meaning | Healthy |
|---|---|---|
| `accelerated_queries` | searches answered on the accelerated path | most of your searches ([below](#how-much-runs-native)) |
| `fallbacks.<reason>` | searches OpenSearch answered, by reason ([the list](feature-matrix.md#falls-back-and-why)) | stable; a reason you did not expect is a query shape to look at |
| `accelerated_errors` | accelerated calls that failed and were re-run on the fallback path | **0**. Each one is logged; see [Failure signatures](#failure-signatures) |
| `open_accelerated_readers` | accelerated readers held (one per open searcher reader) | tracks refreshes; steady growth under steady traffic is a leak |
| `accelerated_fetches`, `accelerated_sequential_fetches`, `fetch_nanos` (`accelerated`, `fallback`) | accelerated stored-field reads and the time on each path | |
| `query_phase_nanos` (`accelerated`, `fallback`), `analyzed_values` (`accelerated`, `fallback`) | query-phase time and analyzed text values, per path | |
| `engines.accelerated`, `engines.standard`, `engines.nrt_replica`, `engines.accelerated_indexer` | engines built on this node, by kind | `accelerated` matches the accelerated-engine shard copies placed here |

### How much runs native

`scripts/fallback-report.py [URL]` reads every node's stats and prints the
native share and each fallback reason's share. Take two readings under real
traffic and report the difference between them:

```
scripts/fallback-report.py http://node:9200 --save before.json
# ... an hour of traffic ...
scripts/fallback-report.py http://node:9200 --since before.json
```

A node restarted between the readings is counted from zero again.

### Logs

Every plugin message starts with `lucene-rust:` (the accelerated engine's own
messages are OpenSearch's engine messages: it is `InternalEngine` with the
writer swapped out).

## Failure signatures

A native search never fails a request OpenSearch could answer. A native
failure is logged and counted, and the search runs again on OpenSearch's path:

```
lucene-rust: native search failed (<status>), re-running on Lucene: <message>
lucene-rust: native sorted search failed (<status>), ...
lucene-rust: native aggregation failed (<status>), ...
lucene-rust: native stored fields failed (<status>), reading with Lucene: <message>
lucene-rust: native reader open failed for [<path>] (<status>): <message>
```

`<status>` is the FFI status code:

| Code | Meaning |
|---|---|
| 4 `Io` | reading a file failed |
| 5 `Decode` | a file did not decode as the format says: a corrupt index, or a format this build does not read |
| 6 `Search` | the query could not run on this index |
| 9 `Panic` | a Rust panic, caught at the boundary. Always a bug: report it with the message |
| 10 `InvalidArgument` | the plugin sent something the library rejected: a plugin/library mismatch, or a bug |
| 11 `HandleLimit` | too many native handles open: a leak. `open_accelerated_readers` will show it |
| 1-3, 7, 8 | internal argument errors; report them |

A few `accelerated_errors` against a corrupt shard are expected: OpenSearch's own
path then fails the same way. A steady rate on healthy shards is a bug. To stop
it while it is investigated, set `index.search.accelerated: false` on
the affected index; this takes effect at once and needs no restart.

**The accelerated engine** reports writer failures as OpenSearch reports
`IndexWriter` failures, with the operation and status in the message:

```
<operation> failed in the accelerated writer (status <code>): <message>
```

- A **document the engine refuses** (a term vector, a vector field, a payload,
  ...; see [the list](opensearch-engine.md#what-it-refuses)) is a 400 on that
  document. The shard carries on.
- A **breaker trip** is a 429 on the request; retry with backoff, as for any
  OpenSearch breaker.
- A **panic** (`status 9`) or an I/O failure inside the writer is *tragic*,
  exactly as a tragic `IndexWriter` exception is for OpenSearch's engine. The
  shard copy fails and OpenSearch recovers it (from its translog, or from
  another copy), then carries on. Acknowledged writes are not lost.
  `scripts/verify-opensearch.sh --engine` proves this with an injected panic.

## Upgrading onto the plugin

A rolling procedure. Rehearsed end to end, and timed, by
`scripts/verify-rollback.sh` (three nodes, every acknowledged write and a fixed
set of answers checked after every step; timings below).

1. **Install the plugin, one node at a time.** For each node: disable shard
   allocation (`cluster.routing.allocation.enable: primaries`), flush, stop the
   node, install the plugin and the JVM option, start it, wait for it to join,
   re-enable allocation and wait for green. Each shard of an existing index
   moves onto the accelerated engine as it starts on an upgraded node: it
   inherits `indices.engine.accelerated.default` (on), stores no setting, and
   keeps its stock segments, which the accelerated engine reads and merges as
   they are. While the restart runs, an index has copies on both engines,
   which replicate to each other. Indices created on OpenSearch 2.x (Lucene 9
   segments) are the exception: they stay on OpenSearch's engine until
   reindexed. Eligible searches run natively from the moment a shard is on an
   upgraded node. To keep existing indices on OpenSearch's engine, start the
   upgraded nodes with `indices.engine.accelerated.default: false`.
2. **Check.** Watch `accelerated_errors` (0) and the fallback mix
   (`fallback-report.py`). To take native search back off one index, set
   `index.search.accelerated: false` on it.
3. **Pin an index to the accelerated engine** (optional). With the default
   on, there is nothing to do. With `indices.engine.accelerated.default: false`,
   an index asks for the engine at creation (`index.engine.accelerated` is
   final), so an existing one moves by reindexing into a new one:
   ```
   PUT new-index   { "settings": { "index.engine.accelerated": true, ... }, "mappings": ... }
   POST _reindex   { "source": { "index": "old-index" }, "dest": { "index": "new-index" } }
   POST _aliases   { "actions": [ { "remove": { "index": "old-index", "alias": "app" } },
                                  { "add":    { "index": "new-index", "alias": "app" } } ] }
   ```
   Writes that arrive during the reindex must go to both, or be replayed, as
   for any reindex migration. An index created this way carries the setting,
   so removing the plugin later needs the reindex in step 3 of rolling back.

## Rolling back

Rust-written segments are ordinary Lucene 10.5 segments, so going back needs
no data conversion. OpenSearch's engine opens them as they are.

1. **Stop using the accelerated engine.** Rolling restart with
   `indices.engine.accelerated.node_enabled: false` on every node. OpenSearch's engine
   now serves the Rust-written shards. Writes and searches continue
   throughout.
2. **Rewrite the Rust-written segments** (optional): force merge each
   accelerated-engine index to one segment. Afterwards everything on disk was written
   by OpenSearch's engine.
3. **Move each index created with `index.engine.accelerated: true` to one
   without it**, by reindex and an alias swap, as in the adoption step:
   ```
   PUT restored-index  { "settings": { ... without the plugin's settings ... }, "mappings": ... }
   POST _reindex        { "source": { "index": "rust-index" }, "dest": { "index": "restored-index" } }
   POST _aliases        { "actions": [ { "remove": { "index": "rust-index", "alias": "app" } },
                                       { "add":    { "index": "restored-index", "alias": "app" } } ] }
   DELETE rust-index
   ```
   This step is required before a rolling removal. The setting is final, so
   it cannot be taken off the index. A node without the plugin refuses a shard
   of an index that carries it: allocation fails with `unknown setting
   [index.engine.accelerated]`, and the replicas stay unassigned. The
   rehearsal hit exactly that before this step existed.

   Indices that only inherited `indices.engine.accelerated.default` carry no setting
   and need nothing. Nor do indices that never used the accelerated engine.
4. **Remove the plugin.** Rolling restart onto nodes without it
   (`bin/opensearch-plugin remove lucene-rust`, and remove the JVM option).

A full-cluster restart can replace steps 3 and 4: stop every node, remove the
plugin, start them all. OpenSearch archives unknown index settings
(`archived.index.engine.accelerated` and the like) when it loads the cluster metadata at startup.
That route costs downtime instead of a reindex, and the rehearsal did not run
it.

To take native search off without restarting anything, set
`index.search.accelerated: false` (and `index.fetch.accelerated:
false`) on the indices concerned.

### Timings

Measured by `scripts/verify-rollback.sh` on three nodes (one host, 1 GB heap
each). The index held 200,077 documents (3 shards, 1 replica), about 200,300
after later writes. Writes, updates and deletes ran between the steps. After
every step each index held exactly the acknowledged documents, and 60 query
shapes answered as before. The final run passed all 1,058 checks.

| Step | Time |
|---|---|
| Rolling restart onto the plugin (per node, flush to green) | 17.9 – 19.5 s |
| Adoption: reindex into a accelerated-engine index, alias swap | 33.8 s |
| Rolling restart with `indices.engine.accelerated.node_enabled: false` (per node) | 17.3 – 20.7 s |
| Force merge of the Rust-written index to one segment | 5.2 s |
| Reindex into an index without the plugin setting, alias swap | 24.0 s |
| Rolling restart onto OpenSearch without the plugin (per node) | 17.4 – 18.7 s |

A reindex scales with the index size and a restart with its recovery time.
Plan them on your own data: these are one host's numbers at 200,000 small
documents.

The rehearsal found two defects, both fixed before these numbers:
- On a accelerated-engine index with deletes, a fetch of adjacent documents failed
  the shard. The soft-deletes reader was not a `CodecReader`, which
  OpenSearch's sequential stored-fields path needs.
- The procedure above lacked step 3.

## Known limits

- **A native search's `timeout` is checked before and after the native call,
  not during it.** A search that runs over its `timeout` answers with every
  hit and is flagged `timed_out` (see
  [the details](accelerated-queries.md#known-limits)). A task cancelled
  while the call runs (the tasks API, search backpressure) stops at the call's
  next checkpoint, within about 100 ms.
- **The native query cache** takes `indices.queries.cache.size` as its node
  budget, but its sets are not in the `query_cache` section of
  `_nodes/stats`; the plugin's stats report them.
- **Fetch is not charged to a breaker.** A stored document is read whole, as
  Lucene reads it; its bytes count toward `indices.memory.off_heap.limit`.
- **Rust engine**: no `index.merge.policy.*`, and no merge statistics in
  `_stats`, because merges run inside commits. Segments are non-compound, so
  there are more files per segment. `_segments` counts only hard deletes. On
  update-heavy indices, scores can differ from OpenSearch's engine until a
  force merge. See
  [where it behaves differently](opensearch-engine.md#where-it-behaves-differently).
- **Only Lucene 10.5's default codec** is read natively, with any of its stored
  fields: LZ4 (`default`), DEFLATE (`best_compression`), or the custom-codecs
  plugin's `zstd` and `zstd_no_dict`. Indices from older Lucene majors, or
  with a custom per-field format, are answered by OpenSearch. The accelerated engine
  writes those four codecs; `zstd` needs the bundled `opensearch-custom-codecs`
  plugin installed, as it does for OpenSearch's own engine.

## Verifying a build

Before rolling a new build out, run the proofs a release is held to:

| | |
|---|---|
| `scripts/docker-test.sh gate` | the workspace gate: lint, tests, coverage, fixtures, licences |
| `scripts/verify-opensearch.sh --yaml` | a real 3.9.0 node: native against OpenSearch's answer, SIGKILL, force merge, OpenSearch's REST YAML suites |
| `scripts/verify-opensearch.sh --engine --yaml` | the same with every index on the accelerated engine |
| `scripts/verify-opensearch-cluster.sh` | three nodes: replication, recovery, failover, relocation between the engines |
| `scripts/verify-rollback.sh` | this page's upgrade and rollback |
