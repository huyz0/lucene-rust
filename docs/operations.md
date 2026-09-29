# Operating the lucene-rust plugin

For whoever installs and runs the plugin on an OpenSearch 3.8.0 cluster:
install, configure, watch, upgrade and roll back. What the plugin does and does
not handle is [`feature-matrix.md`](feature-matrix.md); how the two halves work
inside is [`opensearch-native-queries.md`](opensearch-native-queries.md) (native
search) and [`opensearch-engine.md`](opensearch-engine.md) (the Rust engine).

## Requirements

| | |
|---|---|
| OpenSearch | **3.8.0** exactly (a plugin must match its node's version). Lucene 10.5.0, as that release bundles |
| JDK | the one OpenSearch 3.8.0 bundles (25); the plugin uses the Foreign Function & Memory API |
| Platform | Linux x86_64 or aarch64, glibc ≥ 2.34 (the `opensearchproject/opensearch:3.8.0` image has 2.34). No Windows or macOS |
| Other plugins | none that registers a `QueryPhaseSearcher` (`neural-search` does; OpenSearch accepts one per node). The security plugin works, with native stored-field reads off (below) |
| Storage | a local file system. Remote-backed storage and searchable snapshots are not supported: native search falls back and the Rust engine refuses them |

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
serving eligible searches on every index at once; the Rust engine serves only
indices that ask for it.

## Configure

| Setting | Scope | Default | Meaning |
|---|---|---|---|
| `index.lucene_rust.search.enabled` | index, dynamic | `true` | run this index's eligible searches natively. `false` sends every search to OpenSearch (counted as `disabled`) |
| `index.lucene_rust.fetch.enabled` | index, dynamic | `true` | read stored fields (`_source`, `_id`, `stored_fields`, get/mget) natively |
| `index.lucene_rust.search.native_shapes` | index, dynamic | `fast` | kept for compatibility; `fast` and `all` behave the same since every native shape measures faster |
| `lucene_rust.fetch.reader_wrapper` | node | `true`, `false` with the security plugin | install the reader wrapper native stored-field reads need. The security plugin needs the index's one reader wrapper slot for field- and document-level security, so with it installed the default is off |
| `index.lucene_rust.engine` | index, final | the node default | `true`: the index's shards are written by the Rust engine. Set at creation; creation fails with the reason if the index's configuration is one the engine refuses |
| `lucene_rust.engine.default` | node | `false` | the default of the setting above. An index that only inherits it falls back to OpenSearch's engine where the Rust one would refuse it |
| `lucene_rust.engine.node_enabled` | node | `true` | `false`: this node builds OpenSearch's engine even for Rust-engine indices. Used for rolling back and for mixed clusters |
| `breaker.lucene_rust_writer.limit` | node | `20%` | circuit breaker for the Rust writer's buffered documents. A trip is a 429 on the indexing request |
| `-Dlucene_rust.library.path=FILE` | JVM system property | the zip's `native/<os>-<arch>/` | load `liblucene_ffi.so` from elsewhere; for testing a custom build |

`index.lucene_rust.engine.fault_injection` exists for tests only. Do not set
it: with it on, a document carrying a `__lucene_rust_panic` field makes the
writer panic.

### Memory

The plugin allocates native memory the JVM heap does not see. Leave headroom
beyond `-Xmx` for it:

- **Search**, per segment of 10,000 documents or more: the query cache (at most
  1,000 entries and 16 MB), and for sorted searches decoded sort columns (at
  most 32 MB per segment and 512 MB per process). Both are freed with the
  segment.
- **Indexing**, per Rust-engine shard: the buffered documents until the next
  refresh, bounded by the `lucene_rust_writer` breaker (which counts native
  bytes against a heap-sized limit). The real-memory parent breaker does not
  see them.
- **Reads**: segments are memory-mapped, as OpenSearch's `mmapfs`/`hybridfs`
  do, so they count as page cache, not process memory. A compound (`.cfs`)
  segment is read into memory instead, while a native reader holds it.
  OpenSearch's small, freshly flushed segments are usually compound.

## Watch

### Stats

```
GET /_plugins/lucene_rust/stats
```

This is per node. Its counters are totals since the node started:

| Field | Meaning | Healthy |
|---|---|---|
| `native_queries` | searches answered natively | most of your searches ([below](#how-much-runs-native)) |
| `fallbacks.<reason>` | searches OpenSearch answered, by reason ([the list](feature-matrix.md#falls-back-and-why)) | stable; a reason you did not expect is a query shape to look at |
| `native_errors` | native calls that failed and were re-run on OpenSearch's path | **0**. Each one is logged; see [Failure signatures](#failure-signatures) |
| `open_native_readers` | native readers held (one per open searcher reader) | tracks refreshes; steady growth under steady traffic is a leak |
| `native_fetches`, `native_sequential_fetches`, `fetch_nanos` | native stored-field reads and their time | |
| `engines.rust`, `engines.java`, `engines.nrt_replica`, `engines.rust_indexer` | engines built on this node, by kind | `rust` matches the Rust-engine shard copies placed here |

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

Every plugin message starts with `lucene-rust:` (the Rust engine's own
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
| 11 `HandleLimit` | too many native handles open: a leak. `open_native_readers` will show it |
| 1-3, 7, 8 | internal argument errors; report them |

A few `native_errors` against a corrupt shard are expected: OpenSearch's own
path then fails the same way. A steady rate on healthy shards is a bug. To stop
it while it is investigated, set `index.lucene_rust.search.enabled: false` on
the affected index; this takes effect at once and needs no restart.

**The Rust engine** reports writer failures as OpenSearch reports
`IndexWriter` failures, with the operation and status in the message:

```
<operation> failed in the Rust writer (status <code>): <message>
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
   re-enable allocation and wait for green. Existing indices stay on
   OpenSearch's engine. Their eligible searches run natively from the moment a
   shard is on an upgraded node.
2. **Check.** Watch `native_errors` (0) and the fallback mix
   (`fallback-report.py`). To take native search back off one index, set
   `index.lucene_rust.search.enabled: false` on it.
3. **Adopt the Rust engine, per index, by reindex** (optional). The engine is
   chosen at creation (`index.lucene_rust.engine` is final), so an existing
   index moves by reindexing into a new one:
   ```
   PUT new-index   { "settings": { "index.lucene_rust.engine": true, ... }, "mappings": ... }
   POST _reindex   { "source": { "index": "old-index" }, "dest": { "index": "new-index" } }
   POST _aliases   { "actions": [ { "remove": { "index": "old-index", "alias": "app" } },
                                  { "add":    { "index": "new-index", "alias": "app" } } ] }
   ```
   Writes that arrive during the reindex must go to both, or be replayed, as
   for any reindex migration. `lucene_rust.engine.default: true` makes every
   *new* index use the Rust engine where it can.

## Rolling back

Rust-written segments are ordinary Lucene 10.5 segments, so going back needs
no data conversion. OpenSearch's engine opens them as they are.

1. **Stop using the Rust engine.** Rolling restart with
   `lucene_rust.engine.node_enabled: false` on every node. OpenSearch's engine
   now serves the Rust-written shards. Writes and searches continue
   throughout.
2. **Rewrite the Rust-written segments** (optional): force merge each
   Rust-engine index to one segment. Afterwards everything on disk was written
   by OpenSearch's engine.
3. **Move each index created with `index.lucene_rust.engine: true` to one
   without it**, by reindex and an alias swap, as in the adoption step:
   ```
   PUT restored-index  { "settings": { ... without index.lucene_rust.* ... }, "mappings": ... }
   POST _reindex        { "source": { "index": "rust-index" }, "dest": { "index": "restored-index" } }
   POST _aliases        { "actions": [ { "remove": { "index": "rust-index", "alias": "app" } },
                                       { "add":    { "index": "restored-index", "alias": "app" } } ] }
   DELETE rust-index
   ```
   This step is required before a rolling removal. The setting is final, so
   it cannot be taken off the index. A node without the plugin refuses a shard
   of an index that carries it: allocation fails with `unknown setting
   [index.lucene_rust.engine]`, and the replicas stay unassigned. The
   rehearsal hit exactly that before this step existed.

   Indices that only inherited `lucene_rust.engine.default` carry no setting
   and need nothing. Nor do indices that never used the Rust engine.
4. **Remove the plugin.** Rolling restart onto nodes without it
   (`bin/opensearch-plugin remove lucene-rust`, and remove the JVM option).

A full-cluster restart can replace steps 3 and 4: stop every node, remove the
plugin, start them all. OpenSearch archives unknown index settings
(`archived.index.lucene_rust.*`) when it loads the cluster metadata at startup.
That route costs downtime instead of a reindex, and the rehearsal did not run
it.

To take native search off without restarting anything, set
`index.lucene_rust.search.enabled: false` (and `index.lucene_rust.fetch.enabled:
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
| Adoption: reindex into a Rust-engine index, alias swap | 33.8 s |
| Rolling restart with `lucene_rust.engine.node_enabled: false` (per node) | 17.3 – 20.7 s |
| Force merge of the Rust-written index to one segment | 5.2 s |
| Reindex into an index without the plugin setting, alias swap | 24.0 s |
| Rolling restart onto OpenSearch without the plugin (per node) | 17.4 – 18.7 s |

A reindex scales with the index size and a restart with its recovery time.
Plan them on your own data: these are one host's numbers at 200,000 small
documents.

The rehearsal found two defects, both fixed before these numbers:
- On a Rust-engine index with deletes, a fetch of adjacent documents failed
  the shard. The soft-deletes reader was not a `CodecReader`, which
  OpenSearch's sequential stored-fields path needs.
- The procedure above lacked step 3.

## Known limits

- **A native search cannot be cancelled mid-flight.** Cancellation and the
  `timeout` are checked before and after the native call, not during it. A
  search that runs over its `timeout` answers with every hit and is flagged
  `timed_out` (see [the details](opensearch-native-queries.md#known-limits)).
- **The native query cache is per segment** (1,000 entries, 16 MB). It sits
  outside `indices.queries.cache.size` and its stats.
- **Rust engine**: no `index.merge.policy.*`, and no merge statistics in
  `_stats`, because merges run inside commits. Segments are non-compound, so
  there are more files per segment. `_segments` counts only hard deletes. On
  update-heavy indices, scores can differ from OpenSearch's engine until a
  force merge. See
  [where it behaves differently](opensearch-engine.md#where-it-behaves-differently).
- **Only Lucene 10.5's default codec** is read natively. Indices from older
  Lucene majors, or with a custom per-field format, are answered by
  OpenSearch.

## Verifying a build

Before rolling a new build out, run the proofs a release is held to:

| | |
|---|---|
| `scripts/docker-test.sh gate` | the workspace gate: lint, tests, coverage, fixtures, licences |
| `scripts/verify-opensearch.sh --yaml` | a real 3.8.0 node: native against OpenSearch's answer, SIGKILL, force merge, OpenSearch's REST YAML suites |
| `scripts/verify-opensearch.sh --engine --yaml` | the same with every index on the Rust engine |
| `scripts/verify-opensearch-cluster.sh` | three nodes: replication, recovery, failover, relocation between the engines |
| `scripts/verify-rollback.sh` | this page's upgrade and rollback |
| `scripts/soak-opensearch.sh` | the multi-day soak under chaos, ending in Lucene `CheckIndex` on every shard |
