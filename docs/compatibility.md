# Compatibility ledger

What the plugin keeps identical to stock OpenSearch 3.9.0 (Lucene 10.5.x), what it
adds, and where it deviates, with the reason and the measured cost. It is the
release-facing summary: [`feature-matrix.md`](feature-matrix.md) says what runs
native, [`operations.md`](operations.md) how to run it, and the parity ledger
([`parity.md`](parity.md)) holds the per-class rows. Where this page and a row
disagree, check the code and fix both.

Rules the ledger is held to (project decisions): stock settings, formats and
APIs that we replace stay readable and writable by stock, with stock's defaults;
a new feature or format of ours needs no compatibility and is opt-in, and the
ledger marks it so; nothing is silently remapped to an opt-in feature.

Status of each line is one of **same** (verified identical to stock), **extended**
(a superset: stock's behaviour plus an addition), **opt-in** (our own, off unless
asked), **deviates** (differs from stock, listed with its effect) and **refused**
(rejected loudly, never silently different).

## 1. Pins and platforms

| | |
|---|---|
| OpenSearch | 3.9.0 exactly. A plugin build is pinned to its node version. 2.19.x indices are readable and upgradable (below) |
| Lucene | segments written as Lucene 10.5 (`Lucene104` codec); read: Lucene 9.0 to 10.5 default codecs |
| JDK | the JDK 25 OpenSearch 3.9.0 bundles (Foreign Function and Memory API) |
| CPU | Linux x86-64 (AVX-512 with runtime detection, AVX2 fallback) and aarch64 (NEON/SVE). Windows and macOS: no |
| Storage | local file systems. Remote-backed storage and searchable snapshots: refused by the engine, native search falls back |

## 2. On-disk formats

| Format | Read | Written | Status |
|---|---|---|---|
| `Lucene104` postings and block-tree terms, `Lucene90` stored fields (both modes), doc values, norms, points, live docs, compound, `Lucene94` field infos, `Lucene99` segment info, `segments_N`, `Lucene99` HNSW and flat vectors, term vectors | yes | yes | **same**: Lucene reads the bytes (`verify-write-path.sh`, `verify-interop.sh`); the postings, stored-fields, doc-values and BKD writers are byte-identical to Java on the fixtures |
| Lucene 9.0 to 10.4 formats (`backward-codecs`) | yes | no | **same**: read and merged as they are; a force merge or `_upgrade` rewrites them into 10.5 |
| OpenSearch custom-codecs `zstd`, `zstd_no_dict`, `best_compression` stored fields | yes | yes | **same** codec names and bytes as OpenSearch's engine (`ZSTD104`, `ZSTDNODICT104`); `zstd` needs the bundled custom-codecs plugin installed |
| stock k-NN formats (`KNN9120Codec`, faiss/nmslib files) | no | no | **deviates**: segments the stock k-NN plugin wrote are served by OpenSearch's path (`accelerated_open_failed`); the plugin serves `knn_vector` itself with the `lucene` engine formats and the opt-in `extended` engine |
| `Adaptive90` doc values (`index.codec.adaptive_doc_values`, `auto` by default per index) | yes | yes | **opt-in** by sampling: written only for a shard whose first 64K documents have a column clustered in document order. Stock Lucene refuses the format name, so such an index is not stock-readable until force merged with the setting off and the segments rewritten |
| `ZstdTerms104` terms (`index.codec.terms_compression: zstd`), paged zstd stored fields (`PagedZstd104`, `index.codec.stored_fields.page_aligned`) | yes | yes | **opt-in**, own formats, not stock-readable |
| `index.codec.stored_fields.recompress_on_merge` (`off` by default) | yes | yes | **same format** (plain LZ4 blocks); `on` / `non_force` make merged `.fdt` 9-28% smaller for about 14% of logs indexing throughput on a warmed node |
| lean `_seq_no` (`index.seq_no.index_options`, `disable_sequence_numbers`) | yes | yes | **opt-in**: segments stay Lucene-readable but OpenSearch's engine refuses such an index |
| IVF, disk HNSW, RaBitQ/OSQ vector formats (`extended` k-NN engine), seismic sparse | yes | yes | **opt-in**, own formats |
| encrypted directories (`index.store.encrypted.*`) | yes | yes | **opt-in**, own format, FIPS option (aws-lc-rs) |
| Lucene 8 and earlier; non-default postings formats (`completion`); QAT codecs | no | no | **refused** or served by OpenSearch's path |

Rollback: segments written with the default settings are ordinary Lucene 10.5
and the stock distribution opens them (copy-and-open test in
`opensearch-plugin/e2e/verify_rollback.py`, rerun in the 2026-11-15 sweep: counts,
aggregations, sort and nested hits identical). Indices that carry
`index.engine.accelerated: true` need the reindex step of
[`operations.md`](operations.md#rolling-back) before the plugin is removed.

## 3. Index and node settings

### 3.1 Stock settings the accelerated engine honours

Verified by `AcceleratedEngineTests` (122 tests derived from `InternalEngineTests`) and the settings
audit: `index.merge.policy.*` (tiered and `log_byte_size`, `index.merge.enabled`), `index.merge.scheduler.*`,
`index.merge_on_flush.*`, `index.use_compound_file` and `index.compound_format`,
`index.translog.*`, `index.refresh_interval`, `index.soft_deletes.*`, `index.gc_deletes`,
`index.similarity.*` (scripted and plugin similarities fall back), `index.sort.*` on indices created on 3.2.0 or later,
`index.optimize_auto_generated_id`, `indices.memory.*`, `indices.replication.merges.warmer.enabled`,
`opensearch.shuffle_forced_merge`, `index.check_pending_flush.enabled`, `index.codec` (`default`,
`lucene_default`, `best_compression`, `zstd`, `zstd_no_dict`) and `index.codec.compression_level`. Merge counts and times,
the `max_merge_count` indexing throttle, the flush after a big merge and a failed merge failing the shard are reported as
stock's merge scheduler does. **same**.

### 3.2 Refused at index creation (a loud error, never different behaviour)

An `index.codec` outside the list above (for example `qat_*`); index sorting on an index created before 3.2.0 or with
`index.search.realtime.enabled`; context-aware segments; remote-backed storage;
`index.optimize_doc_id_lookup.fuzzy_set.enabled`; a non-default primary operation policy.
An index that only inherits `indices.engine.accelerated.default` is handed to OpenSearch's engine instead
of failing. Mapped fields the writer cannot take (`completion`, a `term_vector` other than `no`, and `knn_vector` when
the stock k-NN plugin is installed beside it) send an inheriting index to OpenSearch's engine; an index that asked
for the accelerated engine refuses those documents one by one.

### 3.3 Settings the plugin adds (defaults)

| Setting | Default | Status |
|---|---|---|
| `indices.engine.accelerated.default` / `index.engine.accelerated` | `true` / node default | **extended**: every index without its own setting runs on the accelerated engine and stores no setting, so stock still opens it. Indices created on 2.x are included; segments from Lucene 9 are read as they are |
| `indices.engine.accelerated.node_enabled` | `true` | rollback switch |
| `index.search.accelerated`, `index.fetch.accelerated`, `index.highlight.accelerated` | `true` | native search, answers identical to stock in the differential suites |
| `index.highlight.accelerated_extended` | `false` | **opt-in**: also highlights realtime views, derived fields, Thai, unpaired surrogates, `more_like_this` on `plain`, and throws OpenSearch's own errors from the native path; off, those run on OpenSearch's highlighters |
| `index.script.accelerated_extended` | `false` | **opt-in**: also runs natively the script shapes that measured slower than OpenSearch's own classes on a whole-index scan of 300k documents: Painless over `date` values (`java.time` through `def`, 2-5x slower), `random_score` seeded by `_id` (about 2x slower, every scored document's stored `_id` is decompressed), and Lucene expressions that score, sort or filter (1.5-2x slower). Off, those run on OpenSearch's script and function score classes; every other script and function score shape, and expression aggregations, stay native either way |
| `index.search.approximation.exact` | `true` | **deviates, bounded**: a sorted `match_all` or `range` that stock answers with its approximation returns the exact top hits, 4-5x faster. Only which documents of the last hit's value survive a BKD-leaf tie can differ from stock's |
| `index.codec.stored_fields.recompress_on_merge` | `off` | see section 2 |
| `index.codec.adaptive_doc_values` | `auto` | see section 2. Only `true` is stored in a new index; `auto` and `false` leave the index without the setting (it follows `indices.codec.adaptive_doc_values.default`), so a node without the plugin still opens it. Verified by soak `plugin_swap` with the default config: rolling removal and re-adding kept every acknowledged write |
| `index.search.aggregation_segment_cache.enabled` | `false` | **opt-in**, same results |
| `index.engine.accelerated_source_parsing` | `true` | same documents: mappings of keyword, text, numbers, boolean, date, geo_point, nested, objects, multi-fields and `copy_to` (and dynamically mapped fields) are read in the library, everything else goes to the document mapper; `false` leaves every document to the mapper |
| `index.search.realtime.enabled`, `max_staleness` | `false`, `20ms` | **opt-in** |
| `index.merge.reorder.*` (BP reorder) | disabled | **opt-in**, stock-readable, forced merges 3x slower. Tuning knobs, all dynamic: `index.merge.reorder.fields` (every indexed field), `index.merge.reorder.min_doc_freq` (4096), `index.merge.reorder.max_doc_freq` (1.0), `index.merge.reorder.min_partition_size` (32), `index.merge.reorder.max_iters` (20), `index.merge.reorder.ram_budget` (256 MB; a merge needing more keeps its order), `index.merge.reorder.min_merge_docs` (65536), `index.merge.reorder.min_merge_ratio` (0.0) |
| `script.js.enabled`, `script.js.timeout`, `script.js.max_heap` | `false`, `10s`, `128mb` (16 mb to 2 gb) | **opt-in** `"lang": "js"` scripts (cargo `js`); the timeout bounds one search's script run over all its documents, the heap is the cap of each search thread's V8 isolate, held outside the JVM heap and its breakers |
| `index.knn.target_recall` and its sample settings | `0` (off) | **opt-in** |
| `index.segment_partition.field`, `knn.ivf_partition_clusters` | unset, `0` | **opt-in** |
| `index.store.read_mode` | `mmap` | stock's IO |
| `indices.store.page_cache.size` | 1 GB | a cap, filled only by `page_cache` and encrypted shards |
| `indices.aggregation_segment_cache.size` | 256 MB | a cap for opt-in indices |
| `indices.fetch.accelerated_reader` | `true` | with the security plugin installed, stored fields are read natively beneath its field- and document-level security, which still filters and masks what is returned |
| `indices.logs_preset.enabled` | `true` | registers component templates only |
| `index.testing.fault_injection` | `false` | tests only: a document with a `__accelerated_panic` field panics the writer when on |

The k-NN settings (`index.knn*`, `knn.*` and the circuit breaker) equal stock k-NN's
`KNNSettings` (checked against the source when they were added). ml-commons settings are registered under
stock's keys (`opensearch/ml/MlSettings.java`).

## 4. APIs

### 4.1 Search

Every query, aggregation and request feature in [`feature-matrix.md`](feature-matrix.md) is native or falls back to
OpenSearch's own code with a counted reason; a fallback is never an error. Differential evidence against a stock 3.9.0
node (2026-11-15 sweep, [`correctness-sweep`](#6-evidence)): random queries, sorts and aggregations, 94 analyzers,
typed Painless programs, every field type with `search_after` walks, 83 API scenarios (82 identical, 1
cosmetic: the order of two sub-details in a `dis_max` explanation), nested and join queries, edge cases, and the
operation-stream and crash fuzzers. Two bugs found and fixed (#358, #362).

**deviates**: tie order among documents with equal sort keys after a force merge (document order differs from stock's;
stock itself varies across shards), t-digest and floating-point sums that depend on document order, and scores on
update-heavy indices until a force merge (refresh is a commit, so deletes are reclaimed on another schedule).

### 4.2 REST and plugin APIs

| API | Status |
|---|---|
| k-NN: mapping, `knn` query, `lucene` engine with `m`, `ef_construction`, `ef_search`, encoders, `mode`, `compression_level`, `derived_source` | **same** names and DSL (settings equal `KNNSettings`). Faiss and nmslib engines are not provided |
| k-NN `extended` engine (IVF, disk IVF, disk HNSW, RaBitQ, ERQ2-8, OSQ), `target_recall` | **opt-in**, never a silent remap of an engine name |
| ml-commons: models, connectors (HTTP, AWS SigV4, Google), agents (conversational, plan-execute-reflect, AG-UI), memory, MCP, remote and local ONNX/TorchScript inference | a port of ml-commons ([`ml-commons-remote.md`](ml-commons-remote.md), `parity/ml-commons.md`); same API names and setting keys. Not part of the 2026-11-15 differential sweep |
| neural search, neural sparse, seismic | served by this plugin itself (`scripts/check-neural-plugin.py`, `check-sparse-plugin.py`); the bundled `neural-search` plugin registers a `QueryPhaseSearcher` of its own and OpenSearch allows one per node, so it is removed from a node that runs this plugin |
| cross-cluster replication | leader: **same**. Follower: needs `opensearch-plugin/ccr/0001-configurable-follower-engine.patch` for `plugins.replication.follower.engine: delegate`; unpatched, a follower's index writes stock segments (an `extended` k-NN field cannot be followed) |
| `GET /_plugins/accelerated/stats`, `_plugins/accelerated/*` | new endpoints; stock tools ignore them |
| `_stats`, `_segments`, `_cat/segments` | **deviates**: `_segments` counts hard deletes only (soft deletes are not applied on the unwrapped reader, as on `NRTReplicationEngine` replicas); index-level `docs.count` and `docs.deleted` are exact |
| segment replication | Rust primary, replicas run `NRTReplicationEngine`; the plugin needs one reflective write into `IndexModule` unless OpenSearch exposes `Engine#lastRefreshedCheckpoint` (upstream change prepared) |
| remote-backed storage, searchable snapshots | **refused** by the engine |

## 5. Known deviations, in one place

1. Refresh is a commit; recovery replays only the unrefreshed tail, history retention releases sooner.
2. `_segments` and `_cat/segments` count hard deletes only.
3. `index.search.approximation.exact`: exact hits, tie-leaf survivors may differ.
4. Document order after a force merge can differ from stock's, and sort ties across shards, t-digest results and floating-point sums depend on it (stock varies run to run on ties).
5. No `infoStream`.
6. One plugin per node may register a `QueryPhaseSearcher`: `neural-search` cannot run beside this plugin.
7. Opt-in own formats are not readable by stock; the default configuration writes none except `Adaptive90` where `auto` selects it. Set `indices.codec.adaptive_doc_values.default: false` before an adoption that must stay stock-readable.
8. Indices carrying `index.engine.accelerated: true` cannot be opened by a node without the plugin: reindex first.

## 6. Evidence

Differential and fuzz scripts live in `opensearch-plugin/e2e/` (`fuzz_diff.py`, `fuzz_analysis.py`, `fuzz_painless.py`, `fuzz_types.py`,
`verify_sort_types.py`, `verify_api_stock.py`, `verify_edge_cases.py`, `verify_crash_recovery.py`, `verify_knn_exact.py`,
`verify_join_stock.py`, `verify_nested_stock.py`), the soak scenarios in `soak.py` (restarts, rolling upgrade 2.19.6 to 3.9.0, plugin removal and
re-add, snapshots, cross-cluster replication) and `scripts/op-stream-fuzz.sh`, `scripts/crash-fuzz.sh`. Native fuzz targets
are in `crates/lucene-ffi/fuzz`.
