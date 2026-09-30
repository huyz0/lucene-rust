# Feature matrix (operators)

What the lucene-rust plugin does for an OpenSearch 3.8.0 cluster, in
OpenSearch's terms: what runs in Rust, what OpenSearch still does itself, and
what is not supported. Read this to decide whether the plugin fits a workload;
[`operations.md`](operations.md) covers installing and running it, and
[`parity.md`](parity.md) is the per-Lucene-file ledger for contributors.

Pinned versions: **OpenSearch 3.8.0** (Lucene 10.5.0), Linux x86_64 and
aarch64.

## The short version

The plugin has two independent halves. Each is on by default for an index
unless noted, and each falls back to OpenSearch's own code for whatever it
does not handle. A fallback is never an error: the request is answered as
stock OpenSearch would, and the fallback is counted by reason.

| Half | What it replaces | Turned on by | When it does not apply |
|---|---|---|---|
| **Native search** | the shard's query phase (query, sort, aggregations, `post_filter`, `min_score`, `terminate_after`, scroll), and the stored-field reads of fetch and get | installed = on; `index.lucene_rust.search.enabled` (default `true`) per index | per request, for the shapes in the fallback column below |
| **Rust engine** | the shard's indexing engine: writes, refresh, flush, merges, soft deletes, replication | `index.lucene_rust.engine: true` per index, or `lucene_rust.engine.default: true` per node | index configurations and field types it refuses (below) |

Native search works on any index on a local file system, whichever engine
wrote its segments. The Rust engine writes ordinary Lucene 10.5 segments,
which OpenSearch's own engine can take over at any time
([`operations.md`](operations.md#rolling-back)).

## How much of *your* workload runs native

Every search the plugin sees is counted: `native_queries`, or a fallback under
its reason.

```
GET /_plugins/lucene_rust/stats
```

`scripts/fallback-report.py [URL]` turns the counters of every node into
shares, by reason. Take two readings some time apart under real traffic; the
difference between them is that period's mix. The reasons are listed under
[Falls back](#falls-back-and-why). Performance is measured, not assumed:
`docs/milestones/m5-6-native-read.md` has the full read benchmark (174
request shapes, native against OpenSearch's path on the same shard).

## Search

### Queries

Each ✅ row is exercised by `scripts/verify-opensearch.sh`, native against
OpenSearch's answer on the same shard:

| Query DSL | Native | Notes |
|---|---|---|
| `match` (any `operator`, `minimum_should_match`, `boost`) | ✅ | |
| `multi_match` (`best_fields`, `cross_fields`), `query_string` | ✅ | |
| `term`, `terms` on `keyword` | ✅ | |
| `bool` (`must`, `should`, `filter`, `must_not`, nesting, `minimum_should_match`) | ✅ | a negative `minimum_should_match` falls back |
| `constant_score`, `dis_max`, `match_all` | ✅ | |
| `match_phrase` | ✅ | exact and sloppy; a phrase with a gap (a removed stopword) falls back |
| `prefix`, `wildcard`, `regexp`, `fuzzy` | ✅ | `regexp` with non-default flags (e.g. `case_insensitive`) and `wildcard` with a `\` escape fall back |
| `range` on `long`, `date`, `double` | ✅ | OpenSearch's approximate range (an early-terminating range that `track_total_hits` allows) falls back |
| `range` on `integer`, `float`, `short`, `byte`, `half_float`, `scaled_float` | ↩ | falls back (`points_width`) |
| `exists` | ✅ | on a vector field it falls back |
| any other query | ? | runs native when OpenSearch rewrites it into the Lucene queries the rows above produce (term, boolean, constant-score, boost, disjunction-max, match-all/none, phrase, prefix/wildcard/regexp/fuzzy/terms, 8-byte point range, exists) -- e.g. `simple_query_string`, `match_bool_prefix` -- and falls back otherwise, counted as `query_<Class>`/`clause_<Class>` (`nested`, `geo_*`, `knn`, `script_score`, `function_score`, `span_*`, `intervals`, ...). Not verified per query type: measure with the stats below |

Scoring is BM25. A field with another similarity, or BM25 with other
parameters, falls back (`field_similarity`), as does
`search_type=dfs_query_then_fetch` (`dfs`).

### Request features

| Feature | Native | Notes |
|---|---|---|
| top hits by score, `from`/`size`, `track_total_hits` (any value), `track_scores` | ✅ | |
| `sort` on numeric, `date`, `keyword` fields, `_score`, `_doc`; `missing`; modes `min`, `max`, `avg`, `median` | ✅ | index-sorted shards and nested sort keys included |
| `sort` with `mode: sum` on a top-level field | ↩ | `sort_*` |
| `search_after` | ✅ | |
| `scroll` | ✅ | a sorted scroll page without sort values falls back (`scroll_after`) |
| `post_filter` | ✅ | an approximate `range` post filter falls back |
| `min_score` | ✅ | behind a scroll's later pages or `terminate_after`, falls back |
| `terminate_after` | ✅ | an unscored sort, a constant-score or filter-only query's hits, or with aggregations, falls back |
| `timeout`, `allow_partial_search_results` | ✅ | checked before and after the native call; see limits |
| `collapse`, `rescore`, `profile`, `suggest`, `highlight` | ↩ | `collapse`, `rescore`, `profile`; highlighting and suggesters run in OpenSearch's fetch phase as usual |
| fetch: `_source`, `stored_fields`, `_id`, get / mget | ✅ | the stored-field reads (`index.lucene_rust.fetch.enabled`) |
| fetch: `docvalue_fields`, `fields`, `script_fields`, `inner_hits` | ↩ | OpenSearch's fetch sub-phases, unchanged |

### Aggregations

| Aggregation | Native | Falls back when |
|---|---|---|
| `min`, `max`, `sum`, `avg`, `value_count`, `stats` | ✅ | a `script`, `missing` |
| `cardinality` | ✅ | a `script`, `missing` |
| `terms` on `keyword` | ✅ | numeric `terms`, `include`/`exclude`, an order other than the default (count desc), `min_doc_count: 0` |
| `histogram`, `date_histogram` | ✅ | a time zone with daylight saving |
| `range`, `date_range` | ✅ | |
| `filter`, `filters`, `global` | ✅ | |
| any nesting of the above | ✅ | |
| everything else (`top_hits`, `percentiles`, `significant_terms`, `composite`, `nested`, pipeline aggregations, ...) | ↩ | `aggregations` |

## Indexing: the Rust engine

| Feature | Supported | Notes |
|---|---|---|
| index, update, delete, bulk, refresh, flush, force merge | ✅ | |
| realtime get, versioning, `if_seq_no`/`if_primary_term`, external versions | ✅ | |
| soft deletes, history retention, peer recovery | ✅ | |
| document replication | ✅ | primaries and replicas both on the Rust engine |
| segment replication | ✅ | Rust primary; replicas run OpenSearch's `NRTReplicationEngine` |
| mixed clusters (some nodes with `lucene_rust.engine.node_enabled: false`) | ✅ | shards relocate and recover between the engines both ways |
| field types | ✅ for what the Rust writer accepts | tested as the field types OpenSearch's own REST YAML suites index, which pass with every index on the Rust engine (`scripts/verify-opensearch.sh --engine --yaml`); a type those suites do not cover is not separately verified. Refused per document (400): term vectors, vector fields, `completion`, payloads, custom term frequencies, any other per-field postings or doc-values format |
| index settings | | refused at creation: a codec other than `default`/`lucene_default`, index sorting, context-aware segments, remote-backed storage |
| `index.merge.policy.*`, `index.merge_on_flush` | ✗ | the writer's own merge policy; merges run inside commits, not reported in `_stats` merges |
| compound segments | ✗ | writes non-compound segments only (reads compound ones) |

The engine's behaviour where it differs from OpenSearch's (refresh is a commit,
segment stats, scores on update-heavy indices until a force merge) is in
[`opensearch-engine.md`](opensearch-engine.md#where-it-behaves-differently).

## Falls back, and why

Each is a counter under `fallbacks` in `GET /_plugins/lucene_rust/stats`.

| Reason | The request |
|---|---|
| `disabled` | is on an index with `index.lucene_rust.search.enabled: false` |
| `query_<Class>`, `clause_<Class>` | contains a Lucene query the native side does not run (the class names it) |
| `approximate` | uses OpenSearch's approximate `range` |
| `points_width` | is a `range` on a 4-byte (or narrower) numeric field |
| `regexp_flags`, `wildcard_escape`, `phrase_positions`, `boolean_msm_negative`, `boost_invalid` | uses a variant of a supported query the native side does not take |
| `query_too_deep`, `query_too_large` | nests deeper than 32 or has more than 1,024 query nodes |
| `field_similarity` | scores a field with a non-default similarity |
| `term_states` | carries term statistics the native side cannot reproduce |
| `dfs` | is `dfs_query_then_fetch` |
| `aggregations`, `collectors` | has an aggregation or collector the native side does not run |
| `sort_*`, `sort_nested` | sorts by something the native side does not encode |
| `search_after`, `scroll_after` | pages in a way the native side cannot resume |
| `min_score`, `terminate_after` | combines them with something above |
| `collapse`, `rescore`, `profile` | uses them |
| `time_series_order` | is on a time-series shard that visits segments out of order |
| `timeout`, `cancelled` | had already timed out, or was cancelled, before the native call |
| `collector_spec` | runs where another plugin replaces the top-docs collector |
| `postings_format` | is on an index with a postings field in a format the Rust reader does not decode (e.g. `completion`'s `Completion104`); every default format Lucene 9.0-10.5 wrote (`Lucene90`, `Lucene99`, `Lucene912`, `Lucene101`, `Lucene103`, `Lucene104`) is served natively since M8 T8.5 |
| `intra_segment` | uses intra-segment concurrent search slices |
| `reader_*`, `directory_*` | is on a remote-store or wrapped reader the plugin cannot see through |
| `native_open_failed`, `native_error` | hit a native refusal or error; the node log has the message, and the request re-ran on OpenSearch's path |

## Not supported

- **Other Lucene versions and codecs.** Segments with Lucene's default codec
  of 9.0-10.5 are read natively (M8: `Lucene90`..`Lucene104`), so an index
  upgraded from OpenSearch 2.x is served natively
  (`scripts/verify-opensearch-upgrade.sh`). Not read: Lucene 8 and earlier
  (OpenSearch 3.x cannot open them either), non-default postings formats
  (`completion`: `postings_format`), and segments the k-NN plugin wrote with
  `index.knn: true`, which name the plugin's codec (`KNN9120Codec` from 2.19)
  and fall back as `native_open_failed`. The Rust engine refuses non-default
  codecs.
- **Remote-backed storage** (remote store, searchable snapshots): the Rust
  engine refuses it and native search falls back.
- **Windows and macOS.**
- **Other plugins that register a `QueryPhaseSearcher`** (`neural-search`):
  OpenSearch allows one per node.

Limits of what does run natively -- cancellation, the per-segment query cache,
timeouts -- are in [`opensearch-native-queries.md`](opensearch-native-queries.md#known-limits).
