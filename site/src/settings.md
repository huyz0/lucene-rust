# Settings

The settings people reach for first. Everything defaults to a safe, stock-compatible value; the
opt-in rows are features that change a trade-off (and say so). The complete lists, with scope and
exact semantics, are in [Operating it](reference/operations.md#configure) and in the settings tables of
[Compatibility](reference/compatibility.md#3-index-and-node-settings). A repository check fails when
a setting the plugin adds is not named in this documentation.

## Switches

| Setting | Scope | Default | Meaning |
|---|---|---|---|
| `index.search.accelerated` | index, dynamic | `true` | serve this index's eligible searches natively |
| `index.fetch.accelerated` | index, dynamic | `true` | read stored fields (`_source`, `_id`, `stored_fields`) natively |
| `index.engine.accelerated` | index, at creation | node default | write this index with the accelerated engine |
| `index.engine.accelerated_source_parsing` | index, at creation | `true` | read `_source` of mappings with keyword, text, numbers, boolean, date, geo_point, nested, objects, multi-fields and `copy_to` in the library: 1.14x stock's documents per second on a logs bulk at 0.83x its CPU per document, 1.19x on text and 1.14x on nested. Other documents go to the document mapper; `false` turns it off |
| `indices.engine.accelerated.default` | node | `true` | the default of the setting above |
| `indices.engine.accelerated.node_enabled` | node | `true` | `false` rolls the node back to OpenSearch's engine |
| `search.accelerated.direct.enabled` | node, at start | `true` | answer plain searches on a single-shard accelerated index from the REST layer (2-3x on cheap shapes; skips tasks and search stats, never on with the security plugin) |

## SQL

Read-only SQL at `/_plugins/accelerated/sql` (GET or POST, `{"query": "..."}`, `format=jdbc|json`), next to and not instead of the stock `/_plugins/_sql`. One table per statement, an index with one shard.

| Setting | Scope | Default | Meaning |
|---|---|---|---|
| `plugins.accelerated.sql.enabled` | node | `true` | serve the endpoint |
| `plugins.accelerated.sql.direct` | node, at start | `true` | run statements from the REST layer on a started local shard (skips tasks and the security plugin's checks, so never on with the security plugin; there statements go through a security-checked transport action) |
| `plugins.accelerated.sql.max_rows` | node | 10000 | the most rows a statement may return; add a `LIMIT` |
| `plugins.accelerated.sql.memory_limit` | node | 512 MB | bytes the fallback engine's operators may hold |

## Memory

| Setting | Scope | Default | Meaning |
|---|---|---|---|
| `indices.store.page_cache.size` | node | 1 GB | cap of the page cache (filled by the `page_cache` and encrypted stores) |
| `indices.aggregation_segment_cache.size` | node | 256 MB | cap of the aggregation segment cache |
| `breaker.indexing_buffers.limit` | node | derived | circuit breaker for the writer's buffered documents |

## Opt-in features

| Setting | Default | What you get |
|---|---|---|
| `index.search.aggregation_segment_cache.enabled` | `false` | a repeated dashboard collects only the segments added since |
| `index.search.realtime.enabled` | `false` | documents searchable at their acknowledgement; `max_staleness` bounds how old a view may be |
| `index.codec.adaptive_doc_values` | `auto` | columnar numeric doc values: smaller indices, much faster group-bys on log data |
| `index.codec.stored_fields.recompress_on_merge` | `off` | `on` or `non_force`: 9-28% smaller stored fields for about 14% of logs indexing throughput, same LZ4 format |
| `index.merge.reorder.*` | disabled | document reordering at merge for smaller postings; forced merges ~3x slower |
| `index.hot_tier.max_memory`, `min_update_rate`, node `indices.hot_tier.max_memory` (`2%` of heap) | `0` (off), `1.0` (`8mb` turns it on) | opt-in: rows updated faster than the rate keep only their latest version in a bounded buffer; dropped updates leave a no-op with their sequence number |
| `index.segment_partition.field` | unset | segments sorted by a tenant or similar field, so filtered queries read only their share |
| `index.knn.target_recall` | `0` (off) | the vector search adapts its effort to a recall target |
| `script.js.enabled` | `false` | `"lang": "js"` scripts |

Disk-friendly vector methods (IVF, disk HNSW) are chosen per field in the mapping and are never
selected silently; see the [k-NN notes](reference/compatibility.md#3-index-and-node-settings).

## Observing it

```
GET /_plugins/accelerated/stats
```

reports how many searches ran natively and, per reason, how many did not. Counters for the page
cache, the aggregation segment cache and the memory breakers are in the same response.
