# Performance

Every number here comes from a script in the repository, run against a real OpenSearch 3.9.0 node
with and without the plugin, with the response compared to stock's. The reports linked under
**Benchmark reports** state the machine, the data and the method; rerun them on your hardware
before relying on a ratio. A ratio above 1 means the plugin is faster.

## Four baselines, and what the gap between them means

The tables below compare up to four things on the same request:

| | What it is |
|---|---|
| **Lucene** | Apache Lucene 10.6 on its own, in a plain JVM |
| **OpenSearch** | stock OpenSearch 3.9.0 |
| **Plugin** | the same node with accelerated-search installed |
| **Core engine** | the plugin's engine as a library, with no JVM and no OpenSearch around it |

The plugin replaces the work inside a shard: the query phase, aggregations, fetch, indexing and merging.
It cannot replace what OpenSearch does around that work, and that part is the same on both sides: parsing
the REST request, the search action, shard and task bookkeeping, the thread-pool hand-off, building the
response, and for indexing the bulk parser, the document mapper and the translog. On a 1M-document index this costs
about 0.4 ms per search at the median with direct search on, 1.5 ms with it off and, for indexing, about 12 CPU-seconds per million documents.
So a request that the engine answers in 0.1 ms still takes about 0.5-1 ms through OpenSearch, and the
plugin's ratio on cheap requests is bounded by that floor.

The **core engine** numbers show what the engine does without that floor. A node built entirely on the
engine, with no OpenSearch request path in front of it, would sit close to them. Two kinds of core number appear below, and both are measured:

- **Core in plugin**: the time the plugin spent inside the engine for a request, taken from the plugin's own counters while the
  node served that request. It includes the call across the foreign-function boundary (under 1% of a search) and nothing of OpenSearch.
- **Core engine on its own**: the engine run from a benchmark program with no JVM at all, against the same index files as
  Lucene.

A full node's own request handling (network, request parsing, response encoding) is not measured and would add to either figure,
so read core numbers as the engine's share of a request, not as a promise for a standalone server.

## What was measured for this page, and what was reused

| Section | Status |
|---|---|
| Search sweep, 195 shapes, OpenSearch / plugin / plugin with direct search off / core in plugin | measured 2026-11-20 on the current main |
| Lucene against core engine, 87 shapes | measured 2026-11-20 |
| Indexing: OpenSearch against plugin (logs, text, nested) | measured 2026-11-20 |
| Indexing: Lucene writer against core engine writer, 1M log documents | measured 2026-11-20 |
| k-NN (`lucene` engine), neural, sparse, semantic, hybrid, MMR, rerank | measured 2026-11-20 with small local models |
| Confluence-style and PDF-style text search, highlight, big5 | reused from the 2026-11-19 run on the same harness; not re-run |
| Log analytics (BGL), columnar doc values | reused, earlier reports |
| HNSW, IVF, disk HNSW against Lucene and hnswlib | reused, earlier reports (Lucene 10.5) |
| Memory and garbage collection | reused, earlier reports |

All fresh runs: one 4-core 15 GB cloud machine, one run per cell for the heavy shapes and the median of 100 requests for
search shapes, node on 3 cores and client on 1, one shard. Spread between repeats of the same run is 5-10%; a ratio
within about 10% of 1.0 is noise. The full tables and the method are in
[Stock Lucene, stock OpenSearch, the plugin and the core engine](reference/benchmarks/stock-comparison.md).

## Summary

| Workload | OpenSearch to plugin | Core engine on its own or in plugin |
|---|---|---|
| Search, 195 request shapes (geometric mean) | **1.76x** faster with defaults (1.27x with direct search off); 114 shapes at 1.5x or better | 2.8x over OpenSearch for the engine's share (measured, 173 shapes) |
| Search CPU per request | 1.83x less | |
| Indexing, logs (warm) | **1.4x** docs/s, 0.64x the CPU | 200,000+ docs/s and 20 CPU-s per million documents against Lucene's 78,000-93,000 and 44-51 |
| Indexing, text and nested (warm) | 1.5x and 1.3x docs/s | |
| k-NN (`lucene` engine HNSW) | 2.1-3.4x | HNSW in the engine 1.4-1.9x Lucene at the same recall ([HNSW report](reference/benchmarks/hnsw.md)) |
| neural, sparse, semantic, hybrid, MMR, rerank | 1.15-2.07x, geometric mean 1.75x | |
| Long-document highlight (PDF-style) | 19x | |

## Searches: OpenSearch, plugin and core engine

A selection from the 195 shapes on a 1M-document index, one segment, milliseconds at the median over REST:

| Request | OpenSearch (ms) | Plugin (ms) | Plugin, direct off (ms) | Core in plugin (ms) | Plugin vs OpenSearch | Core vs OpenSearch |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent | 3.42 | 1.78 | 5.93 | 0.96 | 1.92x | 3.5x |
| term on a keyword | 1.99 | 0.60 | 1.78 | 0.26 | 3.29x | 7.6x |
| `match` OR, two frequent terms | 10.53 | 3.64 | 6.28 | 3.72 | 2.90x | 2.8x |
| `match` AND, two frequent terms | 8.15 | 4.51 | 4.78 | 3.02 | 1.81x | 2.7x |
| `match_phrase`, two frequent terms | 63.95 | 45.77 | 49.29 | 48.03 | 1.40x | 1.3x |
| bool: `match` + range filter | 3.09 | 2.24 | 3.12 | 1.36 | 1.38x | 2.3x |
| `multi_match` body and title | 4.50 | 2.49 | 3.44 | 1.84 | 1.80x | 2.4x |
| `match` sorted by a numeric field | 6.07 | 1.04 | 3.17 | 0.89 | 5.83x | 6.8x |
| sort by date desc, 100 hits | 10.16 | 1.92 | 3.36 | 1.36 | 5.28x | 7.5x |
| fetch 100 `_source` | 8.40 | 2.46 | 3.46 | 1.62 | 3.42x | 5.2x |
| `terms` on a filtered keyword | 10.00 | 3.87 | 5.62 | 3.70 | 2.59x | 2.7x |
| `stats` on a double | 12.19 | 4.74 | 7.72 | 6.49 | 2.57x | 1.9x |
| `cardinality` on a keyword | 9.32 | 1.68 | 1.69 | 0.36 | 5.56x | 25.8x |
| `terms` > `stats` | 59.85 | 14.60 | 14.95 | 13.35 | 4.10x | 4.5x |
| `date_histogram` by month | 1.72 | 1.68 | 1.70 | 0.43 | 1.02x | 4.0x |
| `date_histogram` by day | 7.64 | 3.45 | 3.50 | 1.74 | 2.21x | 4.4x |
| `percentiles` | 117.06 | 100.70 | 107.58 | 109.98 | 1.16x | 1.1x |
| `post_filter` + `terms` | 33.77 | 6.04 | 6.90 | 5.42 | 5.59x | 6.2x |
| `composite` two keywords | 3.54 | 4.50 | 4.32 | 1.98 | 0.79x | 1.8x |
| unified highlight, 100 hits | 10.45 | 7.24 | 11.24 | 3.49 | 1.44x | 3.0x |

Single cheap shapes are noisy (a fresh node is still compiling Java for its first few shapes; a repeat can move a 1 ms row by 30%), so read the geometric means, not one row.

The last column is the engine's share. Where it is far above the plugin's ratio (`cardinality`, `date_histogram`, `composite`,
the cheap term shapes) the request is dominated by what OpenSearch does around the engine, including, for bucket aggregations, turning the
engine's result into OpenSearch's aggregation objects. Where the two are close (`match_phrase`, `percentiles`, `stats`) the engine's own work is
nearly the whole request.

### Lucene against the core engine

On a 1M-document corpus, in process, top 50, one segment, 87 query shapes (the same Lucene 10.6 hits and top documents are required
before a ratio counts). Geometric mean of queries per second, core over Lucene: **2.69x** (median 1.81x). Term queries 2.96x, `AND` 1.93x, `OR` 1.74x, phrase 1.63x,
prefix and wildcard 5.9x, regexp 12.5x, sorted queries 5.6x. Lucene has no aggregation DSL to compare with; its facet counting is covered in
the repository's own microbenchmarks.

Taking the same shape through the three layers on this machine (term on a frequent word): Lucene 1.15 ms, core engine
0.31 ms, core in plugin 0.59 ms, plugin 0.94 ms over REST, OpenSearch 3.42 ms over REST. The two corpora hold different documents, so compare ratios
between the columns of one corpus, not milliseconds across them.

### The direct search path

A cheap search spends most of its time in OpenSearch's request path. `search.accelerated.direct.enabled` (on by default) answers eligible searches
(single-shard accelerated indices; plain `query`, `size`, `from`, `track_total_hits`, `_source` and field sorts, see
[Operating it](reference/operations.md#direct-search)) in the REST layer instead of OpenSearch's search action. It is
what lifts cheap shapes from level with OpenSearch to 1.8x or more:

| | All 195 shapes | Native shapes under 4.5 ms of stock time | Native shapes of 4.5 ms or more |
|---|---:|---:|---:|
| Direct search off | 1.27x | 0.98x | 1.58x |
| Direct search on (default) | 1.76x | 1.77x | 1.93x |

## Log analytics

On real logs (Loghub BGL, 4.6M documents, one segment) with columnar doc values enabled,
aggregations over numeric fields run several times faster than stock, and the answers match stock's
(exact, or within the sketch's error for percentiles and `cardinality`):

| Aggregation | Stock (ms) | Plugin (ms) | Ratio |
|---|---:|---:|---:|
| `stats(bytes)` | 36 | 3.9 | 9x |
| `histogram` | 85 | 19 | 4.4x |
| `range(bytes)` > `sum` | 57 | 16 | 3.6x |
| `date_histogram(1h)` > `sum` | 147 | 35 | 4.2x |

Honest limits, stated in the same report: keyword group-bys (`terms` on a low-cardinality keyword)
run at roughly half of stock's speed today, and columnar doc values make some scans of unclustered
numeric data slower, which is why the setting is `auto` and decided per index from a sample.

## Indexing

1M documents shaped like an OpenSearch log index, four indexing threads, writer only (no OpenSearch), two runs each:

| | docs/s | CPU seconds per million documents |
|---|---:|---:|
| Lucene 10.6 `IndexWriter` | 78,000-93,000 | 44-51 |
| Core engine writer | 202,000-230,000 | 20-21 |

Index size is identical (227 MB). A force merge to one segment takes 2.3 s and 2.7 CPU-s on the core engine, 4.8 s and 12.0 CPU-s on Lucene.

Through OpenSearch, bulk indexing over REST (5,000 documents per bulk, four clients, one shard, warm node, a new index each pass):

| Documents | OpenSearch docs/s | Plugin docs/s | Ratio | CPU s per million, OpenSearch | CPU s per million, plugin |
|---|---:|---:|---:|---:|---:|
| logs, 10 fields | 68,900 | 97,300 | 1.41x | 41.7 | 26.8 |
| text, 120 words | 30,800 | 47,300 | 1.53x | 92.7 | 57.9 |
| nested, 5 children | 27,000 | 34,800 | 1.29x | 105.3 | 80.7 |

Indexing through OpenSearch is limited by OpenSearch's own bulk path: the REST parser, the document mapper and the translog cost about 12 CPU-seconds
per million log documents on both sides before any engine runs. That floor is why the plugin gains 1.3-1.5x end to end while the engine alone gains 2.2-2.9x
over Lucene, and why the end-to-end gain does not grow past about 1.5x however fast the engine becomes. A run on another day
measured warm logs at 1.3x, so quote 1.3-1.4x. On a cold node both sides are slower while the JVM compiles; see the
[full report](reference/benchmarks/stock-comparison.md) for the first-pass numbers.

## Text search on real documents

Reused from the 2026-11-19 run (Wikipedia reshaped into Confluence-style pages and PDF-style long documents, four shards, node on three cores):

| Workload | Search | Indexing |
|---|---|---|
| Confluence-style pages with permission and space filters | 1.0-1.2x; highlight 1.6x; at 8 clients p99 80 to 29 ms | 1.21x |
| PDF-style long documents (125 KB average) | 1.2-1.4x; highlight 19x (1,276 to 66 ms); stock spends 4.7 CPU-seconds per highlighted request, the plugin 0.12 | 1.26x |
| big5 (1.5M documents, 49 operations) | sorts and ranges 1.2-9.6x; most others near 1.0x | equal |

## Vectors, neural search, hybrid search and rerank

HNSW, quantized HNSW (RaBitQ, OSQ), IVF and disk HNSW are benchmarked on public vector datasets with
recall and latency per memory budget. Disk HNSW reads 4-7x fewer pages than the in-memory layout when
the vectors do not fit in memory and is 2-4x slower when they do, which is why it is opt-in. See the
[HNSW](reference/benchmarks/hnsw.md), [IVF](reference/benchmarks/ivf.md) and
[disk HNSW](reference/benchmarks/disk-hnsw.md) reports.

Measured on 2026-11-20 against stock's own k-NN, neural-search and ml-commons, with small ONNX models served locally (so model inference is a minor and equal
cost; with a large model it dominates both sides and the ratio approaches 1):

| Shape | OpenSearch (ms) | Plugin (ms) | Ratio |
|---|---:|---:|---:|
| `knn` k=10, precomputed vector | 2.86 | 0.90 | 3.20x |
| `knn` k=100, size 100 | 19.52 | 9.89 | 1.97x |
| `neural` | 6.24 | 5.44 | 1.15x |
| `neural_sparse` by analyzer | 3.52 | 1.70 | 2.07x |
| semantic field | 3.50 | 2.85 | 1.23x |
| hybrid `match` + `knn` (min_max) | 6.57 | 3.72 | 1.77x |
| hybrid `match` + `neural` (min_max) | 6.37 | 4.88 | 1.30x |
| MMR | 10.32 | 6.08 | 1.70x |
| rerank, cross-encoder, top 10 | 20.75 | 11.82 | 1.75x |

`lucene`-engine HNSW over 200,000 128-dimension vectors: k=10 in 1.66 ms against 4.23 ms, with a keyword filter 1.78 against 6.03 ms.

## Weak spots

Where the plugin is level with or slower than OpenSearch, measured on the runs above:

- **Percentile-family aggregations** (`percentiles`, `percentile_ranks`, median absolute deviation) run at about 1.1x: the cost is building the digest itself, which
  the engine does at about the speed of the Java code.
- **Cheap bucket aggregations over REST** (`date_histogram` by month 0.97x, `composite` on two keywords 0.83x, `range` 0.89x): the engine answers in 0.2-1.4 ms but
  OpenSearch's conversion of the result into its own aggregation objects costs as much as the whole stock request.
- **`date_histogram` with an hourly fixed interval and a time zone, and `date_histogram_hourly_with_filter` in big5**: 1.03x in the sweep and 0.63x in big5 (reused); the hourly bucketing with a filter is not
  faster natively yet.
- **Plain highlighter** 0.63x (the unified highlighter is 1.4-19x); **`has_child`, `has_parent` and `significant_terms`** 0.6-0.8x.
- **Scripts**: sorting by a Painless script 0.61x; script queries run an interpreter per document on both sides.
- **Aggregations handed back to OpenSearch**: 20 of the 195 shapes (`weighted_avg`, `matrix_stats`, `geo_bounds`, `auto_date_histogram`, `multi_terms`, `rare_terms`, the geo grids,
  samplers and parent/child aggregations) run on OpenSearch's own implementation, so they are level with stock by design.
- **Semantic field and `neural` queries** are 1.15-1.23x with small models; with a real model the shared inference time dominates and the ratio falls toward 1.0.
- **Indexing** is capped near 1.3-1.5x end to end by OpenSearch's own bulk path, and is equal to stock on big5 (reused).
- **Standalone engine**: queries with a `MUST_NOT` clause on a very frequent term (for example `+t2s -t0`) and a few large union phrase mixes are slower in the engine than in Lucene (0.03-0.5x).

## Memory

The plugin allocates with mimalloc and bounds aggregation state, fetch buffers, the page cache and
indexing buffers explicitly. In the 1M-document run above the node's resident set after load and
search was within a few hundred MiB of stock's (2.7 GiB against 2.5 GiB).
