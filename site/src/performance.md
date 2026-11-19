# Performance

Every number here comes from a script in the repository, run against a real OpenSearch 3.9.0 node
with and without the plugin, with the response compared to stock's. The reports linked under
**Benchmark reports** state the machine, the data and the method; rerun them on your hardware
before relying on a ratio. A ratio above 1 means the plugin is faster.

## Searches on a mixed corpus (1M documents, one segment)

A synthetic corpus with Zipf-distributed terms and keywords; REST round trip, median of 50 requests,
cache off. Native search was faster on 19 of 20 shapes, with a median of 1.23x.

| Request | Stock (ms) | Plugin (ms) | Ratio |
|---|---:|---:|---:|
| `match` AND, two frequent terms | 9.1 | 5.3 | 1.71x |
| `match_phrase`, two frequent terms | 73.6 | 42.4 | 1.74x |
| `match` sorted by a numeric field | 6.4 | 3.0 | 2.14x |
| `terms` aggregation on a filtered keyword | 10.6 | 5.6 | 1.91x |
| `stats` on a double | 8.5 | 6.0 | 1.41x |
| term on a keyword | 2.3 | 2.3 | 1.02x |
| `date_histogram` by month | 2.7 | 2.9 | 0.94x |

Source: [Against stock OpenSearch, 1M documents](reference/benchmarks/opensearch-1m.md). The REST
ratios are modest because a 2-5 ms request is mostly OpenSearch's own request handling; the same
shapes measure 1.1-3.3x in process.

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

2M documents shaped like an OpenSearch log index, four indexing threads:

| | docs/s | CPU seconds |
|---|---:|---:|
| stock Lucene writer | 124,700 | 63.2 |
| accelerated writer | 226,300 | 36.1 |

Index size is identical for both. Through OpenSearch end to end, bulk indexing runs at stock's rate
and force merges are faster; see the
[write path report](reference/benchmarks/logs-write.md).

## Vectors

HNSW, quantized HNSW (RaBitQ, OSQ), IVF and disk HNSW are benchmarked on public vector datasets with
recall and latency per memory budget. Disk HNSW reads 4-7x fewer pages than the in-memory layout when
the vectors do not fit in memory and is 2-4x slower when they do, which is why it is opt-in. See the
[HNSW](reference/benchmarks/hnsw.md), [IVF](reference/benchmarks/ivf.md) and
[disk HNSW](reference/benchmarks/disk-hnsw.md) reports.

## Memory

The plugin allocates with mimalloc and bounds aggregation state, fetch buffers, the page cache and
indexing buffers explicitly. In the 1M-document run above the node's resident set after load and
search was within a few hundred MiB of stock's (2.7 GiB against 2.5 GiB).
