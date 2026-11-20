# Benchmark reports

Each report states the machine, the data and the method, and names the script that reproduces it.
Ratios above 1 mean the plugin is faster than the baseline named in the report.

## Against stock OpenSearch

| Report | What it measures |
|---|---|
| [Against stock OpenSearch, 1M documents](reference/benchmarks/opensearch-1m.md) | 20 request shapes, indexing, merging and memory on one corpus, with and without the plugin |
| [Stock Lucene, OpenSearch, plugin and core engine](reference/benchmarks/stock-comparison.md) | the four baselines side by side: 195 request shapes, 87 shapes against Lucene, indexing, k-NN, neural, hybrid and rerank |
| [Environment](reference/benchmarks/environment.md) | the machines and settings the reports were produced on |
| [Write path on log documents](reference/benchmarks/logs-write.md) | indexing throughput and CPU on log-shaped documents against the stock writer |
| [Aggregation sweep](reference/benchmarks/agg-sweep.md) | 22 aggregation, sort and fetch request shapes over REST |
| [Garbage collection](reference/benchmarks/gc-pressure.md) | heap allocation, GC pauses and tail latency against stock at a 1 GB and a 4 GB heap |
| [Aggregation segment cache](reference/benchmarks/agg-segment-cache.md) | dashboards repeated across refreshes: stock, the plugin, and the plugin with the cache |
| [Lean sequence numbers](reference/benchmarks/lean-seq-no.md) | index size saved by the opt-in leaner sequence-number storage on a logs index |
| [Settings defaults review](reference/benchmarks/settings-defaults.md) | the evidence behind each default of the plugin's own settings |

## Text search

| Report | What it measures |
|---|---|
| [Filtered sort](reference/benchmarks/filtered-sort.md) | sorted, filtered searches on merged segments against Lucene |
| [Newest-N queries](reference/benchmarks/newest-n.md) | top N by a time field |
| [Text search with AVX-512](reference/benchmarks/fts-avx512.md) | the effect of the AVX-512 kernels on full-text queries |
| [Partitioned text search](reference/benchmarks/fts-partition.md) | full-text search in a shard shared by many tenants |
| [Partition write cost](reference/benchmarks/partition-cost.md) | what sorting a shard by tenant costs at indexing, refresh and merge |
| [Substring field](reference/benchmarks/substring-field.md) | the substring doc-values field: size and hit parity against stock |

## Vectors and sparse search

| Report | What it measures |
|---|---|
| [HNSW](reference/benchmarks/hnsw.md) | search and build against Lucene and hnswlib, including the filtered-search planner |
| [Quantized HNSW](reference/benchmarks/hnsw-quantized.md) | HNSW walked on RaBitQ and Extended RaBitQ codes |
| [IVF](reference/benchmarks/ivf.md) | recall, latency, build time and memory of IVF with its quantizers, against HNSW and Faiss |
| [Disk HNSW](reference/benchmarks/disk-hnsw.md) | recall, latency and I/O of the disk layouts under a memory budget |
| [HNSW refresh time](reference/benchmarks/hnsw-refresh.md) | how building the graph at add time shortens refresh |
| [Recall targeting](reference/benchmarks/recall-target.md) | tuning search depth to a recall target from sampled queries |
| [Grouped k-NN](reference/benchmarks/knn-grouped.md) | latency and results of `group_by` and `group_size` k-NN searches |
| [Seismic sparse search](reference/benchmarks/seismic.md) | single-thread latency of sparse approximate search |
