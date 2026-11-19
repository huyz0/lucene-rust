# Accelerated Search for OpenSearch

**accelerated-search** is an OpenSearch 3.9.0 plugin that runs the hot paths of a shard — searching,
aggregating, reading stored fields, and indexing and merging — in native code instead of on the JVM.
Anything it does not handle runs on OpenSearch exactly as before, so installing it never makes a
request fail that worked without it.

It is a port of Apache Lucene (pinned at 10.6.0, serving OpenSearch's Lucene 10.5.1 formats), loaded
by the node through Java's Foreign Function & Memory API. The index files it reads and writes are
ordinary Lucene segments: remove the plugin and OpenSearch opens the same data.

## What is unique

Most of these are not in stock OpenSearch at all; the rest are faster or smaller versions of what
stock has. Every number comes from a report in the repository (linked), measured against stock or
against the named baseline; the machines and methods are in the reports.

| Capability | What you get | Measured | On by default |
|---|---|---|---|
| **Realtime search** | A document is searchable when its write is acknowledged, without a refresh, with no extra indexing CPU. Works for k-NN fields too. | Write-to-visible p50 of **4.5 ms** instead of 535-558 ms with a 1 s refresh, same recall, same indexing CPU ([k-NN end to end](reference/realtime-search.md)) | opt-in |
| **IVF vector index** (our own) | An inverted-file index with RaBitQ and OSQ quantizers (1, 2 and 4 bit) and a fast-scan kernel, plus scalar int4 and int7, an automatic choice of quantizer per segment, f16 rerank vectors, tenant-aware clusters and nested k-NN. | At HNSW + BBQ's best recall (0.87-0.88) it is **3-10x faster**, builds 1.2-6x faster and pins a third to nine tenths of the memory; on GloVe-100 it answers at 0.94 recall in 234 µs against 607 µs at 0.92 for Faiss IVF-RaBitQ ([IVF report](reference/benchmarks/ivf.md)) | opt-in per field |
| **Disk-based vector search** | Disk HNSW stores each node's vector and neighbours in one record so a graph walk makes few, batched reads; IVF reads probed clusters in one prefetch. Both run from a page cache with a fixed memory budget, on local disk or an object store with an NVMe tier. | When memory is below the graph plus codes, disk HNSW is **3-24x faster** than stock HNSW + BBQ; IVF needs 2 round trips per query at any budget (1.1 ms against 43 ms for a per-cluster read at 500 µs a round trip) ([disk HNSW report](reference/benchmarks/disk-hnsw.md)). When everything fits in memory the in-memory layout is faster, which is why it is opt-in | opt-in per field |
| **Columnar doc values** | Numeric fields stored in blocks that aggregations fold without per-document work, chosen per index from a sample of its data. | On a 4.6M-document log index: `stats` **9x**, `histogram` 4.4x, `range` + `sum` 3.6x, `date_histogram` 4.2x faster than stock, with identical answers; a taxi-trip index shrinks from 713 to 443 MiB ([columnar report](performance.md#log-analytics)) | `auto` |
| **Native aggregations and sort** | Metrics, percentiles, `terms`, histograms, ranges, `composite`, `top_hits`, `significant_terms`, nested and pipeline aggregations run in the engine, with bounded memory per request. | 19 of 20 shapes faster than stock on one corpus, median 1.23x, up to 2.14x ([1M-document report](reference/benchmarks/opensearch-1m.md)) | on |
| **Aggregation segment cache** | What an aggregation collected from each segment is kept, so a dashboard asked again after a refresh collects only the new segments. | A repeated dashboard runs **1.2-13x faster than stock** and 1.6-6.7x faster than the plugin without the cache ([report](reference/benchmarks/agg-segment-cache.md)) | opt-in |
| **Filtered k-NN planner** | A cost model decides per segment between walking the graph and scanning the filtered set, and a filter's share of the data changes the plan. | A 1% filter answers in 300 µs against Lucene's 1,158 µs at recall 1.0; a 10% random filter reaches recall 1.0 where Lucene returns 0.956 ([HNSW report](reference/benchmarks/hnsw.md)) | on |
| **Recall-targeted k-NN** | Ask for a recall instead of tuning search depth: sampled live queries, checked against an exact scan, set the effort. | For IVF, a 0.90 target meets it at 346 µs where the fixed default spends 1,155 µs ([report](reference/benchmarks/recall-target.md)) | opt-in |
| **Seismic sparse retrieval** | Approximate retrieval for learned sparse vectors (`neural_sparse` with `method_parameters`) with block-summary pruning. | About 0.2 ms a query on a 200k-document SPLADE-shaped segment ([report](reference/benchmarks/seismic.md)) | opt-in |
| **Faster, leaner indexing** | A concurrent writer with a buffer per thread, refresh that does not stop writers, merges on its own threads, zstd stored fields, leaner sequence numbers. | **226k docs/s** against 125k for the stock writer on log-shaped documents, 36 against 63 CPU seconds, same index size ([write path report](reference/benchmarks/logs-write.md)) | on |
| **Bounded memory** | One page cache for index files, caps on aggregation state, fetch buffers, the aggregation cache and indexing buffers, and a native allocator tuned for the node. | Node resident memory 2.7 GiB after load and search against 2.5 GiB for stock in the 1M-document run ([report](reference/benchmarks/opensearch-1m.md)) | on |
| **Encrypted storage** | Per-file AES-256-GCM with keys held in the JVM, a FIPS 140-3 build, and decryption through the page cache. | Described in [Storage and encryption](reference/pluggable-storage.md) | opt-in |
| **Security plugin compatibility** | Native stored-field reads run beneath the security plugin's field- and document-level security, which still filters and masks what is returned. | [Compatibility](reference/compatibility.md) | on |

## Why use it

- **Faster searches and aggregations**, most of all dashboards that bucket and summarise numbers.
- **Smaller indices** from columnar doc values, zstd stored fields and leaner sequence numbers.
- **Vector search that fits your memory**: HNSW in memory when it fits, IVF or disk HNSW when it does not.
- **Drop-in for the plugins around it**: the k-NN, neural-search, sparse-search and ml-commons APIs
  under their stock names.
- **Reversible**: roll back by switching one node setting; no reindex.

## What it is not

It is not a different search engine: query DSL, REST APIs, mappings and index formats are
OpenSearch's. Where behaviour differs from stock, the difference is listed in
[Compatibility](reference/compatibility.md) with its reason and measured cost.

## Where to go next

| If you want to… | Read |
|---|---|
| install it and see it work | [Get started](get-started.md) |
| know what it can and cannot run natively | [What it does](features.md) |
| check what stays identical to stock | [Compatibility](reference/compatibility.md) |
| tune it | [Settings](settings.md) and [Operating it](reference/operations.md) |
| see the numbers | [Performance](performance.md) |
| understand how it is built | [Architecture](architecture.md) |
| build or change it | [Build from source](building.md) and [Contributing](contributing.md) |

Source code and issues: <https://github.com/huyz0/lucene-rust>. Licensed under Apache-2.0.
