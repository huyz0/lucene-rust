# Accelerated Search for OpenSearch

**accelerated-search** is an OpenSearch 3.9.0 plugin that runs the hot paths of a
shard — searching, aggregating, reading stored fields, and indexing and merging — in native
code instead of on the JVM. Anything it does not handle runs on OpenSearch exactly as before, so
installing it never makes a request fail that worked without it.

It is a port of Apache Lucene (pinned at 10.6.0, serving OpenSearch's Lucene 10.5.1 formats), loaded
by the node through Java's Foreign Function & Memory API. The index files it reads and writes are
ordinary Lucene segments: remove the plugin and OpenSearch opens the same data.

## Why use it

- **Faster searches and aggregations.** On the same shard, the native query phase is faster than
  stock on most request shapes, and log-style dashboards that group, bucket and summarise numbers
  run several times faster. See [Performance](performance.md).
- **Smaller, leaner indices.** Optional columnar doc values, leaner sequence numbers, and
  zstd-compressed stored fields cut disk size; see [Settings](settings.md).
- **Bounded memory.** One page cache and explicit budgets for aggregations, fetch and indexing
  buffers, instead of unbounded heap and kernel-cache growth.
- **Drop-in for the plugins around it.** It serves the k-NN, neural-search, sparse-search and
  ml-commons APIs under their stock names, and adds opt-in disk-friendly vector indexes.
- **Reversible.** Roll back by switching one node setting; no reindex.

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
