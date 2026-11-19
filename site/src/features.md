# What it does

The plugin has two independent halves. Each is on by default and each falls back to OpenSearch's
own code for whatever it does not handle. A fallback is never an error: the request is answered as
stock OpenSearch would, and the fallback is counted by reason.

| Half | What it replaces |
|---|---|
| **Native search** | the shard's query phase (queries, sorting, aggregations, `post_filter`, `min_score`, `terminate_after`, scroll) and stored-field reads in fetch and get |
| **Accelerated engine** | the shard's indexing engine: writes, refresh, flush, merges, soft deletes, replication |

## Highlights

- **Search**: `match`, `multi_match`, `query_string`, `term(s)`, `bool`, phrases, prefix, wildcard,
  regexp, fuzzy, ranges on every numeric type, `date` and `ip`, geo queries, `nested`, `has_child`,
  `has_parent`, spans and intervals, `function_score`, `more_like_this`, scripts, every standard
  similarity.
- **Aggregations**: metrics, percentiles, `terms`, histograms, ranges, `filter(s)`, `composite`,
  `top_hits`, `significant_terms`, `cardinality`, pipeline aggregations, `nested`; an opt-in cache
  that makes a repeated dashboard collect only the segments added since.
- **Indexing**: bulk indexing at stock's rate, refresh that does not stop writers, merges on the
  engine's own threads, document and segment replication, peer recovery, mixed clusters.
- **Vectors**: the k-NN API, HNSW with quantization, opt-in IVF and disk HNSW, filtered search
  with a cost-based planner, nested k-NN, multi-vector (`max_sim`), recall targeting.
- **Neural and sparse search**: the neural-search queries and ingest processors, hybrid search,
  semantic fields, Seismic sparse retrieval.
- **ml-commons**: local ONNX and TorchScript models, remote connectors, agents, memory and MCP.
- **Storage**: a demand pager with one memory budget, an object-store reader with an NVMe cache,
  per-file AES-256-GCM encryption with a FIPS 140-3 build.
- **Logs**: a preset of component templates for log indices, columnar doc values, trace-id and
  substring field types.
- **Realtime search** (opt-in): documents searchable right after the acknowledgement, without a
  refresh.

The complete tables — every query, request feature, aggregation and indexing feature, with what
falls back and why — are in the [feature matrix](reference/feature-matrix.md). Which exact shapes
run natively is in [Which searches run natively](reference/accelerated-queries.md).

## What is not supported

- Remote-backed storage and searchable snapshots.
- Windows and macOS.
- Lucene 8 and earlier segments, and other codecs than the default family and zstd.
- Installing next to another plugin that registers a `QueryPhaseSearcher`.

Details and the reasons are at the end of the [feature matrix](reference/feature-matrix.md#not-supported).
