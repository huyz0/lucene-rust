# Seismic query latency

Single-thread latency of a Seismic (`neural_sparse` with `method_parameters`)
query on one clustered segment, measured with `lucene-sparse`'s `seismic_bench`
example:

```
cargo run --release -p lucene-sparse --example seismic_bench -- <docs> <queries> <dir>
```

Each document has 64 distinct tokens drawn from a Zipf-like law over a
30 000-token vocabulary (SPLADE-shaped). The index is force-merged to one
segment, uses default `n_postings`, `cluster_ratio` 0.1 and
`summary_prune_ratio` 0.4. Queries have 16 tokens, with `top_n` 10, `k` 10 and
`heap_factor` 1.0. The digest covers every hit and score, so two builds can
be checked for identical results.

## In-place cluster cursor and forward-index prefetch (2026-11-11)

**Cause.** The query used to decode every cluster of each query token into
owned vectors. About 70% of those clusters are then skipped on their summary
score. Each visited document's forward vector costs two dependent cache misses.

**Change.** The query now reads clusters in place with
`ClusteredPostings::cursor`. It prefetches each document's vector a few
documents before scoring it.

Runs are on a 4-vCPU VM, alternating A/B, with identical digests.

| segment | main mean us | this change mean us |
|---|---:|---:|
| 200k docs | 337 / 331 / 362 / 346 / 342 | 204 / 223 / 214 / 243 / 231 |
| 10k docs | 141 / 141 / 128 | 141 / 144 / 138 |

The cursor alone gives 313 us on the 200k segment. Prefetch distances of 2
and 8 documents measured no better than 4. The 10k segment fits in cache and
shows no change.
