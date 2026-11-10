# HNSW: search and build against Lucene and hnswlib

Lucene-format HNSW (`Lucene99HnswVectorsFormat` and its int7 scalar-quantized
variant) as this port writes and searches it, against Java Lucene 10.5 on the
same index files and against hnswlib on the same data. How to reproduce:
[`benchmarks/hnsw/README.md`](../../benchmarks/hnsw/README.md).

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.1 GHz (AVX-512), a VM. Everything single
  threaded: one query at a time, and builds on one thread.
- Data: GloVe 390 000 x 100, cosine, 10 000 held-out queries with exact top
  100 (`benchmarks/ivf/prepare.py`). 2000 measured queries after 200 (Rust),
  6000 (Java) warm-up ones. Recall@10.
- Graph: M 16, beam 100, one flushed segment. Rust and Java search the same
  files, so recall matches to the query; hnswlib builds its own graph with the
  same M and `ef_construction`.
- `ef` is the candidate count: Rust `knn_top_hits_with(k = 10, candidates =
  ef)`, Java `KnnFloatVectorQuery(field, q, ef)` then top 10, hnswlib
  `set_ef(ef)`.

## Search, mean latency (us)

| ef | recall | Rust before | Rust after | Java Lucene 10.5 | hnswlib (recall) |
|---|---|---|---|---|---|
| 10 | 0.622 | 118-126 | 54-96 | 89 | 45 (0.602) |
| 40 | 0.816 | 275-307 | 134-155 | 242 | 107 (0.809) |
| 100 | 0.893 | 563-619 | 271-307 | 518 | 229 (0.899) |
| 320 | 0.953 | 1529-1580 | 786-818 | 1117 | 674 (0.959) |

int7 (`hnsw-int7`: walk on 7-bit codes, rescore `ef` candidates on floats):

| ef | recall | before | after |
|---|---|---|---|
| 10 | 0.618 | 71-79 | 44-47 |
| 40 | 0.816 | 192-209 | 98-106 |
| 100 | 0.892 | 376-449 | 207-221 |
| 320 | 0.952 | 1043-1108 | 594-639 |

Ranges are two alternating runs of each build on the same files.

## Build, one thread (s)

| | Rust before | Rust after | Java Lucene 10.5 | hnswlib |
|---|---|---|---|---|
| float | 212 | 120 | 330 | 93 |
| int7 | 227 | 119 | | |

## What the time went to

Profiled at ef 100 before the change, cosine scoring was ~70% of a query
and graph decoding ~8%. The kernel itself is not slow (a 100-dim cosine is
~20 ns in cache); a query scores ~1 800 vectors of 400 bytes scattered
through 156 MB, and each one stalled on its own cache misses (~150 ns per
vector). The graph build has the same shape: it is a search per inserted
node.

Two hints fix most of it, with no change to any file or result:

- `VectorScorer::prefetch`: a bulk score starts loading every cache line of
  every candidate before scoring the first, so the misses overlap. Float,
  byte and scalar-quantized scorers implement it, and the int7 float rescore
  uses it. Prefetching only each vector's first two lines, or a few vectors
  ahead of the scoring loop, measured slower.
- `HnswGraphView::prefetch_neighbors`: after popping a candidate the search
  prefetches the next one's neighbour list (off-heap: the `.vex` bytes; on
  heap: the node's level array), which loads while the current neighbours
  score. Worth ~5-10% on top of the first.

Left: hnswlib is still ~1.2x faster at equal recall. It stores each node's
level-0 links next to its vector (one region per hop instead of two) and its
cosine is a dot product over pre-normalized vectors; both are format changes.
During a build, ~30% of the time is the on-heap graph's neighbour arrays (a
chain of three loads per node, plus a copy into scratch); a flat level-0
arena would remove two of the three.
