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

fastText 989 999 x 300, L2 (`ft300`, `prepare.py ... l2`), same method, one
run each (hnswlib built its own graph, which reaches a little more recall per
`ef`):

| ef | recall | Rust before | Rust after | hnswlib (recall) |
|---|---|---|---|---|
| 10 | 0.614 | 143 | 102-106 | 135 (0.608) |
| 40 | 0.793 | 320 | 223-227 | 303-322 (0.798) |
| 120 | 0.869 | 786 | 533-569 | 692-734 (0.883) |
| 200 | 0.892 | 1213 | 831-833 | 1089-1152 (0.908) |
| 400 | 0.914 | 2167 | 1360-1453 | 2049 (0.931) |

At matched recall the two are level on fastText; on GloVe hnswlib is still
~1.2x ahead (below).

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

## The remaining gap to hnswlib

hnswlib's search loop (`searchBaseLayerST`) is the same algorithm. Counting
its distance computations (an instrumented C++ build, `-O3 -march=native`,
prefetch on) against ours on GloVe at `ef` 100:

| | recall | distances / query | us / query | ns / distance |
|---|---|---|---|---|
| this port | 0.893 | 1771 | ~300 | ~170 |
| hnswlib | 0.899 | 1640 | 229-252 | ~150 |

So ~8% is the graph (hnswlib's reaches a little more recall with fewer
distances) and ~10-15% is the cost per distance. Profiled after the change,
a query is: waiting on vector loads ~39%, distance arithmetic ~20% (including
Lucene's per-candidate `f64` sqrt and divide for cosine), the walk and its
heaps ~21%, decoding `.vex` neighbour lists ~18% (offset lookup, group
varints). What hnswlib does differently, and whether it fits Lucene's files:

- One record per node (level-0 links, vector, label adjacent, links as raw
  `u32`): one region per hop, no offset lookup or varint decode. A different
  file format.
- `cosine` is a dot product over vectors normalized at insert; Lucene's
  COSINE keeps the raw vectors and sums three terms per candidate.
  Computing the query's own norm once (done, bit-identical) left two sums
  and measured within noise: the arithmetic is not the bottleneck.
- A `u16` visited-tag array that is never cleared; ours clears only the
  words it touched, which costs little.
- AVX-512 kernels: written bit-identical to the AVX2 ones (each lane holds
  the same partial sum), and measured no change on either set; in-cache
  kernel timings on this VM moved 0-20% at 768-1024 dims within run-to-run
  noise. Not adopted here.

During a build, ~30% of the time is the on-heap graph's neighbour arrays (a
chain of three loads per node, plus a copy into scratch); a flat level-0
arena would remove two of the three.

## Filtered searches

A filtered search scores the filter's documents exactly, without walking
the graph, when they are at most one in eight of the segment's vectors
(`HNSW_EXACT_SHARE` in `crates/lucene-search/src/vector_query.rs`), up to
2^18 of them, the share of the largest segment measured. Lucene
does this only below `perLeafTopK` passing documents, or after the walk
visited as many nodes as pass. In an index sorted by tenant a tenant's
vectors are one contiguous region of the graph, often far from the entry
point, and the walk spends most of its visits on other tenants' nodes.

`knn_tenants --scale 4 --format hnsw` (2M GloVe-100 vectors with jitter,
tenants of 375k, 100k, 10k and 1k, M 16, beam 100, 50 queries per tenant),
recall@10 / mean ms. `any` tenants are queried from anywhere, `local` ones
from inside their region:

| tenant (share of a segment) | 1 segment, before | 1 segment, after | 8 segments, before | 8 segments, after |
|---|---|---|---|---|
| random375000-any (19%) | 0.918 / 6.7 | unchanged (walks) | 0.964 / 16.2 | unchanged (walks) |
| region375000-local (19%) | 0.944 / 1.4 | unchanged (walks) | 0.982 / 2.5 | unchanged (walks) |
| random100000-any (5%) | 0.938 / 10.9 | 1.000 / 4.9 | 1.000 / 21.3 | 1.000 / 5.9 |
| region100000-any (5%) | 0.874 / 8.1 | 1.000 / 5.2 | 0.964 / 14.9 | 1.000 / 5.6 |
| region100000-local (5%) | 0.940 / 1.5 | 1.000 / 4.2 | 0.988 / 5.5 | 1.000 / 5.6 |
| random10000-any (0.5%) | 1.000 / 2.2 | 1.000 / 0.51 | 1.000 / 2.3 | 1.000 / 0.70 |
| region10000-local (0.5%) | 0.992 / 0.75 | 1.000 / 0.57 | 1.000 / 2.2 | 1.000 / 0.72 |
| random1000-any (0.05%) | 1.000 / 1.7 | 1.000 / 0.07 | 1.000 / 1.8 | 1.000 / 0.14 |

- At 19% the walk wins on the mean (exact: 24.5 ms over 8 segments), so the
  share stops at one in eight.
- The one case that gets slower is a mid-size tenant queried from inside
  its own region on a large segment (1.5 -> 4.2 ms), with recall 0.94 ->
  1.0. Walking first with a visit budget of an eighth of the passing
  documents, then scanning, kept it at 2.3 ms but cost the random tenants
  40% and recall 0.96-0.998, so the rule scans directly.
- Graph walks pay the descent on every segment; the scan does not, which is
  why the small tenants gain most with many segments.
- On storage that is not in memory the scan reads the tenant's range of
  `.vec` in order (an index sorted by tenant keeps it contiguous), the walk
  one random page or more per visited node, so the share is conservative
  there.

Raw numbers: `perf/tenant-exact-shard/hnsw-2m.csv` in the project files.
