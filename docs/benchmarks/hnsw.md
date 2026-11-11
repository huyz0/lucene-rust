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

A filtered leaf passing more than `perLeafTopK` documents is searched by a
cost model (`filtered_walk_plan` in
`crates/lucene-search/src/vector_query.rs`) instead of Lucene's rule, which
walks the graph and falls back to an exact scan of the passing vectors
after visiting as many nodes as pass: at a few percent of a segment that
costs the walk and then the scan.

- An unfiltered walk visits about `ln(size) * k` nodes (Lucene's
  `expectedVisitedNodes`, with `k` the collector's queue); under a filter
  unrelated to the query, about that many divided by the filter's share of
  the segment. Measured visits on GloVe-100 390k at a queue of 100:
  10,901 / 6,231 / 3,509 / 2,332 at 10% / 20% / 41% / 69% against
  12,500 / 6,300 / 3,150 / 1,870 predicted.
- A visit costs about four exact scores (`HNSW_VISIT_COST`: 140-160 ns a
  visit against 45-55 ns a scored vector in a scan, which reads the passing
  vectors in order).
- Walk predicted no dearer than the scan: Lucene's walk. Scan cheaper than
  four unfiltered walks (`HNSW_PROBE_GAIN`): the scan. Otherwise the filter
  may still pass most of the query's neighbourhood (a tenant queried from
  inside its own region), which only the walk can see: it walks 256 nodes
  (`HNSW_PROBE_VISITS`), counts the share of them that pass (never below
  the filter's share of the segment), and switches to the scan when the
  rest of the walk is predicted to cost more (`FilteredWalkPlanner` in
  `knn_collectors.rs`). A walk that goes on still gives up past twice the
  scan's cost.
- A scan the model chooses scores the rescoring vectors when the format has
  any (quantized HNSW), so it returns the exact top `k`. Nested
  (diversifying) searches, searches with a patience or deadline (no probe)
  and retired formats keep Lucene's rule.

`knn_filtered --format hnsw --ef 100` (GloVe-100 390k, one segment, M 16,
beam 100, 200 queries, k 10), recall@10 / mean us. `u` filters are random
documents, `c` ones the documents nearest an anchor, queried from near the
anchor (`local`) or from anywhere (`any`). Runs vary 10-25% from one to
the next; the 1-in-8 column is the fixed rule this model replaces (scan at
most one in eight of the segment's vectors):

| filter (share) | Lucene | 1 in 8 | cost model |
|---|---|---|---|
| u4000 (1%) | 1.000 / 1158 | 1.000 / 350 | 1.000 / 300 |
| u10000 (2.5%) | 1.000 / 2430 | 1.000 / 800 | 1.000 / 750 |
| u20000 (5%) | 0.993 / 3450 | 1.000 / 1370 | 1.000 / 1170 |
| u40000 (10%) | 0.956 / 1664 | 1.000 / 2660 | 1.000 / 2300-3000 |
| u80000 (20%) | 0.943 / 1061 | 0.942 / 1000 | 0.942 / 1040-1250 |
| c20000-local (5%) | 0.994 / 311 | 1.000 / 1160 | 1.000 / 1420 |
| c40000-local (10%) | 0.970 / 441 | 1.000 / 2700 | 0.991 / 370-410 |
| c40000-any (10%) | 0.951 / 3101 | 1.000 / 2085-3084 | 0.984 / 1920 |
| c80000-any (20%) | 0.937 / 2084 | 0.914 / 2525-2695 | 0.914 / 2040 |

`knn_tenants --scale 8 --format hnsw` (1M GloVe-100 vectors with jitter,
index sorted by tenant, 50 queries a tenant), recall@10 / mean us:

| tenant | 1 segment, Lucene | 1 segment, cost model | 8 segments, Lucene | 8 segments, 1 in 8 | 8 segments, cost model |
|---|---|---|---|---|---|
| random187500-any | 0.932 / 4094 | 0.932 / 3439-4167 | 0.982 / 12844 | 0.982 / 15645 | 0.982 / 11356 |
| region187500-local | 0.952 / 820 | 0.952 / 774-1059 | 0.992 / 2030 | 0.992 / 1390 | 0.992 / 1711 |
| random50000-any | 0.962 / 7015 | 1.000 / 2495-2824 | 1.000 / 10015 | 1.000 / 3722 | 1.000 / 2733 |
| region50000-any | 0.906 / 5378 | 0.976-0.986 / 2563-2899 | 0.990 / 8643 | 1.000 / 2974 | 1.000 / 2678 |
| region50000-local | 0.940 / 1190 | 0.956-0.962 / 1499-1723 | 0.990 / 5057 | 1.000 / 2529 | 1.000 / 2336 |
| random5000-any | 1.000 / 961 | 1.000 / 317-344 | 1.000 / 1946 | 1.000 / 346 | 1.000 / 321 |
| region5000-local | 0.996 / 431 | 1.000 / 350-587 | 1.000 / 1734 | 1.000 / 506 | 1.000 / 390 |
| random500-any | 1.000 / 874 | 1.000 / 68-79 | 1.000 / 278 | 1.000 / 118 | 1.000 / 96 |

- Against Lucene: 3-13x faster for filters passing 0.05-5% of a segment,
  with recall 1.0 where Lucene's walk gave 0.91-0.96. The loss: a tenant
  just under the probe's floor queried from inside its own region
  (c20000-local, 5%) is scanned where Lucene's walk took 0.3 ms (1.4 ms,
  recall 0.994 -> 1.0).
- Against the fixed 1-in-8 share: the same where the filter is unrelated
  to the query, and 5-7x faster for a tenant queried from inside its own
  region at 10% of a segment (recall 0.99 instead of 1.0). On one 1M
  segment a 5% region tenant probes and walks: faster from inside the
  region (0.96 recall), slower from outside it than the scan. The fixed
  share also scanned up to 2^18 vectors whatever the segment's size; the
  model's walk wins earlier as segments grow.
- ACORN-1 (`FilteredHnswGraphSearcher`, on with a `filteredSearchThreshold`)
  is no candidate here: 0.60 / 0.74 / 0.76 / 0.84 recall at 1% / 2.5% / 5% /
  10% on the 390k set, where Lucene's walk and the scan give 0.96-1.0.
- On storage that is not in memory the scan reads the tenant's range of
  `.vec` in order (an index sorted by tenant keeps it contiguous), the walk
  one random page or more per visited node, so the model leans towards the
  walk there.

Raw numbers: `perf/hnsw-filtered-planner/` and
`perf/tenant-exact-shard/hnsw-2m.csv` in the project files.

## Adaptive beam

`KnnFloatVectorQuery::with_exploration_slack(hits, slack)` (opt-in, no
Lucene equivalent) stops a walk once its best unexplored candidate is
farther than `1 + slack` times the `hits`-th best node found, instead of
once it no longer beats the worst of its `k`
(`AdaptiveBeamCollector`; Al-Jazzazi et al., "Distance Adaptive Beam
Search", the rule behind Vespa's `exploration-slack`). At the same visited
nodes it finds more of the true top 10 than a wider `ef_search`. GloVe-100
390k, one segment, k 10 (queue 100 for the slack rows), recall@10 / nodes
visited:

| filter | ef 100 | ef 200 | slack 0.1 | slack 0.2 | slack 0.3 |
|---|---|---|---|---|---|
| none | 0.876 / 1749 | 0.919 / 3019 | 0.841 / 977 | 0.948 / 3045 | 0.984 / 9129 |
| u40000 (10%) | 0.955 / 10782 | 0.974 / 18649 | 0.933 / 5351 | 0.980 / 16052 | 0.992 / 39217 |
| u80000 (20%) | 0.947 / 6198 | 0.967 / 10941 | 0.913 / 3122 | 0.976 / 9734 | 0.992 / 25407 |
| u160000 (41%) | 0.925 / 3539 | 0.957 / 6200 | 0.893 / 1854 | 0.969 / 5916 | 0.991 / 16518 |
| c80000-any (20%) | 0.931 / 9108 | 0.961 / 14018 | 0.900 / 6752 | 0.980 / 18408 | 0.994 / 43389 |

Latency follows the visits. It changes results, so it stays opt-in. The
OpenSearch plugin takes it as `method_parameters.exploration_slack` on a
`knn` query against a lucene-engine HNSW field (hits = `k`, queue =
`ef_search`); a nested query ignores it and leaving it out keeps stock
results.
