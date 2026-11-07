# Disk HNSW: recall, latency and I/O under a memory budget

`lucene-disk-hnsw` against stock Lucene's HNSW layout and `lucene-ivf`, all
reading their files through one simulated page cache that holds a memory
budget. How to reproduce: [`benchmarks/disk-hnsw/README.md`](../../benchmarks/disk-hnsw/README.md).
Raw CSVs are kept with the project files (`perf/disk-hnsw/`).

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.1 GHz. Searches run one query at a time on
  one thread; GloVe 1000 and fastText 500 measured queries after as many
  warm-up queries. Recall@10 against exact ground truth.
- Every HNSW variant uses the same graph: Lucene's builder, M 16, beam 100,
  on full-precision vectors. Codes are 1-bit BBQ (`osq`, 4-bit query).
- The page cache has 4 KiB pages and CLOCK eviction, and holds the budget
  minus what the index keeps in memory regardless (`pinned`). A budget below
  the pinned size is not run. CPU time includes the simulated cache.
- A round trip is a read that misses, or one batch of ranges announced with
  `prefetch` before they are read. The latency estimates add 100 us
  (local NVMe) or 500 us (EBS-class block storage) per round trip to the
  CPU time. On blob storage behind a local cache, each round trip a RAM miss
  costs is an NVMe read; a cold NVMe miss costs a blob GET instead.

| index | pinned in memory | read through the cache |
|---|---|---|
| `disk` | per-node codes, upper levels | `.dhn` records (vector + neighbours), beam width W |
| `lean` (LM-DiskANN) | upper levels only | `.dhn` records that also hold the neighbours' codes |
| `stock-bbq` | `.vex` offsets | `.vex` adjacency, `.veq` codes, `.vec` vectors (rerank `ef`) |
| `stock-bbq-pinned` | `.vex` and `.veq` | `.vec` vectors only |
| `stock-f32` | `.vex` offsets | `.vex`, `.vec` (neighbours scored on floats) |
| `ivf` | `.ivm` (centroids) | `.ivp` clusters, `.ivv` rerank vectors (rerank 100) |

The stock model gives stock its best case: an expanded node's neighbour
codes are prefetched together (one round trip per hop rather than one per
neighbour), and the rerank vectors are prefetched together.

## GloVe-100 (390 000 x 100, cosine)

Files: `.dhn` 228 MB (7 records a page), `lean` 799 MB, stock `.vec` 156 MB,
`.veq` + `.vex` 30 MB. Fastest setting reaching recall@10 0.85, latency at
500 us a round trip:

| budget MiB | disk | lean | stock-bbq | stock-bbq-pinned | stock-f32 | ivf |
|---|---|---|---|---|---|---|
| 8 | does not fit (15 MB pinned) | **63.9 rt, 32.9 ms** | 637 rt, 321 ms | does not fit (32 MB) | 182 rt, 92 ms | 96 rt, 48 ms (0.851) |
| 16 | 63.9 rt, 33.0 ms | 63.9 rt, 32.9 ms | 542 rt, 273 ms | does not fit | 169 rt, 86 ms | 86 rt, 43 ms |
| 32 | 63.0 rt, 32.6 ms | 63.6 rt, 32.7 ms | 183 rt, 94 ms | **2.0 rt, 2.6 ms** | 164 rt, 83 ms | 69 rt, 35 ms |
| 64 | 60.5 rt, 31.3 ms | 63.0 rt, 32.4 ms | 48 rt, 25.5 ms | **1.0 rt, 2.3 ms** | 151 rt, 76 ms | 45 rt, 23 ms |
| 128 | 48.4 rt, 25.3 ms | 61.3 rt, 31.7 ms | 1.2 rt, 2.4 ms | **1.0 rt, 2.0 ms** | 110 rt, 56 ms | 13 rt, 6.7 ms (0.889) |
| in memory | 1.02 ms (0.879) | 0.91 ms (0.880) | 1.60 ms (0.873) | 1.49 ms (0.873) | 0.84 ms (0.874) | 0.13 ms (0.851) |

HNSW settings: disk and lean L 160, W 8 (recall 0.880); stock ef 320 (0.873),
stock-f32 ef 80 (0.874). With W 32 instead of 8, disk at 32 MiB takes 34.8
round trips (18.3 ms) for 0.884.

## fastText-300 (989 999 x 300, L2)

Files: `.dhn` 1352 MB (3 records a page), `lean` 4055 MB, stock `.vec` 1188 MB,
`.veq` + `.vex` 75 MB. This graph tops out near 0.82 recall@10 at ef/L 160
for every HNSW variant (the runs stopped at 160), so the floor is 0.8:

| budget MiB | disk | lean | stock-bbq | stock-bbq-pinned | ivf (0.872 at nprobe 32) |
|---|---|---|---|---|---|
| 32 | does not fit (61 MB pinned) | **11.2 rt, 6.3 ms** | 302 rt, 152 ms | does not fit (79 MB) | 107 rt, 54 ms |
| 64 | 11.4 rt, 6.7 ms | **11.0 rt, 6.2 ms** | 234 rt, 118 ms | does not fit | 96 rt, 48 ms |
| 128 | 10.6 rt, 6.2 ms | 10.6 rt, 5.9 ms | 26 rt, 14.1 ms | **1.0 rt, 1.8 ms** | 80 rt, 40 ms |
| 256 | 9.6 rt, 5.7 ms | 10.1 rt, 5.8 ms | 1.5 rt, 2.2 ms | **1.0 rt, 1.8 ms** | 58 rt, 29 ms (0.925) |
| 512 | 0.9 rt, 1.2 ms | 1.4 rt, 1.4 ms | 1.0 rt, 2.0 ms | 1.0 rt, 1.9 ms | 1.7 rt, 1.6 ms (0.955) |
| in memory | 0.71 ms (0.824) | **0.67 ms** (0.824) | 1.54 ms (0.813) | 1.26 ms (0.813) | 0.38 ms |

HNSW settings: disk and lean L 160, W 32 (W 8 in memory); stock ef 160.

## Findings

- **Where the layout wins.** When the budget is below stock's graph plus
  codes (about 80 bytes a node in both sets), stock HNSW pages its
  adjacency and codes and pays at least a round trip per hop: 180-640
  round trips a query with BBQ, 110-180 with float vectors. The disk layout
  needs 11-64, so it is **3-24x faster than stock HNSW + BBQ** there at the
  same or better recall, and 2.5-2.8x faster than stock float HNSW (GloVe
  8-32 MiB, fastText 32-64 MiB). `lean` runs in any budget (2.6 MB and
  5.9 MB pinned: the upper levels) with the same round trips as `disk`.
- **Where it does not.** Once stock's graph and codes fit in memory,
  pinning them and streaming only the rerank vectors (`stock-bbq-pinned`)
  is one round trip a query, which no graph walk on disk can match: the
  disk layout's round trips are the search's sequential hops divided by the
  beam width. The page-cache store's plan to pin `.vex`/`.veq` and stream
  `.vec` is the right default for stock HNSW in that regime.
- **In memory** the layout is 1.5-2.3x faster than stock HNSW + BBQ at
  higher recall (stock float HNSW is faster still on GloVe, 0.84 ms, but
  keeps every float vector in memory): every expanded node is scored exactly from its record, so
  no separate rerank pass is needed and the walk follows exact scores.
  `lean` is the fastest, its neighbour codes sitting next to each other in
  the record instead of scattered over the resident code array.
- **Beam width** trades bytes for round trips: GloVe at 32 MiB took 61.9
  round trips at W 8, 42.8 at W 16 and 34.8 at W 32, reading 6-9% more
  records for the same recall. For blob storage W 32 is the better default;
  for local NVMe W 8.
- **IVF** reads one cluster at a time today, so it pays nprobe round trips
  (45-107 here). Announcing all probed clusters with `prefetch` before the
  scan would make that about 2 (one batch of clusters, one of rerank
  vectors), which would put it ahead of every graph layout on blob storage.
- **Page locality.** Breadth-first numbering puts 7 (GloVe) or 3 (fastText)
  records a page; at half the file cached (GloVe 128 MiB) 28% of expanded
  nodes still miss. A block-shuffling order (Starling, SIGMOD 2024) could
  raise hits per page.
