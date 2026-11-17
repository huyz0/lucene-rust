# Faiss codec indexes vs our IVF and HNSW

glove-wiki-gigaword-100 (390 000 vectors, 100 dims, cosine as inner product on
normalised vectors), 1 000 queries, recall@10 against exact search, one query
per call on one thread. Container: 4 cores, 15 GB. Files from
`benchmarks/ivf/prepare.py`.

- `native-faiss-*`: `lucene-faiss` (this repo, `examples/faiss_bench.rs`),
  `IDMap,HNSW16` (efConstruction 100) and `IDMap,IVF1015,Flat`. Single-thread build.
- `faiss-*`: python faiss 1.16 (`faiss_hnsw_bench.py`, `faiss_bench.py`), built with all cores.
- `ivf-*` / `hnsw-bbq`: our `lucene-ivf` (`ivf_bench`), quantized with full-precision rerank.

| index | setting | recall@10 | latency µs | pinned MB |
|---|---|---|---|---|
| native-faiss-hnsw | ef 64 | 0.882 | 238 | 215 |
| native-faiss-hnsw | ef 128 | 0.935 | 445 | 215 |
| native-faiss-hnsw | ef 256 | 0.966 | 920 | 215 |
| faiss-hnsw (python) | ef 64 | 0.857 | 156 | 212 |
| faiss-hnsw (python) | ef 128 | 0.909 | 411 | 212 |
| faiss-hnsw (python) | ef 256 | 0.947 | 684 | 212 |
| native-faiss-ivf | nprobe 32 | 0.942 | 787 | 163 |
| native-faiss-ivf | nprobe 64 | 0.975 | 1111 | 163 |
| faiss-flat (python) | nprobe 32 | 0.939 | 524 | 160 |
| faiss-flat (python) | nprobe 64 | 0.975 | 993 | 160 |
| our ivf-sq7 +rerank 50 | nprobe 64 | 0.956 | 304 | 43 (+156 vectors read) |
| our ivf-osq4 +rerank 50 | nprobe 64 | 0.956 | 295 | 33 (+156 vectors read) |
| our ivf-osq4 +rerank 50 | nprobe 128 | 0.983 | 529 | 33 |
| our hnsw-bbq | ef 320 | 0.796 | 580 | 45 |

Build (390k vectors): native HNSW 156 s single-thread vs 39 s python on four
threads; native IVF-flat 22.5 s vs 7.5 s.

Reading: the native port matches Faiss's recall at the same settings (graphs
differ slightly because the build order is not threaded) and is 1.0-1.5x its
latency per query; index files are within 1.5% of Faiss's. Against our own
engines the Faiss-format indexes cost 4-5x the pinned memory and 2-3x the
latency at equal recall, because they scan full-precision vectors. Faiss
stays an explicit, opt-in engine for stock-Lucene format compatibility; it
is not a default and does not replace IVF/HNSW.

## After optimisation (2026-11-18)

Same data and harness; the container is shared and noisy, so each latency is the
median of three runs (best run in brackets) and python Faiss was re-run in the same
window. What changed: the example and plugin allocator is mimalloc (the HNSW build
was glibc-malloc bound: 2 min of system time), one visited set per build worker and
a thread-local epoch set per search thread (a 390 KB memset per query before),
neighbour vectors prefetched before their distances are computed, fused multiply-add
distances, IVF lists scanned without touching the heap for non-qualifying vectors,
and k-means assignment four vectors per centroid load. Results are unchanged
(same ranked ids as Faiss in the fixture tests; recall matches before/after).

| index | setting | recall@10 | latency µs (median, best) | before |
|---|---|---|---|---|
| native HNSW | ef 64 | 0.876 | 148 (123) | 238 |
| native HNSW | ef 128 | 0.933 | 354 (268) | 445 |
| native HNSW | ef 256 | 0.965 | 624 (508) | 920 |
| python Faiss HNSW | ef 128 | 0.909 | 288 | 411 |
| python Faiss HNSW | ef 256 | 0.947 | 495 | 684 |
| native IVF-flat | nprobe 32 | 0.942 | 711 (663) | 787 |
| native IVF-flat | nprobe 64 | 0.975 | 1337 (1266) | 1111 |
| python Faiss IVF-flat | nprobe 64 | 0.975 | 1102 | 993 |

Build: HNSW 156 s → 34.5 s (python Faiss 39 s, four threads). IVF-flat 22.5 s → 26 s,
unchanged within noise: it is 25 k-means iterations of brute-force assignment
(about 660 GFLOP), where Faiss's BLAS is ahead (7.5 s). A 4x2 register tile was tried
and was 3x slower (spills), so it was dropped. IVF search at nprobe 64 is memory
bound (10 MB of lists per query) and sits within about 15% of Faiss.

## GEMM assignment and shared SIMD kernels (2026-11-18)

k-means training and the final list assignment of `IDMap,IVF<n>,Flat` now rank
centroids with one matrix product per 256-vector block (`lucene_util::gemm::dot_rows`,
the `matrixmultiply` micro-kernels, no C BLAS) as Faiss does with BLAS: Euclidean
ranks `c.c/2 - x.c`, inner product `-x.c`. Distances in searches and scans use the
shared AVX2 (x86) and NEON (aarch64) kernels of `lucene_util::simd`. Same data,
three-run medians:

| index | setting | recall@10 | latency µs | python Faiss µs |
|---|---|---|---|---|
| native HNSW | ef 64 | 0.880 | 121 | 204 |
| native HNSW | ef 128 | 0.933 | 233 | 288 |
| native HNSW | ef 256 | 0.964 | 502 | 495 |
| native IVF-flat | nprobe 32 | 0.942 | 489 | 546 |
| native IVF-flat | nprobe 64 | 0.974 | 990 | 1102 |

IVF-flat build 26 s → 9.5 s (python Faiss 7.5 s, BLAS), same recall. Our own IVF
(`lucene-ivf`) already clusters with a hand-tiled SIMD kernel; swapping its
`full_assign` for the same GEMM was measured (390k x 100, 1693 clusters, 10 full
rounds): 13.8 s vs 13.1 s with identical recall, about 5% on a non-default setting,
so it was not adopted.
