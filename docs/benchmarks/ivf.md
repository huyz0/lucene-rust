# IVF vector index: recall and latency

`lucene-ivf` with its quantizers, compared with this port's Lucene104 HNSW +
BBQ and with Faiss 1.15.1 (`faiss-cpu` wheel, AVX512 build). How to
reproduce: [`benchmarks/ivf/README.md`](../../benchmarks/ivf/README.md).

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.1 GHz with AVX512. Rust builds for
  x86-64-v3.
- Builds use all 4 cores. Searches run one query per call on one thread, for
  1000 queries. Recall@10 is measured against exact ground truth.
- Both IVF implementations use the same cluster count. `ivf_bench` picks it
  with a target of 384 vectors per cluster, and Faiss gets `--nlist` set to
  match.
- `rerank N` re-scores the top N quantized candidates with full-precision
  vectors. Faiss does this with `IndexRefineFlat`.
- `pinned` is what has to stay in RAM or on local NVMe so that a search never
  waits on a remote read:
  - IVF: meta, centroids and postings.
  - HNSW: 1-bit codes and the level-0 graph.
  - Faiss: the index without its refine store.
- Full-precision vectors are read only to rerank: 156 MB for GloVe, 1188 MB
  for fastText.

| set | vectors | dim | metric | clusters |
|---|---|---|---|---|
| GloVe-100 (gensim `glove-wiki-gigaword-100`) | 390 000 | 100 | cosine | 1693 |
| fastText-300 (gensim `fasttext-wiki-news-subwords-300`) | 989 999 | 300 | L2 | 6437 |

## GloVe-100

Each row is the fastest setting that reaches the recall shown.

| index | setting | recall@10 | latency µs | build s | pinned MB |
|---|---|---|---|---|---|
| IVF-BBQ (1-bit) | nprobe 64, rerank 300 | 0.937 | 234 | 5.6 | 15.6 |
| IVF-BBQ (1-bit) | nprobe 128, rerank 300 | 0.955 | 354 | 5.6 | 15.6 |
| IVF-OSQ 2-bit | nprobe 64, rerank 100 | 0.951 | 213 | 5.4 | 22.3 |
| IVF-OSQ 4-bit | nprobe 64, rerank 100 | 0.955 | 327 | 5.0 | 33.4 |
| IVF-OSQ 4-bit | nprobe 128, rerank 100 | 0.983 | 666 | 5.0 | 33.4 |
| IVF-SQ int7 | nprobe 64, rerank 100 | 0.955 | 319 | 4.5 | 42.8 |
| IVF-SQ int4 | nprobe 64, rerank 100 | 0.955 | 338 | 4.6 | 23.3 |
| HNSW + BBQ (M 16, beam 100) | ef 320 | 0.798 | 939 | 36.0 | 44.6 |
| HNSW + BBQ (M 16, beam 100) | ef 640 | 0.870 | 1843 | 36.0 | 44.6 |
| Faiss IVF-Flat | nprobe 64 | 0.963 | 613 | 14.7 | 159.8 |
| Faiss IVF-SQ8 | nprobe 64, rerank 100 | 0.963 | 2163 | 14.5 | 42.8 |
| Faiss IVF-RaBitQ (qb 4) | nprobe 64, rerank 300 | 0.919 | 607 | 15.6 | 12.0 |
| Faiss IVF-PQ50x4 fast-scan | nprobe 64, rerank 300 | 0.953 | 223 | 14.6 | 14.2 |

## fastText-300

| index | setting | recall@10 | latency µs | build s | pinned MB |
|---|---|---|---|---|---|
| IVF-BBQ (1-bit) | nprobe 64, rerank 300 | 0.929 | 810 | 105.5 | 73.5 |
| IVF-BBQ (1-bit) | nprobe 128, rerank 300 | 0.964 | 1091 | 105.5 | 73.5 |
| IVF-OSQ 2-bit | nprobe 128, rerank 100 | 0.964 | 1235 | 92.0 | 117.6 |
| IVF-OSQ 4-bit | nprobe 64, rerank 100 | 0.930 | 1025 | 90.7 | 185.9 |
| IVF-OSQ 4-bit | nprobe 128, rerank 100 | 0.966 | 1636 | 90.7 | 185.9 |
| IVF-SQ int4 | nprobe 128, rerank 100 | 0.965 | 2234 | 87.2 | 164.2 |
| HNSW + BBQ (M 16, beam 100) | ef 640 | 0.880 | 2318 | 127.2 | 83.0 |
| Faiss IVF-Flat | nprobe 64 | 0.946 | 2869 | 321.6 | 1203.7 |
| Faiss IVF-SQ8 | nprobe 64, rerank 300 | 0.946 | 7520 | 331.0 | 312.7 |
| Faiss IVF-RaBitQ (qb 4) | nprobe 64, rerank 300 | 0.945 | 1500 | 368.3 | 61.2 |
| Faiss IVF-RaBitQ (qb 4) | nprobe 128, rerank 300 | 0.973 | 2364 | 368.3 | 61.2 |
| Faiss IVF-PQ150x4 fast-scan | nprobe 64, rerank 300 | 0.929 | 1001 | 329.3 | 98.2 |
| Faiss IVF-PQ150x4 fast-scan | nprobe 128, rerank 300 | 0.954 | 1110 | 329.3 | 98.2 |

## Findings

The 1- and 2-bit rows use fast scan: codes are stored 32 to a block and
scored with 16-entry lookup tables held in SIMD registers, the technique
behind Faiss's PQ fast-scan and RaBitQ. Reranking also prefetches candidate
vectors 8 ahead.

Clustering ends with two full Lloyd rounds (`KMeansParams::full_rounds`)
that compare every vector with every centroid. The hierarchical tree and its
neighbourhood refinement leave about one GloVe vector in ten away from its
nearest centroid, and those rounds fix it. At a fixed number of probes,
recall with exact distances rose from 0.949 to 0.955 on GloVe at nprobe 64,
which matches Faiss's flat k-means for the same number of vectors scanned.
They cost about 1 s per round on GloVe and 20 s per round on fastText.
`full_rounds: 1` keeps most of the gain on fastText (0.964 at nprobe 128)
and builds in 84 s.

- **Against HNSW + BBQ.** IVF-BBQ reaches recall HNSW + BBQ does not reach
  at any ef tried. At HNSW's best recall (0.87 to 0.88) IVF-BBQ is 3 to 10x
  faster, builds 1.2 to 6x faster and pins a third to nine tenths of the
  memory. A probe is one sequential range read where an HNSW search makes
  hundreds of dependent random reads, which matters more on blob storage.
- **Against Faiss IVF-RaBitQ**, the same 1-bit idea: IVF-BBQ is faster at
  equal recall on both sets, by 1.5 to 2.5x, and builds about 3x faster.
- **Against Faiss PQ fast-scan:**
  - On GloVe, 2-bit OSQ is level with it: 0.951 at 213 µs against 0.953 at
    223 µs, pinning 22.3 MB against 14.2. IVF-BBQ needs nprobe 128 for the
    same recall (354 µs), because its estimates need a deeper rerank.
  - On fastText, IVF-BBQ is ahead: 0.929 at 810 µs against 0.929 at 1001,
    and 0.964 at 1091 µs against 0.954 at 1110.
- **Ranking centroids is a fixed cost.** On fastText, ranking 6437 centroids
  of 300 floats costs about 330 µs per query in both engines, because the
  7.7 MB of centroids is read from memory every time. It is now a third of
  IVF-BBQ's latency at 0.93. In batch mode Faiss ranks them for all queries
  with one matrix multiply, which hides this cost; the batch-mode numbers
  are not comparable, so they are not shown. Reduced-precision centroids or
  a two-level centroid tree would cut it.
- **Quantizer choice.** Once rerank is on, 2-bit, 4-bit and int7 codes all
  reach the partition's recall ceiling, so they trade pinned bytes for less
  rerank work. BBQ needs a deeper rerank (300) to get there and pins the
  least. Without rerank, int7 and 4-bit OSQ give the best estimates.
- **SOAR spilling** (`IvfParams::spill`) gave no recall gain per entry
  scanned on GloVe, so it is off by default.

## Quantizer options, search depth and merges

Same machine and setup. Recall@10 over 1000 queries.

**Rotation and Extended RaBitQ.** `rot-<quantizer>` rotates vectors with a
seeded randomized Hadamard transform into the inner quantizer's own code
width, so it adds no code bytes; `erq` is Extended RaBitQ (2 to 8 bits).

| set | index | nprobe | rerank 0 | rerank 100 | rerank 300 | pinned MB |
|---|---|---|---|---|---|---|
| GloVe-100 | BBQ | 64 | 0.464 | 0.876 | 0.937 | 15.6 |
| GloVe-100 | rotated BBQ | 64 | 0.506 | 0.900 | 0.946 | 15.6 |
| GloVe-100 | BBQ | 128 | 0.465 | 0.888 | 0.955 | 15.6 |
| GloVe-100 | rotated BBQ | 128 | 0.509 | 0.915 | 0.968 | 15.6 |
| GloVe-100 | OSQ 2-bit | 64 | 0.679 | 0.951 | 0.955 | 22.3 |
| GloVe-100 | erq 2-bit | 64 | 0.717 | 0.954 | 0.955 | 20.6 |
| GloVe-100 | OSQ 4-bit | 64 | 0.848 | 0.955 | 0.955 | 33.4 |
| GloVe-100 | erq 4-bit | 64 | 0.887 | 0.955 | 0.955 | 33.9 |
| fastText-300 | BBQ | 64 | 0.641 | 0.920 | 0.929 | 73.5 |
| fastText-300 | rotated BBQ | 64 | 0.681 | 0.926 | | 73.5 |
| fastText-300 | OSQ 2-bit | 64 | 0.763 | 0.929 | 0.930 | 117.6 |
| fastText-300 | erq 2-bit | 64 | 0.811 | 0.930 | | 113.2 |

With rerank, every quantizer reaches the partitions' recall ceiling; the
options trade pinned bytes against rerank depth. Rotation and erq pay off
most without rerank or with a shallow one.

**Rerank vector encoding** (`IvfParams::rerank_encoding`). f16 halves the
`.ivv` file (GloVe 159 to 81 MB, fastText 1196 to 602 MB) for under 0.001
recall; int8 (residual from the cluster centroid, per-vector scale) takes it
to 28% for about 0.005. Top-10 overlap of exact search over the rounded
vectors with exact search over f32, as the set grows (a worst case: rerank
only rescores candidates):

| encoding | GloVe 50k | 100k | 200k | 390k | fastText 100k | 250k | 500k |
|---|---|---|---|---|---|---|---|
| f16 | 0.9994 | 0.9996 | 0.9994 | 0.9990 | 0.9995 | 0.9998 | 0.9997 |
| bf16 | 0.9959 | 0.9942 | 0.9948 | 0.9938 | 0.9975 | 0.9979 | 0.9981 |
| int8 | 0.9921 | 0.9902 | 0.9903 | 0.9896 | 0.9948 | 0.9939 | 0.9960 |

int8 loses about 0.001 per doubling on GloVe; f16 does not move.

**Search depth** (`Probes::Auto`, `Rerank::Auto`). Each field is calibrated
at write and merge on 64 of its own vectors; search uses the stored probe
fraction and rerank cap, and stops reranking once no remaining candidate's
estimate plus error bound can beat the k-th exact score. GloVe, calibrated
nprobe 129 of 1693:

| index | rerank | recall@10 | latency µs | reranked |
|---|---|---|---|---|
| BBQ | fixed 300 | 0.955 | 391-421 | 300 |
| BBQ | auto | 0.955 | 396-427 | 264 |
| rotated BBQ | fixed 300 | 0.968 | 391-428 | 300 |
| rotated BBQ | auto | 0.968 | 399-451 | 248 |
| erq 4-bit | fixed 300 | 0.983 | 716-772 | 300 |
| erq 4-bit | auto | 0.983 | 635-684 | 17 |

BBQ's error bounds are loose, so early stop trims little; erq's are tight.

**Merges** (`MergeClustering::ReuseLargest`, the default). When one source
holds at least half of the live vectors, its clusters are kept, the others'
vectors join their nearest centroid, and only clusters that overflow or
empty are split or dissolved. GloVe, 90% + 10% segments, BBQ, nprobe 64,
rerank 100: merge 1.43 s and recall 0.871, against 4.80 s and 0.876 for a
full re-cluster.

Repeated reuse does not drift. An update stream (`ivf_merge_bench
--generations 10 --split 0.5 --delete 0.05`): GloVe starts from half its
vectors, and each of 10 merges adds the next 5% and deletes 5% of the live
documents. After the tenth merge, at nprobe 64, reuse holds recall 0.882 at
117 us a query against 0.878 at 114 us for re-clustering every time, and spends
9.0 s merging in all against 24.5 s. Reuse keeps fewer, larger clusters
(823 of up to 758 vectors against 1183 of up to 668), which costs about 3% more
search time at the same nprobe.

## Centroid routing

Picking the clusters to probe means scoring the query against every
centroid. At 6k-27k centroids that f32 scan dominates a search
(fastText, 27k centroids: about 2 ms of a 2.2 ms query). `routing.rs` cuts it
two ways: 8-bit centroid codes with an exact rescore of the best candidates
(`Routing::Quantized`, always on), and from 8192 clusters a stored two-level
routing tree, k-means over the centroids into parents of about 64
(`Routing::Auto`). Same machine and method as above (one thread, 1000
queries, BBQ codes, rerank 100). The small-cluster runs use `--full 0` to
build tens of thousands of clusters quickly; they stand in for large merged
segments (24k clusters of 16 vectors is the shape of a 9M-vector segment at the
default 384 per cluster).

```
cargo run --release -p lucene-ivf --example ivf_bench -- /tmp/data/fasttext300 l2 \
  --target 96 --full 0 --quantizers bbq --nprobe 16,32,64,128 --rerank 100 \
  --routing exact,int8,auto,tree64
cargo run --release -p lucene-ivf --example routing_bench -- /tmp/data/glove100 cosine
```

fastText-300, 27 438 clusters (recall@10 / µs per query):

| nprobe | exact f32 | int8 | auto (tree, cover 32) | tree, cover 64 |
|---|---|---|---|---|
| 16 | 0.687 / 1844 | 0.687 / 611 | 0.663 / 195 | 0.681 / 203 |
| 32 | 0.777 / 1852 | 0.777 / 772 | 0.766 / 256 | 0.774 / 297 |
| 64 | 0.856 / 1997 | 0.856 / 846 | 0.851 / 408 | 0.855 / 472 |
| 128 | 0.910 / 2087 | 0.910 / 1108 | 0.908 / 694 | 0.910 / 771 |

GloVe-100, 24 203 clusters:

| nprobe | exact f32 | int8 | auto | tree, cover 64 |
|---|---|---|---|---|
| 32 | 0.725 / 624 | 0.725 / 352 | 0.710 / 121 | 0.719 / 137 |
| 64 | 0.800 / 675 | 0.800 / 486 | 0.793 / 187 | 0.797 / 211 |
| 128 | 0.853 / 751 | 0.853 / 512 | 0.848 / 302 | 0.852 / 360 |
| 256 | 0.887 / 870 | 0.887 / 714 | 0.884 / 560 | 0.886 / 752 |

fastText-300 at the default 384 per cluster (6 437 clusters, no tree):

| nprobe | exact f32 | int8 |
|---|---|---|
| 8 | 0.693 / 495 | 0.693 / 228 |
| 16 | 0.794 / 582 | 0.794 / 288 |
| 32 | 0.868 / 602 | 0.868 / 378 |
| 64 | 0.920 / 812 | 0.920 / 564 |

- **8-bit centroids are free recall-wise**: the same recall in every row,
  for 1.2-3.0x less latency. They cost one byte per centroid dimension in
  memory beyond the f32 centroids, which the rescore and the quantizers still
  need.
- **The tree** gives 3-9x less latency than exact routing at large cluster
  counts. At a fixed nprobe it loses up to 0.02 recall at small nprobe and
  0.001-0.005 from nprobe 64 up. Per unit of latency it wins: at 24k-27k
  clusters it reaches a given recall at roughly half the latency of int8
  routing. Calibration (`Probes::Auto`) searches through the tree, so the
  stored probe count already accounts for it. At 3-7k clusters (GloVe
  `--target 192/96`) the tree was only 0-1.4x better at equal recall, hence
  the 8192 threshold.
- **Cost**: the parent list is one vint per centroid in `.ivm`. Opening
  rebuilds the parent means and codes, 60 ms for 27k x 300 centroids. Building
  the tree at write is a k-means over the centroids, small next to clustering
  the vectors.

### Routing micro-benchmark

`benches/routing.rs` (Criterion) times one routing of nprobe 64 over
synthetic clustered centroids, and the 8-bit scan kernel alone:

```
cargo bench -p lucene-ivf --bench routing
```

| centroids | exact f32 | int8 | tree (cover 32) |
|---|---|---|---|
| 6.4k x 300 | 349 µs | 133 µs | 43 µs |
| 27k x 300 | 1572 µs | 552 µs | 62 µs |
| 24k x 100 | 569 µs | 256 µs | 42 µs |

(Euclidean; Cosine is within a few percent.) What makes the 8-bit path fast:

- **A u7 x i8 kernel** (`simd::dot_u7_i8`): the query is coded in 7 bits
  plus an offset of 64, so AVX2 multiplies 32 byte pairs per `maddubs`
  with no sign handling and no saturation, and NEON widens with
  `vmull_s8`/`vmlal_high_s8`. Each row's stored code sum undoes the
  offset. A four-row variant shares the query loads and horizontal sums.
- **A thresholded top-n**: scores below the worst of the current best are
  dropped before they reach a buffer, instead of collecting every score and
  selecting.
- **Branch-free row scoring**: one multiply-add per row with per-row
  constants, the same for Euclidean and dot-product levels.
- **Leaf codes in tree order**: each parent's children are one contiguous
  run of code rows, so the tree scans a few sequential ranges with the
  four-row kernel instead of gathering rows (27k tree route: 145 µs before,
  62 µs after).

The flat int8 scan at 27k x 300 reads 8.6 MB of codes per query, so it is
bound by memory bandwidth (the kernel alone runs at about 21 GB/s here);
beyond that, the tree is what helps.

## Search and build path micro-benchmark

`benches/ivf_path.rs` (Criterion) covers the rest of the path on synthetic
clusters of 100 (Cosine) and 300 (Euclidean) dimensions: one query against
one 384-vector cluster for each quantizer (`set_centroid`, `score_block`,
`error_bounds`); whole searches of a 60k-vector field by rerank encoding
and by fixed, bounded and calibrated depth; four segments with and without
a shared threshold; and writing a field and merging two segments.

```
cargo bench -p lucene-ivf --bench ivf_path
```

Per probed cluster, before and after (300d Euclidean):

| quantizer | set_centroid | score_block (384 codes) |
|---|---|---|
| bbq | 1095 -> 293 ns | 1071 -> 957 ns |
| osq2 | 1314 -> 307 ns | 1780 -> 1464 ns |
| rbbq | 1098 -> 258 ns | 1009 -> 868 ns |
| erq4 | 587 -> 270 ns | 3321 -> 2389 ns |

Setting a centroid (quantizing the query's residual against it and building
the lookup tables) cost as much as scoring the cluster's codes. The fixes:

- **`simd::residual_stats` and `simd::quantize_residual`**: AVX2 kernels for
  the two passes over `query - centroid` (range, mean, norm and the
  dot-product correction in one pass, then the levels and error). The
  autovectorized loops kept scalar conversions with saturation checks.
- **`simd::nibble_luts`**: all 16 entries of two groups' tables per three
  shuffles and masks, instead of entry by entry; higher bit planes are the
  first plane shifted. Extended RaBitQ uses the same kernel.
- **Fast-scan accumulation** (`simd::lut16_accumulate32`): looked-up byte
  pairs are added as 16-bit words and separated once at the end (Faiss's
  trick), and the final reduction is vectorized.
  Codes of 64 groups (256 dimensions) and more use an AVX-512 kernel
  (`simd::lut16_accumulate32_wide`, four groups per shuffle, same sums)
  where `lucene_util::cpu::level` allows it. On Sapphire Rapids,
  `cargo bench -p lucene-ivf --bench ivf_path` score_block at 300
  dimensions is 11-25% faster per block, and at 100 dimensions 0-20%,
  which is no gain end to end, hence the threshold. fastText-300 990k
  cosine (`knn_formats ... --formats ivf`, 3 alternating rounds with
  `LUCENE_RUST_SIMD=avx2` as the baseline): calibrated, recall 0.949,
  970 to 843 us; nprobe 64, recall 0.901, 408 to 343 us.
- **Candidates in a buffer, cut back by selection** rather than a heap:
  a scan offers every code that beats the cut-off.
- **Rerank**: f32 vectors are scored in place (no copy), and a fixed-depth
  rerank reads its candidates in file order.
- **Build**: the neighbourhood refine rounds score each cluster's members
  against its neighbours with the tiled full-assignment kernel, measured
  from the own centroid so Euclidean keeps its precision.

End to end (`ivf_bench`, one query at a time, same index settings, recall
unchanged to the fourth decimal):

| data | quantizer | nprobe / rerank | latency before -> after |
|---|---|---|---|
| GloVe-100 | bbq | 32 / 100 | 89 -> 76 µs |
| GloVe-100 | bbq | 128 / auto | 290 -> 233 µs |
| GloVe-100 | osq2 | 128 / 100 | 319 -> 253 µs |
| GloVe-100 | erq4 | 128 / 100 | 479 -> 420 µs |
| fastText-300 | bbq | 128 / 100 | 553 -> 417 µs |
| fastText-300 | erq4 | 128 / auto | 1481 -> 1136 µs |

Builds got 4-14% faster (fastText bbq 83.6 -> 74.3 s). What remains in a
build is mostly Lucene's document quantization (`OptimizedScalarQuantizer`,
kept bit-exact with Java) and the full Lloyd rounds, which already use the
tiled kernel.

## Through the index: `IvfVectorsFormat`

`knn_formats` (lucene-search example) writes one float field through
`IndexWriter`, opens it with `DirectoryReader` and searches with
`knn_top_hits`, so the numbers include segment opening, the per-segment
collector and doc-id mapping. GloVe-100 (390k, cosine), one flushed segment,
1000 queries after 200 warm-up ones, one thread, k = 10. HNSW is
`Lucene99HnswVectorsFormat(16, 100)` searched with 100 candidates; IVF is
`IvfVectorsFormat(384, encoding)` (BBQ codes) with its calibrated
probe count (`nprobes=0`) or a fixed one.

| format | setting | build | index | recall@10 | mean | p99 |
|---|---|---|---|---|---|---|
| HNSW | ef 100 | 174.2 s | 175.0 MB | 0.889 | 997 µs | 1410 µs |
| HNSW int7 | ef 100 | 171.2 s | 220.2 MB | 0.889 | 1024 µs | 1917 µs |
| IVF f32 | calibrated | 5.8 s | 330.7 MB | 0.968 | 314 µs | 445 µs |
| IVF f32 | nprobes 32 | 5.8 s | 330.7 MB | 0.907 | 163 µs | 238 µs |
| IVF f32 | nprobes 64 | 5.8 s | 330.7 MB | 0.947 | 222 µs | 315 µs |
| IVF f16 | calibrated | 5.3 s | 252.7 MB | 0.968 | 360 µs | 523 µs |
| IVF f16 | nprobes 32 | 5.3 s | 252.7 MB | 0.907 | 182 µs | 302 µs |

IVF builds 30x faster and searches 3-6x faster at higher recall. Its index is
larger because the rerank vectors (`.ivv`) sit beside the flat vectors
(`.vec`) every format keeps; half-float rerank vectors cut that by a quarter
and cost a little latency, since their hits are rescored on the raw vectors
for exact scores.

The same data flushed as 8 segments and force-merged to one (the merge
reuses the largest source's clusters and reranks from the merged raw
vectors):

| format | setting | build + merge | index | recall@10 | mean | p99 |
|---|---|---|---|---|---|---|
| HNSW | ef 100 | 187.2 s | 174.3 MB | 0.886 | 1078 µs | 1540 µs |
| IVF f32 | calibrated | 10.2 s | 332.3 MB | 0.955 | 314 µs | 405 µs |
| IVF f32 | nprobes 32 | 10.2 s | 332.3 MB | 0.904 | 144 µs | 202 µs |
| IVF f16 | calibrated | 9.7 s | 254.3 MB | 0.955 | 319 µs | 404 µs |
| IVF f16 | nprobes 32 | 9.7 s | 254.3 MB | 0.904 | 147 µs | 208 µs |

Reused clusters lose a little recall at the calibrated depth (0.968 to
0.955), as in the merge-stream run above; HNSW pays a full graph rebuild.

```text
cargo run --release -p lucene-search --example knn_formats -- <prefix> cosine \
  --formats hnsw,ivf,ivf-f16 --nprobes 0,32 --segments 8 --merge
```

### Cluster code quantizers

`KnnVectorsFormat::ivf_quantized` (the plugin's `parameters.quantizer`) picks
the cluster codes: `bbq` (default), `osq2`/`osq4`/`osq7`, `erq2` to `erq8`.
GloVe-100 390k, **l2** (truth regenerated with `prepare.py ... l2`), one
segment, f16 rerank vectors, 1000 queries:

| quantizer | index | nprobes 64 | nprobes 128 | nprobes 192 |
|---|---|---|---|---|
| bbq | 255.0 MB | 0.930 @ 238 µs | 0.958 @ 412 µs | 0.962 @ 659 µs |
| osq2 | 261.9 MB | 0.935 @ 210 µs | 0.966 @ 340 µs | 0.973 @ 402 µs |
| erq2 | 260.2 MB | 0.940 @ 300 µs | 0.971 @ 332 µs | 0.978 @ 510 µs |
| erq4 | 274.0 MB | 0.947 @ 375 µs | 0.982 @ 614 µs | 0.990 @ 954 µs |

BBQ levels off near 0.96: its error bounds are loose, so the automatic
rerank stops before the true neighbours are reached. Two-bit codes (osq2,
erq2) cost 2-3% more index and are faster and more accurate than BBQ from
128 probes; erq4 is the one that reaches 0.98-0.99. osq4/osq7/erq8 add bytes
and latency for no recall over erq4.

```text
cargo run --release -p lucene-search --example knn_formats -- <prefix> l2 \
  --formats ivf-f16,ivf-osq2,ivf-erq2,ivf-erq4 --nprobes 64,128,192
```

### Picking the quantizer per segment (`auto`)

Which codes are cheapest at a given recall depends on the data. GloVe-100
390k and fastText-300 1M, **cosine** truth, one segment, f16 rerank vectors,
calibrated depth (recall@10 / mean µs, 1000 queries):

| quantizer | GloVe-100 | fastText-300 |
|---|---|---|
| bbq | 0.955 / 348 | 0.949 / 1243 |
| osq2 | 0.965 / 296 | 0.937 / 1660 |
| erq2 | 0.973 / 355 | 0.968 / 1763 |
| erq3 | 0.970 / 563 | 0.982 / 2522 |
| erq4 | 0.982 / 710 | 0.983 / 3151 |
| **auto** | **0.973 / 317** (erq2) | **0.949 / 1073** (bbq) |

OSQ at 2 bits beats BBQ on GloVe and loses to it on fastText; erq3 matches
erq4 on fastText at 0.8x the time. `quantizer: auto`
(`lucene_ivf::AutoQuantizer`) encodes each field of at least 10,000
vectors with bbq, osq2 and erq2 to erq4 at every flush and merge,
calibrates each against the same 64 sampled queries, and keeps the one that
reaches the calibration recall (0.95) touching the fewest bytes per query:
the codes its probes scan plus a disk page (`IvfParams::auto_rerank_read_bytes`)
per reranked vector. Each segment records the quantizer it picked, so
readers need nothing new. Builds took 1.2x (fastText) to 1.6x (GloVe) as
long as one quantizer's.

```text
cargo run --release -p lucene-search --example knn_formats -- <prefix> cosine \
  --formats ivf-f16,ivf-osq2,ivf-erq2,ivf-erq3,ivf-erq4,ivf-auto --nprobes 0,64,128,192
```

## Through the OpenSearch plugin

`benchmarks/ivf/plugin_knn_bench.py` loads GloVe-100 (390k, cosine) into one
shard of a single node (2 GB heap), force-merges it to one segment and runs
1000 `knn` queries (k = 10, `_source: false`) one at a time over REST. HNSW
is the `lucene` engine (`m` 16, `ef_construction` 100, the default
`ef_search` 100); IVF is the `extended` engine's `method: ivf`
(`target_cluster_size` 384).

| mapping | setting | load | merge | store | recall@10 | took | client |
|---|---|---|---|---|---|---|---|
| HNSW | ef 100 | 154 s (2.5k docs/s) | 28.7 s | 179 MB | 0.887 | 3.73 ms | 5.97 ms |
| IVF half_float | calibrated | 25 s (15.4k docs/s) | 3.0 s | 257 MB | 0.947 | 2.26 ms | 4.19 ms |
| IVF half_float | nprobes 32 | | | | 0.901 | 1.48 ms | 3.27 ms |
| IVF float | calibrated | 25 s (15.5k docs/s) | 3.5 s | 335 MB | 0.972 | 2.21 ms | 3.64 ms |
| IVF float | nprobes 32 | | | | 0.918 | 1.49 ms | 3.08 ms |

IVF loads 6x faster and merges 9x faster. Most of `took` is a fixed cost
that does not depend on the vectors. In a second run on a fresh copy of the
float index, a `match_all` returning 10 hits took 2.10 ms, and an nprobes 32
search took 2.07 ms. Nearly all of that is the fetch phase, at about 0.14 ms
a hit; the plugin's query-phase timer puts the IVF search itself at about
0.1 ms.

## Filtered searches by tenant size

A filter on one tenant of a shared index (a partition field the index is
sorted by) passes a small share of a segment's documents. Each tenant is
measured against its own exact top 10, one segment, `ivf-f16`, calibrated
depth, 200 queries:

- `u<n>`: `n` random documents, unrelated to the vector space;
- `c<n>-local`: the `n` documents nearest an anchor, queried with the
  query vectors nearest that anchor (a tenant searching its own data);
- `c<n>-any`: the same tenant queried with random query vectors.

```
cargo run --release -p lucene-search --example knn_filtered -- /tmp/data/glove100 --format ivf-f16
cargo run --release -p lucene-search --example knn_filtered -- /tmp/data/glove100 --format hnsw --ef 100
cargo run --release -p lucene-search --example knn_filtered -- /tmp/data/glove100 --subset c10000-any
```

Before, a filtered leaf probed the calibrated depth and doubled it only
until `k` accepted documents were in reach, and a filter passing at most
`10 * k` documents went to an exact scan of the whole field. Now:

- the probes grow until they have scanned 2000 accepted vectors for 10
  hits (growing with the square root of k, capped by what the unfiltered
  search scans), starting from the depth the filter's share of the field
  needs on average;
- a cluster holding no accepted vector is not scored;
- a filter whose documents cost less to score exactly than the planned
  probes cost to scan (8 code scans per exact score) goes to the exact
  scan, which now reads only the documents the filter passes (a 300-document
  tenant of 390k: 1192 to 30 µs).

GloVe-100, 390k (recall@10 / mean µs):

| tenant | docs | IVF before | IVF after | HNSW ef 100 |
|---|---|---|---|---|
| none | 390000 | 0.953 / 353 | 0.953 / 336 | 0.953 / 879 |
| u30 | 28 | 1.000 / 10 | 1.000 / 9 | 1.000 / 14 |
| u300 | 308 | 0.666 / 418 | 1.000 / 27 | 1.000 / 63 |
| u1000 | 977 | 0.811 / 444 | 1.000 / 63 | 1.000 / 445 |
| u3000 | 3007 | 0.899 / 427 | 1.000 / 407 | 1.000 / 737 |
| u10000 | 10145 | 0.946 / 514 | 0.989 / 1622 | 1.000 / 2812 |
| u40000 | 40071 | 0.955 / 586 | 0.955 / 676 | 0.995 / 6741 |
| c30-local | 30 | 1.000 / 16 | 1.000 / 9 | 1.000 / 13 |
| c30-any | 30 | 1.000 / 10 | 1.000 / 10 | 1.000 / 14 |
| c300-local | 300 | 0.875 / 491 | 1.000 / 27 | 1.000 / 66 |
| c300-any | 300 | 0.549 / 1368 | 1.000 / 27 | 1.000 / 60 |
| c1000-local | 1000 | 0.918 / 442 | 1.000 / 75 | 1.000 / 431 |
| c1000-any | 1000 | 0.470 / 732 | 1.000 / 66 | 1.000 / 439 |
| c3000-local | 3000 | 0.959 / 354 | 1.000 / 402 | 1.000 / 675 |
| c3000-any | 3000 | 0.528 / 910 | 1.000 / 457 | 1.000 / 690 |
| c10000-local | 10000 | 0.995 / 445 | 0.997 / 1205 | 1.000 / 1598 |
| c10000-any | 10000 | 0.733 / 506 | 0.962 / 1545 | 1.000 / 2506 |
| c40000-local | 40000 | 0.973 / 481 | 0.973 / 578 | 0.985 / 2148 |
| c40000-any | 40000 | 0.908 / 604 | 0.935 / 1006 | 0.994 / 9194 |

fastText-300, 1M (recall@10 / mean µs):

| tenant | docs | IVF before | IVF after | HNSW ef 100 |
|---|---|---|---|---|
| none | 989999 | 0.944 / 714 | 0.944 / 645 | 0.911 / 557 |
| u30 | 37 | 1.000 / 3371 | 1.000 / 17 | 1.000 / 27 |
| u300 | 303 | 0.631 / 1424 | 1.000 / 45 | 1.000 / 884 |
| u1000 | 994 | 0.760 / 813 | 1.000 / 182 | 1.000 / 1198 |
| u3000 | 2971 | 0.860 / 785 | 1.000 / 680 | 1.000 / 1591 |
| u10000 | 10034 | 0.928 / 857 | 1.000 / 1918 | 1.000 / 6089 |
| u40000 | 40178 | 0.952 / 1002 | 0.969 / 1606 | 0.984 / 6828 |
| c30-local | 30 | 1.000 / 4101 | 1.000 / 21 | 1.000 / 24 |
| c30-any | 30 | 1.000 / 3672 | 1.000 / 21 | 1.000 / 25 |
| c300-local | 300 | 0.918 / 645 | 1.000 / 58 | 1.000 / 852 |
| c300-any | 300 | 0.755 / 24276 | 1.000 / 46 | 1.000 / 843 |
| c1000-local | 1000 | 0.932 / 788 | 1.000 / 223 | 1.000 / 1111 |
| c1000-any | 1000 | 0.511 / 9283 | 1.000 / 244 | 1.000 / 1050 |
| c3000-local | 3000 | 0.943 / 527 | 1.000 / 770 | 0.996 / 955 |
| c3000-any | 3000 | 0.786 / 51224 | 1.000 / 764 | 1.000 / 1537 |
| c10000-local | 10000 | 0.980 / 618 | 1.000 / 2185 | 0.959 / 1071 |
| c10000-any | 10000 | 0.541 / 5237 | 1.000 / 2183 | 1.000 / 5780 |
| c40000-local | 40000 | 0.977 / 650 | 0.980 / 970 | 0.927 / 600 |
| c40000-any | 40000 | 0.567 / 4167 | 0.863 / 3615 | 0.965 / 20317 |

- **Before, IVF lost recall on every tenant below a few percent of the
  segment**: 0.63-0.86 for random tenants of 300-3000 documents, and
  0.47-0.79 for a region of the vector space queried from elsewhere, up to
  10k documents. Its mean latency reached 5-51 ms on fastText where the
  doubling probed every cluster.
- **After, every tenant is at 0.93-1.0** except `c40000-any` on fastText
  (4% of the segment, one region, queried from far away): 0.57 to 0.86.
  There the quantized scores of distant vectors are too close to rank, and
  even 1024 probes reach only 0.88. A tenant queried with its own data
  (`-local`) is at 0.98-1.0.
- Tenants of about 10k documents are slower than before (1.9-2.2 ms
  against 0.6-0.9 ms on fastText), the cost of the recall they lacked.
- HNSW keeps its recall under any filter: Lucene's visited limit falls back
  to an exact scan. It is slower for mid-sized tenants (1-7 ms).

An index over just one tenant's documents (`--subset`, standing in for a
graph or clusters built per tenant inside a segment), GloVe:

| tenant | IVF of the tenant | HNSW of the tenant | shared IVF, filtered |
|---|---|---|---|
| u10000 | 0.948 / 55 | 0.983 / 118 | 0.989 / 1622 |
| u40000 | 0.971 / 156 | 0.948 / 204 | 0.955 / 676 |
| c3000-any | 0.878 / 38 | 0.908 / 81 | 1.000 / 457 |
| c10000-any | 0.889 / 59 | 0.907 / 119 | 0.962 / 1545 |
| c40000-any | 0.894 / 81 | 0.943 / 185 | 0.935 / 1006 |
| c10000-local | 0.978 / 53 | 0.988 / 107 | 0.997 / 1205 |

A per-tenant structure is 4-30x faster, but no better on recall: it has
the same approximation as any index of that size, while the filtered
shared index falls back to exact scores for small tenants.

### Mixed tenant sizes at 8M vectors

`cargo run --release -p lucene-search --example knn_tenants -- <glove prefix>
<cache prefix> [--format ivf-f16|hnsw] [--ef L] [--partition-clusters]`
builds 8M vectors (GloVe-100 plus per-dimension jitter, normalized, cosine)
in four segments of 2M, each sorted by tenant: tenants of 1.5M, 400k, 40k,
4k, 300 and 30 documents (each class random or one region of the space,
queried from anywhere or from inside the region). IVF searches ask for 10
candidates, HNSW (m 16, beam 100) for 100. Recall@10 / mean ms, 100 queries
per tenant, on a 4-core box:

| tenant | IVF before | IVF filtered | IVF clusters per partition | HNSW |
|---|---|---|---|---|
| none (unfiltered) | 0.978 / 13.4 | 0.971 / 9.4 | 0.982 / 21.4 | 0.831 / 2.2 |
| random1500000-any | 0.953 / 16.9 | 0.952 / 17.6 | 0.985 / 12.0 | 0.921 / 26.1 |
| region1500000-any | 0.945 / 17.1 | 0.947 / 15.9 | 0.975 / 9.9 | 0.858 / 17.6 |
| region1500000-local | 0.997 / 16.2 | 0.995 / 14.6 | 0.996 / 10.0 | 0.951 / 6.3 |
| random400000-any | 0.945 / 12.2 | 0.955 / 15.5 | 0.972 / 4.1 | 0.963 / 48.1 |
| region400000-any | 0.910 / 12.7 | 0.948 / 15.0 | 0.941 / 3.6 | 0.912 / 37.5 |
| region400000-local | 0.979 / 13.7 | 0.979 / 12.9 | 0.969 / 3.8 | 0.963 / 7.4 |
| random40000-any | 0.875 / 9.5 | 1.000 / 2.9 | 1.000 / 2.8 | 1.000 / 11.4 |
| region40000-any | 0.720 / 24.9 | 1.000 / 2.6 | 1.000 / 2.7 | 0.999 / 11.2 |
| region40000-local | 0.988 / 11.6 | 1.000 / 2.6 | 1.000 / 2.7 | 0.998 / 4.9 |
| random4000-any | 0.934 / 39.1 | 1.000 / 0.42 | 1.000 / 0.38 | 1.000 / 7.4 |
| random300-any | 1.000 / 25.5 | 1.000 / 0.16 | 1.000 / 0.11 | 1.000 / 0.18 |
| random30-any | 1.000 / 49.9 | 1.000 / 0.13 | 1.000 / 0.09 | 1.000 / 0.26 |

("IVF before" asked for 100 candidates; at 10 its unfiltered search is
about 9 ms too.)

- The filtered rules hold at this scale for every tenant of 40k documents
  and fewer: 1.0 recall, where the old doubling gave 0.72-0.93 and up to
  50 ms.
- Tenants of 5-20% of the index stay at 0.95 recall but cost more than an
  unfiltered search: most of each probed cluster belongs to other tenants.
- HNSW at this size reaches 0.83 unfiltered at beam 100, and 0.86-0.96 on
  the large tenants at 6-48 ms: its filtered walk visits many rejected
  nodes before Lucene's visit limit sends it to an exact scan.
- Index sizes: IVF 5.2 GB (f16 rerank vectors beside the raw ones), HNSW
  3.7 GB. Builds: IVF 387 s, IVF per partition 120 s, HNSW 3834 s
  (graphs built single-threaded at flush).

### Clusters per partition

`IndexWriter::set_ivf_partition_clusters(Some(min_vectors))` clusters each
partition of the index sort's first field on its own at flush and merge
(partitions under `min_vectors` together with their neighbours, here
20,000). Each field records it (`.ivm` version 5), the reader keeps each
cluster's document range, and a filtered search probes only the clusters
whose range holds a document it passes, with the calibrated share of
them. A partition of at most 16 times the accepted
target is scored exactly.
In the OpenSearch plugin this is the index setting
`index.knn.ivf.partition_min_vectors` (`min_vectors`, `0` off, the
default), with `index.sort.field` set to the partition field.

- Large tenants search 1.5-4.2x faster with recall up to 0.97-0.985 (the
  400k region queried from far away stays at 0.94).
- Unfiltered searches take 2.3x longer (21 ms against 9.4 ms): a query's
  nearest documents spread over every partition's clusters. The setting
  fits indexes that are always searched within one tenant.
- Builds are 3x faster: k-means runs on each partition alone.

### Composite partition keys

An index sorted by several keys (`index.sort.field: [tenant, sub]`) keeps a
tenant's documents in one run and, inside it, each sub-partition's
documents in a shorter run. A conjunction's required, non-scoring clauses
on a leading prefix of the sort keys (keyword terms; long or date ranges,
the last key of the prefix may be a range) are matched as that one run, by
binary searches over the keys' doc values, instead of by intersecting
their postings or points. Clusters per partition still follow the first
key. `knn_tenants --sort tenant|composite|time` gives every document one of
16 sub-partitions and queries each tenant alone and with one sub-partition
(`+sub`). IVF clusters per partition, 4 segments, mean ms of two builds,
recall@10 in parentheses where below 1.0:

| query | sort by tenant | sort by tenant, sub | sort by time (8 segments) |
|---|---|---|---|
| none | 10.0-10.3 (0.99) | 11.9-14.5 (0.99) | 10.8 (0.99) |
| 1.5M tenant | 6.6-7.4 (0.98) | 7.0-7.4 (0.98) | 16.4-17.1 (0.92-0.99) |
| 1.5M tenant + sub (94k) | 28-33 | 4.7-5.6 | 16.6-25.9 |
| 400k tenant | 2.4-2.8 (0.96-0.98) | 2.3-2.7 (0.96-0.97) | 11.6-28.4 (0.97) |
| 400k tenant + sub (25k) | 2.7-4.2 | 1.3-1.5 | 5.9-6.5 |
| 40k tenant | 2.0-2.5 | 2.0-2.1 | 4.7-7.6 |
| 40k tenant + sub (2.5k) | 0.37-0.38 | 0.22-0.41 | 1.6-1.9 |
| 4k, 300, 30 tenant | 0.29, 0.10, 0.08 | 0.29, 0.10, 0.08 | 1.2, 0.14, 0.10 |

- A filter on the second key costs 4-6x less on large tenants: the
  sub-partition's run is a few hundred kilobytes of vectors scored exactly,
  where the tenant-only sort intersects the sub-partition's postings across
  the tenant and probes its clusters.
- Tenant-only searches are unchanged within noise; the build took
  109-117 s against 98-100 s. Unfiltered searches were 1.2-1.4x slower,
  which this layout's always-filtered searches do not see.
- Sorting by time alone scatters every tenant: 2-12x slower tenant
  searches. (The time-sorted build at 4 segments ran out of memory on the
  15 GB box; it ran with 8.)

Data: `/mnt/project-files/perf/composite-partition/`.

## Nested fields: the best child per parent

A nested search (`knn_top_hits_nested_with`, the plugin's nested `knn`)
returns the best child of each of the best `k` parents. On an IVF field it
used to score every child exactly; now it probes the clusters as a flat
search does, with the shortlist widened by the children per parent (at most
64 times) so siblings do not crowd other parents out, a rerank that keeps
one child per parent and stops once no candidate can beat the `k`-th
parent; while fewer than `k` parents are found, the probes deepen and the
shortlist widens (so a parent with thousands of alike children cannot
crowd the rest out), up to an exhaustive search. A filter keeps the flat
search's rules (above); in a segment where some documents have no vector
(the parents), a filter passing more than 4096 live documents is counted
at 4096 points spread over the segment, since walking all of a filter
passing half the segment cost more (1.7 ms) than probing.

GloVe-100, 390k vectors grouped into parents of 1-8 consecutive rows, one
segment, `ivf-f16`, 200 queries, recall of the exact best 10 parents.
`local` rows are ordered by nearest of 4096 sampled centres, so a parent's
children are alike, as a document's chunks are; `random` rows are not.
Data: `/mnt/project-files/perf/ivf-nested/` (`conv.py` makes the files).

```
cargo run --release -p lucene-search --example knn_nested -- /tmp/data/glove100 --order local --nprobes 0,32,128
cargo run --release -p lucene-search --example knn_nested -- /tmp/data/glove100 --order local --filter half
cargo run --release -p lucene-search --example knn_nested -- /tmp/data/glove100 --format hnsw --ef 100,200
```

| parents | children | filter | before (exact) | IVF calibrated | IVF 32 probes | HNSW ef 200 |
|---|---|---|---|---|---|---|
| 86866 local | 1-8 | none | 1.000 / 16236 | 0.981 / 487 | 0.930 / 352 | 0.947 / 518 |
| 86866 random | 1-8 | none | | 0.982 / 454 | 0.917 / 267 | |
| 86866 local | 1-8 | half the children | 1.000 / 10289 | 0.977 / 1257 | 0.908 / 907 | 0.934 / 1214 (ef 100) |
| 86866 random | 1-8 | half the children | | 0.984 / 1173 | 0.905 / 880 | |
| 23750 local | 1-32 | none | | 0.984 / 1090 | 0.930 / 596 | |
| 390000 local | 1 | none | | 0.957 / 261 | | |

Recall @10 / mean µs. About 700 µs of every filtered row is building the
filter's bit set, the same for HNSW.
