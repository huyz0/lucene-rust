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
