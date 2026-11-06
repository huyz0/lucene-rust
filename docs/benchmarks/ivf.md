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
| IVF-BBQ (1-bit) | nprobe 64, rerank 300 | 0.928 | 245 | 3.5 | 15.6 |
| IVF-BBQ (1-bit) | nprobe 128, rerank 300 | 0.952 | 359 | 3.5 | 15.6 |
| IVF-OSQ 2-bit | nprobe 64, rerank 100 | 0.944 | 246 | 3.4 | 22.2 |
| IVF-OSQ 4-bit | nprobe 64, rerank 100 | 0.949 | 313 | 2.7 | 33.4 |
| IVF-OSQ 4-bit | nprobe 128, rerank 100 | 0.981 | 617 | 2.7 | 33.4 |
| IVF-SQ int7 | nprobe 64, rerank 100 | 0.949 | 288 | 2.3 | 42.8 |
| IVF-SQ int4 | nprobe 64, rerank 100 | 0.949 | 334 | 2.7 | 23.3 |
| HNSW + BBQ (M 16, beam 100) | ef 320 | 0.798 | 939 | 36.0 | 44.6 |
| HNSW + BBQ (M 16, beam 100) | ef 640 | 0.870 | 1843 | 36.0 | 44.6 |
| Faiss IVF-Flat | nprobe 64 | 0.963 | 613 | 14.7 | 159.8 |
| Faiss IVF-SQ8 | nprobe 64, rerank 100 | 0.963 | 2163 | 14.5 | 42.8 |
| Faiss IVF-RaBitQ (qb 4) | nprobe 64, rerank 300 | 0.919 | 607 | 15.6 | 12.0 |
| Faiss IVF-PQ50x4 fast-scan | nprobe 64, rerank 300 | 0.953 | 223 | 14.6 | 14.2 |

## fastText-300

| index | setting | recall@10 | latency µs | build s | pinned MB |
|---|---|---|---|---|---|
| IVF-BBQ (1-bit) | nprobe 64, rerank 300 | 0.926 | 893 | 64.1 | 73.7 |
| IVF-BBQ (1-bit) | nprobe 128, rerank 300 | 0.962 | 1075 | 64.1 | 73.7 |
| IVF-OSQ 2-bit | nprobe 128, rerank 100 | 0.962 | 1301 | 53.3 | 117.9 |
| IVF-OSQ 4-bit | nprobe 64, rerank 100 | 0.927 | 1056 | 52.6 | 185.9 |
| IVF-OSQ 4-bit | nprobe 128, rerank 100 | 0.964 | 1642 | 52.6 | 185.9 |
| IVF-SQ int4 | nprobe 128, rerank 100 | 0.964 | 1788 | 49.7 | 164.2 |
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
vectors 8 ahead. At GloVe nprobe 64, BBQ went from 229 to about 135 µs
without rerank and from 428 to 245 µs with rerank 300. The other rows are
unchanged.

- **Against HNSW + BBQ.** IVF-BBQ reaches recall HNSW + BBQ does not reach
  at any ef tried. At HNSW's best recall (0.87 to 0.88) IVF-BBQ is 3 to 10x
  faster, builds 2 to 10x faster and pins a third to nine tenths of the
  memory. A probe is one sequential range read where an HNSW search makes
  hundreds of dependent random reads, which matters more on blob storage.
- **Against Faiss IVF-RaBitQ**, the same 1-bit idea: IVF-BBQ is faster at
  equal recall on both sets, by 1.5 to 2.5x, and builds 5x faster.
- **Against Faiss PQ fast-scan:**
  - On fastText, IVF-BBQ is now ahead: 0.926 at 893 µs against 0.929 at
    1001, and 0.962 at 1075 µs against 0.954 at 1110.
  - On GloVe, PQ fast-scan is still about 1.6x faster at 0.95. The gap is
    partition quality, not the scan: Faiss's flat k-means partitions give
    0.963 recall at nprobe 64 with exact distances, where these give 0.949
    (IVF-Flat against reranked IVF). So PQ reaches 0.95 at nprobe 64 while
    IVF-BBQ needs 128.
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
