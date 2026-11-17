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
