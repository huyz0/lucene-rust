# IVF benchmarks

Recall and single-thread latency of `lucene-ivf` against Lucene's HNSW + BBQ
(this port's `Lucene104HnswScalarQuantizedVectorsFormat` with 1-bit codes)
and against Faiss, on public word-vector sets the container can download.

## Data

[gensim-data](https://github.com/RaRe-Technologies/gensim-data) releases,
which are GitHub release assets:

| set | vectors | dim | metric |
|---|---|---|---|
| `glove-wiki-gigaword-100` | 390 000 base + 10 000 queries | 100 | cosine |
| `fasttext-wiki-news-subwords-300` | 989 999 base + 10 000 queries | 300 | L2 |

```
curl -LO https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
python3 prepare.py glove-wiki-gigaword-100.gz /tmp/data/glove100 cosine
```

`prepare.py` drops zero and duplicate rows, holds out 10 000 random vectors
as queries and computes the exact 100 nearest neighbours with
`faiss.IndexFlat` (`pip install faiss-cpu numpy`).

## Runs

```
cargo run --release -p lucene-ivf --example ivf_bench -- /tmp/data/glove100 cosine --hnsw
python3 faiss_bench.py /tmp/data/glove100 cosine --nlist <clusters ivf_bench reported>
```

Both print `index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb`.
Builds use every core; searches run one query at a time on one thread.
`pinned_mb` is what has to stay in memory or on local NVMe for a search to
avoid remote reads (IVF: meta and postings; HNSW: 1-bit codes and level-0
adjacency; Faiss: the serialized index without its refine store);
`vectors_mb` is the full-precision vectors, read only for reranking.

Results: [`docs/benchmarks/ivf.md`](../../docs/benchmarks/ivf.md).
