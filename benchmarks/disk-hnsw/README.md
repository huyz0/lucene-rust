# Disk HNSW benchmarks

Recall, CPU time, page misses and round trips of `lucene-disk-hnsw`
against stock Lucene's HNSW layout and against `lucene-ivf`, each reading
its files through the same simulated page cache under a memory budget.

## Data

The IVF benchmark's sets and ground truth
([`benchmarks/ivf/README.md`](../ivf/README.md)):

```
python3 ../ivf/prepare.py glove-wiki-gigaword-100.gz /tmp/data/glove100 cosine
python3 ../ivf/prepare.py fasttext-wiki-news-subwords-300.gz /tmp/data/ft300 l2
```

## Runs

```
cargo run --release -p lucene-disk-hnsw --example disk_hnsw_bench -- /tmp/data/glove100 cosine \
  --budgets 8,16,24,32,48,64,128,inf --lists 80,160,320 --beams 4,8 > glove.csv
python3 summarize.py glove.csv 0.85 500
```

The example's header lists the indexes and options. Every index reads its
files through one page cache per run (4 KiB pages, CLOCK eviction) holding
the budget minus the structures the index keeps in memory anyway; 1000
warm-up queries run before the 1000 measured ones. The cache counts the
pages fetched and the round trips a search waits for: a read that misses
is one, and so are all ranges announced together with `prefetch` before
they are read. `summarize.py` picks, per budget and index, the setting
with the lowest CPU time plus round trips at a given cost among those
reaching a recall floor.

Results: [`docs/benchmarks/disk-hnsw.md`](../../docs/benchmarks/disk-hnsw.md).
