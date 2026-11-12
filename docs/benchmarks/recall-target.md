# Recall-targeted k-NN search

`lucene_search::knn_recall` tunes a field's search depth to a target recall
from sampled live queries, each checked against an exact scan of the field;
the plugin turns it on per index with `index.lucene_rust.knn.target_recall`
(off by default). This page is the evidence for its defaults.

Data: GloVe-100 (gensim-data `glove-wiki-gigaword-100`, cosine, 390k base
vectors, 10k queries removed from the base, `benchmarks/ivf/prepare.py`),
4 flushed segments, k = 10, one search thread, x86-64 (4 vCPU). Formats:
`ivf-f16` (IVF, f16 rerank vectors, calibrated depth by default) and `hnsw`
(m 16, ef_construction 100, 100 candidates by default). Each target was tuned
on 20,000 queries drawn from queries 2,000-9,999, then measured on queries
0-1,999 (held out) at the tuned effort with no sampling ("fixed"). The
tuning ran with a check budget of 0.5 so that it sees in seconds about as
many samples as the default budget (0.05) collects in ten times the traffic.

```text
cargo run --release -p lucene-search --example knn_formats -- <glove100> cosine \
  --formats ivf-f16,hnsw --segments 4 --reuse --dir <dir> --queries 2000 --ef 100 \
  --targets 0.9,0.95,0.98 --tune-queries 20000 --budget 0.5
```

| format | target | tuned effort | recall@10 (held out) | mean us | default recall | default mean us |
|---|---|---|---|---|---|---|
| ivf-f16 | 0.90 | 0.25 | 0.926 | 346 | 0.988 | 1155 |
| ivf-f16 | 0.95 | 0.5 | 0.968 | 612 | 0.988 | 1155 |
| ivf-f16 | 0.98 | 1.41 | 0.993 | 1536 | 0.988 | 1155 |
| hnsw | 0.90 | 0.71 (71 candidates) | 0.934 | 617 | 0.945 | 676 |
| hnsw | 0.95 | 2.0 (200) | 0.961 | 1073 | 0.945 | 676 |
| hnsw | 0.98 | 8.0 (800) | 0.986 | 2151 | 0.945 | 676 |

Every target was met on the held-out queries. IVF's calibrated default is
deeper than 0.95 needs across four segments (each segment calibrates to 0.95
on its own), so a 0.95 target runs 1.9x faster and a 0.90 one 3.3x; HNSW's
default 100 candidates miss 0.95, which the tuner reaches by doubling them.

**Why a confidence bound.** A query's recall is mostly 1.0 with rare hard
queries at 0.4-0.6. Picking the first level whose *mean* reached the target
over a few dozen samples chose levels that looked 0.02-0.03 better than they
were (a lucky run with no hard query in it): HNSW at a 0.98 target settled at
2.83x and measured 0.969 on held-out queries. A level now reaches the target
when its mean less one standard error does, the variance taken as at least
half a coin flip's at the mean, and the run above meets every target.

**Overhead.** An exact check scans every vector of the field: about 20 ms
here (156 MB of floats, memory-bandwidth bound), 30-60x a search. The checks
are capped at 5% of the searches' time after the first 64 samples (which may
take as long as the searches), so on a large shard the budget, not the
sample rate, sets how often they run. Run inline (`knn_top_hits_tuned`) a
sampled query pays the scan in its own latency (p99 20-26 ms in these runs);
the plugin checks on a thread of its own after returning the hits, so the
query's latency is the tuned search's. Unsampled queries pay two atomic
loads and one counter increment.

Raw CSVs: `/mnt/project-files/perf/recall-target/` in the project notes
(`run4.csv` is the table; `run3.csv` the run before the confidence bound).
