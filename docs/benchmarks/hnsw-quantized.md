# HNSW walked on RaBitQ and Extended RaBitQ codes

The `lucene_rust` engine's `hnsw` method (`LuceneRustHnswQuantizedVectorsFormat`)
builds the Lucene99 graph on the raw vectors and stores the quantized codes
next to it (`.vqm`/`.vqd`). A search walks the graph on the codes and reranks
the `ef` candidates on the raw vectors. These numbers compare that with
Lucene's own scalar quantization (OSQ 1, 2, 4 and 7 bits through
Lucene104 HNSW SQ) and with the IVF quantizers.

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.1 GHz with AVX-512, shared, so single runs
  move by about 15%. Builds use 4 threads. Searches run one query at a time on
  one thread, for 1000 queries. Recall@10 is measured against exact ground
  truth.
- Data: GloVe-100 (390 000 x 100, cosine) and fastText-300
  (989 999 x 300, L2), prepared by `benchmarks/ivf/prepare.py`.
- Graph: M 16, beam 100 everywhere.

## Every quantizer on one graph

`crates/lucene-search/examples/hnsw_quant_bench.rs` builds one float graph and
walks it with each quantizer. `codes` is the recall of the code scores alone,
`rerank` the recall after the `ef` candidates are reranked on floats. Bytes are
per vector, corrections included. `rabitq` is BBQ behind a random rotation.

GloVe-100, ef 320:

| quantizer | bytes | codes | rerank | mean us |
|---|---|---|---|---|
| float | 400 | 0.954 | 0.954 | 632 |
| bbq (Lucene 1 bit) | 32 | 0.324 | 0.873 | 637 |
| rabitq | 32 | 0.374 | 0.901 | 723 |
| osq2 (Lucene 2 bit) | 48 | 0.548 | 0.944 | 885 |
| erq2 | 44 | 0.629 | 0.947 | 529 |
| erq3 | 60 | 0.778 | 0.952 | 629 |
| osq4 (Lucene 4 bit) | 80 | 0.804 | 0.951 | 485 |
| erq4 | 76 | 0.855 | 0.952 | 714 |
| osq7 (Lucene 7 bit) | 116 | 0.934 | 0.954 | 518 |

fastText-300, ef 640:

| quantizer | bytes | codes | rerank | mean us |
|---|---|---|---|---|
| float | 1200 | 0.923 | 0.923 | 1865 |
| bbq | 56 | 0.481 | 0.902 | 1819 |
| rabitq | 56 | 0.525 | 0.911 | 1783 |
| osq2 | 96 | 0.572 | 0.910 | 2589 |
| erq2 | 92 | 0.717 | 0.920 | 1430 |
| erq3 | 132 | 0.812 | 0.922 | 1683 |
| osq4 | 176 | 0.736 | 0.920 | 1257 |
| erq4 | 172 | 0.861 | 0.922 | 1820 |
| osq7 | 316 | 0.853 | 0.924 | 1377 |

A graph built on the erq2 codes instead of the floats gives the same recall
(GloVe ef 320: 0.945 against 0.947) and takes twice as long to build, so the
graph stays on floats. That also keeps the graph file byte-for-byte the plain
HNSW one.

## Through the index

`crates/lucene-search/examples/knn_formats.rs` on GloVe-100, one segment.
Index size includes the raw vectors (156 MB).

| format | ef / nprobes | recall@10 | mean us | p99 us | index MB |
|---|---|---|---|---|---|
| hnsw (float) | 400 | 0.959 | 864 | 1346 | 177 |
| hnsw-sq1 (Lucene) | 400 | 0.859 | 634 | 818 | 188 |
| hnsw rabitq | 400 | 0.912 | 922 | 1911 | 189 |
| hnsw-sq2 (Lucene) | 400 | 0.935 | 774 | 1263 | 193 |
| hnsw erq2 | 400 | 0.953 | 680 | 1012 | 194 |
| hnsw erq3 | 400 | 0.959 | 792 | 1261 | 200 |
| hnsw-sq4 (Lucene) | 400 | 0.957 | 610 | 1097 | 202 |
| hnsw-int7 (Lucene) | 400 | 0.960 | 563 | 934 | 222 |
| hnsw erq2 | 200 | 0.926 | 308 | 478 | 194 |
| hnsw-sq2 (Lucene) | 200 | 0.890 | 363 | 919 | 193 |
| ivf erq2 | 64 | 0.948 | 252 | 474 | 259 |
| ivf rabitq | 64 | 0.942 | 217 | 322 | 254 |

Build time is the same for every HNSW format (105-113 s). Encoding the codes
adds 1-7 s on GloVe and 11-43 s on fastText, rising with the bit count.

## Findings

- At 2 bits, erq2 beats Lucene's 2-bit OSQ on both sets: +0.018 recall on
  GloVe at ef 400 with lower latency, and +0.036 at ef 200. Its codes are
  smaller too (44 against 48 bytes on GloVe).
- At 1 bit, rabitq beats Lucene's 1-bit OSQ by +0.05 recall at the same code
  size, at some latency cost from the rotation.
- erq2 gets the recall of Lucene's 4-bit OSQ with about half the code bytes.
  That is the memory case for it: the codes and the graph are what stays in
  RAM, and the raw vectors are read only to rerank.
- At 4 bits and above every quantizer converges after rerank, and Lucene's
  4-bit and 7-bit OSQ are the fastest. So the `hnsw` method offers `rabitq`
  and `erq2`-`erq8`, and Lucene's OSQ stays the `lucene` engine's `encoder`.
- IVF with the same quantizers reaches the same recall faster on this
  single segment of 390 000 vectors, with a larger index.
