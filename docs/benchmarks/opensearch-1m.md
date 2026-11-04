# The plugin against stock OpenSearch, 1M documents

`scripts/bench-opensearch.sh 1000000` (`opensearch-plugin/e2e/bench_nodes.py`)
indexes one synthetic corpus into a stock OpenSearch 3.9.0 node and into a node
with the plugin serving the index on the Rust engine
(`lucene_rust.engine.default: true`) and native search. The nodes run one
after the other, each alone, with a 2 GB heap, on a 4-core, 15 GB container
(2026-11-04).

The corpus: 1,000,000 documents, a body of 20-120 terms and a title of 2-8
terms drawn from a Zipf vocabulary of 200,000 terms (`t0` the most frequent),
a Zipf keyword `tag` (1,000 values), a uniform keyword `category` (50), a
`long`, a `double` and a `date`. One shard, no replica, sent as 200 bulk
requests of 5,000 documents by 4 parallel clients, refresh off until the end.

## Indexing

| | Stock | Plugin (Rust engine) |
|---|---:|---:|
| Bulk indexing | 28.5 s (35,068 docs/s) | 31.0 s (32,238 docs/s) |
| Bulk indexing and the final refresh | 32.7 s | 31.7 s |
| Store after refresh | 484 MiB, 12 segments | 541 MiB, 15 segments |
| Force-merge to one segment | 10.8 s | 10.5 s |
| Node RSS idle / after load / after search | 2.274GiB / 2.402GiB / 2.506GiB | 2.297GiB / 3.038GiB / 2.997GiB |

The Rust engine acknowledges bulk requests at about 0.92 of stock's rate,
within twice the run-to-run spread of a load (about ±5%), and is searchable
sooner, because its final refresh is short. Each shard writer keeps one
buffer per indexing thread (`ConcurrentIndexWriter`), as Lucene keeps one
`DocumentsWriterPerThread` per thread. A refresh commits without stopping
indexing, and merges run on the writer's own merge threads.

The native library allocates with mimalloc (lucene-ffi's `mimalloc` feature,
on in the plugin build). With glibc's malloc, every JVM thread that indexed
kept its own arena and the memory freed in it: the same load left the node at
4.15 GiB.

## Search

Median REST round trip in milliseconds over 50 requests per shape after 30
warm-up requests, `request_cache=false`, top 10 hits. The merged columns
compare the two engines on the same one-segment shape and are the fair ones;
the first pass on each node ran right after loading, and stock's first pass
also overlapped its background merges. "Plugin, Java search" is the plugin
node's index with native search off.

| Request | Stock, merged | Plugin, merged | Ratio | Stock, 12 segments | Plugin, 15 segments | Plugin, Java search |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent (t0) | 5.0 | 2.5 | 1.98× | 7.8 | 6.7 | 4.8 |
| term, mid (t2s) | 3.7 | 2.6 | 1.44× | 5.6 | 5.2 | 2.9 |
| term, rare (t1z4) | 3.5 | 2.4 | 1.49× | 5.3 | 4.0 | 3.1 |
| match OR, 2 frequent | 9.8 | 7.3 | 1.33× | 12.7 | 9.0 | 9.7 |
| match OR, 4 mixed | 6.1 | 4.9 | 1.25× | 8.0 | 6.3 | 5.3 |
| match AND, 2 frequent | 9.8 | 7.5 | 1.30× | 11.0 | 8.5 | 8.8 |
| match AND, frequent + mid | 4.2 | 3.7 | 1.15× | 5.9 | 4.6 | 4.1 |
| match_phrase, 2 frequent | 84.4 | 48.2 | 1.75× | 79.1 | 48.6 | 91.4 |
| multi_match body+title | 8.7 | 5.2 | 1.66× | 9.8 | 6.4 | 8.8 |
| bool match + range filter | 5.1 | 4.8 | 1.05× | 6.3 | 6.2 | 5.4 |
| range on long | 4.6 | 3.3 | 1.39× | 5.0 | 5.2 | 3.8 |
| term on keyword | 3.1 | 1.9 | 1.63× | 4.1 | 2.9 | 2.4 |
| prefix on title | 6.7 | 3.8 | 1.75× | 7.7 | 5.1 | 6.9 |
| match_all, sort by num | 4.1 | 3.2 | 1.28× | 6.3 | 4.8 | 4.1 |
| match, sort by price | 5.0 | 2.8 | 1.77× | 7.0 | 3.8 | 5.4 |
| count, track_total_hits | 2.1 | 1.9 | 1.09× | 3.2 | 2.4 | 4.9 |
| terms agg on category | 3.3 | 2.4 | 1.38× | 5.7 | 3.9 | 3.1 |
| terms agg on tag, filtered | 9.5 | 5.4 | 1.74× | 16.3 | 5.7 | 10.1 |
| date_histogram by month | 5.6 | 5.9 | 0.94× | 6.2 | 6.6 | 5.0 |
| stats on price | 12.3 | 7.8 | 1.59× | 13.2 | 7.6 | 10.8 |

Every request returns the same hit count on both nodes.

After the merge, native search is faster on 19 of the 20 shapes, with a
median of 1.42× and up to 1.98× (frequent `term`). The one slower shape is the
monthly `date_histogram`, at 0.94×.

At this size a round trip is 3-90 ms and most of it is OpenSearch's own
request handling, so ratios near 1.0 are within the run-to-run spread
(about ±10%; an earlier run the same day measured a median of 1.19×).
