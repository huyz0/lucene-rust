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
| Bulk indexing | 33.6 s (29,717 docs/s) | 49.3 s (20,301 docs/s) |
| Store after refresh | 484 MiB, 12 segments | 447 MiB, 5 segments |
| Force-merge to one segment | 11.0 s | 13.9 s |
| Node RSS idle / after load / after search | 2.27GiB / 2.428GiB / 2.516GiB | 2.294GiB / 3.015GiB / 2.735GiB |

The Rust engine indexes at about 0.68 of stock's rate. Each shard writer
keeps one buffer per indexing thread (`ConcurrentIndexWriter`), as Lucene
keeps one `DocumentsWriterPerThread` per thread, and the plugin encodes and
inverts each document before it reaches the writer.

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

| Request | Stock, merged | Plugin, merged | Ratio | Stock, 12 segments | Plugin, 5 segments | Plugin, Java search |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent (t0) | 3.8 | 3.8 | 1.02× | 8.3 | 6.6 | 5.1 |
| term, mid (t2s) | 4.6 | 3.9 | 1.17× | 6.7 | 6.2 | 4.4 |
| term, rare (t1z4) | 4.2 | 2.8 | 1.49× | 5.6 | 5.2 | 3.9 |
| match OR, 2 frequent | 10.9 | 8.1 | 1.35× | 12.8 | 10.5 | 11.0 |
| match OR, 4 mixed | 7.3 | 5.8 | 1.25× | 8.2 | 7.3 | 6.3 |
| match AND, 2 frequent | 10.5 | 7.5 | 1.40× | 11.4 | 9.2 | 9.5 |
| match AND, frequent + mid | 5.0 | 4.3 | 1.17× | 5.8 | 6.0 | 4.8 |
| match_phrase, 2 frequent | 88.1 | 50.8 | 1.74× | 83.5 | 49.9 | 93.8 |
| multi_match body+title | 9.3 | 6.3 | 1.49× | 12.8 | 7.5 | 8.4 |
| bool match + range filter | 6.2 | 5.4 | 1.13× | 9.4 | 6.8 | 6.9 |
| range on long | 4.7 | 4.2 | 1.12× | 6.3 | 6.1 | 4.4 |
| term on keyword | 3.6 | 3.4 | 1.06× | 4.9 | 4.4 | 3.8 |
| prefix on title | 8.2 | 4.7 | 1.74× | 8.1 | 5.8 | 6.9 |
| match_all, sort by num | 5.8 | 4.8 | 1.21× | 6.6 | 6.1 | 3.7 |
| match, sort by price | 5.7 | 5.2 | 1.09× | 6.6 | 5.8 | 7.8 |
| count, track_total_hits | 2.7 | 3.4 | 0.79× | 3.4 | 3.3 | 4.0 |
| terms agg on category | 4.2 | 4.0 | 1.05× | 5.9 | 5.3 | 4.6 |
| terms agg on tag, filtered | 9.7 | 7.2 | 1.35× | 18.6 | 8.4 | 13.2 |
| date_histogram by month | 5.6 | 6.3 | 0.89× | 7.3 | 6.2 | 4.8 |
| stats on price | 12.9 | 10.4 | 1.25× | 12.1 | 9.8 | 9.5 |

Every request returns the same hit count on both nodes.

After the merge, native search is as fast or faster on 18 of the 20 shapes,
with a median of 1.19× and up to 1.74× (`match_phrase`, `prefix`). The two
slower shapes are counting a near-full two-term disjunction
(`track_total_hits`, `size: 0`) and the monthly `date_histogram`, at 0.79×
and 0.89×.

At this size a round trip is 3-90 ms and most of it is OpenSearch's own
request handling, so ratios near 1.0 are within the run-to-run spread
(about ±10%; a run the same day measured a median of 1.38×).
