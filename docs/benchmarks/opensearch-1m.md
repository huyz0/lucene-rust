# The plugin against stock OpenSearch, 1M documents

`scripts/bench-opensearch.sh 1000000` (`opensearch-plugin/e2e/bench_nodes.py`)
indexes one synthetic corpus into a stock OpenSearch 3.9.0 node and into a node
with the plugin serving the index on the Rust engine
(`lucene_rust.engine.default: true`) and native search. The nodes run one
after the other, each alone, with a 2 GB heap, on a 4-core, 15 GB container
(2026-11-03).

The corpus: 1,000,000 documents, a body of 20-120 terms and a title of 2-8
terms drawn from a Zipf vocabulary of 200,000 terms (`t0` the most frequent),
a Zipf keyword `tag` (1,000 values), a uniform keyword `category` (50), a
`long`, a `double` and a `date`. One shard, no replica, sent as 200 bulk
requests of 5,000 documents by 4 parallel clients, refresh off until the end.

## Indexing

| | Stock | Plugin (Rust engine) |
|---|---:|---:|
| Bulk indexing | 31.6 s (31,597 docs/s) | 53.9 s (18,548 docs/s) |
| Store after refresh | 484 MiB, 12 segments | 461 MiB, 11 segments |
| Force-merge to one segment | 11.7 s | 10.8 s |
| Node RSS idle / after load / after search | 2.368GiB / 2.535GiB / 2.641GiB | 2.275GiB / 3.951GiB / 3.926GiB |

The Rust engine indexes at about 0.6 of stock's rate. Each shard writer keeps
one buffer per indexing thread (`ConcurrentIndexWriter`), as Lucene keeps one
`DocumentsWriterPerThread` per thread, and the plugin encodes and inverts each
document before it reaches the writer. The plugin node holds about 1.4 GB more
RSS after loading than stock, outside the Java heap.

## Search

Median REST round trip in milliseconds over 50 requests per shape after 30
warm-up requests, `request_cache=false`, top 10 hits. The merged columns
compare the two engines on the same one-segment shape and are the fair ones;
the first pass on each node ran right after loading, and stock's first pass
also overlapped its background merges. "Plugin, Java search" is the plugin
node's index with native search off.

| Request | Stock, merged | Plugin, merged | Ratio | Stock, 12 segments | Plugin, 11 segments | Plugin, Java search |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent (t0) | 4.4 | 2.6 | 1.66× | 8.1 | 6.5 | 5.3 |
| term, mid (t2s) | 4.0 | 2.5 | 1.65× | 6.4 | 5.6 | 4.2 |
| term, rare (t1z4) | 3.5 | 2.1 | 1.63× | 4.9 | 5.3 | 3.6 |
| match OR, 2 frequent | 13.0 | 7.3 | 1.79× | 15.0 | 10.4 | 10.4 |
| match OR, 4 mixed | 6.1 | 4.4 | 1.40× | 8.7 | 7.1 | 6.5 |
| match AND, 2 frequent | 12.2 | 7.8 | 1.57× | 14.1 | 10.0 | 10.7 |
| match AND, frequent + mid | 4.7 | 4.3 | 1.09× | 7.6 | 5.9 | 4.3 |
| match_phrase, 2 frequent | 83.2 | 49.6 | 1.68× | 83.1 | 53.6 | 93.7 |
| multi_match body+title | 9.6 | 5.7 | 1.70× | 10.6 | 7.8 | 8.8 |
| bool match + range filter | 5.2 | 5.1 | 1.01× | 6.7 | 6.8 | 5.2 |
| range on long | 4.5 | 4.6 | 0.98× | 5.8 | 6.3 | 3.8 |
| term on keyword | 2.8 | 2.0 | 1.35× | 4.5 | 4.4 | 2.2 |
| prefix on title | 7.7 | 4.9 | 1.57× | 8.9 | 6.0 | 7.2 |
| match_all, sort by num | 3.7 | 3.5 | 1.03× | 6.9 | 6.8 | 5.1 |
| match, sort by price | 4.8 | 3.5 | 1.36× | 7.7 | 5.5 | 6.8 |
| count, track_total_hits | 1.6 | 2.4 | 0.66× | 4.8 | 2.9 | 2.7 |
| terms agg on category | 3.0 | 3.0 | 1.00× | 6.1 | 4.9 | 3.7 |
| terms agg on tag, filtered | 9.1 | 5.1 | 1.79× | 13.9 | 6.5 | 11.0 |
| date_histogram by month | 4.9 | 4.7 | 1.04× | 6.6 | 6.5 | 5.8 |
| stats on price | 9.5 | 7.5 | 1.27× | 10.6 | 8.2 | 15.2 |

Every request returns the same hit count on both nodes.

After the merge, native search is as fast or faster on 19 of the 20 shapes,
with a median of 1.38× and up to 1.8× (two-term `OR`, `terms` under a query,
`multi_match`, `match_phrase`). The one slower shape is counting a near-full
two-term disjunction (`track_total_hits`, `size: 0`) on the merged index:
1.6 ms against 2.4 ms. Before the merge, the same count is faster on the plugin.

At this size a round trip is 2-90 ms and most of it is OpenSearch's own
request handling, so ratios near 1.0 are within the run-to-run spread
(about ±10%).
