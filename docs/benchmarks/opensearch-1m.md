# The plugin against stock OpenSearch, 1M documents

`scripts/bench-opensearch.sh 1000000` (`opensearch-plugin/e2e/bench_nodes.py`)
indexes one synthetic corpus into a stock OpenSearch 3.9.0 node and into a node
with the plugin serving the index on the Rust engine
(`indices.engine.accelerated.default: true`) and native search. The nodes run one
after the other, each alone, with a 2 GB heap, on a 4-core, 15 GB container
(2026-11-05).

The corpus: 1,000,000 documents, a body of 20-120 terms and a title of 2-8
terms drawn from a Zipf vocabulary of 200,000 terms (`t0` the most frequent),
a Zipf keyword `tag` (1,000 values), a uniform keyword `category` (50), a
`long`, a `double` and a `date`. One shard, no replica, sent as 200 bulk
requests of 5,000 documents by 4 parallel clients, refresh off until the end.

## Indexing

| | Stock | Plugin (Rust engine) |
|---|---:|---:|
| Bulk indexing | 25.06 s (39,905 docs/s) | 25.5 s (39,222 docs/s) |
| Bulk indexing and the final refresh | 27.46 s | 26.55 s |
| Store after refresh | 484 MiB, 12 segments | 534 MiB, 17 segments |
| Force-merge to one segment | 10.14 s | 7.57 s |
| Node RSS idle / after load / after search | 2.286GiB / 2.393GiB / 2.475GiB | 2.279GiB / 2.793GiB / 2.699GiB |

The Rust engine acknowledges bulk requests at about stock's rate (a single
run; load runs vary by about ±10%) and is searchable sooner, because its final
refresh is short. Each shard writer keeps one buffer per indexing thread
(`ConcurrentIndexWriter`), as Lucene keeps one
`DocumentsWriterPerThread` per thread. A refresh commits without stopping
indexing, and merges run on the writer's own merge threads.

The native library allocates with mimalloc (lucene-ffi's `mimalloc` feature,
on in the plugin build). With glibc's malloc, every JVM thread that indexed
kept its own arena and the memory freed in it: the same load left the node at
4.15 GiB.

## Search

Median REST round trip in milliseconds over 50 requests per shape after 30
warm-up requests, `request_cache=false`, top 10 hits. The merged columns
compare the two engines on the same one-segment index, searched after a
refresh, and are the fair ones; the first pass on each node ran right after
loading, and stock's first pass also overlapped its background merges.
"Plugin, Java search" is the plugin node's index with native search off.

| Request | Stock, merged | Plugin, merged | Ratio | Stock, 12 segments | Plugin, 17 segments | Plugin, Java search |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent (t0) | 3.5 | 2.6 | 1.34× | 7.6 | 5.2 | 3.6 |
| term, mid (t2s) | 2.9 | 2.4 | 1.20× | 4.8 | 4.3 | 3.0 |
| term, rare (t1z4) | 2.8 | 2.5 | 1.13× | 4.0 | 4.3 | 3.0 |
| match OR, 2 frequent | 9.2 | 5.5 | 1.66× | 10.3 | 7.4 | 8.2 |
| match OR, 4 mixed | 4.3 | 3.6 | 1.18× | 6.2 | 5.4 | 5.3 |
| match AND, 2 frequent | 9.1 | 5.3 | 1.71× | 9.9 | 7.4 | 7.2 |
| match AND, frequent + mid | 3.4 | 2.8 | 1.21× | 4.5 | 4.5 | 3.6 |
| match_phrase, 2 frequent | 73.6 | 42.4 | 1.74× | 63.2 | 45.3 | 67.4 |
| multi_match body+title | 5.5 | 4.1 | 1.34× | 9.4 | 5.6 | 7.6 |
| bool match + range filter | 3.7 | 3.5 | 1.06× | 5.4 | 5.2 | 4.0 |
| range on long | 3.4 | 2.7 | 1.25× | 4.0 | 4.5 | 3.3 |
| term on keyword | 2.3 | 2.3 | 1.02× | 3.2 | 3.3 | 2.7 |
| prefix on title | 4.3 | 3.2 | 1.34× | 8.1 | 4.6 | 6.6 |
| match_all, sort by num | 2.9 | 2.7 | 1.08× | 4.4 | 4.4 | 3.3 |
| match, sort by price | 6.4 | 3.0 | 2.14× | 5.2 | 3.6 | 3.8 |
| count, track_total_hits | 2.2 | 2.1 | 1.05× | 2.7 | 2.9 | 3.5 |
| terms agg on category | 2.9 | 2.5 | 1.13× | 4.2 | 3.6 | 3.3 |
| terms agg on tag, filtered | 10.6 | 5.6 | 1.91× | 8.4 | 5.2 | 7.3 |
| date_histogram by month | 2.7 | 2.9 | 0.94× | 4.7 | 6.2 | 4.0 |
| stats on price | 8.5 | 6.0 | 1.41× | 9.5 | 5.4 | 6.6 |

Every request returns the same hit count on both nodes.

After the merge, native search is faster on 19 of the 20 shapes, with a
median of 1.23× and up to 2.14× (`match` sorted by `price`). The one slower
shape is the monthly `date_histogram`, at 0.94×.

At this size a round trip is 3-90 ms and most of it is OpenSearch's own
request handling, so ratios near 1.0 are within the run-to-run spread
(about ±10%). Runs before 2026-11-05 timed the merged pass before a refresh,
so it searched the unmerged segments on both nodes.
