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
| Bulk indexing | 41.5 s (24,069 docs/s) | 159.7 s (6,264 docs/s) |
| Store after refresh | 485 MiB, 12 segments | 457 MiB, 10 segments |
| Force-merge to one segment | 13.9 s | 11.7 s |
| Node RSS idle / after load / after search | 2.291GiB / 2.422GiB / 2.511GiB | 2.307GiB / 3.194GiB / 3.22GiB |

The Rust engine indexes about 3.8 times slower. Its shard writer is one
`IndexWriter` behind one mutex (`lucene-ffi/src/engine_writer.rs`), so the
four bulk clients' documents are laid out one at a time, where OpenSearch's
`IndexWriter` fills one `DocumentsWriterPerThread` per indexing thread.
`lucene-index`'s `ConcurrentIndexWriter` has that shape but does not take the
JVM-inverted documents the engine sends yet. The plugin node also holds about
0.9 GB more RSS after loading, outside the Java heap.

## Search

Median REST round trip in milliseconds over 50 requests per shape after 30
warm-up requests, `request_cache=false`, top 10 hits. The merged columns
compare the two engines on the same one-segment shape and are the fair ones;
the first pass on each node ran right after loading, and stock's first pass
also overlapped its background merges. "Plugin, Java search" is the plugin
node's index with native search off.

| Request | Stock, merged | Plugin, merged | Ratio | Stock, 12 segments | Plugin, 10 segments | Plugin, Java search |
|---|---:|---:|---:|---:|---:|---:|
| term, frequent (t0) | 6.2 | 4.0 | 1.53× | 12.2 | 9.4 | 6.3 |
| term, mid (t2s) | 4.2 | 3.0 | 1.40× | 8.4 | 7.5 | 4.8 |
| term, rare (t1z4) | 4.4 | 3.6 | 1.20× | 7.4 | 8.0 | 3.8 |
| match OR, 2 frequent | 13.8 | 10.9 | 1.27× | 19.4 | 12.6 | 12.6 |
| match OR, 4 mixed | 7.8 | 6.1 | 1.27× | 11.1 | 9.4 | 7.5 |
| match AND, 2 frequent | 12.6 | 9.7 | 1.30× | 16.0 | 14.4 | 11.1 |
| match AND, frequent + mid | 6.3 | 5.6 | 1.14× | 8.6 | 8.1 | 7.7 |
| match_phrase, 2 frequent | 95.3 | 57.0 | 1.67× | 105.1 | 63.6 | 100.0 |
| multi_match body+title | 12.2 | 70.8 | 0.17× | 14.6 | 71.9 | 10.2 |
| bool match + range filter | 7.3 | 7.2 | 1.02× | 10.1 | 8.2 | 8.0 |
| range on long | 5.6 | 5.4 | 1.04× | 7.6 | 7.0 | 4.8 |
| term on keyword | 4.3 | 4.0 | 1.06× | 5.8 | 5.2 | 4.7 |
| prefix on title | 10.1 | 6.1 | 1.64× | 12.0 | 6.7 | 8.8 |
| match_all, sort by num | 5.9 | 4.9 | 1.21× | 8.8 | 7.1 | 5.1 |
| match, sort by price | 7.7 | 5.1 | 1.50× | 11.4 | 6.2 | 7.6 |
| count, track_total_hits | 3.7 | 12.5 | 0.30× | 4.4 | 13.7 | 9.8 |
| terms agg on category | 5.1 | 3.3 | 1.53× | 6.4 | 5.7 | 4.8 |
| terms agg on tag, filtered | 15.1 | 7.7 | 1.97× | 15.0 | 9.5 | 14.9 |
| date_histogram by month | 7.1 | 7.0 | 1.01× | 8.6 | 8.7 | 6.7 |
| stats on price | 19.5 | 9.2 | 2.13× | 21.4 | 10.4 | 8.3 |

Every request returns the same hit count on both nodes.

After the merge, native search is at least as fast on 18 of the 20 shapes
(four of them within the noise) and up to 2.1× faster (`stats`, `terms` under
a query, `match_phrase`, `prefix`). Two shapes were far slower in this run:

- `multi_match` over two fields (a `DisjunctionMaxQuery` of two
  `BooleanQuery` disjunctions): 5.8× slower. The native dismax scored every
  candidate document of each disjunct. It now does what Lucene's
  `DisjunctionMaxBulkScorer` does: with `tie_breaker` 0 and top-k scoring it
  scores each disjunct's bulk scorer over windows of 4096 documents, keeps
  the best score per document, and passes the top-level flag down so the
  disjuncts prune with MaxScore.
- Counting a two-term disjunction that matches nearly every document
  (`track_total_hits`, `size: 0`): 3.4× slower. The count now looks up the
  segment query cache first, as `IndexSearcher.count` does, and the union
  window counts by popcount instead of document by document.

In process (`benchmarks/`, the same merged 1M-document index, one thread):

| shape | Lucene 10.5.2 | native before | native after |
|---|---:|---:|---:|
| dismax of two 2-term disjunctions | 208 qps | 11.6 qps | 642 qps |
| dismax of two disjunctions, second shape | 338 qps | | 620 qps |
| count of a near-full disjunction, repeated | 7 µs (query cache) | | 1.4 µs (query cache) |
| the same count, query cache off | 420 µs | | |

Both engines return the same top documents and counts.

At this size a round trip is 3-20 ms and most of it is OpenSearch's own
request handling, so ratios near 1.0 are within the run-to-run spread
(about ±10%).
