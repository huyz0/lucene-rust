# Garbage collection: stock OpenSearch against the plugin (2026-11-17)

How much heap garbage a node produces, and what that does to pauses and tail latency, with and
without the plugin.

## Setup

A 4-core machine, the node on three cores and the load driver on the fourth, G1 on the JDK 25 bundled
with OpenSearch 3.9.0, a 500k-document corpus, one shard, no replicas. One run per build and heap size
with `-Xlog:gc`, plus a JFR run at a 1 GB heap for allocation by thread. The phases are bulk indexing,
force merge, serial searches (23 shapes), four clients, aggregations, heavy aggregations (high-cardinality
`terms`, histograms, `composite`, `top_hits`), fetch and sort with large sizes, and eight mixed clients.
Each figure is a single run, so treat small differences as noise.

## Results

| | stock, 1 GB heap | plugin, 1 GB | stock, 4 GB | plugin, 4 GB |
|---|---|---|---|---|
| heap allocation (JFR at 1 GB; log estimate at 4 GB) | 26.2 GB | 13.0 GB | ~23 GB | ~9 GB |
| young pauses over the run | 171 | 39 | 48 | 19 |
| total pause in the phases, ms | 4,242 | 681 | 3,126 | 438 |
| longest pause, ms | 141 | 42 | 323 | 145 |
| mixed, 8 clients, p50 / p99 ms | 92.7 / 7,369 | 34.4 / 715 | 84.4 / 6,020 | 30.5 / 698 |
| heavy aggregations, p50 / p99 ms | 54.5 / 3,637 | 28.2 / 161 | 58.3 / 3,339 | 25.9 / 150 |
| fetch and sort, p99 ms | 291 | 190 | 692 | 157 |
| serial search, p50 / p99 ms | 5.4 / 367 | 4.2 / 374 | 5.1 / 388 | 4.0 / 349 |

## Reading it

- On the read side the plugin removes most of the heap garbage: search threads allocate 4.1 GB
  against 18.1 GB for stock. The tail under load is about ten times lower, and the plugin at a 1 GB
  heap has a lower tail than stock at 4 GB.
- The medians are close; the difference is in the tail, which is where pauses show.
- On the write side the plugin is **not** better yet: its write threads allocate 7.5 GB against 6.0 GB
  for stock, and pauses during indexing are of the same order (15 against 36 pauses at 1 GB, 270
  against 211 ms in total at 4 GB). Bulk indexing is faster (16.0 against 20.1 s at 1 GB).
- Part of what is left on the search side is OpenSearch's own reduce phase on the coordinating node,
  which the plugin does not replace.
