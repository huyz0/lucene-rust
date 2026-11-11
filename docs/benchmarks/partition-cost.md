# What partitioning a shard by tenant costs on the write side

`scripts/bench-partition-cost.sh` (2026-11-11, 4 vCPU cloud VM, 15 GB).
One corpus for both engines: `PartitionBench`'s 2M documents, 1000 tenants
with Zipf sizes, interleaved on arrival, 40-word bodies. Every run indexes
from 4 threads with a reader refreshed every second, `TieredMergePolicy`
and `ConcurrentMergeScheduler` defaults, a 256 MB buffer, then commits and
waits for merges; the index is then force-merged to one segment. `java` is
stock Lucene 10.5.2 (`WhitespaceAnalyzer`); `rust` is this port's
concurrent writer configured as the engine writer opens a shard, given
documents already split into words (so absolute docs/s across engines is
not a like-for-like analysis cost; compare layouts within an engine).

Layouts: `plain` (arrival order), `tenant` (index sort on the tenant
keyword), `composite` (tenant, then time newest first) and `segments`
(each tenant's documents in segments of their own, the `partition_field`
mode). FTS rows are medians of 3 runs; run-to-run spread was within 10%.

## Full text, 2M documents

| engine, layout | docs/s | refresh mean / p99 ms | merges (s) | total s | force merge s | small tenant term, natural / merged µs | small tenant and | mid term | large term |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rust plain | 202,675 | 445 / 551 | 3 (5.4) | 10.8 | 4.4 | 506 / 470 | 89 | 570 | 741 |
| rust tenant | 166,035 | 1,292 / 1,649 | 3 (7.5) | 16.1 | 5.9 | 356 / 404 | 31 | 422 | 525 |
| rust composite | 168,667 | 1,478 / 2,022 | 3 (7.7) | 16.7 | 6.6 | 25 / 11 | 16 | 13 | 73 |
| java plain | 147,995 | 821 / 1,452 | 4 (8.0) | 15.5 | 6.1 | 579 / 451 | 121 | 529 | 727 |
| java tenant | 109,177 | 2,649 / 2,930 | 3 (11.0) | 24.8 | 7.7 | 535 / 397 | 47 | 414 | 435 |
| java composite | 107,642 | 4,053 / 5,264 | 2 (7.4) | 26.0 | 11.4 | 526 / 421 | 71 | 485 | 421 |

Query columns after the first are on the merged index. `term` queries also
bound `ts` to a tenth of the span; `and` queries do not. Sizes on disk are
within 2% across layouts (203-209 MB).

- **Indexing** is 18% slower sorted (rust), 26-27% (stock Lucene). The
  whole run, final flush and merges included, takes 1.5x (rust) and
  1.6-1.7x (stock) as long: a sorted flush reorders every column before writing it.
- **Refresh** is where it shows: a refresh flushes what was buffered, and a
  sorted flush takes 3x (rust) to 3-5x (stock) as long. Searches see new
  documents that much later at the same refresh interval.
- **Merges** run about as often; sorted merges take 1.4x the time. Force merge: 1.3-1.5x (rust), 1.3-1.9x
  (stock).
- **Queries**: the composite sort brings a small tenant's text query 20-40x
  down from unsorted. With the tenant alone sorted, queries that filter only
  on the tenant gain 3x (`and`), but a query with another filter (`term`
  and `phrase` here, both with a time range) stays at unsorted speed in
  this port: the tenant's doc-id run is found, but the one-scoring-term bulk
  path (`Bulk::FilteredTerm`) walks the term's postings over the whole
  segment and tests the run per document instead of starting and ending at
  it. Composite sorts avoid it because the time range folds into the run.
- **Own segments per tenant** (`segments`, 200k documents; 2M did not
  finish in 10 minutes): 756 docs/s against 210k, refreshes of 65 s
  (one tiny segment per tenant per refresh), 4,687 segments, term queries
  14 ms against 0.15-0.2 ms, and a force merge that takes 130 s and ends at
  1000 segments. Raw rows: `segments/results-200k.csv`.

## Vectors, 500k x 96d, cosine

32 MB buffer (with 256 MB the whole corpus fits one buffer and the port
never merges), tenant-filtered top 10, ef 100. Recall@10 is 0.999-1.0 in
every cell. `ivf-p` is IVF with clusters per tenant
(`set_ivf_partition_clusters(20_000)`, tenants of 20k+ vectors).

| engine, format, layout | total s | refresh mean ms | merges (s) | force merge s | small µs natural / merged | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|
| rust hnsw plain | 44.5 | 5,362 | 1 (23.6) | 38.2 | 132 / 177 | 647 | 1,452 |
| rust hnsw tenant | 42.8 | 4,469 | 2 (37.2) | 40.9 | 80 / 180 | 606 | 1,368 |
| java hnsw plain | 62.1 | 252 | 21 (183) | 92.4 | | | |
| java hnsw tenant | 64.0 | 214 | 21 (167) | 83.4 | | | |
| rust ivf plain | 4.7 | 620 | 1 (1.5) | 8.2 | 76 / 28 | 520 | 1,176 |
| rust ivf tenant | 6.7 | 588 | 2 (5.2) | 8.1 | 49 / 25 | 195 | 1,132 |
| rust ivf-p tenant | 5.4 | 476 | 2 (3.2) | 4.5 | 64 / 26 | 205 | 3,692 |

- **HNSW**: the tenant sort costs nothing measurable on either engine
  (total and force merge within run noise) and buys little: small and mid
  tenants 10-40% faster on natural segments, the same once merged.
- **IVF**: sorting costs 2 s on a 5 s build and nothing on force merge;
  mid tenants (1-3k vectors) search 2.7x faster. Clusters per tenant halve
  the force merge and keep that gain, but large tenants search 3.3x slower
  (3.7 vs 1.1 ms), on top of the unfiltered slowdown measured before.
- Not from partitioning: this port builds an HNSW graph when a segment
  flushes, so a refresh with vectors buffered takes seconds (4.5-5.4 s
  mean against stock's 0.2 s, which builds the graph as documents arrive).

## Verdict

Sorting by tenant does not improve merge or indexing; it costs 18-27%
indexing throughput, 3x refresh time and 1.3-1.5x merge time for full text,
and about nothing for vectors. It is a query-side layout: composite tenant +
time sorts make small and mid tenants' text queries 20-40x faster and IVF's
mid tenants 2.7x. Own segments per tenant lose on every axis. Keep the
sort opt-in; recommend the composite sort over the tenant alone for
always-filtered tenants until `Bulk::FilteredTerm` (and the phrase path)
clamp to the run; keep `ivf_partition_clusters` opt-in.

Raw data: `/mnt/project-files/perf/partition-cost/`.
