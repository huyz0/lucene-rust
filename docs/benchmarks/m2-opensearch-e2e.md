# M2 end to end: OpenSearch with the Rust query phase

Milestone [M2](../milestones/m2-opensearch-read-path.md), task T2.7: does the
engine-level result hold with a real JVM and a real OpenSearch node in the
loop, and what does one crossing of the JNI boundary cost?

Ratios are `lucene_time / native_time`: **above 1.0 means native is faster.**

## Machine and method

- A 4-vCPU cloud container (Linux 6.18, x86_64), shared; no pinning. Treat
  single-digit-percent differences as noise.
- **REST**: one OpenSearch 3.8.0 node (the official image, bundled JDK 25,
  1 GB heap) with the plugin installed; `scripts/verify-opensearch.sh --docs
  100000 --bench-out FILE --bench-rounds 40`. The `single` index: 1 shard,
  99,749 live documents in 2 segments
  after the run's force merges, with deletions and soft-deleted updates. Each
  shape: 3 warm-up requests, then 40 timed requests per pass in four passes
  alternating Lucene/native/Lucene/native -- 80 samples per engine -- median
  wall time of the HTTP round trip, `request_cache=false`. The switch is the
  index's own `index.lucene_rust.search.enabled`, so both engines answer from
  the same shard, reader and page cache.
- **In process**: `gradle -p opensearch-plugin nativeBench` — one JDK 21 JVM,
  a 500,000-document index written by Lucene's `IndexWriter` (4 segments, 9,891
  deletions), searched through an NRT reader. Lucene runs
  `TopScoreDocCollectorManager(10, 10_000)` — OpenSearch's default
  `track_total_hits` — and native runs `NativeBridge.search` with the same
  threshold. Median of 15 batches of 200 queries, engines alternating by batch.

## The JNI crossing

An empty call (`NativeBridge.abiVersion()`), 1,000,000 per batch: **7.4–9.4 ns**
over three runs.
The M2 budget is 1 µs; a search makes exactly one crossing.

## REST, per request shape

Every shape the engine answers natively, timed natively (the index set to
`native_shapes: all`); the second column is where the plugin sends it by
default.

| shape | routed to | Lucene ms | native ms | ratio |
|---|---|---|---|---|
| match one term | native | 3.02 | 2.77 | 1.09× |
| match rare term | native | 3.38 | 2.63 | 1.29× |
| match two terms | native | 3.78 | 3.10 | 1.22× |
| match four terms | native | 4.70 | 4.15 | 1.13× |
| match operator and | native | 4.14 | 3.97 | 1.04× |
| match minimum_should_match | Lucene (`slower_shape`) | 7.98 | 51.87 | 0.15× |
| term keyword | native | 3.02 | 2.69 | 1.12× |
| term missing | native | 2.41 | 1.80 | 1.34× |
| bool must+should | Lucene (`slower_shape`) | 4.40 | 23.43 | 0.19× |
| bool must_not | Lucene (`slower_shape`) | 4.51 | 16.05 | 0.28× |
| bool filter | native | 3.78 | 3.03 | 1.25× |
| bool filter only | Lucene (`slower_shape`) | 2.54 | 3.17 | 0.80× |
| bool nested | Lucene (`slower_shape`) | 4.13 | 26.14 | 0.16× |
| query_string OR | native | 4.25 | 3.83 | 1.11× |
| size 0 count | native | 2.65 | 1.79 | 1.48× |
| track_total_hits false | native | 2.96 | 2.55 | 1.16× |
| track_total_hits true | native | 6.77 | 5.97 | 1.13× |
| track_total_hits 100 | native | 2.81 | 2.81 | 1.00× |
| from 20 size 15 | native | 3.35 | 2.74 | 1.22× |
| size 200 | native | 9.64 | 8.22 | 1.17× |
| size 0 below threshold | native | 2.27 | 1.93 | 1.18× |
| rare term, exact total | native | 4.00 | 3.77 | 1.06× |
| boosted match | Lucene (`slower_shape`) | 2.65 | 20.51 | 0.13× |
| bool with boosted clause | Lucene (`slower_shape`) | 3.19 | 18.03 | 0.18× |
| constant_score | Lucene (`slower_shape`) | 2.52 | 11.55 | 0.22× |

Two things set the REST figures apart from the engine's own:

1. **The round trip dominates.** A 2.5–4 ms request spends most of its time
   in HTTP, the coordinating node, the fetch phase and JSON — all of it
   identical code on both sides. The query phase is what changed, so a 3×
   faster query phase shows up as 1.3–1.5×.
2. **Shapes the standalone M1 mix never measured are slower.** The M1/M1.6
   query mix (`benchmarks/queries.tsv`) has terms, pure conjunctions, pure
   disjunctions, phrases and multi-term queries — and no boost, no
   `constant_score`, no `must_not`, no `must` + `should` mix, no
   `minimum_should_match`. Those run the Rust engine's exhaustive boolean
   scorer, which scores every candidate, where Lucene's WAND skips; on the
   dense terms of this corpus that is 4–8× slower. The plugin therefore routes
   them to Lucene (`slower_shape`), and they stay correct natively — the
   verify script runs its matrix in both modes. Closing the gap is engine work
   (block-max pruning for mixed booleans and wrapped clauses), the same kind
   M1.6 did for prefix and wildcard, and is out of M2's scope.

## In process, JVM in the loop

| query | Lucene µs | native µs | ratio | native, no count, µs |
|---|---|---|---|---|
| term dense | 323.3 | 189.3 | 1.71× | 137.1 |
| term mid | 175.6 | 90.9 | 1.93× | 40.0 |
| term rare | 99.6 | 30.4 | 3.28× | 20.9 |
| or 2 | 389.6 | 268.6 | 1.45× | 175.5 |
| or 4 | 723.1 | 502.3 | 1.44× | 385.2 |
| and 2 | 840.1 | 776.8 | 1.08× | 361.4 |
| and 3 | 393.5 | 352.3 | 1.12× | 408.8 |

Every routed shape is at least as fast natively with the JVM in the loop. The
Lucene column is noisier than the native one (its rare-term figure has read
341 µs and 100 µs in two runs); the ratios' direction is stable across the
three runs made for this document, their magnitude is not. `and 3`'s no-count
figure above its with-count one is not a transcription slip -- it reproduced
on a second run -- it is the two within noise of each other: that query's
matches are few enough that counting them costs nothing measurable.

**The count was the first finding.** The first cut counted total hits with a
second, exhaustive pass whenever the top hits were full, where Lucene stops
counting at `track_total_hits` (10,000) and reports "at least". That made
dense queries 2.5–4.5× *slower* than Lucene (term dense 1,172 µs vs 289 µs).
The native search now counts inside its collector under Lucene's own
`totalHitsThreshold` rule (`lucene-search`'s
`search_*_multi_segment_counting`), and the "no count" column shows what is
left: counting to 10,000 costs the dense term and disjunction queries
30–130%, the same trade Lucene makes.

## Consistency with the standalone result

M1.6's sweep (`sweep-2026-09.md`) left every query in the M1 mix at or above
Lucene: 1.03–46×. Through OpenSearch the same shapes are 1.00–1.48× over REST
and 1.08–3.3× in process (up to 11× in an earlier run) — the same direction, compressed by the parts of a
request neither engine changes. The discrepancy the milestone asked to watch
for ("the overhead is in the binding") is not there: the crossing is 9.4 ns
and there is one per search. The discrepancy that *is* there is the set of
query shapes the M1 mix never contained, above.
