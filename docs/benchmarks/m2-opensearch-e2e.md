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
  shape: 3 warm-up requests, then 40 timed requests per
  engine, alternating Lucene/native/Lucene/native in four passes, median wall
  time of the HTTP round trip, `request_cache=false`. The switch is the
  index's own `index.lucene_rust.search.enabled`, so both engines answer from
  the same shard, reader and page cache.
- **In process**: `gradle -p opensearch-plugin nativeBench` — one JDK 21 JVM,
  a 500,000-document index written by Lucene's `IndexWriter` (4 segments, 9,891
  deletions), searched through an NRT reader. Lucene runs
  `TopScoreDocCollectorManager(10, 10_000)` — OpenSearch's default
  `track_total_hits` — and native runs `NativeBridge.search` with the same
  threshold. Median of 15 batches of 200 queries, engines alternating by batch.

## The JNI crossing

An empty call (`NativeBridge.abiVersion()`), 1,000,000 per batch: **9.4 ns**.
The M2 budget is 1 µs; a search makes exactly one crossing.

## REST, per request shape

Every shape the engine answers natively, timed natively (the index set to
`native_shapes: all`); the second column is where the plugin sends it by
default.

| shape | routed to | Lucene ms | native ms | ratio |
|---|---|---|---|---|
| match one term | native | 4.21 | 2.89 | 1.46× |
| match rare term | native | 3.89 | 2.96 | 1.31× |
| match two terms | native | 4.32 | 3.39 | 1.27× |
| match four terms | native | 5.34 | 4.63 | 1.15× |
| match operator and | native | 4.51 | 4.29 | 1.05× |
| match minimum_should_match | Lucene (`slower_shape`) | 8.44 | 37.43 | 0.23× |
| term keyword | native | 3.29 | 3.33 | 0.99× |
| term missing | native | 3.05 | 2.34 | 1.30× |
| bool must+should | Lucene (`slower_shape`) | 4.59 | 18.10 | 0.25× |
| bool must_not | Lucene (`slower_shape`) | 4.19 | 14.07 | 0.30× |
| bool filter | native | 4.15 | 3.40 | 1.22× |
| bool filter only | Lucene (`slower_shape`) | 3.35 | 3.53 | 0.95× |
| bool nested | Lucene (`slower_shape`) | 4.55 | 21.52 | 0.21× |
| query_string OR | native | 4.45 | 4.13 | 1.08× |
| size 0 count | native | 2.76 | 2.25 | 1.23× |
| track_total_hits false | native | 3.22 | 2.86 | 1.13× |
| track_total_hits true | native | 6.76 | 6.39 | 1.06× |
| track_total_hits 100 | native | 3.08 | 2.74 | 1.12× |
| from 20 size 15 | native | 3.21 | 3.09 | 1.04× |
| size 200 | native | 8.42 | 8.42 | 1.00× |
| boosted match | Lucene (`slower_shape`) | 3.25 | 16.86 | 0.19× |
| bool with boosted clause | Lucene (`slower_shape`) | 3.68 | 17.98 | 0.20× |
| constant_score | Lucene (`slower_shape`) | 2.82 | 10.87 | 0.26× |

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
   dense terms of this corpus that is 3–5× slower. The plugin therefore routes
   them to Lucene (`slower_shape`), and they stay correct natively — the
   verify script runs its matrix in both modes. Closing the gap is engine work
   (block-max pruning for mixed booleans and wrapped clauses), the same kind
   M1.6 did for prefix and wildcard, and is out of M2's scope.

## In process, JVM in the loop

| query | Lucene µs | native µs | ratio | native, no count, µs |
|---|---|---|---|---|
| term dense | 310.8 | 197.3 | 1.58× | 139.6 |
| term mid | 421.2 | 99.4 | 4.24× | 39.2 |
| term rare | 341.1 | 29.8 | 11.43× | 20.0 |
| or 2 | 431.7 | 320.9 | 1.35× | 201.0 |
| or 4 | 729.0 | 511.8 | 1.42× | 391.4 |
| and 2 | 757.1 | 712.1 | 1.06× | 339.2 |
| and 3 | 384.7 | 337.1 | 1.14× | 391.4 |

Every routed shape is at least as fast natively with the JVM in the loop. The
Lucene column is noisier than the native one (the rare-term figure is a JIT
artefact that moves between runs); the ratios' direction is stable across the
runs made for this document, their magnitude is not.

**The count was the first finding.** The first cut counted total hits with a
second, exhaustive pass whenever the top hits were full, where Lucene stops
counting at `track_total_hits` (10,000) and reports "at least". That made
dense queries 2.5–4.5× *slower* than Lucene (term dense 1,172 µs vs 289 µs).
The native search now counts inside its collector under Lucene's own
`totalHitsThreshold` rule (`lucene-search`'s
`search_*_multi_segment_counting`), and the "no count" column shows what is
left: counting to 10,000 costs the dense queries 40–100%, the same trade
Lucene makes.

## Consistency with the standalone result

M1.6's sweep (`sweep-2026-09.md`) left every query in the M1 mix at or above
Lucene: 1.03–46×. Through OpenSearch the same shapes are 1.0–1.46× over REST
and 1.06–11× in process — the same direction, compressed by the parts of a
request neither engine changes. The discrepancy the milestone asked to watch
for ("the overhead is in the binding") is not there: the crossing is 9.4 ns
and there is one per search. The discrepancy that *is* there is the set of
query shapes the M1 mix never contained, above.
