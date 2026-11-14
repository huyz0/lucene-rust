# Results, 2026-11-13 (main 3f81084f + #309)

One shard, one segment (force-merged), nodes on cores 0-2 with a 2 GB heap, request cache off,
medians of 20 rounds, plugin = `--profile dist --features mimalloc`. Real data: NYC yellow taxi
2025-01/02 (7,052,769 docs, file order) and Loghub BGL (4,631,261 docs, time order). Every response
equal to stock (`exact`, `approx` or `sketch`) in every run below.

## Stage 4: are the aggregated numeric fields sparse or multi-valued?

`survey.py`, first 300k documents of each public corpus sample (OpenSearch Benchmark workloads),
plus a null count over the whole 2025-01 taxi file. "Aggregated" = used under `aggs` in the
dataset's own workload.

| dataset | aggregated numeric/date fields | present | multi-valued |
|---|---|---|---|
| nyc_taxis | dropoff_datetime, tip_amount, total_amount, trip_distance | 100% | 0% |
| big5 | @timestamp, metrics.size, metrics.tmin | 100% | 0% |
| http_logs | @timestamp, size, status | 100% | 0% |
| geonames | geonameid, population | 100% | 0% |
| so | none aggregated (creationDate only numeric) | 100% | 0% |
| taxi 2025 (real) | fare, tip, total, distance, pickup: 100%; passenger_count, congestion_surcharge, Airport_fee: 84.5% | | 0% |
| bgl (real) | @timestamp, bytes | 100% | 0% |

Sparse numeric fields exist but are not aggregated by these workloads (geonames `elevation`,
7.6% present). The only multi-valued numeric-looking fields are geo_point arrays (not aggregated
as numbers). The sparse fields that are in the taxi data (15.5% missing, in blocks) already
aggregate correctly and faster than stock through the dense path (`terms(passengers)`,
`histogram(passengers)` below).

Downside of this evidence: these are benchmark corpora, which are cleaned. Production ECS/OTel
logs are sparse by nature (`http.response.status_code` only on web logs), and none of the public
datasets shows that; a sparse/multi-valued zone design should wait for a real log sample.

## Stage 5: index sort by the main group-by key, end to end (plugin vs stock, today's code)

taxi sorted by `pu_location, pickup`; bgl sorted by `component, @timestamp`. Same shapes both ways.

Indexing and size (taxi / bgl):

| | stock unsorted | stock sorted | plugin unsorted | plugin sorted |
|---|---|---|---|---|
| taxi store MiB | 712.8 | 689.3 | 631.0 | 633.1 |
| taxi bulk load s | 54.2 | 56.7 | 62.2 | 68.6 (+10%) |
| bgl store MiB | 275.6 | 230.0 | 223.8 | 213.0 |
| bgl bulk load s | 42.6 | 42.3 | 45.3 | 50.0 (+10%) |

Plugin median ms, unsorted -> sorted:

| shape | taxi all | taxi 10% | bgl all | bgl 10% |
|---|---|---|---|---|
| terms(key)>sum | 54.3 -> 68.3 | 10.6 -> 14.6 | 41.2 -> 40.8 (component>stats) | 11.2 -> 11.0 |
| terms(low-card int) | 44.1 -> 59.6 | 9.9 -> 8.2 | | |
| histogram | 95.1 -> 101.0 | 11.5 -> 12.5 | 39.9 -> 40.3 | 11.5 -> 11.5 |
| date_histogram | 70.9 -> 75.0 | 9.9 -> 9.4 | 60.7 -> 59.9 | 13.1 -> 14.0 |

Finding: with today's aggregation code, sorting buys disk size (stock bgl -17%, plugin bgl -5%,
taxi ~0) and nothing in speed; taxi group-by is 25% slower and the time-range cases do not
improve. The sorted layout only pays once the aggregators fold whole 1024-doc blocks (the fold
prototype's 21.6x for taxi `terms(pu)>sum` sorted by pickup location was measured on the
in-memory reader, not end to end). Downsides: indexing +10% CPU/wall on the plugin; a sort on the
group-by key removes the time-order locality that `date_histogram` and time-range filters use
(taxi `date_histogram` all +6%); only one sort key can win; stock gains nothing from it.
Recommendation: do not turn it on by default; re-run this A/B after the fold stage lands.

Plugin vs stock (unsorted, whole run): taxi `terms(pu)>sum` 54 vs 204 ms, `histogram>min+max` 95
vs 178, `percentiles` 265 vs 598, date_histogram 71 vs 78, `range(fare)` 2.9 vs 2.7 (parity);
bgl group-bys 0.43-0.57x of stock, `range(bytes)>sum` 1.14x (slower), plain `terms(level)` over a
10% window 14.3 vs 7.5 ms (1.9x slower: stock answers it from the keyword index).

## Stage 6: derived source in the logs preset (bgl, sort @timestamp desc, preset codecs on the plugin)

`message` is mapped `text` with `store: true` in the derived runs (stock cannot derive an unstored
text field). Loading CPU is the node's process CPU over the bulk load.

| | stock `_source` | stock derived | plugin `_source` | plugin derived |
|---|---|---|---|---|
| store MiB | 240.1 | 182.3 (-24%) | 171.8 | 134.2 (-22%) |
| load wall s / CPU s | 38.9 / 110 | 41.6 / 119 (+8%) | 55.1 / 157 | 48.7 / 140 (-11%) |
| force merge s / CPU s | 21.5 / 23.7 | 16.9 / 21.7 | 8.4 / 11.5 | 9.4 / 12.6 |
| newest 100 hits ms | 3.1 | 5.5 | 5.3 | 6.2 |
| match "error" 100 hits ms | 3.8 | 5.7 | 4.4 | 5.4 |
| 100-hit time window ms | 2.8 | 4.8 | 3.1 | 3.8 |

Hits (source) are equal between stock and plugin in both modes. Derived source costs +0.9 to
+2.4 ms per 100-hit fetch and saves 22-24% on this six-field log. Plugin's preset costs 43% more
indexing CPU than stock (157 vs 110 s) and gives 28% smaller storage and group-bys about 2x faster (stats and date_histogram 4-10x).

Downsides: the saving here is small because BGL is six fields and the message dominates; wide JSON
logs save more (docs/logs-preset.md: 715 -> 532 MiB on the synthetic corpus). A derived `_source`
is rebuilt from mapped fields, so it is not the document as sent (field order, number and date
formatting, unmapped and `ignore_above` fields, `_source` includes of fields not in the mapping);
this run compares stock derived with plugin derived and does not measure fidelity to the original
document. Text fields need `store: true` or a keyword subfield (the stored copy stays).
Recommendation: keep it opt-in and leave the preset as is, since it changes what `_source` returns;
document it as the lever for storage-bound deployments.

## Filtered k-NN: block statistics for the filter (2026-11-14)

`cargo run --release -p lucene-search --example knn_filter_stats -- DIR 500000 32 clustered|shuffled [skip]`
(in-process, one segment, 500k x 32-d HNSW, `ts` dense `Adaptive90` with zones, k=10, 30 rounds, one
run each). The harness above needs Docker, which the build container lacks, so this is the library
path of the plugin's filtered search: `filter_bitsets` (materialise the range filter) + the search.

The planner already receives the exact filter cardinality, so a statistics *estimate* cannot improve
its choice; the cost is the materialisation, 80-95% of a filtered query on main (7-14 ms of 7-14 ms
total). The statistics now make the exact set cheap instead (`dense_range_bits`: whole windows in
range set as a run, windows outside skipped, the rest decoded a window at a time).

Filter ms, main -> branch (total query ms in `perf/knn-filter-stats/ab-main-vs-branch.txt`):

| layout | 0.05% | 0.5% | 2% | 10% | 30% | 60% | 90% |
|---|---|---|---|---|---|---|---|
| clustered, no skip index | 6.9 -> 0.02 | 6.7 -> 0.01 | 6.6 -> 0.02 | 6.5 -> 0.02 | 7.0 -> 0.01 | 7.0 -> 0.02 | 7.1 -> 0.01 |
| clustered, skip index | 0.11 -> 0.02 | 0.14 -> 0.02 | 0.21 -> 0.02 | 0.46 -> 0.02 | 1.1 -> 0.02 | 2.1 -> 0.01 | 2.5 -> 0.01 |
| shuffled, no skip index | 9.2 -> 3.1 | 8.8 -> 2.8 | 7.8 -> 1.9 | 11.5 -> 2.1 | 11.5 -> 2.7 | 9.8 -> 1.9 | 8.6 -> 1.7 |

Downsides: shuffled (unclustered) data gets no pruning (every window spans the value range, so
min/max say nothing), only the direct decode, 3-5x; a field without zones (sparse, multi-valued,
not `Adaptive90`) takes the old path unchanged; a skip-indexed field without zones also keeps the old
path. No indexing cost beyond the zones already written (+1-1.5% `.dvd`). Not measured: the full
OpenSearch request path, stock OpenSearch, recall (the filter set is bit-identical, tested against
the values on 4 formats x 2 layouts).
