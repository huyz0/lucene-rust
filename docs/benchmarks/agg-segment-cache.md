# Aggregation segment cache: dashboards across refreshes

The aggregation segment cache keeps what an aggregation collected from each
segment, so a dashboard asked again after a refresh collects only the segments
that changed. OpenSearch's shard request cache keys a whole shard's answer by
its reader, so a refresh or a moved time window misses it. The cache is
`index.lucene_rust.search.aggregation_segment_cache` (index, dynamic, default
`false`), within the node's `lucene_rust.search.aggregation_segment_cache.size`
(default `256mb`). See [operations](../operations.md).

How to reproduce: [`benchmarks/agg-segment-cache`](../../benchmarks/agg-segment-cache/README.md)
(`run.sh` runs all three configurations in about 10 minutes).

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.1 GHz, 16 GB. One node in Docker
  (OpenSearch 3.9.0, 4 GB heap, security off), one shard, no replica.
- Corpus: 6,454,808 Loghub lines (the first 250 MB of BGL, HDFS and
  Thunderbird, and all of OpenSSH), interleaved, as log events over 7 days:
  `@timestamp` (date), `source`, `host` (81,266 values), `level` (keyword),
  `bytes` (long), `message` (text). 6,254,808 are loaded in 10k-event bulks
  (4 at a time) with refresh off, then refreshed; 200,000 arrive in the live
  rounds.
- Configurations:
  - **stock**: OpenSearch 3.9.0 without plugins.
  - **main**: the plugin built from main, every shape native.
  - **cache**: the plugin built from this change, cache on.
  - main and cache start from a copy of the same loaded data directory (6
    segments after the load's merges); stock loads its own (20 segments).
- Dashboards, each one request with `size: 0`:
  - **d24**: last 24 h; hourly `date_histogram` > top-5 `source` terms; top-10
    `host` terms; `filters` for error and warn.
  - **d7**: the same over the last 7 days.
  - **d24_bgl**: last 24 h of `source: bgl`; 10-minute `date_histogram` >
    `stats(bytes)`; `level` terms.
  - **d7_bytes**: last 7 days; 6-hourly `date_histogram` > `sum`, `max`, `avg`
    of `bytes`; top-5 `host` terms > `avg(bytes)`; `range` of `bytes`.
- Phases, all latencies the client's (median of 15, or of 20 live rounds):
  - **repeat**: the same request again, which the shard request cache
    answers in every configuration.
  - **sliding**: `now` one second later each request, as a dashboard
    refreshing itself: the request cache misses.
  - **miss**: a tree never seen before each request (an interval a few ms
    off), so every segment misses the segment cache: its cost when cold.
  - **append**: 20 rounds of a 5,000-event bulk, a refresh, then each
    dashboard once at the newest event's time. Merges run as they come
    (main and cache: 6 to 9 segments, 5 merges over both live phases).
  - **mixed**: 20 rounds that also re-index 300 and delete 100 events picked
    at random over the whole index (soft deletes in every segment, every
    round).
  - **refresh probe**: after each refresh and before the dashboards, a terms
    aggregation over the last second on the three keyword fields. It takes
    what a new reader costs whatever the request (global ordinals of the
    81k hosts): 20 to 30 ms in every configuration, kept out of the
    dashboards' numbers.

## Results (ms, p50)

| dashboard | phase | stock | main | cache | cache vs main |
|---|---|---:|---:|---:|---:|
| d24 | sliding | 56.6 | 48.9 | 17.0 | 2.9x |
| d7 | sliding | 287.7 | 119.3 | 32.1 | 3.7x |
| d24_bgl | sliding | 22.1 | 30.6 | 18.8 | 1.6x |
| d7_bytes | sliding | 478.1 | 249.7 | 37.1 | 6.7x |
| d24 | append | 57.6 | 46.8 | 21.7 | 2.2x |
| d7 | append | 276.3 | 131.5 | 37.4 | 3.5x |
| d24_bgl | append | 20.8 | 29.3 | 16.3 | 1.8x |
| d7_bytes | append | 461.4 | 241.5 | 49.2 | 4.9x |
| d24 | mixed | 97.6 | 46.7 | 48.8 | 0.96x |
| d7 | mixed | 327.4 | 137.2 | 142.2 | 0.97x |
| d24_bgl | mixed | 28.0 | 24.0 | 24.4 | 0.98x |
| d7_bytes | mixed | 614.9 | 245.3 | 270.0 | 0.91x |
| d24 | miss | 55.6 | 46.6 | 55.1 | 0.85x |
| d7 | miss | 256.1 | 107.2 | 111.2 | 0.96x |
| d24_bgl | miss | 20.6 | 25.1 | 24.9 | 1.0x |
| d7_bytes | miss | 383.6 | 114.8 | 142.5 | 0.81x |
| all | repeat | 4.4-5.3 | 4.5-6.1 | 4.3-6.4 | (request cache) |

p90 and every other number: `out/*.json` of a run. A second cache run
right after (the verify configuration in `run.sh`) gave the same within
noise: sliding 21.3 / 31.5 / 17.4 / 38.3, append 21.4 / 37.7 / 14.2 / 47.8.

- **Dashboards over appended logs**: 1.6-7x faster than main, 1.2-13x faster than
  stock. A refresh adds a small segment, and a merge one segment; everything
  else is read back.
- **Updates spread over the whole index** (mixed): every segment's live
  documents change every round, so nothing can be read back. A segment whose
  live documents changed since a tree last read it is collected without the
  cache until two requests find it unchanged, which keeps the cost within
  2-10% of main.
- **Cold**: a tree's first request costs up to 24% more than main (building
  and storing each segment's part), most for the 81k-host terms.

## How a moving window keeps hitting

A part is keyed by the segment's identity, the tree, and the query with each
range narrowed to the segment's own values: a segment the window covers keys
the same wherever the window's ends are. The window's lower end falls inside
a merged segment, which spans days. Such a segment is cached in blocks of
consecutive documents, 65,536 at the smallest and four times larger at each
level up, each keyed by its own values: a block the window covers or misses
is one entry, a block it cuts is its four children. Logs arrive in time
order, so a block spans a short time and the window's end collects at most
one smallest block per segment it cuts (several segments, since the load's
four concurrent bulks give overlapping segments).

Measured on the way here (same corpus, d7 sliding): segment-level keys only,
128 → 88 ms; flat 2^18-document blocks, 46 ms; nested blocks, 32 ms.

## Memory

Parts are counted at their in-memory size. After the static phase (which adds
60 never-seen trees) the cache was at its 256 MB budget with 770 parts; after
the append rounds 555 parts, 234 MB, 3,219 hits and 1,843 misses (1,288
evictions over the run, mostly the never-seen trees). The `host` terms hold up
to 81k keys per part and are most of the bytes. A part larger than an eighth
of the budget is not kept. Counters: `aggregation_segment_cache` in
`GET /_plugins/lucene_rust/stats`.

## Correctness

- **Same reader, cache off and on**: after every live refresh (append and
  mixed) and for every sliding request, each dashboard ran with the cache off,
  then on twice, with the request cache bypassed: 220 checks, 0 differences.
- **Against stock**: the dashboards at four times with terms made exact (a
  shard size past every key): bitwise identical across stock, main, cache
  cold and cache warm (16 of 16). As asked (default shard size), stock and
  the plugin differ in `top_hosts` counts and error bounds, because their
  segments differ (20 vs 6) and each slice truncates its own terms. main and
  cache are bitwise identical there too.
- Rust tests (`crates/lucene-search/tests/agg_segment_cache.rs`): every tree
  and query over refreshes, subsets of segments, deletes, a doc-values update,
  a force merge and eviction, whole-segment and in 64-document blocks, against
  the pass without the cache; sums near 2^51 fall back to the uncached pass.
  Each of these breaks the tests: dropping the live documents or doc-values
  generations from the identity, the blocks' live documents, a wrong block
  bound, the narrowing, or the terms ordinal remap.

## Recommendation

Turn it on for log and metrics indices queried by dashboards (append-mostly,
time-ranged aggregations): 1.6-7x on the dashboards here, and identical
answers. On an index updated all over it gains nothing and costs up to 10%;
leave it off there.
