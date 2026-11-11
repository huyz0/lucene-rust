# Newest N: top N by a time field

`search_sorted` now answers "the N newest (or oldest) matching documents" by
bounding the key first and searching only the documents past the bound, instead
of collecting with comparator pruning across the whole index.

## How it works

1. A best-first walk over every segment's BKD cells (or doc-values skip-index
   blocks when the field has no points) finds a value `t` such that at least `k`
   documents have a key at least as good as `t`. Segments the query cannot match
   (zero estimated matches, or a required range outside the segment's bounds)
   are left out, and the walk is clipped to the query's own ranges on the key
   and to the `search_after` value.
2. The ordinary sorted search runs over `query AND key in [t, previous bound)`.
   If that slice yields fewer than N hits, `k` grows (×8 after an empty slice,
   otherwise at the observed hit rate, at least ×2) and the next slice runs.
3. Once the slices would cover more than an eighth of the values (or the
   query's estimated matches), the rest runs as the ordinary search after the
   last bound, so a filter that matches only old documents costs about what it
   did before.
4. The total is counted separately up to the track-total-hits threshold, with
   `Weight::count` shortcuts per segment. It is exact whenever counting finishes
   and a lower bound (`GreaterThanOrEqualTo`) otherwise, as in Lucene.

It applies to one `LONG` or `INT` key with a `MIN`/`MAX`/`MIDDLE_*` selector
whose missing value sorts after the bound, with no `index_sort` prefix,
`min_score` or custom comparator. Everything else takes the ordinary path
unchanged. Hits are always the ordinary search's hits; the unit test
`newest_n::tests` compares both paths over ties, missing values, multi-valued
fields, a shuffled segment, an `INT` key, deletions, filters that never match
the newest documents and `search_after` paging.

## Setup

- Machine: 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, Linux 6.18; rustc 1.97.1;
  Lucene 10.5.1 on JDK 21.
- Corpus: Loghub BGL, HDFS, OpenSSH and Thunderbird (6,454,808 lines)
  interleaved in a seeded random order, one document a line, with times spread
  over 7 days in arrival order. Fields: `ts` (LongPoint + doc values), `ts_s`
  (doc values with a skip index), `tsec` (seconds, so ties), `tmiss` (missing on
  one document in ten and on the newest 0.5%), `src` (keyword).
- The index is written by Lucene (`benchmarks/newest-n/NewestN.java build`) and
  both engines read the same files. Two layouts: 26 segments (250k lines each, no merges) and 5 segments (`-Dmerges=tiered`).
- Warm: median of at least 1.2 s of runs after 0.6 s warm-up, one thread
  (`examples/newest_n.rs`, `NewestN.java query`). Track-total-hits 1000.
- Cold: `benchmarks/newest-n/cold.sh` drops the page cache and starts a fresh
  process per run, timing reader open plus one query; median of 3. Java's figure
  excludes JVM start but includes class loading and JIT warm-up.
- Results: `benchmarks/newest-n/results/` holds the CSVs and every shape's hit
  list. **All 51 shape runs in both layouts return the same hits (doc and sort
  value, in order) as Lucene 10.5.1 and as main.**

## Warm, 26 segments (ms)

| shape | N | Lucene 10.5.1 | main | this change | vs main |
|---|---:|---:|---:|---:|---:|
| match_all ts desc | 10 | 349 | 539 | 0.16 | 3410.2x |
| match_all ts asc | 10 | 2.68 | 0.04 | 0.04 | 1.0x |
| match_all ts_s(skipper) desc | 10 | 305 | 528 | 0.31 | 1717.8x |
| match_all ts_s(skipper) asc | 10 | 2.72 | 0.08 | 0.08 | 1.0x |
| src=openssh ts desc | 10 | 31 | 56 | 3.81 | 14.8x |
| src=hdfs ts desc | 10 | 104 | 204 | 7.28 | 28.0x |
| src=openssh ts_s(skipper) desc | 10 | 35 | 75 | 4.90 | 15.4x |
| range24h ts desc | 10 | 40 | 70 | 0.19 | 369.3x |
| range24h&hdfs ts desc | 10 | 14 | 24 | 7.63 | 3.2x |
| range24h&openssh(0 hits) ts desc | 10 | 0.13 | 0.01 | 0.01 | 0.8x |
| match_all tsec(ties) desc | 10 | 346 | 391 | 0.17 | 2247.2x |
| match_all tsec(ties) asc | 10 | 2.72 | 0.03 | 0.04 | 0.8x |
| match_all tmiss desc missing_last | 10 | 509 | 477 | 0.09 | 5441.6x |
| match_all tmiss asc missing_last | 10 | 4.44 | 0.04 | 0.03 | 1.3x |
| match_all tmiss desc missing_first | 10 | 4.91 | 0.04 | 0.05 | 0.8x |
| match_all ts desc search_after | 10 | 1271 | 1029 | 0.35 | 2907.8x |
| src=hdfs tsec(ties) desc search_after | 10 | 348 | 405 | 8.54 | 47.4x |
| match_all ts desc | 100 | 978 | 777 | 0.21 | 3646.3x |
| match_all ts asc | 100 | 2.70 | 0.04 | 0.09 | 0.5x |
| match_all ts_s(skipper) desc | 100 | 942 | 771 | 0.48 | 1622.1x |
| match_all ts_s(skipper) asc | 100 | 2.73 | 0.10 | 0.13 | 0.7x |
| src=openssh ts desc | 100 | 120 | 88 | 7.79 | 11.3x |
| src=hdfs ts desc | 100 | 233 | 265 | 13 | 19.9x |
| src=openssh ts_s(skipper) desc | 100 | 78 | 108 | 10 | 10.8x |
| range24h ts desc | 100 | 137 | 104 | 0.20 | 516.4x |
| range24h&hdfs ts desc | 100 | 47 | 37 | 15 | 2.5x |
| range24h&openssh(0 hits) ts desc | 100 | 0.05 | 0.01 | 0.01 | 0.8x |
| match_all tsec(ties) desc | 100 | 1289 | 969 | 0.19 | 5184.9x |
| match_all tsec(ties) asc | 100 | 2.74 | 0.05 | 0.06 | 0.9x |
| match_all tmiss desc missing_last | 100 | 1055 | 673 | 0.06 | 11094.3x |
| match_all tmiss asc missing_last | 100 | 3.83 | 0.05 | 0.05 | 1.1x |
| match_all tmiss desc missing_first | 100 | 3.67 | 0.13 | 0.14 | 0.9x |
| match_all ts desc search_after | 100 | 2164 | 1566 | 0.40 | 3947.5x |
| src=hdfs tsec(ties) desc search_after | 100 | 679 | 640 | 18 | 36.3x |
| match_all ts desc | 1000 | 1462 | 1090 | 0.24 | 4474.5x |
| match_all ts asc | 1000 | 3.05 | 0.24 | 0.25 | 1.0x |
| match_all ts_s(skipper) desc | 1000 | 1306 | 1085 | 0.75 | 1447.3x |
| match_all ts_s(skipper) asc | 1000 | 6.77 | 0.29 | 0.31 | 0.9x |
| src=openssh ts desc | 1000 | 182 | 122 | 15 | 8.1x |
| src=hdfs ts desc | 1000 | 422 | 322 | 24 | 13.2x |
| src=openssh ts_s(skipper) desc | 1000 | 131 | 123 | 18 | 6.9x |
| range24h ts desc | 1000 | 182 | 150 | 0.24 | 620.2x |
| range24h&hdfs ts desc | 1000 | 51 | 54 | 26 | 2.1x |
| range24h&openssh(0 hits) ts desc | 1000 | 0.05 | 0.01 | 0.02 | 0.5x |
| match_all tsec(ties) desc | 1000 | 1644 | 1404 | 0.24 | 5929.5x |
| match_all tsec(ties) asc | 1000 | 4.04 | 0.27 | 0.26 | 1.0x |
| match_all tmiss desc missing_last | 1000 | 1428 | 1021 | 0.35 | 2880.2x |
| match_all tmiss asc missing_last | 1000 | 5.62 | 0.26 | 0.29 | 0.9x |
| match_all tmiss desc missing_first | 1000 | 5.46 | 2.92 | 2.90 | 1.0x |
| match_all ts desc search_after | 1000 | 3252 | 2203 | 0.85 | 2587.1x |
| src=hdfs tsec(ties) desc search_after | 1000 | 1312 | 821 | 31 | 26.8x |

## Warm, 5 segments (ms)

| shape | N | Lucene 10.5.1 | main | this change | vs main |
|---|---:|---:|---:|---:|---:|
| match_all ts desc | 10 | 171 | 251 | 0.16 | 1566.2x |
| match_all ts asc | 10 | 60 | 14 | 12 | 1.1x |
| match_all ts_s(skipper) desc | 10 | 154 | 222 | 0.46 | 485.7x |
| match_all ts_s(skipper) asc | 10 | 56 | 0.33 | 0.48 | 0.7x |
| src=openssh ts desc | 10 | 16 | 42 | 42 | 1.0x |
| src=hdfs ts desc | 10 | 44 | 124 | 7.24 | 17.2x |
| src=openssh ts_s(skipper) desc | 10 | 12 | 54 | 56 | 1.0x |
| range24h ts desc | 10 | 40 | 72 | 0.19 | 381.7x |
| range24h&hdfs ts desc | 10 | 13 | 26 | 7.42 | 3.4x |
| range24h&openssh(0 hits) ts desc | 10 | 0.44 | 0.00 | 0.25 | 0.0x |
| match_all tsec(ties) desc | 10 | 154 | 212 | 0.16 | 1348.3x |
| match_all tsec(ties) asc | 10 | 58 | 14 | 11 | 1.2x |
| match_all tmiss desc missing_last | 10 | 191 | 321 | 0.07 | 4343.1x |
| match_all tmiss asc missing_last | 10 | 78 | 25 | 0.04 | 672.1x |
| match_all tmiss desc missing_first | 10 | 79 | 0.04 | 0.04 | 1.1x |
| match_all ts desc search_after | 10 | 342 | 542 | 0.29 | 1897.5x |
| src=hdfs tsec(ties) desc search_after | 10 | 147 | 228 | 8.79 | 25.9x |
| match_all ts desc | 100 | 340 | 360 | 0.19 | 1940.2x |
| match_all ts asc | 100 | 62 | 14 | 13 | 1.1x |
| match_all ts_s(skipper) desc | 100 | 318 | 356 | 0.76 | 467.7x |
| match_all ts_s(skipper) asc | 100 | 61 | 0.35 | 0.39 | 0.9x |
| src=openssh ts desc | 100 | 33 | 54 | 55 | 1.0x |
| src=hdfs ts desc | 100 | 88 | 145 | 15 | 9.8x |
| src=openssh ts_s(skipper) desc | 100 | 28 | 67 | 68 | 1.0x |
| range24h ts desc | 100 | 93 | 110 | 0.21 | 514.2x |
| range24h&hdfs ts desc | 100 | 29 | 39 | 15 | 2.6x |
| range24h&openssh(0 hits) ts desc | 100 | 0.29 | 0.00 | 0.24 | 0.0x |
| match_all tsec(ties) desc | 100 | 411 | 460 | 0.19 | 2436.2x |
| match_all tsec(ties) asc | 100 | 66 | 14 | 13 | 1.1x |
| match_all tmiss desc missing_last | 100 | 327 | 403 | 0.05 | 8547.1x |
| match_all tmiss asc missing_last | 100 | 80 | 24 | 0.05 | 465.4x |
| match_all tmiss desc missing_first | 100 | 78 | 0.14 | 0.14 | 1.0x |
| match_all ts desc search_after | 100 | 628 | 751 | 0.40 | 1887.3x |
| src=hdfs tsec(ties) desc search_after | 100 | 248 | 326 | 16 | 20.7x |
| match_all ts desc | 1000 | 432 | 531 | 0.21 | 2477.0x |
| match_all ts asc | 1000 | 71 | 81 | 69 | 1.2x |
| match_all ts_s(skipper) desc | 1000 | 455 | 488 | 0.70 | 692.7x |
| match_all ts_s(skipper) asc | 1000 | 67 | 26 | 26 | 1.0x |
| src=openssh ts desc | 1000 | 48 | 64 | 67 | 1.0x |
| src=hdfs ts desc | 1000 | 135 | 169 | 31 | 5.4x |
| src=openssh ts_s(skipper) desc | 1000 | 42 | 76 | 70 | 1.1x |
| range24h ts desc | 1000 | 142 | 151 | 0.27 | 564.3x |
| range24h&hdfs ts desc | 1000 | 41 | 49 | 25 | 2.0x |
| range24h&openssh(0 hits) ts desc | 1000 | 0.29 | 0.00 | 0.23 | 0.0x |
| match_all tsec(ties) desc | 1000 | 542 | 599 | 0.21 | 2795.3x |
| match_all tsec(ties) asc | 1000 | 70 | 75 | 68 | 1.1x |
| match_all tmiss desc missing_last | 1000 | 459 | 541 | 0.28 | 1926.2x |
| match_all tmiss asc missing_last | 1000 | 80 | 131 | 0.24 | 552.6x |
| match_all tmiss desc missing_first | 1000 | 79 | 97 | 92 | 1.1x |
| match_all ts desc search_after | 1000 | 814 | 1036 | 0.84 | 1236.7x |
| src=hdfs tsec(ties) desc search_after | 1000 | 359 | 384 | 28 | 13.6x |

On merged segments the selective `src=openssh` filter (0.6% of documents, none
in the last day) falls back to the ordinary search almost at once and runs at
main's speed, which is about 2.5x slower than Lucene there; that gap is in the
ordinary filtered sorted search and is a follow-up. The 0-hit
`range24h&openssh` shape pays about 0.25 ms for the walk.

## Cold, 26 segments (ms, reader open + one query)

| shape | N | Lucene 10.5.1 | main | this change |
|---|---:|---:|---:|---:|
| match_all ts desc | 10 | 1176 | 695 | 28 |
| match_all ts asc | 10 | 656 | 22 | 27 |
| src=openssh ts desc | 10 | 674 | 101 | 43 |
| src=hdfs ts desc | 10 | 768 | 237 | 54 |
| range24h ts desc | 10 | 664 | 101 | 26 |
| range24h&hdfs ts desc | 10 | 695 | 50 | 53 |
| match_all tsec(ties) desc | 10 | 1012 | 520 | 29 |
| match_all tmiss desc missing_last | 10 | 970 | 627 | 27 |
| match_all ts desc search_after | 10 | 1214 | 1262 | 29 |
| match_all ts desc | 100 | 1224 | 904 | 23 |
| src=openssh ts desc | 100 | 702 | 132 | 58 |
| range24h ts desc | 100 | 704 | 153 | 26 |
| match_all ts desc | 1000 | 1519 | 1223 | 27 |
| range24h ts desc | 1000 | 718 | 193 | 28 |

## Through OpenSearch (REST, ms)

`benchmarks/newest-n/os_bench.py` against OpenSearch 3.9.0 in Docker (3 CPUs,
client pinned to the fourth), index of the same 6.45M lines in 4 segments,
`sort: [{ts: desc|asc}]`, `request_cache=false`, median round trip of 50 after
5 warm-ups. Default track-total-hits (10,000) unless `tth=false`.

Stock OpenSearch rewrites a sorted range or `match_all` into its
`ApproximatePointRangeQuery`, which the plugin used to hand back to Java. The
new index setting `index.lucene_rust.search.native_approximate` (default
`false`, dynamic) keeps those queries native.

| shape | size | stock OpenSearch | plugin (default) | plugin, `native_approximate` | hits identical |
|---|---:|---:|---:|---:|---|
| match_all desc | 10 | 15.1 | 13.9 | 3.8 | yes |
| match_all asc | 10 | 4.3 | 4.3 | 3.4 | yes |
| range24h desc | 10 | 11.5 | 12.1 | 3.2 | yes |
| match_all desc tth=false | 10 | 5.2 | 4.9 | 3.0 | yes |
| match_all asc tth=false | 10 | 3.6 | 3.5 | 2.9 | yes |
| range24h desc tth=false | 10 | 4.1 | 4.1 | 2.8 | yes |
| src=hdfs desc | 10 | 200.3 | 12.8 | 12.1 | yes |
| src=openssh desc | 10 | 70.3 | 75.3 | 78.9 | yes |
| range24h&hdfs desc | 10 | 31.0 | 14.2 | 14.1 | yes |
| match_all desc | 100 | 17.6 | 16.5 | 3.3 | yes |
| match_all asc | 100 | 4.4 | 4.5 | 3.4 | yes |
| range24h desc | 100 | 16.9 | 15.6 | 3.3 | yes |
| match_all desc tth=false | 100 | 4.4 | 4.8 | 4.6 | yes |
| match_all asc tth=false | 100 | 3.4 | 3.6 | 3.2 | yes |
| range24h desc tth=false | 100 | 3.9 | 4.6 | 2.9 | yes |
| src=hdfs desc | 100 | 300.1 | 21.8 | 20.2 | yes |
| src=openssh desc | 100 | 107.7 | 108.3 | 105.3 | yes |
| range24h&hdfs desc | 100 | 47.5 | 23.2 | 22.5 | yes |

Hits and totals are identical to stock on every shape here.

## Recommendation

- The Lucene-layer change is on by default: it returns the ordinary search's
  hits by construction and is tested against it.
- Keep `native_approximate` opt-in for now. OpenSearch's approximation collects
  documents in BKD order and can pick a different document among equal sort
  values at a leaf boundary; the native path returns the exact Lucene order.
  They agreed on every query above, and switching the default is a candidate
  once tie behavior is accepted as a documented deviation.
