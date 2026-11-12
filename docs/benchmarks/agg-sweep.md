# Aggregations, sort and fetch sweep (2026-11-12)

A pass over the native aggregation, sort, doc-values and stored-field fetch
paths for CPU, allocations, copies and peak memory. Two benches:

- `benchmarks/rust-runner` bin `sweep <index> [filter]`: each shape through the
  FFI entry points the plugin calls (`ffi_jvm_reader_aggregate_tree_alloc`,
  `search_sorted_alloc`, `document_alloc`, `doc_values_alloc`) over the bench
  corpus (1M docs), with a counting allocator over mimalloc: time per call,
  allocations, bytes, peak KiB and a digest of the result. `SWEEP_BIG=<bytes>`
  prints a backtrace per allocation at least that large.
- `benchmarks/agg-sweep/run.sh base.so new.so`: 22 REST shapes on OpenSearch
  3.9.0, 1M documents of `bench_nodes.py`'s corpus force-merged to one
  segment, `request_cache=false`, median of 30 after 10 warm-up requests;
  stock, the plugin with native search off (Java Lucene) and two plugin
  builds alternated twice (best of two).

## In process (merged corpus, main -> branch, digests identical)

| shape | time | allocations | peak |
|---|---|---|---|
| filters > stats | 0.63x | 994k -> 87 | 4.3 MB -> 0.4 MB |
| range > stats | 0.30x | same | 4 MB -> 0.16 MB (filtered) |
| percentiles, terms > percentiles | 0.54-0.60x | 1161 -> 48 | |
| cardinality keyword | 0.40x | 196k -> 65 | 5 MB -> 1.2 MB (filtered) |
| terms keyword > terms | 0.65-0.80x | 124k -> 330 | |
| histogram > stats | 0.76-0.79x | | |
| any agg under a query | | | 4 MB -> ~0.15 MB |

What changed:

- `filters`: the per-document list of matching filters is gone (a loop and a
  flag).
- keyword `cardinality`: terms hashed in place from the dictionary (no
  `String` and copy per distinct ordinal); a window path reads the
  prefetched ordinals.
- sub-aggregation finish groups only the requested owners' buckets instead
  of building a list per parent bucket.
- a query's matches for the aggregation pass are a segment bit set
  (`aggs::segment_match_bits`, 1 bit a document) instead of an `i32` list (32
  bits a match).
- `range` has a window path; disjoint ranges (the usual shape) find their
  range by counting starts, not by `RangeAggregator`'s binary search.
- t-digest's stable sort keys each value with its index in one `u128` (an
  unstable sort of distinct keys), not an indirect comparator; NaN keeps the
  comparator.
- `histogram`/`date_histogram` windows remember recent keys' buckets, and turn
  that off when it misses (no slower on high-cardinality keys).
- `composite` whose first source OpenSearch reads sorted (a keyword over a
  `match_all`, `TermsSortedDocsProducer`): native now walks that field's terms
  in order from `after`'s and stops after a term that adds no competitive
  key, as Java does, instead of reading every document.

## End to end (one merged segment, ms)

| shape | stock | plugin Java | main | branch |
|---|---|---|---|---|
| terms category > stats price | 41.7 | 36.1 | 14.3 | 13.8 |
| terms tag > terms category | 43.7 | 40.7 | 29.0 | 29.8 |
| terms tag, lone | 3.3 | 3.4 | 3.4 | 3.3 |
| terms category, filtered t8 | 9.1 | 9.5 | 6.0 | 5.0 |
| filters 3 > stats price | 48.2 | 47.7 | 60.2 | 45.5 |
| range num 4 > avg price | 14.2 | 10.8 | 42.8 | 13.4 |
| percentiles price | 90.0 | 83.0 | 71.8 | 42.8 |
| terms category > percentiles price | 77.5 | 77.6 | 77.9 | 48.4 |
| cardinality tag | 7.0 | 8.3 | 13.4 | 5.8 |
| cardinality num | 11.1 | 11.8 | 9.2 | 9.0 |
| histogram price 10 > stats num | 34.9 | 36.1 | 20.2 | 16.3 |
| date_histogram day > terms category | 74.9 | 76.6 | 40.9 | 37.9 |
| composite category+tag 100 | 5.8 | 5.0 | 33.2 | 4.5 |
| composite category+tag 100 > avg price | 5.9 | 5.3 | 36.4 | 4.6 |
| composite category+tag 100 after | 3.6 | 4.0 | 40.1 | 4.0 |
| composite category 10 | 5.1 | 5.8 | 5.7 | 5.9 |
| terms num (numeric) 25 | 140.4 | 167.8 | 133.1 | 110.7 |
| t8, terms tag > top_hits 3 | 30.4 | 27.6 | 22.4 | 22.5 |
| t0 sort num desc, 10 | 7.6 | 5.3 | 2.1 | 2.1 |
| t0 sort tag asc, 100 | 6.6 | 5.1 | 4.3 | 4.0 |
| t5 fetch 100 _source | 6.1 | 6.1 | 3.7 | 3.5 |
| t5 fetch 100 docvalue_fields | 5.1 | 6.6 | 4.6 | 3.7 |

Every shape's hits and aggregations are byte-equal between main, the branch
and the plugin's Java path on the same index. Stock differs only where its
own merge ordered documents differently (last-digit float sums, tie order).

## Checked and left alone

- Sorts: faithful. A descending numeric sort over a ranged query cannot use
  the competitive iterator, in Lucene too.
- Stored fetch: LZ4 decompression dominates (~2.8 us a document). Doc-values
  fetch: 0.7 us a hit.
- Open: `top_hits` buffers every scored match before ranking (16 MB peak at
  1M matches); numeric `terms` peaks at 69 MB for 630k keys; the range shape
  is still ~1.25x the plugin's Java path, which counts ranges from the points
  with sub-aggregations (filter rewrite).
- Highlighting runs in Java Lucene in the plugin (no native path).
