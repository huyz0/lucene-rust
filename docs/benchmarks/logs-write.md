# Write path on log-shaped documents

`scripts/bench-logs-write.sh` (2026-11-11, 4 vCPU cloud VM, 15 GB). 2M
documents shaped like an OpenSearch log index: `_id` (stored and indexed),
a ~300-byte `_source`, six keyword fields (indexed without norms, sorted-set
doc values, Zipf cardinalities 6 to 1000), four longs (a point and a
sorted-numeric doc value each), `_seq_no`, `_version`, `_primary_term`, and
a 20-word `message` with positions. Both engines index from 4 threads with a
reader refreshed every second, a 256 MB buffer and default merging, then
commit and wait for merges. `java` is stock Lucene 10.5.2
(`WhitespaceAnalyzer` on `message`); `rust` is this port's concurrent writer
with mimalloc, given `message` already split into words. Each thread builds
its documents as it adds them. Medians of 3 runs.

| | docs/s | total s | CPU s | refresh mean ms | peak RSS | force merge s (CPU s) |
|---|---:|---:|---:|---:|---:|---:|
| java | 124,700 | 17.7 | 63.2 | 545 | - | 6.5 (13.4) |
| rust before | 200,300 | 11.9 | 42.5 | 1,025 | 2.77 GB | 4.3 (5.4) |
| rust after | 226,300 | 10.0 | 36.1 | 546 | 2.40 GB | 4.3 (5.4) |

Peak RSS includes the 380 MB corpus held in memory. Index size: 453 MB
for both engines, and the files this port writes are byte-identical before
and after.

What changed (flush and add path only):

- **Field lookup by number** was a SipHash map probe for every field of
  every document, at add (validation, index options) and at flush. It is an
  array now. About 2% docs/s here and on the 2M-document text bench
  (`bench-partition-cost.sh fts`, plain layout).
- **Doc-values columns at flush** were built one field at a time, each
  scanning every document's values, with a vector per document for
  SORTED_NUMERIC and SORTED_SET and a copy of every SORTED_SET value. One
  pass now routes each value to its field's column, keeps SORTED_NUMERIC
  values flat and copies each distinct SORTED_SET value once (the column
  shapes merges already write). Refresh 1,025 -> 680 ms, CPU -10%.
- **The flush's decoded doc values and points** grew one segment-sized
  array by doubling, faulting its pages in again at each step. It is sized
  from the first document now. Refresh 680 -> 546 ms, CPU -5%.

The text bench (`partition-cost.md`, plain): refresh 451 -> 431 ms mean,
docs/s about +2%. The plugin-path allocation profile (`alloc-profile
200000`): 44 -> 43 allocations per indexed document, flush bytes allocated
125 -> 115 MB per flush; search and fetch allocations unchanged.
