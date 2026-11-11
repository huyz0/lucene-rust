# Full-text search in a shard shared by tenants

`scripts/bench-fts-partition.sh` (2026-11-11, 4 vCPU cloud VM, one thread,
query cache off on both sides). 2M documents, 1000 tenants with Zipf sizes
(rank 1 about 360k documents, rank 300 about 700), interleaved on arrival;
40-word bodies over a 50k-word Zipf vocabulary with planted two-word phrases.
Stock Lucene 10.5.2 writes three indexes, each force-merged to 4 segments:
unsorted (`plain`, 191 MB), sorted by `tenant` then `ts` newest first
(`composite`, 190 MB), and that sort with document blocks of 0-2 children
per document and a parent field (`blocks`, 260 MB; the OpenSearch nested
layout, with `_primary_term` as the non-nested filter in every query).

Every query filters by its tenant; a third of the text queries also bound
`ts` to a tenth of the span. Kinds: `term`, `and` / `or` of two terms,
`phrase`, `newest` (the tenant's 10 newest, sorted by `ts` desc) and
`newest_term` (the same with a body term). Classes: large (ranks 1-5), mid
(20-60), small (300+). `java` is stock Lucene, `base` this port before the
change (main 7a6ea04b), `rust` after it.

## Hot mean latency, microseconds

| class, kind | plain java | plain rust | composite java | composite base | composite rust | blocks java | blocks base | blocks rust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large term | 739 | 1352 | 393 | 74 | 73 | 473 | 660 | 118 |
| large phrase | 1091 | 1937 | 479 | 99 | 105 | 633 | 745 | 162 |
| large and | 995 | 1040 | 235 | 176 | 177 | 353 | 370 | 312 |
| large or | 1631 | 1772 | 375 | 277 | 283 | 683 | 727 | 625 |
| large newest | 608 | 607 | 88 | 35 | 35 | 123 | 367 | 94 |
| large newest_term | 453 | 533 | 215 | 155 | 90 | 228 | 193 | 137 |
| mid term | 616 | 1088 | 372 | 16 | 16 | 427 | 625 | 51 |
| mid phrase | 654 | 1252 | 388 | 26 | 28 | 525 | 646 | 72 |
| mid newest | 545 | 767 | 95 | 77 | 16 | 137 | 227 | 69 |
| small term | 490 | 1051 | 345 | 13 | 13 | 412 | 617 | 44 |
| small phrase | 609 | 1140 | 424 | 22 | 21 | 442 | 622 | 61 |
| small or | 1093 | 809 | 92 | 32 | 33 | 145 | 92 | 83 |

Raw CSVs and every query's top hits: `/mnt/project-files/perf/fts-partition/`.

- **Composite sort** makes a tenant's filter one doc-id run: text queries
  inside small and mid tenants are 20-70x faster than unsorted and 10-25x
  faster than stock Lucene on the same sorted index (which still
  intersects the tenant's postings).
- **Sorted by the second key** (`newest`) inside a pinned tenant ends each
  segment at the first non-competitive document: mid tenants 77 -> 16 us,
  large with a term 155 -> 90 us. Large tenants alone were already fast
  through the time points' competitive iterator.
- **Nested blocks** used to turn the run off (`blocks base`, at unsorted
  speed). Runs of whole blocks bring term and phrase queries 5-14x down,
  within 3-4x of the flat composite index (the parents' bitset is walked
  instead of a dense range).
- **Bytes read cold** (whole query set once on a dropped page cache,
  `/proc/self/io` read_bytes): plain 82.7 MB, composite 64.9 MB (stock Lucene
  on it 73.3 MB), blocks 89.9 MB after vs 98.2 MB before. A tenant's
  postings, norms and doc values sit together, so fewer pages are touched;
  with blocks the run skips reading the tenant's postings.
- **Scores**: every engine and layout returns the same top hits and score
  bits as stock Lucene on the unsorted index (360/360 queries; blocks
  against stock on the blocks index). BM25 statistics are the shard's
  (`docFreq`, `docCount`, average length over all tenants) in all of them;
  the sort changes doc ids, not statistics.
- Not from this change: on the unsorted index this port's filtered `term`
  and `phrase` are about 2x slower than stock Lucene (separate follow-up).
