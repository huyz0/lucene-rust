# Parity ledger

What of Apache Lucene **10.5.0** (OpenSearch 3.8.0's pin, `gradle/libs.versions.toml`)
this port has, file by file: ported, partial, deliberately not ported, or
Rust-only. This page is the index; the rows live in one file per area under
[`docs/parity/`](parity/). Process: the `parity-tracking` skill. Checked by
`scripts/check-parity.py` (every Rust path and `::item` exists, every
`crates/*/src/*.rs` has a row, the index links every area file, budgets).

## Reading a row

Every area file is one table, `| Java | Rust | Status |`:

- **Java** -- the Lucene class(es), relative to `org/apache/lucene/`
  (`codecs/lucene90/Lucene90NormsFormat`), or `--` for Rust-only machinery.
  `scripts/check-java-refs.py` checks them against the 10.5.0 sources.
- **Rust** -- `crate/src/file.rs::{items}`, without the `crates/` prefix.
  Every path and item must exist.
- **Status** -- a status word and milestone, then only current facts:
  **Differs:** (behaviour that is not Java's, and why), **Gap:** (what is
  missing), **Tests:** (fixture generator -> test file, named tests),
  **Bench:** (current `scripts/bench-micro.sh` ratio, Rust speed / Java
  speed, so above 1.0 is faster). History -- which batch did what, superseded
  claims, test-count logs -- is not kept here: it is `git log` and
  [`docs/sweep/`](sweep/).

## Status vocabulary

| Word | Meaning |
|---|---|
| **ported** | Java's behaviour on Java's bytes, verified against a Java fixture; any deviation listed under Differs. |
| **partial** | Ported for a stated scope; the rest listed under Gap. |
| **rust-only** | No Java counterpart (boundary, test or performance machinery). |
| **not-needed** | Deliberately not ported, with the reason. |
| **generated** | Produced by a script from Lucene's own sources (tables, automata). |
| **native** / **falls back** | Plugin rows only: served by Rust, or left to Lucene with the reason OpenSearch's stats count. |

Milestones (M1-M10, task ids such as T9.4) are [`docs/roadmap.md`](roadmap.md)'s.

## Areas

| File | Covers | Rows |
|---|---|---|
| [matrix.md](parity/matrix.md) | The supported matrix: formats read and written, search and write features, what is out of scope | 14 |
| [util.md](parity/util.md) | `lucene-util`: bits, packed ints, `BytesRef*`, FST, automata, hashing, sorting | 32 |
| [util-geo.md](parity/util-geo.md) | `lucene-util` geo: encodings, tessellator, polygons, distance (M9 T9.1) | 10 |
| [store.md](parity/store.md) | `lucene-store`: directories, inputs/outputs, locks, checksums | 22 |
| [codecs.md](parity/codecs.md) | `lucene-codecs`: the `Lucene104` codec, every format read and written | 67 |
| [backward-codecs.md](parity/backward-codecs.md) | `lucene-backward-codecs`: Lucene 9.0-10.4 formats (M8) | 18 |
| [analysis.md](parity/analysis.md) | `lucene-analysis`: analyzers, tokenizers, filters | 18 |
| [analysis-common.md](parity/analysis-common.md) | `lucene-analysis-common` (M11) | 15 |
| [index.md](parity/index.md) | `lucene-index`: documents, `IndexWriter`, flush, merge, deletes, segment files, NRT snapshots | 45 |
| [search-queries.md](parity/search-queries.md) | `lucene-search`: query types, rewrites, parser, vector queries | 37 |
| [search-execution.md](parity/search-execution.md) | `lucene-search`: scorer tree, bulk scorers, similarities, multi-segment fan-out | 29 |
| [search-collectors.md](parity/search-collectors.md) | `lucene-search`: collectors, sorting, facets, aggregations, highlighting, fetch | 33 |
| [search-readers.md](parity/search-readers.md) | `lucene-search`: `DirectoryReader`, NRT readers, `CheckIndex`, searcher management | 26 |
| [geo-points.md](parity/geo-points.md) | Geo point fields and queries (M9 T9.2) | 9 |
| [geo-shapes.md](parity/geo-shapes.md) | Geo shape fields and queries (M9 T9.3) | 8 |
| [spatial3d.md](parity/spatial3d.md) | `lucene-spatial3d` (M9 T9.4) | 16 |
| [spatial-extras.md](parity/spatial-extras.md) | `lucene-spatial-extras` (M9 T9.5) | 25 |
| [joins.md](parity/joins.md) | Document blocks and block joins, `lucene-join` (M10) | 11 |
| [grouping.md](parity/grouping.md) | `lucene-grouping` (M10 T10.4) | 11 |
| [functions.md](parity/functions.md) | `lucene-queries` function queries (M10 T10.5) | 11 |
| [queries.md](parity/queries.md) | `lucene-queries` intervals, payloads, more-like-this, common terms, spans (M10 T10.6) | 12 |
| [analysis-lang.md](parity/analysis-lang.md) | `lucene-analysis-common` synonyms, classic, language packages (M11 part 3) | 10 |
| [ffi.md](parity/ffi.md) | `lucene-ffi`: the general C ABI (handles, searches, writer) | 16 |
| [plugin.md](parity/plugin.md) | `lucene-ffi`: the OpenSearch plugin boundary (JVM reader, FFM, query phase), M10 query shapes | 10 |
| [plugin-geo.md](parity/plugin-geo.md) | OpenSearch plugin: geo queries and geo sorting (M9 T9.6) | 7 |
| [engine.md](parity/engine.md) | The OpenSearch engine (M5): writer features it needed | 12 |

## Supported, in one paragraph

Lucene 10.5.0's default codec (`Lucene104`) is read and written in full, and
real Lucene reads what Rust writes (`scripts/verify-write-path.sh`,
`scripts/verify-interop.sh`); every format Lucene 9.0-10.4 wrote is read
(M8). Searching covers the query types OpenSearch builds, BM25 and the other
similarities, sorting, OpenSearch's shard aggregations and fetch; the
OpenSearch plugin serves them natively and the M5 engine writes through
Rust. Lucene 8, other codecs and features with no OpenSearch consumer are
out of scope. Details, with the rows behind each claim:
[matrix.md](parity/matrix.md).

## Where a new row goes

In the file of the **crate and module the Rust code lives in** -- not of the
Java package: a `lucene-search` collector goes in `search-collectors.md`,
even for a `lucene-grouping` class. The M9/M10 module files (geo, spatial,
joins, grouping, functions) and `engine.md` are the exceptions, grouped by
feature because a feature spans crates. Next to the rows for the same Java
class; bump the row count above. A row past 2,000 characters or a file past
80 KB fails the check: split the row (read side / write side, one class per
row) or the area, and link [`docs/sweep/`](sweep/) for rationale instead of
writing it out.
