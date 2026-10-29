# Parity ledger

What of Apache Lucene **10.5.2** (OpenSearch `main`'s pin, `gradle/libs.versions.toml`)
this port has, file by file: ported, partial, deliberately not ported, or
Rust-only. This page is the index; the rows live in one file per area under
[`docs/parity/`](parity/). Process: the `parity-tracking` skill. Checked by
`scripts/check-parity.py` (every Rust path and `::item` exists, every
`crates/*/src/*.rs` has a row, the index links every area file, budgets).

**Status (2026-10-28): the port is complete** by the roadmap's definition
(M14's last criterion): every class of every 10.5.2 module is `ported` or
`not-needed` with a reason in `docs/inventory/*.tsv`, and
`scripts/check-port-inventory.py` passes with nothing allowlisted. The
52 rows below still marked **partial** name narrower behaviour inside
ported classes (each under its **Gap:**), not missing classes.

## Reading a row

Every area file is one table, `| Java | Rust | Status |`:

- **Java** -- the Lucene class(es), relative to `org/apache/lucene/`
  (`codecs/lucene90/Lucene90NormsFormat`), or `--` for Rust-only machinery.
  `scripts/check-java-refs.py` checks them against the 10.5.2 sources.
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
| **not-supported** | Cannot be ported faithfully (licence, missing data): refused with the named typed error, or no Rust type. |
| **generated** | Produced by a script from Lucene's own sources (tables, automata). |
| **native** / **falls back** | Plugin rows only: served by Rust, or left to Lucene with the reason OpenSearch's stats count. |

Milestones (M1-M12, task ids such as T9.4) are [`docs/roadmap.md`](roadmap.md)'s.

## Areas

| File | Covers | Rows |
|---|---|---|
| [matrix.md](parity/matrix.md) | The supported matrix: formats read and written, search and write features, what is out of scope | 14 |
| [util.md](parity/util.md) | `lucene-util`: bits, packed ints, `BytesRef*`, FST, automata, hashing, sorting | 33 |
| [util-geo.md](parity/util-geo.md) | `lucene-util` geo: encodings, tessellator, polygons, distance (M9 T9.1) | 10 |
| [store.md](parity/store.md) | `lucene-store`: directories, inputs/outputs, locks, checksums | 22 |
| [codecs.md](parity/codecs.md) | `lucene-codecs`: the `Lucene104` codec, every format read and written | 75 |
| [backward-codecs.md](parity/backward-codecs.md) | `lucene-backward-codecs`: Lucene 9.0-10.4 formats (M8) | 18 |
| [analysis.md](parity/analysis.md) | `lucene-analysis`: analyzers, tokenizers, filters | 18 |
| [analysis-common.md](parity/analysis-common.md) | `lucene-analysis-common` (M11) | 22 |
| [index.md](parity/index.md) | `lucene-index`: documents, `IndexWriter`, flush, merge, deletes, segment files, NRT snapshots | 52 |
| [search-queries.md](parity/search-queries.md) | `lucene-search`: query types, rewrites, parser, vector queries | 39 |
| [queryparser.md](parity/queryparser.md) | `lucene-queryparser`: the classic, multi-field, simple, extendable, surround and complex-phrase, XML and flexible query parsers (M13 T13.1) | 19 |
| [search-execution.md](parity/search-execution.md) | `lucene-search`: scorer tree, bulk scorers, similarities, multi-segment fan-out | 30 |
| [highlighter.md](parity/highlighter.md) | `lucene-highlighter`: the classic `Highlighter`, the unified highlighter and the `FastVectorHighlighter` (M13 T13.2) | 11 |
| [search-collectors.md](parity/search-collectors.md) | `lucene-search`: collectors, sorting, facets, aggregations, highlighting, fetch | 33 |
| [search-readers.md](parity/search-readers.md) | `lucene-search`: `DirectoryReader`, NRT readers, `CheckIndex`, searcher management | 28 |
| [monitor.md](parity/monitor.md) | `lucene-search`: the percolator: query analysis, presearcher, query index and facade (M13 T13.5a-d) | 11 |
| [expressions.md](parity/expressions.md) | `lucene-expressions`: JavaScript expressions as `DoubleValuesSource`s (M14 T14.2) | 5 |
| [classification.md](parity/classification.md) | `lucene-classification`: naive Bayes, k-NN and perceptron classifiers, their evaluation and dataset splitting (M14 T14.3) | 5 |
| [replicator.md](parity/replicator.md) | `lucene-replicator`: near-real-time replication -- primary and replica nodes, copy jobs, file ref-counting (M14 T14.3) | 9 |
| [sandbox.md](parity/sandbox.md) | `lucene-sandbox` (M14.f): half-float, big-integer and bounding-box points, nearest neighbours, multi-range point and doc-values queries, combined-field, covering, term-automaton and phrase-wildcard queries, the large-hits collector and the collector profilers, `PayloadSpanUtil`, `MergeOnFlushMergePolicy`, the `IDVersion` postings format, k-means clustering, the facet package (collectors, cutters, recorders, builders, orchestrators, histograms) | 19 |
| [codecs-extra.md](parity/codecs-extra.md) | `lucene-codecs` (M14.g): the non-default postings formats -- bloom, memory, bitvectors, blocktreeords, uniformsplit (with sharedterms), blockterms and simpletext | 33 |
| [test-framework.md](parity/test-framework.md) | `lucene-test-framework` (M14.h): randomized generators, mock analysis/directories, index/format suites, readers and writers | 31 |
| [test-framework-codecs.md](parity/test-framework-codecs.md) | `lucene-test-framework` (M14.h): asserting/cranky/compressing and other test codecs, installed writer seams | 33 |
| [test-framework-search.md](parity/test-framework-search.md) | `lucene-test-framework` (M14.h): query/scoring assertions, search bases, scorer wrappers | 11 |
| [test-framework-geo.md](parity/test-framework-geo.md) | `lucene-test-framework` (M14.h): geo helpers, geo-point/XY/range bases | 4 |
| [misc.md](parity/misc.md) | `lucene-misc`: index and search utilities (M14 T14.3; the rest of the jar is M14.b-i) | 25 |
| [tools.md](parity/tools.md) | `lucene-tools` (M14.i): the command-line tools of `lucene-misc` and `lucene-demo` (its toy word-vector dictionary included), Luke's index inspector models, and lucene-benchmark's search-quality tools | 24 |
| [suggest.md](parity/suggest.md) | `lucene-search`: the suggester base layer (`Lookup`, `InputIterator`, the dictionaries), `TSTLookup`, `WFSTCompletionLookup` and `FSTCompletionLookup` (M13 T13.3a-b, d: the completion postings format through `IndexWriter`, `SuggestField` and the completion queries; the analyzing suggesters `AnalyzingSuggester`/`FuzzySuggester` and `SuggestStopFilter` (T13.3c); `FreeTextSuggester` (T13.3c); `AnalyzingInfixSuggester` and `BlendedInfixSuggester` (T13.3c, in memory); the spell checkers `DirectSpellChecker`, `WordBreakSpellChecker`, `SpellChecker`, their distances and dictionaries (T13.3e); the completion formats of Lucene 9.0-10.4 (read)) | 34 |
| [geo-points.md](parity/geo-points.md) | Geo point fields and queries (M9 T9.2) | 9 |
| [geo-shapes.md](parity/geo-shapes.md) | Geo shape fields and queries (M9 T9.3) | 8 |
| [spatial3d.md](parity/spatial3d.md) | `lucene-spatial3d` (M9 T9.4) | 16 |
| [spatial-extras.md](parity/spatial-extras.md) | `lucene-spatial-extras` (M9 T9.5) | 25 |
| [joins.md](parity/joins.md) | Document blocks and block joins, `lucene-join` (M10) | 11 |
| [grouping.md](parity/grouping.md) | `lucene-grouping` (M10 T10.4) | 12 |
| [facet.md](parity/facet.md) | `lucene-facet`'s taxonomy index (T13.4a), counting (T13.4b), drill-down (T13.4c), value, range and facet-set counting (T13.4d1), and sorted-set counts, sampling and taxonomy utilities (T13.4d2) | 40 |
| [functions.md](parity/functions.md) | `lucene-queries` function queries (M10 T10.5) | 11 |
| [queries.md](parity/queries.md) | `lucene-queries` intervals, payloads, more-like-this, common terms, spans (M10 T10.6); sandbox fuzzy-like-this (M13) | 14 |
| [analysis-lang.md](parity/analysis-lang.md) | `lucene-analysis-common` synonyms, classic, language packages (M11 part 3), Thai, collation, `SegmentingTokenizerBase` | 14 |
| [analysis-icu.md](parity/analysis-icu.md) | `lucene-analysis-icu` (M12 T12.4): the ICU4J runtime port, normalization, folding, `ICUTokenizer`, collation keys, transliteration, the break rule compiler, character names | 16 |
| [analysis-opennlp.md](parity/analysis-opennlp.md) | `lucene-analysis-opennlp` (M12 T12.5): the OpenNLP 2.5.3 runtime port (models, maxent/perceptron, beam search, the six tools), `OpenNLPTokenizer` and the POS, chunker and lemmatizer filters | 14 |
| [analysis-lang-modules.md](parity/analysis-lang-modules.md) | The M12 language modules: `lucene-analysis-phonetic` (with Commons Codec), Stempel (with Egothor), Morfologik, `lucene-analysis`' `morph` base, Kuromoji, Nori, SmartCN | 19 |
| [ffi.md](parity/ffi.md) | `lucene-ffi`: the general C ABI (handles, searches, writer) | 17 |
| [plugin.md](parity/plugin.md) | `lucene-ffi`: the OpenSearch plugin boundary (JVM reader, FFM, query phase), M10 query shapes, M13 term ranges and completion-field indexes (suggester/highlighter execution deferred: `NamedRegistry` forbids replacing them) | 10 |
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
