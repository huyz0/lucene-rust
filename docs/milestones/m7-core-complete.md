# M7 — `lucene-core` complete, and "fully ported" made checkable

> **Goal:** every class in Lucene 10.5.0's `lucene-core` jar is ported, or
> recorded as not needed with a reason a reviewer can check -- and a gate that
> fails when one is neither.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M5.6](m5-6-native-read.md); can run alongside [M6](m6-production-candidate.md) |
| **Unblocks** | [M8](m8-backward-codecs.md), [M9](m9-geo-and-spatial.md), [M10](m10-joins-grouping-queries.md), [M11](m11-analysis-common.md) |
| **Status** | in progress -- delivered: T7.1 inventory gate; T7.3 similarities ported and wired into search (`search_boolean_query_multi_segment_with_similarity`) and indexing (`IndexWriter::set_similarity`); the public `util/automaton` API (`lucene-util::automaton`, fixture-verified; term intersection still on the codecs byte DFA); `store` locks and directories. `scripts/check-port-inventory.py --milestone M7 --summary` is the live count |

---

## Why this milestone exists

The port was built outward from what OpenSearch's read and write paths need,
so its coverage of `lucene-core` is deep where those paths go and absent
elsewhere. Measured on 2026-09-30 against the 10.5.0 sources jar (1,213
files, 176k lines of code): `index`, `search`, `codecs/lucene90`,
`codecs/lucene104` and `util/packed` are covered row by row in
[`parity.md`](../parity.md), while `document` (92 files), `geo` (28),
`search/similarities` (48, BM25 only), `search/comparators`, `codecs/perfield`,
`util/quantization`, `analysis/standard` and `analysis/tokenattributes` have
no ledger rows at all. Of core's concrete `*Query` classes, 27 are referenced
by the port and 28 are not.

There is also no mechanical way to answer "is Lucene fully ported?".
`parity.md` lists what someone chose to record; nothing lists what nobody
recorded. This milestone adds that inventory first, so every later milestone
closes against a list rather than against memory.

---

## Scope

### In scope

- **An inventory gate**: every public or package-private class in the
  10.5.0 source jars of the modules this roadmap covers appears in
  `parity.md` with one status (`ported`, `partial`, `not needed: <reason>`,
  `deferred to M<n>`).
- The rest of `lucene-core`:
  - `document`: the field-type API (`IntField`, `LongField`, `KeywordField`,
    `FeatureField`, the `*Range` fields, `ShapeField`'s core half, stored
    and doc-values variants), as Rust types over the existing writer.
  - `search/similarities`: every similarity (`ClassicSimilarity`,
    `BooleanSimilarity`, the DFR/IB families, LM Dirichlet and
    Jelinek-Mercer, `Axiomatic*`, `MultiSimilarity`,
    `PerFieldSimilarityWrapper`), each scoring bit-for-bit against Lucene.
  - `search/comparators` and the rest of `search`: the 28 unreferenced
    queries that are not geo (M9) -- among them `CombinedFieldQuery`,
    `FeatureQuery`, `SortedSetDocValuesRangeQuery`,
    `SortedNumericDocValuesSetQuery`, `IndexSortSortedNumericDocValuesRangeQuery`,
    the `*RangeSlowRangeQuery` family, `LongDistanceFeatureQuery`,
    `NGramPhraseQuery`, `RescoreTopNQuery` and the fusion queries -- plus the
    standalone queries brought into the scorer tree (`FuzzyQuery`,
    `MultiPhraseQuery`, spans).
  - `codecs/perfield`: `PerFieldPostingsFormat`/`PerFieldDocValuesFormat`/
    `PerFieldKnnVectorsFormat` reading and writing more than one format per
    segment.
  - `util/quantization` and the scalar-quantized vector formats.
  - The writer gaps `parity.md` names: compound segments from `IndexWriter`,
    `STRING`/`SortedSet`/`Binary` index sorts, the write lock
    (`NativeFSLockFactory`), pending deletes.
  - `analysis`, `analysis/standard`, `analysis/tokenattributes`:
    `StandardTokenizer` (the UAX#29 grammar) and the attribute model.

### Out of scope

- Everything in other jars (M8–M14).
- `internal/hppc` and `internal/tests`: Java collection and test shims with no
  Rust counterpart to write; recorded as `not needed` with the Rust type used
  instead.
- A second OpenSearch integration pass; new native shapes follow in the
  milestones that add the queries OpenSearch builds.

---

## Tasks

### T7.1 — The inventory gate · delivered 2026-09-30

`scripts/check-port-inventory.py` over `docs/inventory/lucene-core.tsv`: one
row per top-level class of the compiled 10.5.0 jar (1,196 with the Java 21
multi-release variants), each `ported` (a Rust file and symbol, checked to
exist), `partial` (the same, plus ` -- M<n>: ` and the gap), `not-needed`
(a reason), `todo:M<n>` or `deferred:M<n>`. It runs in the gate and in CI;
`--milestone M7` is this milestone's "done", `--summary` its progress. Seen to
fail on a deleted row, a missing symbol and an untagged partial
([`mechanical-gates.md`](../mechanical-gates.md#port-inventory)).

The first classification (five parallel passes, each row checked against the
Java source and the Rust tree) measured M7's real size, which is larger than
the scope list below: **475 ported, 227 not needed, 7 deferred to M8, 60 to
M9 (geo), and 381 open for M7** -- 226 `todo:M7` and 155 `partial`. The open
ones beyond the scope list include the `Matches` API, rescorers and value
sources, the NRT managers (`SearcherManager`, `ReaderManager`), reader
wrappers (`Filter*Reader`, `MultiReader`, `ParallelLeafReader`,
`ExitableDirectoryReader`, `SortingCodecReader`), the other merge and
deletion policies, lock factories and the remaining directories, the
`document/column` API, the public automaton API, the in-memory packed-ints
family, offline sorting, and the analysis attribute model.

### T7.2 — `document` field types

Delivered (ported, not yet benchmarked): every M7 row of `document` and
`document/column` (67) -- the `Document`/`Field`/`FieldType` API indexed by
`IndexWriter::add_fields_document` and `add_batch`, the point, range,
doc-values, feature, date and late-interaction fields in
`lucene-index/src/document/`, and their queries (points, slow doc values with
the skipper path, range relations, feature, distance feature) in
`lucene-search/src/document/`; differential against `GenDocumentFields` and
`GenDocumentColumns`.
### T7.3 — Similarities, bit-for-bit
### T7.4 — The remaining core queries and comparators, in the scorer tree
### T7.5 — Per-field formats, read and write
### T7.6 — Scalar-quantized vectors
### T7.7 — Writer gaps: compound segments, string index sorts, write lock
### T7.8 — `StandardTokenizer` and the attribute model · delivered 2026-09-30

`crates/lucene-analysis` is now Lucene's streaming model: `TokenStream`/
`Tokenizer`/`TokenFilter`/`FilteringTokenFilter`/`CachingTokenFilter`/
`GraphTokenFilter`, every core attribute, `CharFilter` offset correction,
`Analyzer` with components, reuse, `normalize` and both wrappers,
`StopwordAnalyzerBase`/`CharArraySet`/`WordlistLoader`, and
`StandardTokenizer` as Lucene's own JFlex scanner over tables extracted from
the 10.5.0 class. `GenStandardTokenizer` checks `zzCMap` for every code point
and 17,073 analyses (Unicode's word-break and emoji conformance inputs from
Lucene's test framework, the tokenizer, `StandardAnalyzer`, a custom chain
with a char filter) token for token, plus the graph filter and both automaton
converters. `TokenStreamToAutomaton`/`AutomatonToTokenStream` now build and
read `lucene-util`'s `Automaton` (delivered 2026-09-30). Not in this task: the inverter reading a filter's payload and term
frequency attributes (on `index/IndexingChain`'s row).

### Search, analysis and document gaps · delivered 2026-09-30

- `ColumnBatch`: an empty column registers its `FieldInfo` as Java's does
  (`GenDocumentColumns` gained three empty columns).
- `NRTCachingDirectory`: `IndexWriter` passes its flush and merge size estimates (`Directory::create_output_with_estimate`, `EstimatedWrites`), so small segments are cached.
- One point range/set query: the document package runs the scorer tree's `PointRangeQuery`/`PointInSetQuery` (hits and scores unchanged against `GenDocumentFields`).
- `IndexOrDocValuesQuery` in the scorer tree, choosing points or doc values by the boolean's lead cost (`GenM7Queries` gained seven searches).
- `RescoreTopNQuery` as a clause (rewritten to a `DocAndScoreQuery` by the searcher), and values-source sorts over query-backed and vector sources (`rewrite_sort`); `GenValuesRescore` gained both.
- Comparator skipping: `DocComparator`'s competitive iterator (Lucene's exact lower bound), `NumericComparator`'s `DVSkipperCompetitiveDISIBuilder` and `TermOrdValComparator`'s `SkipperBasedCompetitiveState` (skip-index sorts added to `GenSortedSearch`/`GenKeywordSort`).
- `CompiledAutomaton.getTermsEnum` (`FieldTerms::compiled_terms`, fixture-verified against `GenRegexpIntersect`) and `visit` (`query_visitor::visit_compiled`).
- `QueryBuilder` and `GraphTokenStreamFiniteStrings` over the streaming analysis model (`GenQueryBuilder`: 29 cases, `toString` and hits/scores bit for bit).

Each follows [`port-workflow`](../porting-workflow.md): the closest-to-Java
port with a Java-fixture differential test, a `bench-micro` pair against
Lucene, then optimisation to a ratio of at least 1.0.

---

## Acceptance criteria

- [x] `check-port-inventory.py` runs in the gate and has been seen to fail.
- [ ] `check-port-inventory.py --milestone M7` passes: no `lucene-core` class
      is `todo:M7` or `partial` with an M7 gap (geo is M9's, older formats M8's).
- [ ] Every new query and similarity matches Lucene's hits and scores
      bit for bit on a generated fixture, and is no slower than Lucene on its
      benchmark.
- [ ] Real Lucene reads a Rust-written index using two postings formats and a
      compound segment (`verify-write-path.sh`).
- [ ] `StandardTokenizer` agrees with Lucene token for token on the UAX#29
      conformance tests and a large corpus sample.

## Risks and unknowns

- **`document` is an API, not a format.** Its value is ergonomic; a Rust
  shape that differs from Java's is fine if every field indexes the same
  bytes. The differential test is the bytes, not the class graph.
- **Similarity float order.** Each similarity must reproduce Lucene's float
  expression order to match bit for bit (the `BM25` lesson in
  `similarity.rs`).

## Exit artifacts

- `scripts/check-port-inventory.py` and its allowlist
- `docs/parity.md` rows for every `lucene-core` class
- One fixture generator per new format and query family under `fixtures/src/`
