# M7 — `lucene-core` complete, and "fully ported" made checkable

> **Goal:** every class in Lucene 10.5.0's `lucene-core` jar is ported, or
> recorded as not needed with a reason a reviewer can check -- and a gate that
> fails when one is neither.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M5.6](m5-6-native-read.md); can run alongside [M6](m6-production-candidate.md) |
| **Unblocks** | [M8](m8-backward-codecs.md), [M9](m9-geo-and-spatial.md), [M10](m10-joins-grouping-queries.md), [M11](m11-analysis-common.md) |
| **Status** | not started |

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

### T7.1 — The inventory gate

`scripts/check-port-inventory.py`: reads the 10.5.0 sources jars (fetched by
`scripts/lib-lucene-jars.sh`), lists every top-level class, and fails when one
has no `parity.md` row. Starts as a report with a checked-in allowlist of
today's gaps, each tagged with the milestone that closes it; each milestone
shrinks the allowlist, and the gate fails if the list grows. Negative
control: delete a ported row and watch it fail.

### T7.2 — `document` field types
### T7.3 — Similarities, bit-for-bit
### T7.4 — The remaining core queries and comparators, in the scorer tree
### T7.5 — Per-field formats, read and write
### T7.6 — Scalar-quantized vectors
### T7.7 — Writer gaps: compound segments, string index sorts, write lock
### T7.8 — `StandardTokenizer` and the attribute model

Each follows [`port-workflow`](../porting-workflow.md): the closest-to-Java
port with a Java-fixture differential test, a `bench-micro` pair against
Lucene, then optimisation to a ratio of at least 1.0.

---

## Acceptance criteria

- [ ] `check-port-inventory.py` runs in the gate and has been seen to fail.
- [ ] Its allowlist holds no `lucene-core` class.
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
