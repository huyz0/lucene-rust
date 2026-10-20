# M14 — The remaining modules and the tooling

> **Goal:** the last modules of the Lucene 10.5.0 distribution -- non-default
> codecs, expressions, classification, replication, `misc`, `sandbox` -- plus
> Rust counterparts of the test framework, `luke`, the demo and the
> benchmark suite. At the end, `check-port-inventory.py`'s allowlist is empty.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md)–[M13](m13-search-application-modules.md) |
| **Unblocks** | "fully ported", as the inventory gate defines it |
| **Status** | in progress: T14.2 and T14.3's `classification`, `misc` (bar the CLI tools), `replicator` and `sandbox` (M14.f) done |

---

## Why this milestone exists

After M7–M13 what is left is either self-contained or tooling:

| Module | Files | Lines of code | Note |
|---|---|---|---|
| `codecs` | 95 | 19,242 | `SimpleText`, bloom filters, `BlockTreeOrds`, `UniformSplit`, memory postings |
| `expressions` | 19 | 2,697 | JavaScript expressions compiled to JVM bytecode |
| `classification` | 19 | 1,961 | k-NN and naive Bayes text classifiers |
| `replicator` | 13 | 2,439 | near-real-time replication (`nrt`: 11 classes and two `package-info`s) |
| `misc` | 39 | 4,107 | `IndexSplitter`, `HighFreqTerms`, sorting and store utilities |
| `sandbox` | 113 | 11,849 | whatever M10 did not need |
| `test-framework` | 237 | 60,830 | random indexing, asserting codecs, mock analyzers |
| `luke` | 163 | 15,251 | Swing index browser |
| `demo` | 20 | 2,585 | indexing and search examples |
| `benchmark` | 128 | 9,200 | the `byTask` benchmark framework |

---

## Scope

### In scope

- `codecs`, `classification`, `replicator`, `misc`, and the rest of `sandbox`,
  ported and differentially tested like everything else.
- `expressions`: the same language and results, compiled to a Rust
  closure tree or bytecode interpreter instead of JVM bytecode.
- **Test framework**: the pieces that make Lucene's own tests valuable,
  as Rust test utilities -- `RandomIndexWriter`, `MockAnalyzer`,
  `MockDirectoryWrapper` (fault injection), asserting codecs and readers,
  `LuceneTestCase`'s randomized configuration -- and a port of the
  highest-value Lucene test classes on top of them.
- **Tooling equivalents**, as command-line tools rather than ports of Java
  UIs: an index inspector covering `luke`'s views, the demo's indexing and
  search examples, and `byTask`-style benchmark scripts on the existing
  `bench-micro`/`bench-compare` harness.

### Out of scope

- `luke`'s Swing UI and `benchmark`'s Java-specific plumbing, recorded as
  `not needed` with the Rust tool that covers each capability.

---

## Tasks

- **T14.1** — The non-default codecs. **Done** at module level (M14.g).
- **T14.2** — `expressions`. **Done**: [`docs/parity/expressions.md`](../parity/expressions.md).
- **T14.3** — `classification`, `replicator`, `misc`, `sandbox`.
  `classification` **done**: [`docs/parity/classification.md`](../parity/classification.md)
  (its `DatasetSplitter` brought term vectors to the explicit writer).
  `replicator` **done**: [`docs/parity/replicator.md`](../parity/replicator.md).
- **T14.4** — Test framework utilities, then a first batch of ported Lucene
  tests chosen by the bugs they historically caught.
- **T14.5** — Index inspector CLI, demo, benchmark scripts.
- **T14.6** — Empty the allowlist; publish the final `parity.md`.

## Sub-milestones (M14 plan)

Each batch follows the port workflow (port, differential fixtures against
real Lucene 10.5.0 bytes, benchmark, optimise) and updates the inventory row
or ledger row in the same commit. Order is by dependency and size.

| Sub | Scope | Size (Java lines) | Lives in | Status |
|---|---|---|---|---|
| M14.a | `classification` (naive Bayes, kNN, fuzzy kNN, perceptron) | 1,961 | `lucene-search::classification` | ported; `cls_bm25nb` and `cls_perceptron_train` below parity (see the ledger) |
| M14.b | `misc` utilities: `TermStats`, `GetTermInfo`, `HighFreqTerms`, `CollectorMemoryTracker`, `SweetSpotSimilarity`, `MemoryTracker`, `document/LazyDocument`, `util/fst/*` | ~1,100 | `lucene-search::misc` | ported: `HighFreqTerms`, `TermStats`, `CollectorMemoryTracker`, `MemoryTracker`, `SweetSpotSimilarity`, `LazyDocument`, `ListOfOutputs` and `UpToTwoPositiveIntOutputs`, each equal to Lucene's where there is output to compare (`misc_fixtures.rs`, `misc_fst_fixtures.rs`, `misc_lazy_fixtures.rs`); `GetTermInfo`'s command line is M14.i's (`lucene-tools`) |
| M14.c | `misc` index tools: `IndexSplitter`, `MultiPassIndexSplitter`, `PKIndexSplitter`, `IndexRearranger`, `IndexMergeTool`, `BinaryDocValueSelector`, `BPIndexReorderer`, `BPReorderingMergePolicy` | ~3,000 | `lucene-search::misc` | done; `IndexMergeTool`'s command line is M14.i's (`lucene-tools`) |
| M14.d | `misc` search and store: `DiversifiedTopDocsCollector`(+manager), `DocValuesStats`(+collectors), `HumanReadableQuery`, memory-accounting bitset collectors, `DirectIODirectory`, `RAFDirectory`, `HardlinkCopyDirectoryWrapper`, byte-tracking wrappers | ~2,400 | `lucene-search::misc`, `lucene-store` for directories | done |
| M14.e | `replicator` (near-real-time replication; 10.5.0 has only the `nrt` package -- the revision replicator, its taxonomy flavour and HTTP service are gone from the jar) | 2,439 | `lucene-search::replicator` | done: all 11 classes ported ([`docs/parity/replicator.md`](../parity/replicator.md)); a Rust replica copies Java primaries' points and Java's `FileMetaData` and `infosBytes` read back equal, and Java bytes that are damaged or disagree with Java's checksum are refused (`replicator_nrt_fixtures.rs`) |
| M14.f | `sandbox` (113 classes, many small; triaged to `ported` or `not needed` per class; `queries/FuzzyLikeThisQuery` already ported in M13 for `CorePlusExtensionsParser`). Also deferred here from M13: the classic `QueryParserBase`'s and the flexible parser's date-resolution ranges and `MultiTermQuery` rewrite-method settings (`docs/inventory/lucene-queryparser.tsv`) | 11,849 | per area | done: 85 classes ported and 11 `not-needed` (`check-port-inventory.py --module sandbox --require-jar`), each area against Lucene's own output ([`docs/parity/sandbox.md`](../parity/sandbox.md)); both query parsers' rewrite method, determinize work limit and date-resolution ranges ported (`queryparser_*_fixtures.rs`); the flexible parser's points configuration (`PointsConfig`, `NumberDateFormat`, the points nodes and builders, over a black-box `DecimalFormat`) and `EscapeQuerySyntaxImpl` with every node's `toQueryString` ported, `check-port-inventory.py --module queryparser --milestone M14` clean; date bounds take fixed-offset time zones only (no daylight saving) and refuse a Buddhist-calendar locale (`th-TH`) once a bound is read as a date |
| M14.g | `codecs` non-default codecs (`SimpleText`, bloom, `BlockTreeOrds`, `UniformSplit`, memory postings) | 19,242 | `lucene-codecs` | done at module level: 86 classes, 81 ported, 1 `not-needed` (`RamUsageUtil`), 3 partial, 1 `todo:M14` (`check-port-inventory.py --module codecs --require-jar`); every format read and rebuilt byte for byte from Java's flushed and merged segments, and Lucene reads the Rust-written files back (`scripts/verify-write-path.sh`); ledger [`docs/parity/codecs-extra.md`](../parity/codecs-extra.md). Deferred: `IndexWriter`/`DirectoryReader` route no field to these formats (`SimpleTextCodec` only names its formats, partial); sharedterms `merge(MergeState)` and `STMergingTermsEnum` (partial, todo); bitvectors merge; the SimpleText `.si` index sort kept as provider/`toString`/bytes, not decoded (partial); blocktreeords, uniformsplit and blockterms do not read `.psm` and have no impacts |
| M14.h | test framework: `RandomIndexWriter`, `MockAnalyzer`, `MockDirectoryWrapper`, asserting readers, plus the chosen batch of Lucene tests | 60,830 (batch by acceptance) | `lucene-util` / test support crate | planned |
| M14.i | tooling: index inspector CLI, demo, `byTask`-style bench scripts | ~11,800 (luke's Swing UI and benchmark's Java plumbing are `not needed`) | `lucene-tools` (one `lucene-tools` binary), `benchmarks/` | in progress: the `misc` command lines (`GetTermInfo`, `HighFreqTerms`, `IndexMergeTool`, `IndexSplitter`, `MultiPassIndexSplitter`) byte-equal to Lucene's output; the demo's `IndexFiles` and `SearchFiles` (with its KNN dictionary) printing Lucene's hits (`docs/parity/tools.md`, `docs/inventory/lucene-demo.tsv`) |
| M14.j | empty the inventory allowlist; publish `parity.md` | -- | `docs/` | planned |

`misc` is not yet in `docs/inventory/`, so `check-port-inventory.py` cannot
see it: M14.b adds `docs/inventory/lucene-misc.tsv` and the module choice,
and the gate then counts its classes like the others.

## Acceptance criteria

- [ ] Every ported module passes Java-fixture differential tests.
- [ ] Real Lucene reads indices written with each non-default codec.
- [x] `expressions` evaluates Lucene's expression test corpus to the same
      doubles.
- [ ] `check-port-inventory.py` passes with an **empty allowlist**: every
      class of the 10.5.0 distribution is `ported` or `not needed` with a
      reason.

## Risks and unknowns

- **The test framework is open-ended.** Lucene has tens of thousands of
  tests; T14.4 ports the infrastructure and a chosen batch, not all of them.
  The acceptance bar is the infrastructure plus the batch, stated up front.

## Exit artifacts

- `docs/parity.md` with no `deferred` rows
- The inspector, demo and benchmark tools under `tools/` or `benchmarks/`
