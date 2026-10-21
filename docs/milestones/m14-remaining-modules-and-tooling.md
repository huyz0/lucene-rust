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
| M14.h | test framework: `RandomIndexWriter`, `MockAnalyzer`, `MockDirectoryWrapper`, asserting readers, plus the chosen batch of Lucene tests | 60,830 (batch by acceptance) | `crates/lucene-test-framework` (a dev-dependency only) | in progress: h1, h2 and h3 done (212 classes triaged, `check-port-inventory.py --module test-framework --require-jar`); batches below; ledger [`docs/parity/test-framework.md`](../parity/test-framework.md) |
| M14.i | tooling: index inspector CLI, demo, `byTask`-style bench scripts | ~11,800 (luke's Swing UI and benchmark's Java plumbing are `not needed`) | `lucene-tools` (one `lucene-tools` binary), `benchmarks/` | done: the `misc` command lines (`GetTermInfo`, `HighFreqTerms`, `IndexMergeTool`, `IndexSplitter`, `MultiPassIndexSplitter`) byte-equal to Lucene's output; the demo's `IndexFiles` and `SearchFiles` (with its KNN dictionary) printing Lucene's hits; Luke's models (overview, documents, commits, search, analysis, index tools, 20 Newsgroups) through `lucene-tools Luke`, 73 cases byte-equal to Lucene's models; lucene-benchmark's `quality` package (`QueryDriver`, `QualityQueriesFinder`) byte-equal to Lucene's report and submission. Deferred, by design: Luke's Swing `app/*` and lucene-benchmark's `byTask`/Java harness are `not needed` (the project benchmarks with `scripts/bench-micro.sh` and `benchmarks/rust-runner`); Luke's `repairIndex` (it is `CheckIndex -exorcise`); no bench pairs for the command lines; `Trec1MQReader` unit-tested only; the Luke writer's `SegmentInfos.version` and one post-delete generation differ from Java's (masked, in the ledger) -- `docs/parity/tools.md`, `docs/inventory/lucene-{misc,demo,luke,benchmark}.tsv` |
| M14.j | empty the inventory allowlist; publish `parity.md` | -- | `docs/` | planned |

`misc` is not yet in `docs/inventory/`, so `check-port-inventory.py` cannot
see it: M14.b adds `docs/inventory/lucene-misc.tsv` and the module choice,
and the gate then counts its classes like the others.

### M14.h batches

Order: h1, then h2 and h3, then h4, then h5, h6 and h7, then h8. Each batch
ports its infrastructure, the Lucene tests named, and a differential check
against a Java twin (`fixtures/src/testframework/`, run by
`scripts/gen-test-framework-fixtures.sh`).

| Batch | Infrastructure | Lucene tests | Acceptance | Status |
|---|---|---|---|---|
| h1 | the crate, inventory (212 classes; the plan counted 211), gate/CI, generator jars, the `test-framework-dev-only` invariant; seeds and reproduce line, `RandomNumbers`, `TestUtil`'s generators, `English`, `LineFileDocs` (200-line excerpt), bit-set and doc-id-set suites, `AutomatonTestUtil`, `FSTTester` | `TestFixedBitSet`, `TestSparseFixedBitSet`, `TestRoaringDocIdSet`, `TestIntArrayDocIdSet` | `GenTestFrameworkRandom`: 200 seeds equal | done |
| h2 | `MockDirectoryWrapper`, mockfile behaviours, output fault hook, `BaseDirectoryTestCase`, `BaseLockFactoryTestCase` | `TestMockDirectoryWrapper`, directory and lock-factory subclasses | 300 seeds of `IndexWriter` at 10% I/O errors: no panic, Rust and Java `CheckIndex` clean (`scripts/test-framework-faults.sh`) | done |
| h3 | mock analysis chain, `BaseTokenStreamTestCase` | `TestMockAnalyzer`, `TestMockCharFilter`, `TestLookaheadTokenFilter`, `TestMockSynonymFilter`, `TestToken` | `GenTestFrameworkAnalysis`: 100 seeds of token dumps equal | done (deferrals below) |
| h4 | `RandomIndexWriter`, `newIndexWriterConfig`, `RandomCodec`, merge policies, reader wrappers, `newSearcher`, `BaseTestCheckIndex`, `BaseMergePolicyTestCase` | `TestCheckIndex`, `TestAssertingLeafReader`, merge-policy tests | 300 seeds pass Rust `CheckIndex`; on 50, Java's too | planned |
| h5 | `BaseIndexFileFormatTestCase`; doc values, postings, stored fields, term vectors suites | the default codec's format tests | 30 seeds per format read back by Lucene | planned |
| h6 | points, norms, segment/field infos, live docs, compound, KNN suites; the test postings formats' read side | the default codec's format tests | as h5 | planned |
| h7 | `QueryUtils`, `CheckHits`, asserting search wrappers, search/similarity/geo bases | equivalence, explanation, similarity and geo tests | top docs, score bits and explanations equal Java's | planned |
| h8 | writer stress, exceptions, NRT | `TestNRTThreads`, `TestStressNRT`, `TestIndexWriterExceptions`, ... | `scripts/test-framework-stress.sh` over 200 seeds | planned |

h1's known gaps (rows in the ledger):

- `BaseDocIdSetTestCase`: `intoBitSet` and `ramBytesUsed` are not ported, because `DocIdSet` has no Rust counterpart for them.
- `TestRoaringDocIdSet`: the range-add and `docIDRunEnd` tests are not ported.
- `TestFixedBitSet`: the `andNot(DocIdSetIterator)` shapes are reduced to `and_not(&FixedBitSet)`, `asReadOnlyBits` is not ported, and the set is built over an exact-length word array (Java over-allocates it; the draw is kept).
- `FSTTester`: the UTF-32 enum checks are missing, and `ListOfOutputs` is not exercised.
- `TestUtil.randomPattern` is not ported.
- `randomlyRecaseCodePoints` maps through `lucene-util`'s JDK 25 case tables, while the generator runs JDK 21. Its fixture sticks to ASCII and Latin-1, where the two agree.
- `TESTS_LINEDOCSFILE` defaults to the committed excerpt rather than the 9.7 MB file.

Port bug found: `FixedBitSet::into_array` indexed past a full destination array (fixture `intoArrayCap`).

h3's deferrals (rows in the ledger):

- `checkRandomData`'s `RandomIndexWriter` pass makes all of Java's draws, so a seed still yields Java's texts, but it does not index yet. That waits for h4, as does `TestMockAnalyzer.testChangeGaps`, which is entirely an indexing test (term-vector positions and offsets).
- `newAttributeFactory` and `TestToken.testTokenAttributeFactory` are not ported: the port's attributes are a fixed struct with no factory to vary.
- `checkResetException`'s missing-`close()` half has no counterpart, because an `AnalyzerTokenStream` closes itself on drop.
- `VocabularyAssert`'s zip-file overloads are not ported.
- `CollationTestBase`, `MockBytesAnalyzer` and `MockUTF16TermAttributeImpl` are not needed (see the inventory).
- `checkRandomData` over every analyzer with a default construction in `lucene-analysis` (the language analyzers, CJK, Classic, UAX29URLEmail, English, Standard, Whitespace, UnicodeWhitespace, Simple and Stop) and every M12 module's analyzer, but not the wrappers (`ShingleAnalyzerWrapper`, `LimitTokenCountAnalyzer`, `PerFieldAnalyzerWrapper`) or `CustomAnalyzer` (`tests/check_random_data_all_analyzers.rs`; `ThaiAnalyzer` excepted, since its tokenizer refuses by design) found no analyzer bug.

h2's known gaps:

- `BaseDirectoryTestCase`: `testDetectClose`, the double-close tests, `isLoaded` and GroupVInt's overflow `ArithmeticException` are not ported, because the Rust directories have no counterpart for them.
- `MockDirectoryWrapper` is seeded as Java's is, but draws at other sites: there are no `newIOContext` draws, a crash damages files through the delegate, and under a rate limiter the write hook runs once per chunk. `corruptUnknownFiles`, `assertNoUnreferencedFilesOnClose`, `getLevelForCheckOnClose`, input clone tracking, the read-side input checks and the READONCE check are not ported; the inventory marks it `partial`.
- `BaseChunkedDirectoryTestCase` is not needed, because `MmapDirectory` has no chunk size. `SerialIOCountingDirectory` is not needed either, because there is no `IOContext` read advice. The mock filesystem's NIO plumbing, `ShuffleFS`, `VerboseFS` and `WindowsPath` are not needed.
- The fault run never saw a failed commit leave its `segments_N` behind, so that branch of its model is untested.

Port bugs found: `FsDirectory`/`MmapDirectory::create_output` truncated an existing file (Java: `CREATE_NEW`), and an output under a missing root failed (Java's constructor creates it). Both are fixed in 5bcebd83, and `create_output_refuses_an_existing_name_and_creates_a_missing_root` covers them. The create-new outputs then exposed writer state a failure left behind: a failed commit kept its generation, a failed `.liv` write kept its partial file and delete generation, and a failed packet apply dropped the segment lists. All three are fixed in 0af77546.

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
