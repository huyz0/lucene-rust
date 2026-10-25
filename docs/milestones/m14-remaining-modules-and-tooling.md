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
| M14.g | `codecs` non-default codecs (`SimpleText`, bloom, `BlockTreeOrds`, `UniformSplit`, memory postings) | 19,242 | `lucene-codecs` | done at module level: 86 classes, 84 ported, 1 `not-needed` (`RamUsageUtil`), 1 partial (`check-port-inventory.py --module codecs --require-jar`); every format read and rebuilt byte for byte from Java's flushed and merged segments; SimpleText segment info additionally covers all four typed sort providers and prerelease versions in 184 isolated Java files; sharedterms also merges independent deleted sources with field-number remapping and sorted fallback, and Lucene reads the Rust-written files back (`scripts/verify-write-path.sh`); ledger [`docs/parity/codecs-extra.md`](../parity/codecs-extra.md). Partial: SimpleText `DirectoryReader` opens metadata, compound files, stored fields, terms/postings, generation-aware doc values, norms and term vectors, points, float/byte vectors and skippers; search execution and `IndexWriter` routing remain. Other format routing deferred; bitvectors merge; blocktreeords, uniformsplit and blockterms do not read `.psm` and have no impacts |
| M14.h | test framework: `RandomIndexWriter`, `MockAnalyzer`, `MockDirectoryWrapper`, asserting readers, plus the chosen batch of Lucene tests | 60,830 (batch by acceptance) | `crates/lucene-test-framework` (a dev-dependency only) | in progress: h1 to h5 and h7 done (212 classes triaged, `check-port-inventory.py --module test-framework --require-jar`); batches below; ledger [`docs/parity/test-framework.md`](../parity/test-framework.md) |
| M14.i | tooling: index inspector CLI, demo, `byTask`-style bench scripts | ~11,800 (luke's Swing UI and benchmark's Java plumbing are `not needed`) | `lucene-tools` (one `lucene-tools` binary), `benchmarks/` | done: the `misc` command lines (`GetTermInfo`, `HighFreqTerms`, `IndexMergeTool`, `IndexSplitter`, `MultiPassIndexSplitter`) byte-equal to Lucene's output; the demo's `IndexFiles` and `SearchFiles` (with its KNN dictionary) printing Lucene's hits; Luke's models (overview, documents, commits, search, analysis, index tools, 20 Newsgroups) through `lucene-tools Luke`, 73 cases byte-equal to Lucene's models; lucene-benchmark's `quality` package (`QueryDriver`, `QualityQueriesFinder`) byte-equal to Lucene's report and submission. Deferred, by design: Luke's Swing `app/*` and lucene-benchmark's `byTask`/Java harness are `not needed` (the project benchmarks with `scripts/bench-micro.sh` and `benchmarks/rust-runner`); Luke's `repairIndex` (it is `CheckIndex -exorcise`); no bench pairs for the command lines; `Trec1MQReader` unit-tested only; the Luke writer's `SegmentInfos.version` and one post-delete generation differ from Java's (masked, in the ledger) -- `docs/parity/tools.md`, `docs/inventory/lucene-{misc,demo,luke,benchmark}.tsv` |
| M14.j | empty the inventory allowlist; publish `parity.md` | -- | `docs/` | planned |

`misc` is tracked in `docs/inventory/lucene-misc.tsv`; the gate checks its
class inventory with `check-port-inventory.py --module misc --require-jar`.

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
| h4 | `RandomIndexWriter`, `newIndexWriterConfig`, `RandomCodec`, merge policies, reader wrappers, `newSearcher`, `BaseTestCheckIndex`, `BaseMergePolicyTestCase` | `TestCheckIndex`, `TestAssertingLeafReader`, merge-policy tests | 300 seeds pass Rust `CheckIndex`; on 50, Java's too (`scripts/test-framework-index.sh`) | done (deferrals below) |
| h5 | `BaseIndexFileFormatTestCase`; doc values, postings, stored fields, term vectors suites | the default codec's format tests | 30 seeds per format read back by Lucene (`scripts/test-framework-formats.sh`) | done (deferrals below) |
| h6 | points, norms, segment/field infos, live docs, compound, KNN suites; the test postings formats' read side | the default codec's format tests | as h5 (330 indexes, the segment info, field infos, compound, norms, points and float/byte KNN suites added) | in progress: segment/field infos, live docs, compound, norms, points, KNN done (deferrals below); block-terms test postings readers and writers ported; every output file matches actual Java test formats; bloom reader/writer ported with sparse/saturated Java byte parity; RAMOnly tokens, instance state, postings/payloads and seek behavior ported vs Java; MockRandom seeded dictionary/selector branches and successive writes match Java bytes and postings; both ROT13 uniform-split formats match Java files, flags and postings on empty/singleton/many groups; codec routing and inherited gaps remain |
| h7 | `QueryUtils`, `CheckHits`, asserting search wrappers, search/similarity/geo bases | equivalence, explanation, similarity and geo tests | top docs, score bits and explanations equal Java's | done (deferrals below) |
| h8 | writer stress, exceptions, NRT | `TestNRTThreads`, `TestStressNRT`, `TestIndexWriterExceptions`, ... | `scripts/test-framework-stress.sh` over 200 seeds | done: `ThreadedIndexingAndSearchingTestCase` (partial) and `SuppressingConcurrentMergeScheduler` (deferrals below) |

h1's known gaps (rows in the ledger):

- `BaseDocIdSetTestCase`: `intoBitSet` and `ramBytesUsed` are not ported, because `DocIdSet` has no Rust counterpart for them.
- `TestRoaringDocIdSet`: the range-add and `docIDRunEnd` tests are not ported.
- `TestFixedBitSet`: the `andNot(DocIdSetIterator)` shapes are reduced to `and_not(&FixedBitSet)`, `asReadOnlyBits` is not ported, and the set is built over an exact-length word array (Java over-allocates it; the draw is kept).
- `FSTTester`: Rust uses the typed `do_test_lists` entry point for Java's runtime
  list-output detection. Actual tester fixtures cover scalar expansion, both
  input modes, saved bytes, compiler counts and RNG continuation.
- `TestUtil.randomPattern` is not ported.
- `randomlyRecaseCodePoints` maps through `lucene_util::automaton::java_to_{upper,lower}_case`, the JDK 25 simple mapping rather than Rust's own tables, while the generator runs JDK 21. Its fixture sticks to ASCII and Latin-1, where the two agree.
- `TESTS_LINEDOCSFILE` defaults to the committed excerpt rather than the 9.7 MB file.

Port bug found: `FixedBitSet::into_array` indexed past a full destination array (fixture `intoArrayCap`).

h3's deferrals (rows in the ledger):

- `newAttributeFactory` and `TestToken.testTokenAttributeFactory` are not ported: the port's attributes are a fixed struct with no factory to vary.
- `checkResetException`'s missing-`close()` half has no counterpart, because an `AnalyzerTokenStream` closes itself on drop.
- `VocabularyAssert`'s zip-file overloads are not ported.
- `CollationTestBase`, `MockBytesAnalyzer` and `MockUTF16TermAttributeImpl` are not needed (see the inventory).
- `checkRandomData` over every analyzer with a default construction in `lucene-analysis` (the language analyzers, CJK, Classic, UAX29URLEmail, English, Standard, Whitespace, UnicodeWhitespace, Simple and Stop) and every M12 module's analyzer, but not the wrappers (`ShingleAnalyzerWrapper`, `LimitTokenCountAnalyzer`, `PerFieldAnalyzerWrapper`) or `CustomAnalyzer` (`tests/check_random_data_all_analyzers.rs`; `ThaiAnalyzer` excepted, since its tokenizer refuses by design) found no analyzer bug.

h2's known gaps:

- `BaseDirectoryTestCase`: `testDetectClose`, the double-close tests, `isLoaded` and GroupVInt's overflow `ArithmeticException` are not ported, because the Rust directories have no counterpart for them.
- `MockDirectoryWrapper` is seeded as Java's is, but draws at other sites: there are no `newIOContext` draws, a crash damages files through the delegate, and under a rate limiter the write hook runs once per chunk. `corrupt_unknown_files` scans every retained commit before damaging unknown codec files and pending commits, with `corrupt_unknown_files_with` supplying Java's overridable corruption callback. Scan errors prevent the callback; the Rust scan requires quiescent directory mutations because it does not hold Java's synchronized monitor across both listings and corruption. The optional close assertion uses a delegate writer with `NoDeletionPolicy`, compares file sets, and verifies document counts around a default-policy writer close without committing; this second pass can prune old commits. Rust delegate resources close when their owners drop. `getLevelForCheckOnClose`, input clone tracking, the read-side input checks and the READONCE check are not ported; the inventory marks it `partial`.
- `BaseChunkedDirectoryTestCase` is not needed, because `MmapDirectory` has no chunk size. `SerialIOCountingDirectory` is not needed either, because there is no `IOContext` read advice. The mock filesystem's NIO plumbing, `ShuffleFS`, `VerboseFS` and `WindowsPath` are not needed.
- The fault run never saw a failed commit leave its `segments_N` behind, so that branch of its model is untested.

Port bugs found: `FsDirectory`/`MmapDirectory::create_output` truncated an existing file (Java: `CREATE_NEW`), and an output under a missing root failed (Java's constructor creates it). Both are fixed in 5bcebd83, and `create_output_refuses_an_existing_name_and_creates_a_missing_root` covers them. The create-new outputs then exposed writer state a failure left behind: a failed commit kept its generation, a failed `.liv` write kept its partial file and delete generation, and a failed packet apply dropped the segment lists. All three are fixed in 0af77546.

h4's deferrals (rows in the ledger):

- `RandomCodec` has constructor and cached UTF-16 field-selection descriptors, including separate identities for equal candidates; heterogeneous native dispatch, identity-based per-field suffix groups and randomized BKD writers remain absent. Explicit alternative component writers exist, but the segment writer still installs the `Lucene104` codec. Cranky field-info, live-doc, segment-info, compound, doc-values, norms, points, postings, stored-field and term-vector adapters, asserting component adapters and direct CheapBastard postings exercise real formats without installing them through `Codec`/`IndexWriter`.
- `AssertingCodec` composes eight eager asserting adapters with cached selector hooks, suppression and delegate callbacks. Its native default graph, including binary Lucene90 term vectors and a KNN writer, and operational per-field routing remain absent. `CompressingCodec` and its six concrete variants preserve constructor settings, mode/format identities, delegation and random/error order; custom-header/geometry stored-field and term-vector I/O still need configurable lower-crate factories. Both remain partial, without SPI/IndexWriter installation.
- The bounded codec pairs measure composition/configuration rather than native I/O or constructors: [AssertingCodec](../benchmarks/asserting-codec.md) retains 312/328 Rust/Java bytes in its adapter scope, with typed router identity backed by shared selector state; [CompressingCodec](../benchmarks/compressing-codec.md) shares immutable settings between distinct selected formats and retains 88/112 bytes, or 112/128 with a preset mode. These are scoped retained bytes, excluding native graphs and allocator/runtime overhead.
- `RandomIndexWriter`: soft deletes and update-by-query are not ported, since the writer has no configured soft-deletes field. A soft update of an unregistered field therefore still refuses where Java creates it.
- `newSearcher` and `wrapReader` belong to h7, and `SuppressingConcurrentMergeScheduler` to h8.
- `AssertingLeafReader` wraps postings, terms, doc values and norms, but not stored fields, term vectors, impacts, the doc-values skipper, points or vectors. Its closed-listener key check cannot fire, because a reader's cache key cannot outlive it.
- `IndexWriter::force_merge` merges committed segments only; Java flushes first. Tests that need a buffered document merged commit before they merge (`BaseTestCheckIndex.testDeletedDocs`).
- A writer outside the explicit-documents mode, and the concurrent writer, still refuse a doc-values update of an unknown field.
- `checkRandomData` now indexes every text through a `RandomIndexWriter` (term vectors, offsets, positions, payloads and pile-ups as Java draws them), and `TestMockAnalyzer.testChangeGaps` is ported.

Port bugs found: once a parent field was set, the stored-only `add_document(s)` entry points did not mark each add's last document as its parent, so a block had none (`stored_only_adds_mark_their_parent`). `updateDocValues` and `tryUpdateDocValue` did not create a doc-values-only field the way Java's `buildDocValuesUpdate` does (fixture `dvonly.txt`). The SORTED_NUMERIC doc-values reader refused to go backwards, so a reordering merge (`MockRandomMergePolicy.reverse`) of a sparse SORTED_NUMERIC or SORTED_SET column panicked (`a_reordering_merge_carries_sparse_multi_valued_columns`). A reordering merge wrapped each source twice, so a drawing `wrapForMerge` (`MockRandomOneMerge`) ran its stream ahead of Java's (`a_reordering_merge_wraps_each_source_once`). Each fix is in its own commit.

h5's deferrals (rows in the ledger and the inventory):

- Every suite runs on the one codec this port writes (`Lucene104`, with `Lucene90` doc values, stored fields and term vectors); `getCodec()` has nothing to choose.
- `RandomPostingsTester` writes its postings as explicit documents through `IndexWriter`, not through a `FieldsConsumer`, and does not exercise `TermState`, enum reuse, `intoBitSet` or `nextPostings`. `testInvertedWrite` and `testMultiClose` are not ported: the writer has no codec plug-in point and no codec-level reader to close twice.
- `testCheckIntegrityReadsAllBytes` damages each file in turn instead of recording which bytes `checkIntegrity` read: this port's directories hand out whole files.
- The merging reader wrappers (`maybeWrapWithMergingReader`, `MismatchedCodecReader`, `testMergeFilterReader`'s reversed stored fields) have nothing to wrap, because the merge reads the source files. Mismatched field numbers are reached by adding fields in a different order to each index.
- Term vectors are read as a value per document, so the enum seeks, reuse and the "past the last position" checks have nothing to call.
- `DocIdSetIterator.intoBitSet` is not ported; the doc values `intoBitSet` tests run Java's default `nextDoc` loop against `advanceExact`.
- `@Nightly` tests take their sizes as parameters and run smaller by default.
- `testRandomExceptions` reopens the writer after every injected error, so it never tests a writer writing on after a non-tragic one, as Java's does.
- `RandomPostingsTester` checks impacts against the reader's `SlowImpactsEnum` only.

Port bugs found: the postings writer refused a term repeated at one position (a position increment of 0), which Java writes as a delta of 0 (`rejects_decreasing_positions_within_a_doc_but_not_repeated_ones`). `add_indexes_merged` (`addIndexes(CodecReader...)`) took each source's write lock, which only `addIndexes(Directory...)` does, so it refused an index whose writer was still open (`add_indexes_refuses_incompatible_sources`). A segment reader passed a damaged `.fdm`, `.tvm`, `.vemf` or `.psm` at open and a damaged `.tim`, `.tip` or `.dvs` in `check_integrity`, where Java's open and `checkIntegrity` refuse them (`check_integrity_and_skippers`). A segment whose `.fnm` indexed positions on a field with no terms wrote a `.psm` without the `.pos`/`.pay` lengths Java's reader reads off the FieldInfos, so Lucene's `CheckIndex` refused it (`psm_frames_pos_and_pay_from_the_segment_fields_not_the_inputs`). Each fix is in its own commit.

h6's deferrals so far (rows in the ledger and the inventory):

- The segment info, field infos and live docs formats are byte functions, not format objects, so the suites drive `write`/`parse` through a directory. A parsed `SegmentInfo` or `FieldInfo` is a value the caller owns, so Java's immutable-view checks have nothing to refuse, and `testExceptionOnCloseInput` has no input `close` to fail (a directory hands out whole files).
- `CompoundReader::list_all` also lists the directory underneath (its `open` falls through to it), so the suite's `listAll` is `member_files`. A headerless sub-file is refused as "bad magic", not Java's "codec header mismatch". `testCheckIntegrity` flips a `.cfs` byte opening does not read instead of counting reads; `CompoundReader::check_integrity` was added for it. `testDoubleClose`, `testPassIOContext` and `testResourceNameInsideCompoundFile` are not ported (no explicit close, `IOContext` or resource names).
- The norms suite checks values with `advanceExact`, since the bulk `NumericDocValues.longValues` is not ported; no merging reader wrapper.
- `@Monster` tests (`testLargeCFS`, live docs' `testOverflow`) take their sizes as parameters and run smaller.
- The points suite walks the index tree as values: a node has no `size()`, so its count is checked as its children's sum and every visited point against its cell. `switchIndex` merges the side writer's committed directory rather than an NRT reader, and `MismatchedCodecReader` is an index that numbers the fields in the other order. `@Nightly testRandomBinaryBig` runs at 5,000 values.

Port bugs found by the points suite, each fixed in its own commit: a merge dropped a field whose every term belonged to deleted documents, so the merged `.fnm` claimed postings with no `.tim` (`fix(index)`, `per_field_postings_write_path.rs`); and `.kdm`'s footer was verified neither at open nor by `check_integrity` (`fix(search)`, `a_damaged_points_meta_file_is_refused_at_open`). The segment info, field infos, live docs, compound and norms suites found none.

The KNN vectors suite (`Lucene99HnswVectorsFormat`) ports 44 of `BaseKnnVectorsFormatTestCase`'s 49 tests (40 runners). Graph searches open the default format's graph (including compound files); `testSearchWithVisitedLimit` checks sparse vectors and deletions, and 268 Java leaf searches verify budgets, zero/oversized k, visit counts/relations, hit order and score bits on dense, sparse, byte and tiny fields; 36 further Java probes cover empty, missing and non-vector fields. Not ported: off-heap byte accounting and `testMergeOffHeapByteSizeMaps` (no accounting API), `testFieldSetValue` and the setters of `testInvalidKnnVectorFieldUsage` (vector fields are immutable), `testDifferentCodecs1`/`2` (no `SimpleText` writer), and `testReadQuantizedVectorWithEmptyRawVectors` (skipped in Java too). Port bugs it found, each fixed in its own commit: the search reader verified neither the HNSW graph's `.vem` footer at open nor `.vex` in `check_integrity`, as Java's reader does (`fix(search)`, `a_damaged_hnsw_graph_file_is_refused`); a second writer did not check a field's schema against the committed segments, so it accepted a vector dimension or similarity change Java's `FieldNumbers` refuses (`fix(index)`, `a_changed_vector_schema_is_refused_with_javas_message`); and a merge dropped a vector field whose every vector was deleted, where `KnnVectorsWriter.merge` keeps it (`fix(index)`, `a_merge_keeps_a_vector_field_whose_every_vector_was_deleted`).

h7's deferrals (rows in the ledger and the inventory):

- Every base indexes one segment under a plain `IndexWriter`, with deletes in a last commit; `RandomIndexWriter`, random `IndexWriterConfig`s, `forceMerge` (its coin is drawn and ignored), random field types and `newSearcher`'s reader wrappers wait for h4. The bases therefore do not replay a Java test seed's index. Each generator draws in Java's order, but a stream leaves Java's after a consumer the port skips: `newIndexWriterConfig` and `newSearcher` in the geo and range bases; `QueryUtils`' `wrapUnderlyingReader` passes with the `checkBulkScorerSkipTo` each makes; and `RandomApproximationQuery`, whose weight takes `new Random(random.nextLong())` from the test's stream at `createWeight` and each scorer another from the weight's, where `exec/test_queries.rs` seeds each scorer from the query's own `JavaRandom`, which `test_approximation_search_equivalence.rs` seeds from the test's stream as it builds each query.
- `QueryUtils`/`CheckHits` skip the `wrapUnderlyingReader` passes: there is no `MultiReader` over arbitrary readers.
- With no `Weight`, `AssertingWeight`/`AssertingQuery` are a guard that wraps every scorer of a search (`lucene-search`'s `asserting` feature, absent from `lucene-ffi`'s build). `BlockScoreQueryWrapper` and `DisablingBulkScorerQuery` use feature-only query values; `ScorerIndexSearcher` stores scorer-only dispatch on the searcher, including manager workers. Their Java differential checks pass; scorer-only/default-bulk complete/no-score traversal measures 4.47–4.98x Java and constant-term competitive pruning 1.71–1.91x, while block-score bounds measure 1.16–1.39x and construction/iteration 0.62–0.69x with differing scorer setup and teardown scopes documented. Field-sort/facet dispatch and leaf subranges remain outside the scorer-only wrapper. `BulkScorerWrapperScorer` preserves cursor/refill/error semantics and has a paired traversal benchmark; its indexed bulk adapter still requires caller-supplied cost. `ShardSearchingTestBase` remains unported (deferred again in h8, below). Production bulk scoring supplies the leaf's `maxDoc` to `AssertingBulk::installed`, so its "past `maxDoc` is exhausted" check also runs in real searches when the guard is installed.
- `NamedQuery` is not a clause: `MatchesTestBase` takes the named clauses alongside.
- `BaseSimilarityTestCase`'s explanation checks wait for `SimScorer.explain` beyond BM25; their `rarely()` draws are kept, so the streams stay Java's.
- The geo point bases verify boxes and circles only (no polygon or geometry verifies, no fixed-input tests); `BaseXYPointTestCase` has six faithful point-array generators, overridable shape hooks, four ordered capture callbacks and rectangle/distance oracles, with Java raw-bit/RNG fixtures in both nightly modes; its full query verifiers, fixed/multivalued tests and contextual codec setup remain open. `EarthDebugger` is not needed. All eight `ShapeTestUtil` helpers are ported with explicit RNG/nightly inputs, Java raw-float/RNG-continuation fixtures and a paired benchmark; see `docs/benchmarks/shape-test-util.md`.
- Tests of the ported bases not ported: `BaseRangeFieldQueryTestCase.testRandomBig`; of `BaseGeoPointTestCase`, `testMultiValued`, `testRandomBig`, `testRandomDistance`, `testRandomDistanceHuge`, `testRectBoundariesAreInclusive`, `testEquals`, the fixed-input `testBoxBasics`, `testDistanceBasics` and `testSmallSet*` (11), the polygon tests (`testPolygon*`, `testMultiPolygonBasics`, `testNarrowPolygonCloseToNorthPole`) and the extreme- and invalid-value ones (`testIndex{ExtremeValues,OutOfRangeValues,NaNValues,InfValues}`, `testBoxNull`, `testBoxInvalidCoordinates`, `testDistance{Null,Illegal,Negative,NaN,Inf}`). Every other `test*` of the bases and of `TestSimpleExplanations`, `TestComplexExplanations`, `TestSimpleSearchEquivalence` and `TestApproximationSearchEquivalence` is ported.
- The geo/range verifier suites have no benchmark pair. The standalone `ShapeTestUtil` helpers have five checked Java/Rust workloads; their ratios and memory scope are in `docs/benchmarks/shape-test-util.md`. The bounded `BaseXYPointTestCase` generators/hooks/oracles have eight checked paired workloads; scope and ratios are in `docs/benchmarks/base-xy-point.md`.
- Every one of the 22 similarities' cases matched Java bit for bit. Three search bugs surfaced, each fixed in its own `fix(search)` commit: `IndexSearcher` searched a query without `BooleanQuery.rewrite`'s structural rules, so `w1 w1^1000` scored as a sum while Java (and `explain`) score `w1^1001` (`bq_dup` in `search.txt`); and a `DisjunctionMaxQuery` explained its value in float arithmetic where Java's explain and scorer use doubles, one ulp off its score (`dmq8`); and a phrase and a multi-phrase explained BM25's value under `ClassicSimilarity` (`phrase`, `phrase_sloppy`, `mpq`). The explanation fixture runs Java's searcher without a query cache and its booleans in occur order, and compares only the top line of `mpq`'s, `synonym`'s and (under Classic) boosted queries' explanations, and only the hits of Classic `dmq8` (its boosts' own nodes round the explained sum one ulp off the score): documented limits of Rust's `explain`.

h8's deferrals (rows in the ledger and the inventory):

- `ThreadedIndexingAndSearchingTestCase`'s indexing threads share the writer behind a `Mutex`, since the port's document API takes `&mut IndexWriter`: their operations interleave one at a time, while the search threads search NRT readers concurrently. `getCurrentSearcher`/`doSearching` are a simplified `ReaderMode`: each search thread opens its own NRT reader per iteration, after a commit half the time, where `TestNRTThreads` reopens one searcher per main-loop pass and searches it from 100-iteration threads (its open-but-deleted-files check is not ported); or every search commits and reopens one shared reader, without `TestSearcherManager`'s reopen thread, `SearcherLifetimeManager` or non-NRT branch. The port's writer merges on commit, not on flush, so the merged segments come from the commits `doSearching` makes. `advanceSegmentInfosCounter`, `FailOnNonBulkMergesInfoStream`, `ensureSaneIWCOnNightly` and `reduceOpenFiles` are not ported.
- `SuppressingConcurrentMergeScheduler` filters the failures the scheduler hands back (`merge`, `sync`, `close`); the port's `ConcurrentMergeScheduler` has no `handleMergeException` hook.
- `ShardSearchingTestBase` is not ported: its `ShardIndexSearcher` overrides `termStatistics`/`collectionStatistics` with other nodes' statistics, and the port's searcher has no override hook.
- No bench pair: test-only code.

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
