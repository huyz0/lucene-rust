# M13 — Search-application modules: query parsers, highlighting, suggesters, facets

> **Goal:** the modules applications build on top of search -- query parsing,
> highlighting, suggestions and autocomplete, faceting, in-memory and reverse
> search -- ported with Lucene's behaviour.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M10](m10-joins-grouping-queries.md), [M11](m11-analysis-common.md) |
| **Unblocks** | native `highlight`, completion suggesters, `query_string`/`simple_query_string` parsing |
| **Status** | in progress, 1 of 4 criteria -- T13.2: the classic, match, unified and fast vector highlighters ported, every fixture row equal; T13.1: the classic `QueryParser`, `MultiFieldQueryParser`, `SimpleQueryParser`, the `ext` package, the surround language, `ComplexPhraseQueryParser`, `CoreParser` (the `xml` package's core builders) and the `flexible` framework (`StandardQueryParser`, `PrecedenceQueryParser`, `QueryParserUtil`) are ported (every fixture row equal); `CorePlusQueriesParser` and `CorePlusExtensionsParser` are not |

---

## Why this milestone exists

The port has a subset of the classic query parser, a `UnifiedHighlighter`
subset, sorted-set and range facets, and the suggester's FST shape. The
modules themselves are much larger:

| Module | Files | Lines of code | Port today |
|---|---|---|---|
| `queryparser` | 250 | 18,359 | classic parser, subset |
| `highlighter` | 95 | 8,900 | `UnifiedHighlighter`, subset |
| `suggest` | 81 | 8,205 | FST completion shape |
| `facet` | 101 | 11,007 | SSDV and range facets |
| `memory` | 3 | 1,762 | `MemoryIndex` ported (T13.5) |
| `monitor` | 41 | 3,102 | none |

OpenSearch highlights, parses `query_string` and serves completion suggesters
through these modules; each of those falls back to Lucene today.

Each module has a per-class inventory, `docs/inventory/lucene-<module>.tsv`
(`queryparser`, `highlighter`, `suggest`, `facet`, `monitor`, `memory`);
`scripts/check-port-inventory.py --module <module> --milestone M13
--summary` lists what M13 still owes in it.

---

## Scope

### In scope

- `queryparser`: classic, `simple`, `flexible` (standard and precedence),
  `complexPhrase`, `surround`, `xml`, `ext`.
- `highlighter`: `UnifiedHighlighter` in full, `FastVectorHighlighter`
  (needs the term-vector reader wired into `DirectoryReader`, a gap
  `parity.md` records), the classic `Highlighter`, matches.
- `suggest`: `AnalyzingSuggester`, `FuzzySuggester`,
  `AnalyzingInfixSuggester`, `FreeTextSuggester`, `BlendedInfixSuggester`,
  the document suggesters and the completion postings format, spell checkers.
- `facet`: taxonomy facets (the taxonomy index and its writer/reader),
  `DrillDownQuery`, `DrillSideways`, range and long-value facets, the
  `facetset` package.
- `memory` (`MemoryIndex`) and `monitor`.
- OpenSearch: highlighting, `query_string`/`simple_query_string` parsing,
  completion and term/phrase suggesters as native paths.

### Out of scope

- Modules in M14.

---

## Tasks

- **T13.1** — Query parsers: a differential harness that parses a corpus of
  query strings in Java and in Rust and compares the query trees.
  *Progress:* the harness is `fixtures/src/GenQueryParserClassic.java` ->
  `crates/lucene-search/tests/queryparser_classic_fixtures.rs`: a query's
  `toString` and class tree, or its exception message, row for row. The
  classic parser is the JavaCC 7 lexer and parser ported state for state
  (`crates/lucene-search/src/queryparser/classic/`), so its 9,000-odd
  failing rows carry Lucene's token, column and expected tokens; with
  `MultiFieldQueryParser` it matches every row under fourteen
  configurations (operators, analyzers, synonyms, per-field analyzers,
  boosts), and `SimpleQueryParser` (`GenQueryParserSimple`, twelve
  configurations, each flag off in turn), the `ext` package
  (`GenQueryParserExt`: five extensions, two delimiters) and the surround
  language (`GenQueryParserSurround`: trees, exceptions, and the rewrite
  against a Java-written index) likewise, and `ComplexPhraseQueryParser`
  (`GenQueryParserComplexPhrase`: in order and not, trees, exceptions, the
  span rewrite and its hits); the three JavaCC lexers and parsers share one
  runtime (`queryparser/javacc.rs`), and the XML query parser's core builders
  (`GenQueryParserXml`: 104 documents, each's tree and hits over an index with
  an `IntPoint` field, or its exception, through `quick-xml`). Remaining:
  date-resolution ranges, and the `xml` package's `LikeThisQuery` and
  `FuzzyLikeThisQuery` builders (`CorePlusQueriesParser`,
  `CorePlusExtensionsParser`). The `flexible` framework is ported
  (`GenQueryParserFlexible`: its lexer token for token over 8,246 queries,
  352 queries parsed under eleven configurations and 7,894 fragment joins); its points
  and date-resolution configuration is deferred to M14
  (`docs/inventory/lucene-queryparser.tsv`).
- **T13.2** — Highlighting, including term vectors in `DirectoryReader`.
  **Ported and benchmarked** (`scripts/bench-micro.sh --bench highlight`, pair `HighlightMicro`
  vs Lucene 10.5.0; ratio > 1 is Rust faster): `uh_postings` 0.96x (inside the 1.18x noise
  floor), `fvh` 0.75x (noise 2.06x), `uh_analysis` 0.36x, `uh_tv` 0.36x. Stage 3 so far: the
  `MemoryIndex` strategies no longer write the one-document codec segment on the
  weight-matches path (0.17x -> 0.36x). What remains is that same `MemoryIndex`: the Rust one
  encodes each document's postings into a codec segment for `IndexSearcher` to read, where
  Lucene searches the in-heap postings directly. A lightweight in-memory leaf reader is a
  `MemoryIndex`-wide change (T13.5's, `memory/reader.rs`), recorded as the reason for the two
  ratios below 1.0, not done here. The classic `Highlighter` (`GenHighlight`), the
  `matchhighlight` package (`GenMatchHighlight`, with the JDK's word and
  sentence breaks fuzzed against `BreakIterator`), the `UnifiedHighlighter`
  (`GenUnifiedHighlight`: every offset source, passage and break option;
  five classes partial, their gaps deferred to M14 in
  `docs/inventory/lucene-highlighter.tsv`) and the `FastVectorHighlighter`
  (`GenVectorHighlight`: 10,036 rows, eight configurations, term vectors with
  and without offsets or positions, matched fields, multi-valued and discrete
  fields), every fixture row equal.
- **T13.3** — Suggesters and the completion postings format, five sub-tasks in order:
  - **T13.3a** — the base layer: `Lookup`, `LookupResult`, the `CHARSEQUENCE_COMPARATOR`,
    `LookupPriorityQueue` (Java's heap, so ties match), `InputIterator` and its
    wrapper, `BufferedInputIterator`, `SortedInputIterator` (in memory). **Done**
    (the dictionaries and `UnsortedInputIterator` have Java fixtures, `GenSuggestFst.java`:
    `FileDictionary`, `DocumentDictionary`, `DocumentValueSourceDictionary`, `java.util.Random`).
    (`BitsProducer` came with T13.3d.)
  - **T13.3b** — TST and FST lookups, **done**. Benchmark pair `fst_lookup` vs Lucene 10.5.0: `wfst_lookup` 1.29x (inside the noise floor), `fstc_lookup` 1.50x, `wfst_build` 2.16x, `fstc_build` 2.27x. `TSTLookup` (792 queries, `get`, and
    `store()` bytes equal Lucene's, `GenSuggest.java`); `WFSTCompletionLookup` and
    `FSTCompletionLookup` (396 and 792 lookups, `store()` bytes equal Lucene's and Lucene's
    bytes load, `GenSuggestFst.java`). Sorting is in memory, so `ExternalRefSorter` is `InMemorySorter`.
  - **T13.3c** — analyzing suggesters. **C1 done**: `AnalyzingSuggester`, `FuzzySuggester`, `SuggestStopFilter` and its factory
    (11 configurations x 480 lookups, `store()` bytes and `load()` equal Lucene's, `GenAnalyzingSuggest.java`).
    Benchmark (`scripts/bench-micro.sh analyzing`, Rust vs Lucene 10.5.0, ratio > 1 is faster): `analyzing_build` 1.96x,
    `analyzing_lookup` 1.14x (inside the noise floor), `fuzzy_build` 1.86x, `fuzzy_lookup` 2.81x.
    **C2 done**: `FreeTextSuggester` (7 configurations, `store()` bytes and `load()` equal Lucene's, `GenFreeTextSuggest.java`);
    `TopNSearcher` gained `skip_label` (Java's `addIfCompetitive` override). Benchmark (`--bench freetext`):
    `freetext_lookup` 1.43x, `freetext_build` 26.85x.
    **C3 done**: `AnalyzingInfixSuggester` and `BlendedInfixSuggester`, in memory (the documents in weight order with
    their postings; no `Directory`, so no persistence or merges -- the ledger lists each difference). 14 configurations
    of the lookup grid, the NRT flow, payload and error paths against `GenInfixSuggest.java`. Benchmark (`--bench infix`,
    20,000 suggestions): `infix_lookup` 2.60x, `infix_build` 1.61x (inside the noise floor). **T13.3c done.**
  - **T13.3d** — `suggest.document`: the completion postings format and its queries (what OpenSearch uses; prioritised).
    **D1 done**: the `.cmp`/`.lkp` writer (bytes equal Lucene's), the reader, `FSTUtil` and `NRTSuggester.lookup`
    with `TopSuggestDocsCollector` (180 prefix and 96 regex queries equal Lucene's, `GenCompletion.java`).
    **D2 done**: `Completion104` routed through `PerFieldPostingsFormat` (`IndexWriter` flush and merge,
    `SegmentReader::suggester`) and `SuggestField`; the flushed and force-merged `.lkp`/`.cmp` equal
    Lucene's, and `VerifyCompletion` has Lucene answer every prefix and regex suggestion over the
    Rust-written index as over its own. It moved `forceMerge(1)` to Lucene's largest-first merge order.
    **D3 done**: `PrefixCompletionQuery`/`RegexCompletionQuery`/`FuzzyCompletionQuery`/`ContextQuery`,
    `CompletionWeight`, `SuggestIndexSearcher`, `CompletionAnalyzer`, `ContextSuggestField`, `BitsProducer`:
    2960 queries equal Lucene's key, doc, score bits and context (`GenCompletionQuery.java`). It fixed
    `SuggestQueue::insert_with_overflow` to Java's strict `lessThan`.
    Benchmarked (`bench-micro.sh --bench completion`, 3 reps, noisy shared machine): prefix 1.08x (inside
    the noise), fuzzy 1.32x, regex 0.89x, context 0.79x of Lucene 10.5.0 (noise floor 1.15x). Optimised from
    0.82x/0.68x/0.74x: the `TopNSearcher` queue is a deque, queued paths drop their arc output, sequence
    outputs append in place, and `NrtSuggester` loads once per field (Java's cache). What remains in the
    context and regex cases is automaton construction (`concatenate`/`determinize`/`union`) and the
    per-start-path copies of the FST's `Vec` outputs; both are cases where Java shares the `BytesRef`.
    Written up rather than closed: a query takes 4-40 us either way.
    **D4 done**: the completion formats of older indexes, read: `Completion90`, `Completion99`, `Completion912` and
    `Completion101` (over `Lucene101` and `Lucene103`), each written by its own release (`gen-bwc-fixtures.sh
    --completion`: 9.0.0, 9.11.1, 9.12.2, 10.2.2, 10.3.2), answer `GenCompletionQuery`'s 2960 queries as Lucene 10.5.0
    does, and force-merge into `Completion104`; `CompletionTerms`' weights and `TopSuggestDocs.merge` equal Lucene's.
    `Completion50`/`Completion84` are not needed (Lucene 8.x formats; a Lucene 8-created index does not open). **T13.3d done, and with it T13.3.**
  - **T13.3e** — `spell`, **done**: the string distances, `SuggestWord` and its queue and comparators,
    `PlainTextDictionary`, `LuceneDictionary`, `HighFrequencyDictionary`, `DirectSpellChecker`,
    `WordBreakSpellChecker` and `SpellChecker` (its n-gram index written by this port's `IndexWriter`, and Lucene's
    read), against `GenSpellDistances`, `GenSpellChecker`, `GenSpellWordBreak` and `GenSpellCheckerIndex`, scores to the
    bit. Benchmark (`--bench spell`): `direct_suggest` 11.66x, `word_break` 5.94x, `spell_suggest` 1.09x (inside the
    noise floor), `spell_index` **0.42x** -- the `IndexWriter` document API's cost (per-document field registration and
    inversion), owed by the writer: the explicit-document flush now hashes terms (`TermHash`) instead of a `BTreeMap`
    per occurrence, from 0.29x.
- **T13.4** — Taxonomy facets and drill-sideways, four sub-tasks in order (`lucene-facet`, 80 classes
  `todo:M13` in `docs/inventory/lucene-facet.tsv`; the existing `crates/lucene-search/src/facets.rs`
  holds the earlier sorted-set and range subset, folded into the real classes as each sub-task lands):
  - **T13.4a** — the taxonomy index: `FacetLabel`, the `writercache` package, `TaxonomyWriter`,
    `DirectoryTaxonomyWriter`, `DirectoryTaxonomyReader`, `TaxonomyIndexArrays`,
    `ParallelTaxonomyArrays`, `Consts`, `FacetField`, and `FacetsConfig.build` for taxonomy
    dimensions. Acceptance: Lucene reads a Rust-written taxonomy index and the Rust reader reads a
    Lucene-written one (`CheckIndex` plus ordinals, parents and children equal).
    **Done**: `GenTaxonomy` -> `taxonomy_fixtures.rs` (Java's taxonomies read; Rust's writer replaying
    Java's calls under four caches to the same ordinals and dump (the segment layout is not compared); `addTaxonomy`,
    `replaceTaxonomy`; `FacetsConfig.build`'s documents field for field; label hashes and order;
    the refusals), and `write_taxonomy_fixture` -> `VerifyTaxonomy` (`scripts/verify-write-path.sh`:
    CheckIndex, Lucene's reader dumps what this port's read, and Lucene's writer appends to the
    Rust-written index keeping every ordinal and the epoch). Found on the way: `addIndexes` refused a
    compound-file source (`check_addable` read the segment's field infos outside its `.cfs`).
    Benchmark (`--bench taxonomy`, two runs, noise floor 1.39x/1.41x on a loaded host):
    `taxo_add_new` 4.32x/3.92x, `taxo_add_lookup` 1.80x/1.93x, `taxo_ordinal` 2.88x/2.95x,
    `taxo_path` 6.11x/5.14x, `taxo_add_cached` 1.05~/1.09~ (inside the noise floor), from 0.40x:
    `FacetLabel` now caches each component's hashes as Java's `String` caches `hashCode`, a clone
    compares equal by identity (Java's `HashMap` finds the key it was given by `==`), the LRU maps
    hash with `FxHasher`, and the unbounded cache's `get` no longer reorders (Java builds a plain
    `HashMap` for it). Partial: `FacetsConfig.build` without `AssociationFacetField` (T13.4b);
    `SearcherTaxonomyManager`, `ReindexingEnrichedDirectoryTaxonomyWriter`, `TaxonomyMergeUtils`,
    `OrdinalMappingLeafReader` and `PrintTaxonomyStats` stay with T13.4d2.
  - **T13.4b** — counting: `Facets`, `MultiFacets`, `FacetResult`/`LabelAndValue`,
    `FacetsCollector`/`FacetsCollectorManager`, `TaxonomyFacets`, `FastTaxonomyFacetCounts`,
    `IntTaxonomyFacets`/`FloatTaxonomyFacets`, the association fields and aggregations,
    `TaxonomyFacetLabels`, `FacetCountsWithFilterQuery`, `FacetUtils`.
    **Done**: `GenFacetCounts` -> `facet_counts_fixtures.rs` (a 2,400-document multi-segment index
    with deletions; every public `Facets` call of the counts, both association facets (SUM/MAX, by
    field and by `DoubleValuesSource`) and `MultiFacets`, over match-all, selective and empty
    queries, dense and sparse; `FacetsCollectorManager.search`/`searchAfter`;
    `TaxonomyFacetLabels`; `FacetsConfig.build` of associations; Java's refusals), plus unit tests
    after `TestTaxonomyFacetCounts`, `TestTaxonomyFacetAssociations`, `TestTaxonomyFacetLabels`,
    `TestMultipleIndexFields` and `TestFacetsConfig`. The earlier sorted-set and range code returns the shared
    `FacetResult` (values are `FacetValue`s) and `SortedSetFacetCounts` implements `Facets`; the
    FFI's primitives are unchanged. Java's three taxonomy
    base classes are one `TaxonomyFacets` told apart by `Aggregation`. Bench
    `scripts/bench-micro.sh --bench facet_counts` (`FacetCountsMicro.java` / `micro_facet_counts.rs`:
    200,000 documents in four segments, a 2,000-author flat dim, a three-level hierarchy, a
    multi-valued tag dim, int and float associations; 3 reps, noise floor 1.09x):
    `facet_count_rare` 1.33x (sparse counts over 1% of the documents), `facet_top_children` 1.40x,
    `facet_count_all` 1.04~, `facet_int_assoc_sum` 1.31x, `facet_float_assoc_sum` 1.39x,
    `facet_collect_all` 352x (first pass: 1.28x, 0.91~, 0.86x, 0.70x, 0.70x, 0.27x). The first
    optimisation pass: the hits walked directly (`facet_utils::for_each_hit`, led by the cheaper side
    as `ConjunctionDISI` is), a segment's decoded values taken in bulk
    (`SortedNumericDocValues::take_values`), the dense counts and association arrays borrowed once
    per segment, and a full queue's least count short-circuiting `insertWithOverflow`. The second,
    each cause measured first with temporary per-step cases (`tmp_*`, both engines, since removed):
    - `facet_collect_all` was the driver, not the collector: a match-all called the collector once
      per hit through `LeafCollector` and `PerSegment`. Ported Lucene's path --
      `ConstantScoreScorerSupplier.matchAll` -> `DenseConjunctionBulkScorer` -> `collectRange` --
      as `ScoringCollector::collect_range`/`SegmentCollector::collect_range` (default: a `collect`
      per document, Java's), `Scorer::constant_score` and `exec::bulk::constant_runs_score` (runs of
      at least `WINDOW_SIZE / 2` documents, no deletions, no pruning); `FacetsCollector` sets a run
      in its `DocIdSetBuilder` at once (the result is the per-document one). Java's own
      `FacetsCollector` takes the default, so the Rust side now skips work Java does: the ratio is
      not a like-for-like collector comparison.
    - The association sums were the `BINARY` decode (`tmp_bin_decode` 0.34x): each value re-derived
      its `DirectMonotonic` block from the meta. `direct_monotonic::Reader` resolves the blocks once
      (`DirectMonotonicReader.getInstance`); `BinaryReader` and `SortedNumericReader` hold one,
      and the latter reuses the previous document's end address as the next start. 0.66-0.72x ->
      1.31-1.39x.
    - `facet_top_children` was the sibling walk (`tmp_walk` 0.71x: a virtual `IntArray::get` and
      two dependent loads per step) and the labels (`getBulkPath` 0.24x: two `Arc` counts per
      cached `FacetLabel` clone and drop, a second `Vec`, a `DimConfig` copy per call).
      `IntArray::chunks` lets the walk read `ChunkedIntArray`'s chunks directly, one slice when
      the taxonomy is one chunk (`IntArrayReader`); `FacetLabel` keeps one `Arc`; a fully cached
      `get_bulk_path` returns its hits as they are; `FacetsConfig::dim_config` returns the shared
      config as `getDimConfig` does. 0.76x -> 1.40x.
    - `facet_count_all` 0.81x -> 1.04~: the same address resolution and the label and walk
      changes. What is left is the per-document `SORTED_NUMERIC` iteration (`tmp_sn_decode` 0.82x:
      a dispatched `next_doc` and `take_values` and a `Vec` refill per document, where the JIT
      inlines `nextValue` into the counting loop); the case sits inside the noise floor (1.05x).
  - **T13.4c** — drill-down: `DrillDownQuery`, `FacetQuery`, `MultiFacetQuery`, `DrillSideways`,
    `DrillSidewaysQuery`, `DrillSidewaysScorer`.
    **Done.** Fixture `GenDrillDown.java` -> `drill_down_fixtures.rs` (`sideways.tsv`,
    `paging.tsv`, `queries.tsv`, `errors.tsv`): workloads whose costs pick each scorer strategy
    (query-first, drill-down-advance, union; the choice itself is not asserted), the
    concurrent (per-dim query) form, paging, custom collector managers, `toString`/rewrite and
    Java's refusals, matched byte for byte. Every fixture segment has deletions, so the advance
    path's skipped two-phase check on dim 1 (no deletions) is covered by the scorer's unit tests
    against a brute-force reference only. Deviations (ledger rows): taxonomy only (the
    `SortedSetDocValuesReaderState` constructors are T13.4d2); the executor form runs its queries
    in turn; no `Sort` overloads; the `buildFacetsResult`/`scoreSubDocsAtOnce` hooks are not
    overridable. `drill_sideways` bench (noise 1.07x): advance 1.36x, query-first 1.21x, union
    0.97~, `#check` digests equal. Union is inside the noise: its time is the sideways
    `FastTaxonomyFacetCounts` over ~200k docs, the per-document `SORTED_NUMERIC` iteration
    `facet_count_all` already measures (above), not the scorer.
  - **T13.4d1** — value and range counting: `LongValueFacetCounts`, `StringValueFacetCounts`
    (`StringDocValuesReaderState`), the `range` package (`LongRange`/`DoubleRange` with `getQuery`
    and the values-source forms, `RangeFacetCounts`, the exclusive and overlapping
    `LongRangeCounter`s, `DynamicRangeUtil`), `rangeonrange`, `facetset`,
    `Multi{Long,Double}Values[Source]`.
    **Done.** Fixture `GenValueFacets.java` -> `value_facets_fixtures.rs` (`facets.tsv`,
    `queries.tsv`, `dynamic.tsv`, `errors.tsv`), a single- and a multi-segment index each, every
    count, top-N, `getQuery` hit list and refusal matched exactly. Deviations (ledger rows): the
    hash-table orders of `LongValueFacetCounts`/`StringValueFacetCounts` (`getAllChildren`,
    `toString`) come out sorted (Java's are seeded per process); `DynamicRangeUtil` reads segments
    in turn, not on an executor; `getQuery`'s fast-match is a filter beside the range check, not
    the leading scorer; `rangeonrange` refuses an empty or mixed-dimension range list with
    `IllegalArgument` (Java: `ArrayIndexOutOfBoundsException`). No form needs an `Expression`
    source, so none is deferred to M14. The FFI keeps calling the older dim-less range
    primitives in `facets.rs`. `value_facets` bench (noise 1.08x, `#check`
    digests equal): long 1.40x, long multi 0.96~, string 1.13x, long range 1.37x, overlapping
    long range 1.06~, double range 1.73x. The faithful port first measured string 0.48x and
    long range 0.50x: both read a `SORTED`/`NUMERIC` field through the multi-valued adapter
    (two dispatches a document) where Java takes `unwrapSingleton`'s single-valued values, and
    the range counter's branchy binary search cost the rest; the fix reads the single-valued
    form, walks the hits as a set and finds the interval with `partition_point`. Long multi is
    inside the noise: per document it is the `SORTED_NUMERIC` iteration `facet_count_all`
    already measures (above).
  - **T13.4d2** — the rest of the package: the sorted-set classes still `todo` (the abstract and
    concurrent SSDV counts) and the SSDV `DrillSideways` constructors,
    `RandomSamplingFacetsCollector`, `OrdinalMappingLeafReader`, `TaxonomyMergeUtils`,
    `SearcherTaxonomyManager`, `PrintTaxonomyStats`, `ReindexingEnrichedDirectoryTaxonomyWriter`.
  Each sub-task: fixture-first (a `Gen*.java` writes the index and the expected results), then the
  port, tests, the ledger and inventory rows, and a benchmark pair for the counting paths.
- **T13.5** — `MemoryIndex` and `monitor`. `MemoryIndex` is ported
  (`GenMemoryIndex`: 16 documents under eight configurations, every query's
  score bit for bit, `toStringDebug`, the refusals), searched in memory over
  the codec's own encoding: 1.08x Lucene to build, 1.8x to search. Its
  benchmark moved `FuzzyQuery` onto Lucene's parametric Levenshtein
  automata, 4x cheaper to build per query.
  `monitor` (41 classes, 5,374 lines) is planned as four sub-tasks, in order:
  - **T13.5a** `QueryAnalyzer`, `QueryTree`, `QueryDecomposer`,
    `TermWeightor`, `TermsEnumTokenStream`, `TermsHashBuilder`: each query's
    term tree, the input to the presearcher. **Done**: `QueryAnalyzer`,
    `QueryTree`, `QueryDecomposer`, `TermWeightor` ported and tested
    (`GenMonitor.java`, `monitor_fixtures.rs`); `TermsEnumTokenStream` and
    `TermsHashBuilder` are part of T13.5b, which needs them.
  - **T13.5b** `TermFilteredPresearcher`, `MultipassTermFilteredPresearcher`,
    `Presearcher`, `PresearcherMatch(es)`: the term-filter index that narrows
    a document batch to candidate queries. **Done** for the presearcher
    itself (`presearch.tsv`, `batch.tsv`); `PresearcherMatch(es)` are the
    result types of T13.5c's matching path.
  - **T13.5c** `Monitor`, `MonitorQuery`, `MonitorConfiguration`,
    `QueryIndex` and its readonly/writable forms, `MatchingQueries`,
    `CandidateMatcher`, `CollectingMatcher`, `ParallelMatcher`,
    `PartitionMatcher`, `MatcherFactory`, `QueryMatch`, `ScoringMatch`,
    `ExplainingMatch`, `HighlightsMatch`: the matching path, end to end.
  - **T13.5d** `MonitorQuerySerializer`, `ConcurrentQueryLoader`, `SlowLog`,
    `QueryTimeListener`, `MonitorUpdateListener`, `CustomQueryHandler`,
    `RegexpQueryHandler`, `SuffixingNGramTokenFilter`: persistence and
    operations.
  Each sub-task is a port with its own differential fixtures (queries and
  documents from `GenMonitor.java`), and its own inventory rows.
- **T13.6** — Plugin wiring for the OpenSearch features above.

**Status by task:** T13.1 done; T13.5 `MemoryIndex` done, `monitor` in
progress (T13.5a-c done; T13.5d done -- batches, the four matchers, partitioned matching, persistence verified both ways against Lucene, the regexp handler and debug()); T13.2 ported and benchmarked (`uh_analysis`/`uh_tv` 0.36x, reason above); T13.3 done; T13.4a-c and T13.4d1 done (T13.4d2 not started; `facet_count_all` 1.04~ and `drill_sideways_union` 0.97~ inside the noise, reasons above); T13.6
not started. M13 closes when all four acceptance criteria hold.

## Acceptance criteria

- [ ] Every parser yields Lucene's query tree for the whole T13.1 corpus,
      including its syntax errors (same failure, same position).
- [x] Highlighters return Lucene's fragments and offsets on a fixture corpus
      (T13.2; benchmarked, the two `MemoryIndex`-bound ratios written up).
- [ ] Real Lucene reads a Rust-written completion field and taxonomy index.
- [ ] OpenSearch's highlighting and suggester YAML suites fail identically
      with and without native execution.

## Risks and unknowns

- **The flexible parser's size.** It is most of `queryparser` and heavily
  object-oriented (processor pipelines); port by behaviour and query tree,
  not class by class.

## Exit artifacts

- Query-string, highlighting and suggester fixture corpora
- `docs/parity.md` rows for the six modules
