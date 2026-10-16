# M13 — Search-application modules: query parsers, highlighting, suggesters, facets

> **Goal:** the modules applications build on top of search -- query parsing,
> highlighting, suggestions and autocomplete, faceting, in-memory and reverse
> search -- ported with Lucene's behaviour.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M10](m10-joins-grouping-queries.md), [M11](m11-analysis-common.md) |
| **Unblocks** | native `highlight`, completion suggesters, `query_string`/`simple_query_string` parsing |
| **Status** | in progress, 0 of 4 criteria -- T13.1: the classic `QueryParser`, `MultiFieldQueryParser`, `SimpleQueryParser`, the `ext` package, the surround language, `ComplexPhraseQueryParser`, `CoreParser` (the `xml` package's core builders) and the `flexible` framework (`StandardQueryParser`, `PrecedenceQueryParser`, `QueryParserUtil`) are ported (every fixture row equal); `CorePlusQueriesParser` and `CorePlusExtensionsParser` are not |

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
- **T13.3** — Suggesters and the completion postings format.
- **T13.4** — Taxonomy facets and drill-sideways.
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
progress (T13.5a-c done; T13.5d done -- batches, the four matchers, partitioned matching, persistence verified both ways against Lucene, the regexp handler and debug()); T13.2, T13.3, T13.4 and T13.6
not started. M13 closes when all four acceptance criteria hold.

## Acceptance criteria

- [ ] Every parser yields Lucene's query tree for the whole T13.1 corpus,
      including its syntax errors (same failure, same position).
- [ ] Highlighters return Lucene's fragments and offsets on a fixture corpus.
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
