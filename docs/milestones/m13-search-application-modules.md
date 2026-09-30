# M13 — Search-application modules: query parsers, highlighting, suggesters, facets

> **Goal:** the modules applications build on top of search -- query parsing,
> highlighting, suggestions and autocomplete, faceting, in-memory and reverse
> search -- ported with Lucene's behaviour.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M10](m10-joins-grouping-queries.md), [M11](m11-analysis-common.md) |
| **Unblocks** | native `highlight`, completion suggesters, `query_string`/`simple_query_string` parsing |
| **Status** | not started |

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
| `memory` | 3 | 1,762 | none |
| `monitor` | 41 | 3,102 | none |

OpenSearch highlights, parses `query_string` and serves completion suggesters
through these modules; each of those falls back to Lucene today.

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
- **T13.2** — Highlighting, including term vectors in `DirectoryReader`.
- **T13.3** — Suggesters and the completion postings format.
- **T13.4** — Taxonomy facets and drill-sideways.
- **T13.5** — `MemoryIndex` and `monitor`.
- **T13.6** — Plugin wiring for the OpenSearch features above.

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
