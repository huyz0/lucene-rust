# M10 — Nested documents, joins, grouping, and the `queries` module

> **Goal:** block joins, query-time joins, grouping, function and interval
> queries, as `lucene-join`, `lucene-grouping` and `lucene-queries` have them,
> so OpenSearch's `nested`, `has_child`, `collapse`, `function_score` and
> `intervals` run natively.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md) |
| **Unblocks** | native `nested`, `function_score`, `intervals`, `combined_fields`, field collapsing |
| **Status** | in progress (T10.0, T10.1, T10.3 done; T10.2 queries ported; T10.4, T10.5 ported) |

---

## Why this milestone exists

None of `lucene-join` (31 files, 5.2k lines) or `lucene-grouping` (32 files,
3.2k lines) is ported, and 9 of `lucene-queries`' concrete queries are not
(its span queries are). These back some of OpenSearch's most used features:
`nested` fields are block joins (`ToParentBlockJoinQuery`), `function_score`
is `FunctionScoreQuery`, `intervals` is `IntervalQuery`, field collapsing is
grouping. Each of them falls back to Lucene today.

---

## Scope

### In scope

- `lucene-join`: `ToParentBlockJoinQuery`, `ToChildBlockJoinQuery`,
  `ParentChildrenBlockJoinQuery`, `CheckJoinIndex`, the diversifying child KNN
  queries, query-time joins (`JoinUtil`, `TermsQuery`,
  `GlobalOrdinalsQuery`), and indexing blocks of documents atomically.
- `lucene-grouping`: `GroupingSearch`, first- and second-pass collectors,
  `AllGroupsCollector`, `BlockGroupingCollector`, `TopGroups`.
- `lucene-queries`: `function` (value sources, `FunctionScoreQuery`,
  `FunctionRangeQuery`), `intervals`, `payloads`, `mlt` (`MoreLikeThis`),
  `CommonTermsQuery`, the remaining spans.
- The `sandbox` queries OpenSearch builds (`CombinedFieldQuery`'s sandbox
  form, `CoveringQuery`, `PhraseWildcardQuery`, the doc-values multi-range
  queries).
- OpenSearch: `nested`, `has_child`/`has_parent`, `function_score`,
  `intervals`, `combined_fields`, `more_like_this` and `collapse` as native
  shapes.

### Out of scope

- The rest of `sandbox` (M14).

---

## Tasks

- **T10.0** — Port inventories for `lucene-join`, `lucene-grouping` and
  `lucene-queries` (`docs/inventory/lucene-{join,grouping,queries}.tsv`,
  `check-port-inventory.py --module join|grouping|queries` in the gate and
  CI). **Done.** `--milestone M10 --module <m>` lists what each still owes.
- **T10.1** — Document blocks in the writer (atomic add and update of a
  parent with its children) and `CheckJoinIndex`. **Done.** Blocks were
  already atomic (one sequence number, never split by a flush, `hasBlocks`);
  this adds `IndexWriterConfig.setParentField` (`set_parent_field`), index
  sorting with blocks at flush and merge (whole blocks moved by their
  parents' keys), index sorting for explicit documents, the document API's
  `softUpdateDocuments`, `CheckIndex.testSort` over parents, and
  `BitSetProducer`/`QueryBitSetProducer`/`CheckJoinIndex`. Proven by
  `VerifyJoin` (real Lucene reads, checks, appends to and force-merges a
  Rust-written block index, sorted and unsorted), the merge stress test
  (`block_join_merge_stress.rs`) and block ops in `op-stream-fuzz.sh`. The
  concurrent writer takes native documents only, so it has no parent field
  and still refuses blocks in a sorted index.
- **T10.2** — Block-join queries and their scoring modes. **Queries
  ported** (`lucene-search/src/join`, `exec/join.rs`):
  `ToParentBlockJoinQuery` (all five score modes, scorer and bulk scorer),
  `ToChildBlockJoinQuery`, `ParentChildrenBlockJoinQuery`,
  `ParentsChildrenBlockJoinQuery`, `ToParentBlockJoinSortField` with
  `BlockJoinSelector`/`ToParentDocValues`, explain and matches,
  `DiversifyingChildrenFloat/ByteKnnVectorQuery`; 1056 differential searches
  (`GenBlockJoin`) bit for bit. Benchmark pair `scripts/bench-micro.sh
  --bench join` (`JoinMicro.java` / `micro_join.rs`, 60 000 blocks, ~300 000
  children with vectors; every case's hits digest equal to Lucene's): unfiltered
  diversifying KNN 1.24x, filtered 1.09x, `ParentChildren` 1.60x, `ToParent`
  `Max` 1.60x, `Total` 1.11x; `Avg`, `Min`, `None`, `ToChild`,
  `ParentsChildren` and the sort field 0.94x-1.06x, inside the 1.07x noise
  floor; `ToParent` inside a boolean 0.90x. Stage 3 made the sort field lazy
  (a block folded only when the collector reads its parent, as
  `ToParentDocValues` does: 0.52x before), gave the KNN filter its bulk
  scorer and a rank walk from documents to ordinals (0.73x before), and let
  the block-join bulk scorer's child collect into a concrete collector (the
  per-child hit was a virtual call). The boolean case's remaining time is the
  generic `ReqOptSumScorer`/conjunction scorers around the join, which
  mirror Java's; the join's own scorer is a small share of it.
- **T10.3** — Query-time joins. **Ported** (`lucene-search/src/join/query_time.rs`,
  `exec/query_join.rs`): `JoinUtil`'s four `createJoinQuery` overloads with
  every collector and to-side query -- `TermsCollector`/`TermsWithScoreCollector`
  over `SORTED`/`SORTED_SET` doc values into `TermsQuery` (a seeking term-set
  multi-term query) or `TermsIncludingScoreQuery`, the numeric join into
  `PointInSetQuery`/`PointInSetIncludingScoreQuery`, and the global-ordinal
  joins (`GlobalOrdinals{,WithScore}Collector`/`Query`, `min`/`max`). The from
  side runs through a new per-segment collector driver
  (`leaf_collector.rs`: `getLeafCollector`/`collect`/`finish` for every
  segment) and the `DocValues` getters (`reader/doc_values.rs`). 897
  differential searches (`GenQueryTimeJoin`, four segments with deletions,
  every score mode, alone and boosted in booleans) bit for bit.
  Benchmark pair `scripts/bench-micro.sh --bench query_join`
  (`QueryJoinMicro.java` / `micro_query_join.rs`: 200 000 documents in four
  segments, half from-documents referencing 20 000 keys; per word, the join
  built from `+type:from +body:word` and its top ten; every case's hits
  digest equal to Lucene's). Before stage 3: terms joins 0.63x-0.83x, global
  ordinals `None` 0.59x, numeric `None` 0.02x. Stage 3 read the from side's
  terms once per segment ordinal in ordinal order after the segment (the
  dictionary walked forward instead of an LZ4 block decompressed per
  document; each term's scores still combined in document order), reused one
  postings cursor across the to side's terms, translated the collected
  global ordinals to segment ordinals once per segment, made
  `GlobalOrdinalsQuery`'s scorer the constant-score scorer Java's
  `ConstantScoreWeight` gives it (it now stops once ten hits beat it, as
  Lucene's does), and gave the one-dimensional `PointInSetQuery` Java's
  single merged tree walk (it walked the tree once per point). After: terms
  `None` 1.39x, `Avg` 1.08x, `Max`/`Min`/`Total` 0.99x-1.05x (inside the
  1.07x noise floor), multi-valued `Max` 1.21x, global ordinals `None` 1.42x,
  `Avg` with min/max 1.13x, numeric `Max` 1.06x. Left below 1.0: global
  ordinals `Max` 0.90x -- not constant-scored, so it walks all 100 000
  to-documents as Java does, and its time is that two-phase walk (postings,
  a per-document ordinal read through a boxed doc-values iterator, the
  bulk loop), each piece Java's own -- and numeric `None` 0.90x-0.94x,
  inside its 1.09x-1.17x noise floor (it was 0.02x).
- **T10.4** — Grouping. **Ported** (`lucene-search/src/grouping`): every
  class of `lucene-grouping` (`ValueSourceGroupSelector` came with T10.5's
  value sources) -- the selectors (term, long range, double range), the first
  and second passes with their reducers, `TopGroupsCollector`,
  `AllGroupsCollector`, `AllGroupHeadsCollector`, `DistinctValuesCollector`,
  `BlockGroupingCollector`, the grouped facets, every collector manager,
  `SearchGroup.merge`/`TopGroups.merge`/`mergeBlockGroups`, and
  `GroupingSearch` with caching. 668 differential searches (`GenGrouping`:
  `GroupingSearch` by selectors and by blocks, the managers over one slice and
  two, the facets) equal to Lucene's.
  Benchmark pair `scripts/bench-micro.sh --bench grouping`
  (`GroupingMicro.java` / `micro_grouping.rs`: 200 000 documents in four
  segments, in blocks of one to eight, 2 000 groups; per word, one grouping
  search; every case's digest equal to Lucene's). The faithful port measured
  0.21x-0.82x (facets 0.24x). Stage 3, each step re-run against the 668
  differential searches: `lookupTerm` through the doc-values dictionary's
  `seekCeil` instead of a binary search of `lookupOrd`s (the facets'
  per-segment re-mapping and every selector's); each distinct term of the
  facet pairs looked up once per segment; a sparse doc-values field's
  documents decoded once per reader with a rank index (`advanceExact` was a
  binary search of the whole set) and its column read at that rank; the
  selector's ordinal map a table, filled as ordinals show up (or by one
  forward walk of the dictionary once many miss) instead of seeking every
  known term per segment; the comparator slots compared against a
  document's values as Java's `compareBottom` does -- by score, number or
  ordinal, without building the document's values unless it wins a slot
  (a keyword slot's term resolved to this segment's ordinal once); the
  group maps hashed with `FxHasher`. After: grouped facets 2.31x (single-
  valued) and 2.48x (multi-valued), all groups with group heads 2.30x,
  distinct values 1.95x, cached 1.28x, long range 1.23x. Left below 1.0:
  by relevance 0.92x, field-sorted (`n` descending, `s` within the group)
  0.80x, and blocks 0.86x. Their time is the two passes' walk of every
  matching document (about 80 000 per word), each reading the group
  field's ordinal through a boxed doc-values iterator and comparing it
  through the generic multi-key comparator -- per-document work Java's JIT
  inlines monomorphically; the algorithms and the documents visited are
  Lucene's.
- **T10.5** — Function queries and value sources. **Ported**
  (`lucene-search/src/function`, `exec/function.rs`): every class of
  `lucene-queries`' `function`, `function.docvalues` and
  `function.valuesource` packages -- `ValueSource`/`FunctionValues` (with
  `ValueFiller` and `MutableValue`), the typed `*DocValues` bases, every
  value source (constants, the doc-values field sources with their
  multi-valued selectors, `joindf`, the arithmetic, range-map, scale and
  boolean functions, `docfreq`/`idf`/`termfreq`/`tf`/`totaltermfreq`/
  `sumtotaltermfreq`/`numdocs`/`maxdoc`/`norm`, `query()`, the vector
  sources and similarities), `FunctionQuery`, `FunctionRangeQuery` with
  `ValueSourceScorer`, `FunctionMatchQuery`, `FunctionScoreQuery` with
  `boostByValue`/`boostByQuery`, `IndexReaderFunctions`, the bridges to
  `DoubleValuesSource`/`LongValuesSource` and sorting by a value source.
  Reader-wide state (`createWeight`) is computed once per search in the
  statistics pass. `GenFunction` records 96 value-source specs read for every
  document (every getter's bits, strings, objects, vectors and filled
  mutable values, or Lucene's exception) and 214 searches (hits and score
  bits, alone, boosted and in booleans, and their explanations), all equal
  to Lucene's; `(float) Math.pow` needed no tolerance. With them,
  `lucene-grouping`'s `ValueSourceGroupSelector`: 28 grouped searches
  (`groups.tsv`, 14 sources by all documents and by `t:red`) equal to
  Lucene's.
- **T10.6** — Intervals, payload queries, `MoreLikeThis`, `CommonTermsQuery`.
- **T10.7** — Plugin wiring for the OpenSearch shapes above.

## Acceptance criteria

- [ ] Every query matches Lucene's hits and scores bit for bit on generated
      fixtures, including empty and single-child blocks and deleted parents.
- [ ] A Rust-written block index passes Lucene's `CheckJoinIndex`.
- [ ] OpenSearch's `nested` and `function_score` YAML suites fail
      identically with and without native execution.
- [ ] Each new query is no slower than Lucene on its benchmark.

## Risks and unknowns

- **Block integrity under merges.** A merge that splits or reorders a block
  breaks every join silently. It needs a resource-bound-style test: many
  merges, then `CheckJoinIndex`.
- **Value-source breadth.** `function` has many small classes; most are
  one-liners, but each needs a scoring differential.

## Exit artifacts

- Join, grouping and function fixture generators
- `docs/parity.md` rows for `lucene-join`, `lucene-grouping`, `lucene-queries`
  and the ported `sandbox` classes
