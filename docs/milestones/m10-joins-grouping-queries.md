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
| **Status** | in progress (T10.0, T10.1, T10.3 done; T10.2 queries ported; T10.4, T10.5 ported; T10.6 done: intervals, common terms, more-like-this, spans, payloads, two benchmark cases below 1.0 written up; T10.7 wired, its REST and YAML runs pending) |

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
  per-child hit was a virtual call). Then (2026-10-05) `ToParent` inside a
  boolean (one scoring join as the `MUST` beside one optional term) got its
  own bulk path, `Bulk::ReqScorerOpt`: `ReqOptSumScorer` when the required
  scorer's maximum is unbounded, so no threshold can skip a document of it
  and the generic leap-frogging only cost a virtual call per document on each
  side -- 6.91 ms -> 6.00 ms, 1.10x -> 1.26x (interleaved against the build
  before, noise floor 1.12x; every other join case unchanged).
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
  `Avg` with min/max 1.13x, numeric `Max` 1.06x. Then (2026-10-05, the
  same-machine A/B against the build before): `GlobalOrdinalsQuery`'s
  scorer reads the join field's ordinals straight off the codec's column
  (no boxed iterator per document) and confirms its approximation's
  documents a batch at a time -- global ordinals `Max` 0.94x -> 1.21x-1.28x,
  `Avg` with min/max 1.07x -> 1.39x-1.42x, `None` 1.41x; numeric `None`
  1.10x-1.15x (it was 0.90x-0.94x, inside its noise floor), every other case
  1.07x-1.62x. Nothing left below 1.0. The T10.3-T10.4
  review bounded the from side's memory -- `TermsCollector` marks ordinals in
  a per-segment bit set, `TermsWithScoreCollector` drains its pending
  `(ordinal, score)` pairs every 65 536 and keeps drained ordinals' ids for
  the segment, `GlobalOrdinalsWithScoreCollector` allocates Java's 4096-slot
  blocks lazily -- with the 897 searches unchanged (also with a drain every
  two pairs) and no case moving outside the noise in an interleaved
  same-machine A/B of the two builds (five repetitions: every case within
  0.90x-1.03x of before, spreads 1.06x-1.25x).
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
  distinct values 1.95x, cached 1.28x, long range 1.23x. Then (2026-10-05):
  the group field's ordinals read straight off the codec's column
  (`reader/doc_values.rs::SortedOrds`, the numeric sort key likewise), each
  document's group found by the selector's term id in a table instead of
  by hashing its bytes (`grouping::IdIndex`), and a one-key numeric group
  sort compared without the generic multi-key loop. By relevance 1.00x ->
  1.19x-1.23x, field-sorted 0.83x-0.92x -> 1.12x-1.16x, blocks 0.89x-0.90x ->
  1.05x-1.07x (inside its 1.04x-1.07x noise floor), distinct values
  2.22x-2.43x, cached 1.47x-1.55x, every case now at or above 1.0 (two
  interleaved runs against the build before; noise floors 1.03x-1.22x).
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
  mutable values, or Lucene's exception) and 220 searches (hits and score
  bits, alone, boosted and in booleans, and their explanations), all equal
  to Lucene's; `(float) Math.pow` and `idf`'s `Math.log` needed no tolerance
  on any fixture value (equal here, not guaranteed across libm/JIT
  implementations: neither is `StrictMath`). With them,
  `lucene-grouping`'s `ValueSourceGroupSelector`: 28 grouped searches
  (`groups.tsv`, 14 sources by all documents and by `t:red`) equal to
  Lucene's. The review after the port found the reader-wide state computed
  only on scoring paths -- a sorted, counted, aggregated or terminated search
  read `scale`/`docfreq`/`maxdoc`/`IndexReaderFunctions` from one segment --
  and a boolean's explanation scored without the searcher's similarity. Every
  entry point now prepares its function queries over all its segments, a
  leaf without them errors (gate `toplevel-whole-reader`), and 16 more
  searches (`gfilter`, `gfilterclassic`, `fqboolclassic`; 236 in all) pin it.
  Benchmark pair `scripts/bench-micro.sh --bench function`
  (`FunctionMicro.java` / `micro_function.rs`: 200 000 documents in four
  segments, 16 words, one top-10 search per word, query cache off on both
  sides). First run: 0.56x-0.87x, plus a digest mismatch on
  `FunctionMatchQuery` as a filter. That was the Rust runner leaving the
  segment query cache on where Java's `setQueryCache(null)` turns it off: a
  cached filter changes where a top-10 search stops counting hits, never the
  hits. Fixed in the runner, and covered by a new `fmqfilter` fixture search.
  Stage 3: field sources read the segment's `NumericReader` directly,
  instead of through a boxed `NumericDocValues` iterator. That took
  `fn_query_field` from 0.56x to 0.75x.

  Before the batches (3 interleaved reps, noise floor 1.09x): `fn_score_field`
  0.62x, `fn_boost_composite` 0.88x, `fn_query_field` 0.75x,
  `fn_query_composite` 0.80x, `fn_range` 0.75x, `fn_range_filter` 0.84x,
  `fn_termfreq` 0.83x, `fn_tf_idf` 0.73x, `fn_match` 0.92x. Every case was
  a per-document chain of virtual calls -- `Bulk::score` -> `dyn
  Scorer::score` -> `dyn FunctionValues`/`DoubleValues` -> the doc-values
  reader, each link returning a `Result` through memory -- which Java's JIT
  inlines into one loop at these monomorphic call sites; the decode was
  14%-34% of each profile.

  Stage 3, the batches (M10 optimisation pass): one virtual call per batch
  of documents instead of several per document.
  `FunctionValues::{float_val_batch, double_val_batch, range_batch}` and
  `DoubleValues::fill_batch` are each defined as the per-document getters,
  asked in the same order (so forward-only values read exactly as before),
  with the per-document loop as their default -- statically dispatched
  inside each implementation -- and overrides where the batch is cheaper:
  the arithmetic functions ask each child for the whole batch, and the
  field columns read the 64-aligned window covering a dense batch through
  `NumericReader::fill_window` (a chunked decode; one `IndexedDISI` walk for
  a sparse column). `AllScorer`, `ValueSourceScorer` and
  `FunctionScoreScorer` score a batch at a time (`Scorer::prefers_batches`,
  collected by `DefaultBulkScorer`'s loop; they ignore thresholds, so
  nothing a threshold could skip is read). A `DoubleValues` only batches
  when nothing under it reads the scorer's score through the per-document
  cell (`batch_capable`; a wrapped value source checks that no
  `fromDoubleValuesSource` kept its scorer view). `+term #filter` with a
  two-phase filter over every document (`frange`, `FunctionMatchQuery`)
  became `Bulk::TwoPhaseTerm`: the term's postings a level-0 block at a
  time, the filter confirming the run in one call
  (`Scorer::matches_batch`), the confirmed documents scored as a batch --
  replaying `ImpactsDISI` exactly (a check of the run's block after any hit
  that raises the threshold, which may drop the rest of the run), because
  the first version, which only re-checked at block boundaries, counted
  different `totalHits` than Lucene. `function/tests.rs::batches` checks every
  batch getter against the per-document getters and 165 searches with the
  batches on and off.

  After (3 interleaved reps, Java and the pre-change binary interleaved,
  noise floor 1.12x):

  | Case | Before | After |
  |---|---|---|
  | `fn_score_field` | 0.71x | 1.92x |
  | `fn_boost_composite` | 0.98x | 2.44x |
  | `fn_query_field` | 0.81x | 2.05x |
  | `fn_query_composite` | 0.94x | 2.94x |
  | `fn_range` | 0.70x | 1.83x |
  | `fn_range_filter` | 0.88x | 1.24x |
  | `fn_termfreq` | 0.77x | 1.41x |
  | `fn_tf_idf` | 0.67x | 2.07x |
  | `fn_match` | 0.92x | 1.61x |
  | `fn_joindf` | 1.12x | 1.31x |

  (`fn_joindf`, added by the review: it keeps `joindf`'s map precomputed in
  `createWeight` -- Java's per-document `seekExact` needs the top-level
  reader at `getValues`, and the map is faster on a field of ~20 000 keys
  per segment.)
- **T10.6** — Intervals, payload queries, `MoreLikeThis`, `CommonTermsQuery`.
  Intervals **ported** (`lucene-search/src/intervals`, `exec/intervals.rs`):
  every class of `lucene-queries`' `intervals` package -- the `Intervals`
  factories and every `IntervalsSource`, their iterators, `IntervalQuery`
  with its saturation and sigmoid scoring, `IntervalMatches` and
  `IntervalBuilder`'s analyzed text. `GenIntervals` records 120 sources
  (`toString`, `minExtent`, five scoring variants with hits, score bits and
  explanations, the `Matches` of every hit and explained document), all 6 538 lines equal to
  Lucene's. `CommonTermsQuery` **ported** (`lucene-search/src/common_terms.rs`),
  rewritten by the searcher: `GenMoreLikeThis`'s fourteen term sets and four
  boosted booleans of term clauses, 99 lines equal to Lucene's -- which took `BoostQuery.rewrite`'s unit-boost drop,
  boosts folded into term weights' explanations, `BooleanQuery.toString`'s
  parentheses, a deleted document's explanation and `MatchNoDocsQuery`'s.
  `MoreLikeThis` and `MoreLikeThisQuery` **ported** (`lucene-search/src/mlt.rs`):
  344 lines of `mlt.tsv` equal to Lucene's, Java's `HashMap` iteration order
  over the candidate terms reproduced. The span queries and the payload
  queries **ported** (`lucene-search/src/spans`, `exec/spans.rs`): `SpanNode`
  (first, position range, not, containing, within, field masking, the
  multi-term wrapper, payload check and payload score, over terms, nears and
  ors) on lazy `Spans`, with `SpanWeight.matches` -- `GenSpans`'s 95 queries
  with the `Matches` of their hits, 4 794 lines equal to Lucene's but the 18
  where Java's `toString` throws on a `null` payload.
- **T10.7** — Plugin wiring for the OpenSearch shapes above. **Wired**
  (ABI 33; `opensearch-plugin/.../ExtendedEncoder.java`, query-tree nodes
  20-24 decoded by `lucene-ffi/src/jvm_nodes.rs` and
  `jvm_function_score.rs`): `nested` (OpenSearch's
  `OpenSearchToParentBlockJoinQuery`, every `score_mode`, its parent filter
  -- OpenSearch's bit-set-cache producer or a `QueryBitSetProducer` -- a node
  whose bit sets the native side caches across requests per segment and
  filter; `inner_hits` is the fetch phase and does not affect it), the span
  queries (`span_term`/`near`/`or`/`not`/`first`/`containing`/`within`,
  `field_masking_span`, `span_multi` once rewritten), `intervals` (every
  Lucene `IntervalsSource`, read field by field, a multi-term source's
  automaton sent as bytes), `combined_fields`, and OpenSearch's own
  `function_score` -- not Lucene's `FunctionScoreQuery`: OpenSearch combines
  in its own `double` arithmetic, so it runs as the ported
  `FunctionScoreQuery` around a values source computing exactly that
  (`weight`, `field_value_factor` with every modifier, `random_score`,
  numeric and date `gauss`/`exp`/`linear` with every `multi_value_mode`,
  function filters, every `score_mode` and `boost_mode`, `max_boost`, a
  `boost` moved onto the sub-query as OpenSearch's weight does).
  `more_like_this`, `common` and `parent_id` need nothing: OpenSearch
  rewrites them to booleans of term queries before the query phase.
  **Falls back**, each by name (`docs/opensearch-native-queries.md`):
  `has_child`/`has_parent` (the rewrite runs `JoinUtil`'s from side in Java
  over OpenSearch's own `OrdinalMap` and keeps the collected ordinals; the
  ported global-ordinal joins would have to receive the from query and the
  ordinal map instead), `collapse` (a collector with its own collapsed
  result, which the native top-hits path does not return), `function_score`
  with `min_score`, scripts (`script_score`, script functions, the intervals
  `script` filter), geo decays, field data other than plain sorted-numeric
  doc values (keyword for `random_score`), a `span_near` with `span_gap`.
  Proven in process by `NativeSelfTestM10` (`gradle -p opensearch-plugin
  selfTest`): 1,068 queries -- 216 `nested`, 212 span, 212 `intervals`, 216
  `combined_fields`, 212 `function_score`, alone, beside a scored clause and
  as filters, on an NRT block index with OpenSearch's nested layout and
  whole-block deletions over four refreshes and a force merge -- every one
  encoded and compared with Lucene by `NativeSelfTest.compare` (top hits,
  counts and the total-hits threshold, sorted pages, `terminate_after`,
  `min_score`, aggregations): 13,479 of 13,479 compared scores bit-exact,
  0 failures. Seen to fail when a decoder mapped `Max` to `Total`, offset
  `span_first`'s end by one, or skewed the `sum` boost mode by 0.1%.
  Rust unit tests `jvm_nodes::tests`, `jvm_function_score::tests` (lines
  99.5% and 98.8%). The REST matrix gained `e2e/m10_matrix.py` (44 rows on
  the main indices, 11 `nested` rows, 3 on a parent-join index), and `mlt` joined
  the YAML suites `scripts/verify-opensearch.sh --yaml` runs; neither has
  been run (Docker unavailable where this was done).

## Stage-3 status (2026-10-06)

Every M10 benchmark case at or above Lucene's speed, or inside the run's
noise, but two `queries` cases (written up below the table). Ratios are Lucene's
time over ours (`scripts/bench-micro.sh --bench <name>`, interleaved with
the build before; `~` inside the run's noise floor):

| Bench | Cases | Range | Lowest |
|---|---|---|---|
| `join` (T10.2) | 12 | 1.03x-1.79x | `ToChild` 1.03~, `Min` 1.05~, filtered KNN 1.08~ |
| `query_join` (T10.3) | 11 | 1.14x-1.61x | terms `Min` 1.14x |
| `grouping` (T10.4) | 9 | 1.05x-2.98x | blocks 1.05x-1.17x (two 7-rep runs of two builds each; floors 1.14x-1.30x) |
| `function` (T10.5) | 10 | 1.39x-2.82x | `FunctionRangeQuery` as a filter 1.39x |
| `aggs` (terms behind a filter) | 6 | 1.45x-3.11x | dense range on a keyword 1.45x; see below |
| `queries` (T10.6) | 19 | 0.89x-1.68x | a phrase over a disjunction 0.89x, `SpanPositionRangeQuery` 0.90x; see below; eleven more inside the noise (0.93~-1.03~); `MoreLikeThisQuery` 1.68x |

Lucene's side of the two term-filtered `cat` cases (a single-valued
`SORTED_SET`) is bimodal: in some JVM runs (one of six, three of five in
two runs on 2026-10-05) it settles about 2x faster than its usual 5-7 ms,
at 2.5-2.7 ms, against our 3.1-3.8 ms -- 0.72x-0.80x against those runs.
The median ratios above count every run; that faster mode is the gap left.

Block grouping is the one case not clearly above 1.0. Its profile
(callgrind, `MICRO_CASE=grp_blocks`): the collector's per-hit bookkeeping
(append the document and score, one float compare against the bottom
group, `processGroup` per block) about 45%, the term's postings and BM25
scores about 25%, and finding each block's end about 18% -- the
`lastDocPerGroup` matches marked in a bit set per segment, which Java
instead advances its scorer through. A lazily advanced scorer, as Java's,
and reusing the evicted group's buffers measured the same instruction count
(1.110G vs 1.114G) and no wall-clock change (1.41 ms vs 1.44 ms, 7 reps), so
neither was kept: nothing in it is a port inefficiency; Lucene does the same
work per hit.

The interval, span and payload queries (`--bench queries`, 200 000
documents, a pair of words per query, every case's digest equal to
Lucene's) started at 0.73x-0.87x, the payload ones at 0.11x-0.2x. Stage 3,
in order: a term's positions read one at a time off the lazy cursor
(`nextPosition()`), its payloads streamed with them
(`PositionsCursor::read_payloads`; they had walked the skip data per
document), a disjunction's top list reused; then the call structure, which
the profile showed as the whole remaining cost -- every step a virtual call
through `Box<dyn Spans>`/`Box<dyn IntervalIterator>` returning a `Result`,
where Java's JIT inlines the monomorphic sites. A parent now holds a term
leaf inline (`BoxSpans::Term`, `BoxIntervals::Term`) and calls it
statically, with its per-position steps `#[inline]`; the per-document
frequency sums are trait methods, monomorphised per spans or iterator; and
a scorer is generic over its root for the common shapes (term, first,
range, not, near and containment spans; block, ordered, unordered,
filtered and containment intervals). Together: `spanFirst` 0.79x -> 0.96~,
`spanNot` 0.80x -> 0.94~, `ordered` 0.83x -> 0.98~, `phrase` 0.90x ->
0.99~, `SpanWithin` 0.88x -> 1.01~. Two tries made it slower and were
dropped: boxing the `Result`'s error so it comes back in registers (the
96-byte `Error` returns through memory) cost 5-13% on every case, and
making a near spans or a disjunction an enum variant of its own cost
5-18% (the larger dispatch defeats inlining).

Two cases stay below the noise floor. A phrase over a disjunction (0.89x,
`MICRO_CASE=iv_or_phrase`, perf): the disjunction's `nextInterval` 11%, the
term iterators' 10%, the block conjunction's 9%, `.pos` positioning per
document 9% and `nextPosition` 6%, `.doc` advancing 6%, and the
disjunction's queues (`DisiQueue` advance, top list, down heap, the
interval queue's add) 13% between them -- the disjunction is the one level
still behind a virtual call, and holding it statically measured slower.
`SpanPositionRangeQuery` over an ordered near (0.90x,
`MICRO_CASE=sp_pos_range`): the near's `stretchToOrder` 14%, the term
spans' `nextStartPosition` 13%, `.pos` positioning 10% and `nextPosition`
8%, `.doc` advancing 8%, the near's `matches` 7%, the conjunction's
`doNext` 5%: the same work per document as Lucene's `NearSpansOrdered`, the
near behind one virtual call. Neither has a single cost left to remove.

## Acceptance criteria

- [ ] Every query matches Lucene's hits and scores bit for bit on generated
      fixtures, including empty and single-child blocks and deleted parents.
- [x] A Rust-written block index passes Lucene's `CheckJoinIndex`. (T10.1:
      `fixtures/src/VerifyJoin.java` runs Lucene's `CheckJoinIndex` over
      Rust-written block indices, sorted and unsorted, before and after a
      force merge.)
- [ ] OpenSearch's `nested` and `function_score` YAML suites fail
      identically with and without native execution. Open: not run yet
      (`scripts/verify-opensearch.sh --yaml` needs Docker). OpenSearch's core
      REST spec has no suite of its own for either: `nested` is exercised by
      `search.inner_hits` (in `YAML_SUITES`), `function_score` only by the
      `lang-painless` module's `script_score` suites, which fall back.
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
