# M5.6 — The whole read path native, faster than Lucene

**Goal.** Every part of a search request that a shard answers — the query
phase for every query shape OpenSearch builds, sorting, aggregations, fetch and
get, scroll and the remaining request features — runs in Rust through the
plugin, and each is measured at least as fast as Lucene on the same bytes.

M2 moved the query phase for term and boolean shapes and routed the rest to
Lucene; its benchmark found the shapes M1 never measured (boosts, `must_not`,
mixed booleans) 4–8× *slower*, because the port had fast paths for three
shapes and a materializing path for everything else. M5 moved indexing. This
milestone finishes the read side.

**Status.** In progress. R1 and R2 delivered; R3 mostly, R4 (numeric, score, `_doc` and keyword keys) and R5 (the metrics and keyword `terms`) partly delivered (below). R6 and R7 open.

## Tasks

| ID | Task | Status |
|---|---|---|
| R1 | Query execution engine: Lucene's scorer tree and bulk scorers, every boolean shape at least as fast as Lucene | ✅ delivered |
| R2 | General query wire format and Java encoder for every Lucene query OpenSearch builds | ✅ delivered for the shapes R1 runs (term, boolean, constant score, boost, dismax, match-all, match-none); leaf queries arrive with R3 |
| R3 | Leaf queries as streaming scorers: phrase, the multi-term family, points and doc-values ranges, exists, terms-in-set, dismax, synonym | mostly delivered: phrase, prefix/wildcard/terms, points ranges, dismax, and a native query cache (R3b); open: `exists` (with R4's doc-values wiring), `regexp` over the wire, fuzzy speed (q25) |
| R4 | Sort and `search_after` natively (`TopFieldCollector`) | numeric, score, `_doc` and keyword keys and `track_scores` delivered (below); open: `avg`/`sum`/`median` modes, nested sorts, index-sorted shards |
| R5 | Aggregations natively: terms, histogram, date_histogram, range, the metrics, cardinality, filter/filters | metrics (`min`, `max`, `sum`, `avg`, `value_count`, `stats`) and keyword `terms` delivered (below); open: the bucket aggregations, `cardinality`, sub-aggregations |
| R6 | Fetch (`_source`, stored fields, `docvalue_fields`) and get natively | open |
| R7 | scroll, `post_filter`, `min_score`, `terminate_after`, timeouts; the full read benchmark (in process and REST) with every native shape at least 1.0× Lucene | `post_filter`, `timeout`, scroll, `terminate_after` and `min_score` (by score) delivered (below); open: `min_score` behind a sort, the full read benchmark |

## R1 — the scorer tree (delivered)

`lucene-search/src/exec/` is Lucene 10.5.0's scorer composition, class for
class: `ConjunctionScorer`, `BlockMaxConjunctionScorer`, `DisjunctionSumScorer`,
`DisjunctionMaxScorer`, `WANDScorer`, `ReqExclScorer`, `ReqOptSumScorer`,
`ConstantScoreScorer`, `TermScorer`, with two-phase iteration, built the way
`BooleanWeight.scorerSupplier` and `BooleanScorerSupplier` build them. Bulk
scorers are chosen as `BooleanScorerSupplier.booleanScorer()` chooses them,
plus two of our own where Lucene has none or answers conservatively: a batch
`ReqOptSumScorer` and a precise `docIDRunEnd`. `docs/parity.md` has the rows.

Acceptance:

- [x] Every mixed shape agrees with real Lucene, pruned and exact, doc ids and
      score bits: `tests/mixed_boolean_fixtures.rs` (68 queries since R2, two segments,
      deletions), and every bulk scorer and `ReqOptBulk` path is reached by
      one of them (`exec/tests.rs::fixture`).
- [x] 1,500 random trees, with two-phase leaves, agree with brute force in all
      three score modes (`exec/tests.rs`).
- [x] Both suites seen to fail on seeded defects.
- [x] Every mixed shape the M2 REST benchmark measured slower (boosts,
      `must_not`, `must` + `should`, `constant_score`, nesting) at least as
      fast as Lucene in process, on the merged index — 1.3× to 6.7×.
- [x] Every boolean shape in the query file at least 1.0× on both indexes
      (after R2's run below: 1.04–13.7×; q11 0.99× merged with overlapping
      runs). q25 (fuzzy, 0.76× segmented) is a leaf query, R3's.

## R2 — the query tree on the wire (delivered)

The plugin sent two shapes before: a lone `TermQuery` and a flat clause list
of terms. R2 sends the rewritten Lucene query as a tree (JVM ABI 7, the
`QUERY_TREE` blob): term, boolean with `minimumNumberShouldMatch`, constant
score, boost, dismax, match-all and match-none, nested to Lucene's depth.
`QueryEncoder` writes it, `jvm_reader::decode_node` reads it, and a query
with anything else in it falls back to Lucene under `query_<Class>` (the
root) or `clause_<Class>` (a nested clause). Count-only requests run the
scorer tree without scores; totals follow OpenSearch's own
`shortcutTotalHitCount` (a match-all counts `numDocs`, an exact term its
`docFreq`s when nothing is deleted), so every total agrees with a stock node.

The REST benchmark's shapes, measured in process first, found three places
where the tree lost to Lucene and one where it lost badly:

- **A lone term took an old unpruned path** (0.08×): the pre-R1 shortcut
  for one-clause booleans scored every posting. It now takes the scorer
  tree's term bulk scorer: 1.6–3.5×.
- **`MaxScoreBulkScorer` only over terms.** Lucene runs it over any
  `SHOULD` scorers in `TOP_SCORES`; a nested boolean fell to the tree.
  `MaxScore` is now generic over its clauses (`MaxScoreLeg`), with
  `LegConjunctionScorer` filling batches in a tight loop: nested bool
  0.52× → 1.04×.
- **Dismax** (0.85×; `multi_match`, a tie-breaker of 0, 0.36×): Lucene
  has no skipping for a tie-breaker other than 0. `DisMaxBulk` scores terms
  a block at a time into a window and drops windows whose dismax bound
  cannot compete, and with a tie-breaker of 0 lets each clause skip its own
  blocks as `DisjunctionMaxBulkScorer` does: 4.1× (REST corpus), 13×
  (q51), `multi_match` 1.3–3.5×.
- **`MUST` + `SHOULD` with a minimum** (0.77×): Lucene runs
  `ConjunctionScorer(req, opt)` a document at a time; the same scorers as a
  block-max conjunction score identically and skip: 1.3×.

Acceptance:

- [x] The tree searches exactly like the clause list it replaces (same hits
      and score bits), malformed trees are invalid arguments, the depth
      (32) and node (1,024) limits agree between the encoder and the
      decoder at the boundary, and a match-all counts live documents only
      (`jvm_reader` tests).
- [x] The real node agrees with Lucene on every matrix row, on three
      indices, one of them with no deletions and more than 10,000 documents
      so every shortcut total applies (`scripts/verify-opensearch.sh`).
      What it cannot catch: with the shortcut disabled the matrix still
      passed (1,360 checks), because a REST response caps `hits.total` at
      `track_total_hits` whichever way the shard counted. The shortcut
      changes the work a shard does (it stops collecting after the top
      hits), not a response a client can see.
- [x] Every shape in the REST benchmark at least 1.0× in process on a
      corpus built like the REST index (1M documents, 24-word vocabulary,
      two fields), merged and 8-segment (`benchmarks/rest-shapes.tsv`,
      `benchmarks/corpus/src/GenRestCorpus.java`): r1 msm 2 1.10×, r2 nested
      1.04×, r3 `OR` 1.39×, r4 dismax 4.1×, r5 `must` + msm 1.3×, r6
      `must` + `should` 1.4–3.0×, r7 one term 1.6×, r8 `multi_match`
      (dismax, tie-breaker 0) 1.30× merged and 3.45× segmented.
- [ ] Over REST (`docs/opensearch-native-queries.md`), 20 of 23 shapes are
      1.04–1.51×; msm 2 (0.97×), `must` + msm (0.99×) and `multi_match`
      (1.00×) are inside the run-to-run spread but not clearly above it.
      Carried to R7.

## R3 — leaf queries (in progress)

**Phrase (delivered).** `exec::phrase::PhraseScorer` is Lucene's two-phase
`PhraseScorer`, so a phrase composes anywhere in a tree instead of being
resolved up front; the JVM sends it as a tree node (ABI 8). Ten phrase shapes
agree with real Lucene, pruned and exact (`tests/mixed_boolean_fixtures.rs`,
91 queries: sloppy, repeated terms, pulsed singleton terms), and seen to fail
on a seeded defect. They found one bug that predated R3: a phrase's idf was
summed in `f32`, where `BM25Similarity.idfExplain` sums in `double` and casts
once, so a phrase of three or more terms could score an ulp off (the
self-test's random phrases showed it as 10 of 9,135 scores not bit-exact). In process (q56–q61,
1M documents): 0.95–1.30×, with one exception that is not the phrase's.

**The query cache.** q59 (`+t1 -"t0 t2"`) measured 0.20×. Lucene makes
exactly as many `matches()` calls (70,435 per query, counted with a wrapping
query), but `IndexSearcher`'s default `LRUQueryCache` caches the repeated
`MUST_NOT` phrase as a bitset after a couple of uses: Lucene runs q59 at 472
qps with the cache and 88 without, against our 94. OpenSearch caches the same
way (`IndicesQueryCache`, on by default), so matching Lucene on repeated
filters needs a native query cache: R3b, on `query_cache.rs`'s existing
`LRUQueryCache` port. Every Java number in this document was measured with
the cache on, as Lucene ships.

**Sloppy phrases.** `SloppyPhraseMatcher.maxFreq` (the sum of the terms'
frequencies) is ported, and the sloppy matcher reuses its buffers across
documents: a fresh set per candidate was 30% of a sloppy phrase's time in
`malloc`/`free`. In process: `(ps 2 t1 t0)` 1.03–1.14×, `(ps 2 t0 t1)`
1.18–1.24×, q61 1.05–1.11×; `(ps 1 t2 t3)` 0.84–0.91× is open.

**The multi-term family (delivered).** `prefix`, `wildcard`, `terms` and
`regexp` run inside the tree as Lucene rewrites them (a constant-scored
disjunction up to 16 terms, the blended 16-iterators-and-a-bitset past
that), through a union scorer that holds the term cursors directly; the
plugin sends prefix, wildcard and term sets (ABI 9). In process: filters
over the family 7–39× (q62, q65, q67), a `SHOULD` prefix 1.1–1.4× (q63), a
lone term set 0.92× merged and 3.4× segmented (q66). Open: a term set as a
scored `MUST` beside a scoring `SHOULD` (q64), 0.47–0.58× -- Lucene makes the
same 58,075 iterator moves, so the gap is the codec's docs-only `advance`.

**Points ranges (delivered).** `range` on `long`, `date` and `double`
fields runs natively (ABI 10): a constant-scored range anywhere in a tree,
Lucene's every-document shortcut, a bitset or sorted list by density, the
reader-level rewrite to match-all or match-none, and conjunctions that test
a bitset filter by membership. In process: filter 1.0–1.2× (q68),
exclusion 1.2–1.3× (q69), every-value filter 1.4–1.5× (q70), a small lone
range 0.97–1.0× (q71).

**R3b, the native query cache (delivered).** `exec::cache` caches a
non-scoring clause per segment under Lucene's own policy, built from the
clause's scorer without live docs and kept with the segment's core across
reopens. q59 went from 0.20× to 1.20× merged and 2.27× segmented; the
other `FILTER`/`MUST_NOT` shapes (q41, q47–q49, q52, q54, q55) stay at
1.2–4.3×.

## R4 — sorted search (numeric, score, `_doc` and keyword keys delivered)

`lucene-search/src/top_field.rs` is Lucene's `TopFieldCollector`: the hit
queue, `SimpleFieldCollector`/`PagingFieldCollector`, the relevance, document
and numeric comparators over any number of keys, and `NumericComparator`'s
competitive iterator -- once the queue is full and the total-hits threshold
passed, the sort field's points are intersected with the range that can still
compete, and the scorer is advanced past everything else. The plugin sends a
sort blob beside the query (ABI 11, `SortEncoder`) for the sorts OpenSearch
builds for `long`, `integer`, `short`, `byte`, `double`, `float` and `date`
fields (`SortedNumericSortField`, `min`/`max` mode, any `missing`), for
`_score` and `_doc`, and for `search_after` over them.

It agrees with Lucene on 1,512 fixture runs (hits and values exact, totals
exact wherever Lucene's are) and on 5,804 random sorted pages in the plugin's
self test.

Past the threshold the two report different lower bounds for the total:
Lucene's match-all and filter conjunctions collect whole 4,096-document windows
before they consult the competitive iterator, and this collector consults it
per document. REST responses cap the total at `track_total_hits` either way.

In process (`benchmarks/queries.tsv` q72-q81, the `sorted` kind in both
runners: `TopFieldCollectorManager(sort, 50, null, 1000)` against
`search_sorted`, 1M documents, five interleaved runs, median qps, Lucene's
default query cache on):

| query | shape | merged | 15 segments |
|---|---|---|---|
| q72 | match-all by `num` asc | 27.7x | 16.0x |
| q73 | match-all by `num` desc | 20.1x | 13.7x |
| q74 | `t0` by `num` | 1.67x | 1.55x |
| q75 | rare term by `num` desc | 1.42x | 1.32x |
| q76 | `+t1 -t2` by `num` | 1.42x | 1.98x |
| q77 | `t1 OR t2` by `num` | 1.52x | 13.45x |
| q78 | `t0` by `_doc` | 2.45x | 3.15x |
| q79 | `t1 OR t2` by `_score`, `num` | 1.71x | 1.55x |
| q80 | `t1` by `num`, `_score` | 1.32x | 1.17x |
| q81 | match-all by `_doc` | 2.40x | 2.50x |

The first port measured 0.34-0.91x on six of these. What closed it, in
order: a dense competitive set kept as a bit set, not a sorted list
(`DocIdSetBuilder`'s form); the points walk's buffers kept across the ~250
competitive updates a query makes; a scorer outside `TOP_SCORES` reading no
impacts (Lucene's plain `FREQS` postings); a doc-led sort counting the rest
of a segment in runs once its queue is full; a sort that reads the score only
on ties iterating without scores and scoring just those documents; the
competitive set walked by membership over a cached or match-all scorer; and
whole inside leaves handed to the visitor at once. The match-all rows gain
most from counting per document where Lucene collects 4,096-document windows
first.

Over REST on a real 3.8.0 node (`scripts/verify-opensearch.sh --docs 100000
--bench-rounds 40`; mean latency, Lucene over native, 2 segments): every sorted
row the matrix runs native is faster -- `_doc` 1.19x, `long` desc 1.03x,
`double` 1.42x, `float` 1.20x, `date` 1.10x, `integer`,`long` 1.07x,
multi-valued `min` 1.04x, sparse with each `missing` 1.01-1.04x, `_score` then
a field 1.02x, a field then `_score` 2.50x, from 20 size 15 1.30x,
`track_total_hits: true` 1.23x, `search_after` 1.03x and 1.30x, size 0 1.27x.
The same run: 2,197 checks against a stock node, 0 failures; the self test's
5,804 random sorted pages all agree with `TopFieldCollectorManager`.

OpenSearch answers a top-level `range`, and a `match_all` sorted by one numeric
field without `missing`, approximately (`ApproximateScoreQuery` resolved to its
`ApproximatePointRangeQuery`/`ApproximateMatchAllQuery`, which break ties in
BKD order); those stay on OpenSearch's path (`approximate`), since returning
Lucene's exact answer would differ from a stock node's.

### Keyword keys

`keyword` fields sort natively too (`SortedSetSortField`, `min`/`max` mode,
`missing` `_first`/`_last`, `search_after` by term; ABI 12 carries the terms
both ways). It is `TermOrdValComparator`: ordinals compared within a segment,
the bottom and the search-after term looked up in each new segment through a
random-access port of the doc-values terms dictionary
(`lucene-codecs/src/terms_dict.rs::TermsDict`), and
`PostingsBasedCompetitiveState` -- the postings of up to 1,024 competitive
terms as a disjunction -- as the competitive iterator. Verified against 936
Lucene runs and 4,072 `lookupTerm`/`lookupOrd` probes
(`tests/keyword_sort_fixtures.rs`, `fixtures/src/GenKeywordSort.java`) and by
the self test's random keyword sorts.

In process (q82-q89, same method as above):

| query | shape | merged | 15 segments |
|---|---|---|---|
| q82 | match-all by `keyword` asc | 203x | 131x |
| q83 | match-all by `keyword` desc | 5.10x | 4.02x |
| q84 | `t0` by `keyword` | 2.09x | 2.25x |
| q85 | rare term by `keyword` desc, missing first | 1.56x | 1.54x |
| q86 | `t1 OR t2` by `keyword` | 2.08x | 78.6x |
| q87 | match-all by `cat` (no postings) | 1.31x | 1.30x |
| q88 | `t1` by `cat` max desc, `num` | 1.54x | 1.66x |
| q89 | `t1` by `keyword`, `_score` | 1.83x | 1.43x |

The faithful port measured 0.44-0.84x on five of these. What closed it: the
term of a copied hit read once per segment for the slots still queued, not per
copy (Java's `copy` calls `lookupOrd` every time), while no earlier segment's
slot is in the queue to compare it with by term; the dictionary's block buffer
reused; a hit the leading key alone rules out dropped before `collect` (a dense
column's ordinal or value compared with the bottom directly, and on a tie the
score read and compared there); a run of matches (match-all) walked without
moving the scorer, asked for only while runs keep turning up; a lone term
scored by the cursor that iterates it rather than a second tree; and the
sparser of the scorer and the competitive iterator leading the leapfrog. The
last three also lifted the numeric rows: after them q72-q81 measure 1.12-25.1x
merged and 1.14-14.7x segmented.

A copy of the REST node's own shard (100K documents, two segments, OpenSearch's
mappings) found a shape neither corpus has: a sparse `long` sorted with
`missing: _first` behind a tie-break, 0.46x in process. `NumericComparator`
skips nothing while the missing value can still compete, so both engines
compared every match. Here the documents that can compete are the ones without
a value plus the ones whose value is in range, and those are now the
competitive iterator (a bit set of the documents without a value, built once
per segment straight from the `IndexedDISI` words, joined with the points in
range); with single-valued `SORTED_NUMERIC` columns read as the numeric column
they are (`DocValues.singleton`), the shape measures 1.15x.

The same shard at REST's own parameters (`size` 10, `track_total_hits`
10,000) showed three more: a keyword sort, a date sort under a filter and a
`long` sort, each over a term matching fewer documents than the threshold, so
nothing can be skipped and every match is compared, at 0.50-0.78x. The columns
are sparse there (the shard's update tombstones have no fields), and a sparse
per-document read went through the general decode path. Four changes: a
forward read inside the current `IndexedDISI` block without the header logic
(`DisiCursor::advance_exact_in_block`), a sparse column's values read by
ordinal with the dense column's packed reader, the fast reject extended to
sparse columns, to a total still below the threshold and to search-after
pages, and a run of matches rejected in one loop. Then a per-segment column
cache: a sort column that a per-document read has to decode is decoded once,
on its second use in the segment, and read from memory afterwards (within 32 MB
per segment, next to the query cache). The rows now measure 1.02-1.65x in process
(keyword 1.49x, date under a filter 1.02x, `long` 1.65x), and the
match-all rows 5.4-41x.

Over REST after this round (the same node and matrix, 100K documents, 400
requests per engine per row, interleaved, median wall latency, Lucene over
native): `_doc` 1.13x, `long` desc 1.10x, `double` 1.23x, `float` 1.14x,
`date` 1.04x, multi-valued `min` 1.21x, sparse with `missing` `_first`/`_last`/a
value 1.25x/1.08x/1.14x, a field then `_score` 2.41x, from 20 size 15 1.17x,
`track_total_hits: true` 1.07x, size 0 1.13x, `search_after` 1.03x and 1.34x,
keyword 1.05x, keyword then a field 1.11x, keyword `max` missing first then
`_score` 1.35x, keyword `search_after` 1.16x; at parity: `integer` then `long`
over match-all 0.98x and `_score` then a field 0.99x (server-side `took`
0.89/1.00 ms and 1.71/2.01 ms, where the same searches measure 2-3x faster
than Lucene in process on a copy of the shard -- the difference is not in the
search and is not yet explained). The matrix's 2,278 checks against a stock
node pass, and the self test's 6,576 random sorted pages (numeric, score,
`_doc` and keyword keys) all agree with `TopFieldCollectorManager`.

`track_scores` behind another key runs natively as OpenSearch runs it: the
collector beside a `MaxScoreCollector` in a `MultiCollector`, which exposes no
competitive iterator, so every match is scored for the max score and handed to
the collector (ABI 13: a sort-blob options byte, the max score back in a fourth
count slot). 224 Lucene runs through the same `MultiCollector` agree exactly --
hits, totals, max-score bits -- and so do the self test's tracked pages.

Falls back, deliberately for now:

* `avg`/`sum`/`median` modes. OpenSearch sorts them with its own
  `LongValuesComparatorSource` (and the `Double`/`Float` ones): a Lucene
  `NumericComparator` over `MultiValueMode.select` of the values with the
  missing value filled in, and it keeps the comparator's points skipping. The
  points hold the individual values, not their sum, so with negative values a
  `sum` sort can skip a document that competes; and the docs-with-value
  fallback it would hand the collector is an iterator that cannot iterate. A
  native port would have to reproduce both to match a stock node, so these
  stay on Lucene, where they run exactly as they do today.
* nested sorts: they need the block-join parent/child sets, not ported yet.
* index-sorted shards: Lucene stops a segment early when the index sort is a
  prefix of the search sort (`canEarlyTerminate`), which changes the totals;
  not ported yet.

## R5 — aggregations (metrics and keyword `terms` delivered)

`min`, `max`, `sum`, `avg`, `value_count` and `stats` at the top level of a
request run natively when every aggregation of the request is one of them, on a
mapped `long`/`integer`/`short`/`byte`/`double`/`float` field or a millisecond
`date`, with no `missing`, script or sub-aggregation (ABI 14,
`ffi_jvm_reader_aggregate`), and `terms` on a keyword field (below). The
rest -- the histograms, `range`, `cardinality`, `filter(s)`, `global`, `terms`
outside its supported shape, anything nested -- stays on OpenSearch's
aggregators, as does a request mixing it with native ones.

How it hooks in. `QueryPhase` asks the `QueryPhaseSearcher` for its
`AggregationProcessor`; the plugin keeps OpenSearch's, whose `preProcess`
builds the aggregators and registers their collector manager, which `QueryPhase`
turns into the one collector context `searchWith` receives. When
`NativeAggregations.plan` accepts every factory (read through reflection: the
factory classes are package-private), the native path runs the aggregation
pass beside the top-hits search and stores OpenSearch's own shard results --
`InternalMin` and the rest, with the factories' names, formats and metadata --
before `postProcess`, which returns early when the result already has
aggregations. The aggregators OpenSearch built are never fed and are released
with the context, as they are after any search. A failure anywhere re-runs the
whole request on Lucene.

What is computed (`lucene-search/src/aggs.rs`): one pass over the live matches
keeps, per field, the value count, the `CompensatedSum` value and delta, the
minimum and maximum over every value and over each document's first and last
(`MultiValueMode.MIN`/`MAX`), with Java's `Math.min`/`Math.max`. A `min`/`max`
whose query is a bare `MatchAllDocsQuery` -- a request without a query; 3.8
turns an explicit `match_all` into an `ApproximateScoreQuery`, which does not
qualify -- on a field with points reads each segment's bound off the points
as `MinAggregator.findLeafMinValue`/`MaxAggregator.findLeafMaxValue` do,
including the minimum's give-up after 1,024 deleted points; this is not only
faster but a different answer over a `double` field holding a `NaN` document
(`NaN` sorts last among the points and wins `Math.min`).

Concurrent segment search. OpenSearch 3.8's `auto` mode turns concurrent
search on for aggregation requests (two slices by default on this node): each
slice gets its own aggregators, and `NonGlobalAggCollectorManager` reduces their
shard results with `InternalAggregations.reduce` -- so a slice's sum reaches the
reduce without its compensation delta, and a multi-segment shard's `sum` rounds
differently from a one-pass answer (the REST matrix caught it: one float `sum`
in the last bit). The native pass therefore takes OpenSearch's slices
(`IndexSearcher.getSlices()`, segments in each slice's order), keeps a state per
slice, and the plugin builds each slice's results and hands them to the same
`InternalAggregations.reduce(..., partialOnShard())`. Intra-segment slices
(part of a segment) fall back. A sorted search under concurrent search is sliced
the same way (`top_field::search_sorted_sliced`): a collector per slice, run on
rayon's pool as Lucene runs them on its executor, hits merged as
`TopDocs.merge` merges them.

Verified: `GenMetricAggs` (4 segments, 7 queries, 7 fields including `NaN`,
per-slice states over the non-contiguous slices `[[0,2],[1,3]]`, a document
with 300 values;
signed zeros, infinities and values that round when widened) bit for bit,
including the points answers and a segment where the points give up; seen to
fail with the compensation dropped, a document's last value in place of its
first, the points ignored, the give-up removed, and deletions ignored by the
minimum's walk (the maximum's cell pruning is a speed property only: without
it the last live point is still the maximum, so no result can show it). The
self test compares 1,862 query x field states through JNI; the REST matrix runs
its aggregation rows natively against a stock node (the metrics, and `terms`
alone, beside metrics and hits, under a sort, multi-valued, with `shard_size`
and `min_doc_count`) and the ones that must fall back (`missing`, `global`,
`terms` by `_key`, with `min_doc_count` 0, with `include`, on a numeric field,
and a sort by `@timestamp` ascending,
whose segments OpenSearch visits last first -- the time-series optimisation,
which the native pass does not port, so such searches stay on OpenSearch's); and `search_sorted_sliced` agrees with Lucene over every untracked run of
both sort fixtures (seen to fail without the doc tie-break, with keyword
missing values flipped, with `reverse` ignored, and with a slice's segments
searched out of doc-base order).

### `terms` on keyword fields

`terms` runs natively on a keyword field with the default order (`_count`
desc, `_key` asc), `min_doc_count` of 1 or more, `shard_min_doc_count` 0 and no
`include`/`exclude` or `_doc_count` field (`lucene-search/src/terms_agg.rs`).
It is `GlobalOrdinalsStringTermsAggregator`: a keyword field's global
ordinals (`GlobalOrds`, Lucene's `OrdinalMap`: the segments' dictionaries
merged once per reader and cached on the `DirectoryReader`), every live match
counted once per distinct term into an array by global ordinal, the top
`shard_size` kept by count and then ordinal (term), the rest summed into
`otherDocCount`, only the kept terms' bytes read. A segment every document of
which matches is counted from its postings when the field has at most 30,000
terms (`tryCollectFromTermFrequencies`). The plugin reads OpenSearch's own
effective thresholds (the `shard_size` heuristic, `ensureValidity`) off the
factory and builds `StringTerms` per slice, reduced with the metrics.
Verified against `GenTermsAggs` (4 segments, 6 keyword fields including raw
`0x00`/`0xff` bytes, a `SORTED` field and one with postings, 7 queries x 4
shard sizes, whole shard and slices: 504 results), seen to fail with the
tie-break reversed, deletions ignored, `otherDocCount` dropped, the buckets
unsorted, segment ordinals used as global ones and the postings count off by
one; the self test compares 18,018 results through JNI.

### Speed

The straight port read the matches through the scorer and each value through
the general doc-values path, one field at a time per document, and lost to
OpenSearch (0.54x on `sum`/`avg`/`value_count` over a multi-valued field,
0.71-0.81x on a float `sum` under a sort). What changed, in order of effect:

* every aggregation of a request runs in one native pass (one JNI call): each
  segment's matches are collected once, with the bulk scorer (a disjunction a
  window at a time), and every metric and `terms` column read over them;
* a column is streamed (`NumericReader::for_each_value`, the new
  `SortedNumericReader::for_each_doc`: one address per document, values
  decoded a chunk at a time) for a match-all, and for any query whose matches
  cover more than a quarter of the segment, tested against a bit set of the
  matches; otherwise each match seeks;
* the metric fold is compiled for the parts the request reads (a `min` alone
  skips the compensated sum), and a field asked for several ways is read once;
* `terms` counts into cached global ordinals (above);
* the slices run concurrently, as OpenSearch's do -- the first on the calling
  thread while rayon's pool takes the rest, so the handoff overlaps work -- and
  a sorted search beside aggregations is sliced the same way; a points-only
  request stays on one thread;
* a segment keeps its parsed points metadata instead of re-reading `.kdm` per
  search (35 -> 10 us per request on a shard of OpenSearch's many points
  fields).

Over REST (median wall latency, Lucene over native, 600 requests per engine per
row interleaved and the engine order alternated; 100,000 documents in 8
segments, and 60,000 with deletions):

| row | clean | with deletions |
|---|---|---|
| `sum` + `avg` + `value_count`, multi-valued, match-all | 2.03x | 1.80x |
| `min` + `max`, a term query, size 0 | 1.22x | 1.32x |
| `stats` with the hits | 1.25x | 1.26x |
| explicit `match_all` `min` + `max` | 1.28x | 1.24x |
| float `sum` under a sort by `n` | 1.17x | 1.02x |
| `terms` (`tag`), with the hits | 1.33x | 1.15x |
| `terms` + `avg` + hits | 1.19x | 1.14x |
| `terms` under a sort | 1.21x | 1.03x |
| `terms` on a multi-valued field | 1.11x | 1.15x |
| `terms` with `shard_size`, a disjunction | 0.95x | 1.13x |

(The `shard_size` row's shard search takes about 1.8 ms on both engines on the
clean index -- `took` 1.78 against 1.82 -- and the rest is the same ~0.1 ms of
native per-request cost as below; it measured 0.98x in the run before.)

| `terms`/`min`/`max` whose shard search takes < 0.3 ms (match-all `terms`, no-query points, nothing matching, `min_doc_count` 3) | 0.94-0.99x | 0.97-1.13x |

The last rows are requests of about 2 ms end to end whose shard search takes a
fraction of a millisecond on both engines (`took` rounds to 0 or 1, native's
average never above Lucene's); what separates them is under 0.1 ms, inside the
run-to-run spread (the same row measured 0.91-1.03x across runs).

## Benchmark

`benchmarks/queries.tsv` q40–q55 are the mixed shapes (the `sexpr` kind, both
runners): the same S-expressions the fixture uses, run by
`search_boolean_query_multi_segment_maxscore_counting` (what the plugin calls)
and by `IndexSearcher.search(query, 50)`, both with Lucene's default
1,000-hit threshold. 1M-document corpus (`scripts/bench-corpus.sh --docs
1000000`), merged and 15-segment. Three interleaved Rust/Java runs per query
(1 s warm-up, 1.5 s measured), median qps, pinned to two cores of a shared
4-vCPU container; the ratio is Rust over Java, so above 1.0 is faster. Recall
matched on every query.

| query | shape | merged | 15 segments |
|---|---|---|---|
| q40 | `+t0 ?t1` | 2.45× | 1.23× |
| q41 | `+t0 -t1` | 2.49× | 2.45× |
| q42 | `?t1 ?t2 -t0` | 2.16× | 2.33× |
| q43 | 2 of `t0 t1 t2 t3` | 1.02× | 0.96× |
| q44 | `t0^2` | 1.44× | 1.58× |
| q45 | `?t0^2 ?t1` | 1.46× | 1.32× |
| q46 | `constant_score(t1)` | 1.33× | 1.60× |
| q47 | `+(t0 t2) +t1 -t3` | 1.30× | 1.17× |
| q48 | `#t0 ?t1 ?t2` | 1.88× | 1.91× |
| q49 | `#t0 ?t1 ?t2`, msm 1 | 1.28× | 1.29× |
| q50 | `+tz ?t0 ?t1` | 6.70× | 6.16× |
| q51 | dismax 0.3 of three | 1.03× | 1.15× |
| q52 | `+t0 +t1 ?t2 -t3` | 2.31× | 1.82× |
| q53 | `?constant_score(t0) ?t1` | 2.42× | 2.79× |
| q54 | `#t1 -t0` | 3.53× | 3.58× |
| q55 | `+t2s -t0` | 3.98× | 3.78× |

Before R1 the same file measured q42 at 0.03×, q54 at 0.05×, q49 at 0.28×,
q47 at 0.37× and q52 at 0.47× (the materializing path).

The whole file (q01–q55) on both indexes after R1: every query from the M1
mix at 1.0× or above on the 15-segment index except q25 (fuzzy, 0.76×); on
the merged index q07/q08/q11/q14 sit at 0.90–0.99×, as they did before R1.
Both runs are in the R1 commit message.

Two things beat Lucene by construction rather than by constant factor, and
are where most of the `must_not` and `must` + `should` gains come from:

- **`docIDRunEnd` reports the real run.** Lucene's default is `doc + 1`
  outside a fully dense block, so `ReqExclBulkScorer` steps through a nearly
  dense excluded term one document at a time; the cursor here reports the
  run of consecutive documents in its decoded block and the scorer jumps it.
- **`ReqOptSumScorer` a batch at a time.** Lucene has no bulk scorer for
  `MUST` + `SHOULD`; this one drops batch documents on block maxima before
  touching the optional clauses, leads with the optional clauses once they
  are required and cheaper, and turns a filter-only required side into a
  filtered `MaxScoreBulkScorer` once a threshold exists.

## R7 — the request features around the query (in progress)

`RustQueryPhaseSearcher` answers these the way `QueryPhase` and its collector
contexts do, in the plugin's Java; the native searches underneath are R1-R5's.

- **`post_filter`**: `QueryPhase` wraps only the top-docs collector in a
  `FilteredCollector`, so the hits are the query's matches the filter also
  matches, each at the query's score, and the aggregations see every match
  of the query. Natively: the hits search `+query #filter` (a `FILTER`
  clause does not score), the aggregations the query's own blob. The filter
  is a filter collector, so no total is answered from index statistics
  (`hasFilterCollector ? -1 : shortcutTotalHitCount`).
- **`timeout`**: `ContextIndexSearcher` checks the deadline before each
  segment. The native search checks it before the call (already past: Lucene
  answers, with nothing, `timed_out`) and after it: a native search that ran
  over is flagged `timed_out` as `QueryPhase` flags one, or fails without
  `allow_partial_search_results`. Its hits are complete, where Lucene would
  have stopped at a segment boundary -- a partial answer allows either.
- **scroll**: `ScrollingTopDocsCollectorContext`: each page is `size` hits,
  the first counted exactly and its total and max score kept for the later
  pages, which search after the last emitted hit (remembered here on one
  shard, by the fetch phase on more). A sorted scroll pages with its sort; a
  scroll by score as the `_score` sort, whose `PagingFieldCollector` skips
  exactly what `PagingTopScoreDocCollector` skips, its hits handed back as
  `ScoreDoc`s.

- **`terminate_after`**: OpenSearch never runs it concurrently, and puts an
  `EarlyTerminatingCollector` (forced) in a `MultiCollector` beside the
  top-docs collector: the first `n` matches in index order are let through,
  the next one -- or the next segment -- ends the search. Natively
  (`lucene-search/src/terminate.rs`): find where the `n`th match falls, then
  run the sorted search over that prefix of the index (the cut segment's live
  documents masked past it) with the whole shard's statistics; a search by
  score runs as the `_score` sort. `terminated_early` is whether a match or a
  segment followed the cut. `size: 0` counts as `TotalHitCountCollector`
  does, a term's `docFreq` or a match-all's `numDocs` per visited segment
  whole. Lucene's own answer depends on how its bulk scorer hands matches
  out: `DenseConjunctionBulkScorer` gives `collectRange` batches, which
  `MultiCollector` hands to the terminating collector first, so the top-docs
  collector loses the whole batch the cut falls in (a match-all stopped at
  3,001 reports 2,571). The native side cuts at the document, so those
  searches -- an unscored sort, a constant-score or filter-only query -- stay
  on Lucene, as do aggregations (an aggregator answering a segment from
  index statistics sees past the cut), `search_after` and scroll.
  `tests/terminate_after_fixtures.rs` checks 864 Lucene runs through Lucene's
  own collectors (`fixtures/src/GenTerminateAfter.java`), including the
  ranged ones it leaves to Lucene; seen to fail with the cut one document
  short (92 runs) and with a later segment not ending the search (6).
- **`terminated_early` without `terminate_after`**: under concurrent search
  OpenSearch counts a `size: 0` request through the same collector, not
  forced, with a limit of 0 (a disabled total, or one answered from index
  statistics) or `track_total_hits`, and reports `terminated_early: true`
  when a slice's collector stopped. The REST matrix now compares the field,
  which showed the native path leaving it out on every `size: 0` aggregation
  row. With a limit of 0 it is set whenever the shard has a segment; past
  `track_total_hits` it depends on which segments the count collector
  answers from `Weight.count` (and so never shows the terminating collector)
  -- the plugin asks Lucene's own weight, query cache included, and the
  native side replays each slice over the others (`count_terminates`).

- **`min_score`**: `MinimumScoreCollector` sits outside every other collector,
  so the hits, the total, the `size: 0` count and the aggregations all see
  only documents scoring at least the minimum. The plugin prefixes every
  query blob with the minimum (`QUERY_MIN_SCORE`); natively the top-docs
  collector is wrapped (`MinScoreCollector`, pruning as Lucene's does), the
  count and the aggregations score the query `COMPLETE`, and the concurrent
  count replay counts only passing documents. By score only: behind a field
  sort (where the sorted collector's competitive iterators and lazy scoring
  would need it inside), a scroll's later pages or `terminate_after`, Lucene
  answers. `tests/min_score_fixtures.rs` checks 140 Lucene runs; the self test
  random queries at two minimums each.

Acceptance, `scripts/verify-opensearch.sh` against a stock node's answers:
`post_filter` with terms and metric aggregations, sorted, paged, `size: 0`,
`track_total_hits: false`, matching nothing; `timeout` with a sort and
aggregations; five scrolls to the end on one shard and three (by score, by
`_doc`, by two keys, by score with `track_scores`, with a `post_filter`),
page for page the same, every page's query phase native on every shard;
eleven `terminate_after` rows native (scored hits, a disjunction, `size: 0`
over a term, a match-all, no query and a `bool` filter, uncounted, a field
sort with `track_scores`, `_score` then a field, under a `post_filter`, not
reached) and three left to Lucene (a field sort, a match-all's hits,
aggregations), `terminated_early` compared.
