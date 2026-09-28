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

**Status.** In progress. R1–R6 delivered; R7 partly delivered (below).

## Tasks

| ID | Task | Status |
|---|---|---|
| R1 | Query execution engine: Lucene's scorer tree and bulk scorers, every boolean shape at least as fast as Lucene | ✅ delivered |
| R2 | General query wire format and Java encoder for every Lucene query OpenSearch builds | ✅ delivered for the shapes R1 runs (term, boolean, constant score, boost, dismax, match-all, match-none); leaf queries arrive with R3 |
| R3 | Leaf queries as streaming scorers: phrase, the multi-term family, points and doc-values ranges, exists, terms-in-set, dismax, synonym | ✅ delivered: phrase, prefix/wildcard/regexp/terms, fuzzy, points ranges, `exists`, dismax, a term's own `docFreq` (`cross_fields`), and a native query cache (R3b), every in-process shape at 1.0× or above but q64 (0.8–0.96×, reason below) |
| R4 | Sort and `search_after` natively (`TopFieldCollector`) | ✅ delivered: numeric, score, `_doc` and keyword keys, `track_scores`, the `avg`/`median` modes, index-sorted shards and nested keys (below); on Lucene by design: the `sum` mode on a top-level field (its points skipping is not exact), a nested sort on a shard large enough for Lucene's comparator to skip |
| R5 | Aggregations natively: terms, histogram, date_histogram, range, the metrics, cardinality, filter/filters | ✅ delivered: the metrics (`min`, `max`, `sum`, `avg`, `value_count`, `stats`), keyword `terms`, `histogram`, `date_histogram`, `range`/`date_range`, `filter`, `filters`, `global`, `cardinality`, and any nesting of them (below); outside: scripts, `missing`, non-default `terms` orders, zones with daylight saving, other aggregation types |
| R6 | Fetch (`_source`, stored fields, `docvalue_fields`) and get natively | ✅ delivered: every stored-fields read of the fetch phase and the get API (`_source`, `_id`, `stored_fields`, highlighting's source) is native, 1.24× Lucene per document (below); `docvalue_fields` stays on Lucene's doc-values readers by design |
| R7 | scroll, `post_filter`, `min_score`, `terminate_after`, timeouts; the full read benchmark (in process and REST) with every native shape at least 1.0× Lucene | `post_filter`, `timeout`, scroll, `terminate_after` and `min_score` (by score and behind a sort) delivered; the query-phase REST benchmark at median 1.58× on eight segments (worst 0.89×) and 1.60× on one merged segment of 100,000 documents, where 17 of 132 shapes are still under 1.0× (below); open: those shapes |

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
      runs). q25 (fuzzy, 0.76× segmented) is a leaf query, R3's -- since
      1.16–1.30× segmented (R3, fuzzy).

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

## R3 — leaf queries (delivered)

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
1.18–1.24×, q61 1.05–1.11×; `(ps 1 t2 t3)`, 0.84–0.91× then, now measures
1.40–1.46× (and `(ps 2 t1 t0)` 1.62–1.78×, q61 1.59×).

**The multi-term family (delivered).** `prefix`, `wildcard`, `terms` and
`regexp` run inside the tree as Lucene rewrites them (a constant-scored
disjunction up to 16 terms, the blended 16-iterators-and-a-bitset past
that), through a union scorer that holds the term cursors directly; the
plugin sends prefix, wildcard and term sets (ABI 9). In process: filters
over the family 7–39× (q62, q65, q67), a `SHOULD` prefix 1.1–1.4× (q63), a
lone term set 1.17–1.43× merged (q66, below). A term set as a scored `MUST`
beside a scoring `SHOULD` (q64) is the one shape left under 1.0×: 0.8–0.96×
(below).

**A term set's disjunction in the query cache.** A JFR profile of Lucene on
q64 showed where its time went: `FixedBitSet.nextSetBit`, not postings.
`rewriteAsBooleanQuery` turns up to 16 terms into `ConstantScoreQuery`
around a `BooleanQuery` of `SHOULD` terms, and `ConstantScoreQuery` creates
that query's weight without scores -- so `IndexSearcher` wraps it in
`CachingWrapperWeight`, and a term set used a few times is read from a
cached bitset *whatever the mode of the clause around it*, a scoring `MUST`
included. `exec::multi_term` now does the same, keyed on the rewritten
`BooleanQuery` (compound, so cached on its fourth use), and a top-level term
set of 2 to 16 terms on a segment with a cache runs through the scorer tree
instead of the streaming union so its uses are counted. A constant score
over a cached bitset is walked word by word by the default bulk scorer
(`Scorer::constant_bits`, Lucene's `DocIdStream`), not stepped through three
virtual calls per document. q64 went from 0.50× to 0.73–0.88× (both engines
now score the same 58,048 documents; what is left is per-document cost
spread over the `ReqOptSumScorer`, the impacts and BM25), q66 from 0.27×
(its streaming union decoded whole windows of three dense terms to keep
1,001 documents) to 1.17–1.43×. A prefix, wildcard or regexp stays on the
streaming union: knowing its term count means walking the term dictionary,
the whole cost of a regexp -- routing them too cost q32 and q34 40%.

What q64 has left is per-document cost: both engines score the same 58,048
candidates through `ReqOptSumScorer`, and the Rust chain (the required
cached set, the optional term's impacts and BM25, the collector) is a
dynamic call per step where the JIT inlines Lucene's. Two attempts that did
not pay are recorded so they are not retried: testing the cached set by
membership instead of leapfrogging (no gain, and it complicates keeping the
required iterator positioned), and a per-term `(freq, norm)` score table in
place of BM25's division (6–12% slower: the division overlaps with the
postings work, the table's check does not).

**Fuzzy (q25).** Its up-to-50 expansions run as a `MaxScoreBulkScorer`
disjunction, and the essential clauses' next document was found by scanning
every one of them per document step, each read from a `TermLeg` six
kilobytes from the next -- Lucene keeps them in `DisiPriorityQueue`. The
essential clauses are now a binary heap on a contiguous array of their
current documents, rebuilt when the partition changes and sifted after the
top moves. q25: 0.59–0.68× segmented to 1.16–1.30×, 0.93× merged to
1.53–1.62×; the disjunction shapes q08/q11/q14 stay at 1.0–1.3×.

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

## R4 — sorted search (delivered)

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

### Sort modes, index-sorted shards and nested keys

* `avg` and `median` modes (ABI 22) run natively: OpenSearch sorts them with
  its own `LongValuesComparatorSource` (and the `Int`, `Double` and `Float`
  ones), a `NumericComparator` over `MultiValueMode.select` of a document's
  values with the missing value filled in, skipping with the points of the
  individual values; an average or median lies between a document's least and
  greatest value, so that skipping is exact for them and the native answer is
  Lucene's. `sum` stays on Lucene: a sum can pass the points, so which
  documents it keeps depends on where skipping starts.
* Index-sorted shards (ABI 23): a segment whose index sort begins with the
  search's sort ends at its first non-competitive document
  (`TopFieldCollector.canEarlyTerminate`), decided per segment by the plugin
  with `SortField.equals`, as Lucene decides it.
* Nested keys (ABI 25, `top_field::NestedSort`): a root document's value is
  `MultiValueMode.select(values, missing, rootDocs, innerDocs, maxDoc,
  maxChildren)` -- the `min`, `max`, `sum` or `avg` of the values of the
  children between the previous root and it that the inner query matches
  (deletions ignored, as the `BitSetProducer` and the inner weight ignore
  them), at most `max_children` with a value, the missing value for a root
  without any; longs for a whole-number key, doubles for a floating-point one
  (`Math.min`/`Math.max`, a `float` result cast back). The plugin sends the
  root filter (`Queries.newNonNestedFilter()`, an `exists` on
  `_primary_term`, or the parent object's type filter) and the inner query
  (the nested type filter and any `filter`) as query blobs. Lucene's
  comparator keeps skipping with the field's points -- the children's -- once
  more than the threshold's hits are in, which can drop roots; the native key
  does not skip, so the plugin runs a nested sort only on a shard that cannot
  hold more documents than the threshold (`sort_nested` otherwise), where the
  two agree. `median` has no nested pick in OpenSearch and stays there.
  Verified by `top_field.rs`'s test against an independent pick over a fixture
  (every mode, both `max_children`, five field types) and by the REST matrix's
  nested rows against a stock node on a one-shard and a two-shard nested
  index (each mode and type, a nested `filter`, `max_children`,
  `search_after`, beside aggregations, and the two that fall back).

## R5 — aggregations (delivered)

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

### Bucket aggregations and sub-aggregations

Everything else runs as a tree (ABI 24, `ffi_jvm_reader_aggregate_tree`,
`lucene-search/src/bucket_aggs.rs`): `histogram`, `date_histogram`, `range`
and `date_range`, `filter`, `filters`, `global`, keyword `terms` and
`cardinality`, with the metrics above, nested in any combination. The flat
path above stays for the requests it takes (its column streaming and
points shortcuts are faster there).

The shape is OpenSearch's `BucketsAggregator` tree, collected document by
document: each aggregator gets `collect(doc, owningBucketOrd)`, a keyed one
(the histograms, `terms`) assigns a bucket ordinal when a bucket's first
document arrives, as `LongKeyedBucketOrds.add` does, a fixed one (`range`,
`filters`) uses `owning * width + index`, and each hands the document on to
its sub-aggregations under the bucket's ordinal. The documents come segment
by segment in document order, so every bucket's metric sums are added in
Java's order, and the slices of a concurrent search keep states of their own,
reduced by the plugin with OpenSearch's `InternalAggregations.reduce`. Ported
per aggregator:

* `histogram` (`NumericHistogramAggregator`): each distinct
  `Math.floor((v - offset) / interval)` of a document's values, the previous
  key starting at `-Infinity` -- so a `-Infinity` value is never bucketed, as
  in Java -- keyed by `Double.doubleToLongBits`, in the hard bounds
  (`key * interval`, both ends inclusive);
* `date_histogram` (`DateHistogramAggregator`): `Rounding.Prepared.round`
  for the calendar units and fixed intervals in a fixed-offset zone, with an
  offset (`DateUtils.roundFloor`, `roundWeekOfWeekYear`, and the month,
  quarter and year floors of `DateUtilsRounding`, checked against a
  proleptic-Gregorian oracle over eight centuries), a multi-valued document's
  equal rounded values once, hard bounds `[min, max)`;
* `range` (`RangeAggregator`): the aggregator's own sorted ranges and
  `maxTo`, each value binary-searched from where the previous one's search
  ended (`MatchedRange`);
* `filter`/`filters` (`FilterAggregator`/`FiltersAggregator`): each filter's
  matches in a segment, deletions ignored as the weight ignores them, a
  bucket per matching filter and the `other` bucket for a document matching
  none;
* `global`: the global query (`buildFilteredQuery(match_all)`) over every
  segment, with the main search's slices -- `ConcurrentAggregationProcessor`
  runs it with them and adds it to whatever is already there, so the plugin
  removes its collector manager once the native pass has answered it;
* `terms` under a bucket (`RemapGlobalOrds`): per owning bucket the top
  `shard_size` by count then term, the rest into `otherDocCount`;
* `cardinality`: the distinct values per bucket -- the terms of a keyword,
  the longs of a whole-number field, the `double` bits of a floating-point
  one -- hashed as `CardinalityAggregator` hashes them (`MurmurHash3`'s
  `h1`, `BitMixer.mix64`) into a native port of `HyperLogLogPlusPlus`
  (`cardinality_sketch.rs`: linear counting in its open-addressing table,
  the upgrade to registers past its threshold) and handed over as
  `writeTo` writes a sketch, which the plugin reads with `readFrom` as a
  coordinating node reads a shard's. The first version shipped the values
  and let the plugin hash them into a Java sketch: the JVM then did all of
  stock OpenSearch's per-value work plus the transfer, and `terms` >
  `cardinality` at precision 10 measured 0.76-0.95x;
* a top-level `min`/`max` keeps its points shortcut.

The plugin reads every parameter off the aggregators OpenSearch built for the
request (the sorted ranges, the rounding, the bounds, the effective
`shard_size`, the precision; a deferred sub-aggregation behind its
`WrappedAggregator`), and builds the results through their own empty results
and factories where OpenSearch exposes them (`InternalHistogram.create`,
`InternalRange.Factory`), by reflection where it does not (`InternalFilter`,
`InternalGlobal`, `InternalCardinality`, and `InternalDateHistogram`, whose
empty-bucket rounding drops the offset only in `buildAggregations`).

Verified: `bucket_aggs.rs`'s tests (the calendar rounding against the oracle;
a match-all `filters` and `global` bit for bit against the flat metrics pass;
histograms, ranges and date histograms against a scan of the columns, with
the cardinality of each bucket; `terms` under a filter against the flat
`terms`), and the REST matrix against a stock node: 28 rows of bucket trees
(every aggregation above, nested three deep, keyed, with hard and extended
bounds, offsets, `min_doc_count: 0`, a `+05:30` zone, multi-valued and float
fields, beside hits and a sort, behind `min_score`) on every index of the
matrix, and the shapes that stay on OpenSearch (a zone with daylight saving,
`terms` ordered by a sub-aggregation).

Speed. The straight port lost on six of the new rows (0.54-0.90x). OpenSearch
counts a top-level `date_histogram` or `range` without sub-aggregations from
the BKD tree on a segment every document of which matches (its filter
rewrite); the native pass now does too (`count_from_points`: a segment
without deletions, a field with one point per document, at most 1,024 buckets,
each bucket's interval counted by an intersect whose cells inside it are
counted whole). The bucket ordinals and distinct values use a multiplicative
hash finished with MurmurHash3's `fmix64` (SipHash cost the most; a bare
multiply left a widened float's zero low bits zero, and the table buckets by
the low bits -- `cardinality` over a float field went to 0.54x until the
finisher); a `date_histogram` keeps its last bucket's bounds and ordinal; a
metric under a bucket folds only the parts it reads; a dense column is read
inline. Over REST (same node and method as below, 21,703 documents in two
segments, Lucene over native, median of 60 requests per engine):

| row | ratio |
|---|---|
| `histogram` | 1.04x |
| `histogram` multi-valued + `stats` | 1.09x |
| `histogram` on a float | 1.07x |
| `date_histogram` by day | 1.26x |
| `date_histogram` week, year and quarter | 1.10x |
| `range` | 1.18x |
| `date_range` | 1.01x |
| `filters` + `other` + `sum` | 1.32x |
| `terms` + sub-metrics | 1.21x |
| `histogram` > `terms` | 1.15x |
| `terms` > `date_histogram` > `max` | 1.15x |
| `cardinality` of five fields | 1.07x |
| `global` + a metric | 1.11x |
| `date_histogram` by month + `sum` | 0.94x |
| `terms` > `cardinality`, precision 10 | 0.95x |

The last two are a match-all read document by document under a
sub-aggregation, about 3 ms end to end on both engines; the same rows measured
0.87x and 0.91x in the run before and 1.00x/0.99x on neighbouring rows in this
one, so the gap is within the run-to-run spread but not yet shown to be
closed.

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
Both runs are in the R1 commit message. Since R3's fuzzy work (the
`MaxScore` essential heap), q25 is 1.16–1.30× segmented and
q07/q08/q11/q14 measure 0.97–1.24× across both indexes, within the noise
this machine shows between runs (the Java side of one shape varies by up
to 20%).

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

## R6 — fetch and get (delivered)

The fetch phase and the get API read a hit's stored fields through
`StoredFields.document(docID, StoredFieldVisitor)`; OpenSearch's visitors
(`FieldsVisitor` and its subclasses) turn them into `_source`, `_id`,
`_routing` and the requested `stored_fields`. The plugin installs an index
reader wrapper (`IndexModule.setReaderWrapper`, `NativeStoredFieldsReader`)
on every index, so every searcher OpenSearch hands out -- the fetch phase's,
the get API's, `mget`'s -- has leaves whose `storedFields()`, and whose
sequential reader (`SequentialStoredFieldsLeafReader`, which the fetch
phase asks for on runs of adjacent hits), answer from the native reader of
the same segments: `ffi_jvm_reader_document` decodes the document
(`SegmentReader::visit_stored_document`, the fixture-verified
`stored_fields` reader, its `.fdm` metadata parsed once per segment core as
Java's reader keeps it) into one blob of `(number, type, value)` fields in
stored order, and the plugin replays it into OpenSearch's visitor with its
`needsField` answers (`NO` skips a field, `STOP` ends the document), as
`Lucene90CompressingStoredFieldsReader.document` does. Everything else the
wrapped reader is asked goes to Lucene, and both cache helpers are the
wrapped reader's, so the query cache, the request cache and the
`_id` lookups' per-segment caches are unchanged. The native handle is the
query phase's own (`NativeReaders.peek`, keyed by the reader's cache key),
never opened for the fetch: a reader no search opened natively -- the
realtime get's internal reader -- is read by Lucene rather than paying a
native open for a handful of documents. A document the native side fails to
read (a damaged segment, one without stored-fields files) is read by Lucene
too, and `index.lucene_rust.fetch.enabled` (dynamic, default on) sends every
document to Lucene.

An index takes one reader wrapper (`IndexModule.setReaderWrapper` is
set-once), and the security plugin claims it for field- and document-level
security. The node setting `lucene_rust.fetch.reader_wrapper` decides
whether this plugin installs its wrapper: on by default, off by default when
the security plugin is installed beside it (stored fields are then Lucene's),
and a wrapper another plugin set first is left in place with a warning.

The blob carries every stored field of the document; the visitor's
`needsField` answers are applied in Java. A visitor that wants only `_id`
(`_source: false`) still has the whole document copied across the boundary,
which Lucene skips with a `skipBytes` -- measured within the numbers below,
and the obvious next step (a field mask from the visitor) if a workload of
id-only fetches shows it.

`docvalue_fields` is not a stored-fields read: OpenSearch's
`FetchDocValuesPhase` reads each value through the field's doc-values
iterator, one leaf reader call per value, which the wrapper leaves on
Lucene. Moving it across the JNI boundary would pay a call per value to save
a few nanoseconds of doc-values decoding each, so it stays on Lucene by
design (the REST rows compare its output all the same).

Verified: `jvm_fetch.rs`'s tests (every document of the Lucene-written
`stored_fields_index` through the FFI blob equals the segment reader's
decoding; every value type round-trips; a damaged chunk and a segment
without stored-fields files are errors, not empty documents),
`directory_reader.rs`'s (the segment reader answers what the codec reader
answers; a partial `.fdt/.fdx/.fdm` set is an error; metadata that does not
parse fails every read and is never kept), the plugin's self test (a
Lucene-written index with every stored type read through the wrapper --
random access and the sequential reader -- and by Lucene with the same
visitors: every field, `NO` on some including the last, `STOP` midway;
every `needsField` and value call compared, about 750 of its 3,000
documents under each visitor), and
`verify_opensearch.py`'s fetch rows on the same node with the setting
toggled: whole hits for `_source`, `_source` includes and excludes, no
`_source`, `stored_fields` (with and without `_source`), `docvalue_fields`,
`from`, `version`/`seq_no_primary_term`, highlighting and a run of adjacent
hits (the sequential reader), then `get` (plain, `_source_includes`,
`stored_fields`, `realtime=false`) and `mget` (realtime and not) -- after
the query matrix and again after the SIGKILL restart. Every search and
non-realtime get must be served natively throughout: the native counter
moves, and neither Lucene's count nor the error count does.

**Speed.** The plugin times `StoredFields.document` on both paths
(`fetch_nanos` in its stats): over the benchmark's fetch rows, native
16.4 µs per document against Lucene's 20.4 µs, **1.24×**. The first version
opened the stored-fields reader (parsing `.fdm`) and built a `Document` per
call; keeping the parsed metadata with the segment and encoding straight
from the decompressed bytes took it from parity to 1.24×. Over REST, where
a request's fetch is a small part of its ~2 ms, the rows measure
0.95–1.12× (median 1.04×) and `get` 1.12×, within the round trip's noise.

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
  run the sorted search over that prefix of the index (the cut segment
  searched below its last collected match: the bulk scorers get the cut as
  their `max`, an iterating scorer is wrapped to end there) with the whole
  shard's statistics; a search by
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
  count replay counts only passing documents. Behind a sort
  (`top_field::search_sorted_min_score`) the wrapper's leaf collector passes
  on no competitive iterator, so no key skips anything; it asks for
  `COMPLETE` scores unless the sort's collector wants `TOP_SCORES` (a
  score-led sort, which still prunes by score): every match is scored by the
  bulk scorers and only the passing ones reach the sort's collector and the
  `MaxScoreCollector` beside it. Behind a scroll's later pages or
  `terminate_after`, Lucene answers. `tests/min_score_fixtures.rs` checks 140
  Lucene runs; the self test random queries at two minimums each, and a third
  of its random sorted pages behind a minimum (2,644 pages, `MinimumScoreCollector`
  around Lucene's `TopFieldCollector` and its `MaxScoreCollector` when
  tracking).

- **An approximate range in a `post_filter`**: OpenSearch rewrites a `range`
  to `ApproximateScoreQuery` and, where it may, resolves it to the
  approximation, which stops after `track_total_hits` matches in BKD order.
  In a `post_filter` that is a filter narrower than the range: on one merged
  segment of 99,534 documents stock OpenSearch answered a `range` post filter
  with 9,429 hits "exactly" and without the best-scoring ones, where the same
  range as a `filter` clause gives the full answer. The native engine gives
  the full answer, so a `post_filter` that resolves to an approximation stays
  on OpenSearch (`approximate`), as the main query already did. Found by the
  first verification on 100,000 documents after a force merge.

### The query-phase benchmark over REST

The plugin's own `query_phase_nanos` counters, per path (no HTTP or fetch
time in either): every native row of `verify_opensearch.py`'s matrix, 116
request shapes, on one shard of 100,000 documents in eight segments, the
index switched between the native path and Lucene six times, 20 requests per
shape per turn after three warm-up requests, median per shape. Lucene over
native:

| | ratio |
|---|---|
| median | 1.58× |
| 10th percentile | 1.11× |
| worst | 0.89× (`constant_score` over `match_all`: 40 µs against 45 µs) |

Shapes under 1.0× at the last run: that one; `agg terms shard_size` (0.87×
to 1.10× across runs, Lucene's own time moving by 400 µs); and three at
0.98-0.99×. Rows under 100 µs move ±10% between runs.

**The same benchmark after a force merge, warmed** (`opensearch-plugin/e2e/phase_bench.py`,
on the node `scripts/verify-opensearch.sh --docs 100000 --keep` leaves: one
segment of 99,534 documents and one of 200, every shape first run 40 times on
each path, then the six alternating turns). Of the 132 non-aggregation
shapes, median 1.60×, 10th percentile 0.92×; 17 under 1.0×, the worst
`exists` over a sparse field with hits (0.47×, 50 µs against 107 µs) and a
multi-valued keyword (0.51×), `regexp` with an interval (0.53×), `fuzzy` on
a keyword (0.60×), a lone `exists` counted (`size: 0`, 0.70×), then
eleven between 0.82× and 0.99×. The one-segment shard is not the
eight-segment one: per-request costs that eight small segments hid now show.

The warm-up matters. Measured on a node that had served a few thousand
requests, the plugin's own Java (planning, eligibility, encoding) had not
been compiled yet: a term query's native query phase was 133 µs, of which
53 µs before the native call; after 20,000 more requests, 66 µs and 5 µs,
Lucene's path barely moving (its code is warm from every request the node
serves). Earlier per-shape numbers on a freshly started node understate the
native path by that much.

What this round changed, each confirmed by the in-process search on a copy
of the merged shard (`ffi_jvm_reader_search`, no JVM):

- A keyword field has no norms, so its norm is one constant for every
  document; the batch scorer looked it up per document through the general
  path. A term on `tag` 69 µs to 43 µs, a dense one 154 µs to 86 µs.
- `TopDocsCollector.collect`'s common case -- a full queue the document
  loses to, below the count threshold -- is inline and does nothing else
  (43 µs to 39 µs).
- A lone `exists` counted is `FieldExistsQuery.count` from the index
  statistics (0.08× to 0.70×), with the points' document count read from the
  metadata the segment keeps rather than parsed per request.
- `constant_score`'s inner clause goes through the query cache, as
  `ConstantScoreWeight`'s does; the per-segment cache holds 1,000 entries
  (it held 64: a matrix of a few hundred shapes evicted each before its next
  use, where OpenSearch's node-wide cache, 10,000 entries, keeps it).

The next round (same node and shapes, after `verify-opensearch.sh --docs
100000 --keep` rebuilt it: 6,863 checks, 0 failures): of the 132
non-aggregation shapes, median 1.67×, 10th percentile 1.09× (from 0.92×), 10
under 1.0× (from 17). What the in-node gap turned out to be, and what closed
most of it:

- **A native call in the node runs with cold caches.** Timed inside the JNI
  entry, a no-hit term's search took 21 µs of CPU per call in the node and
  1.2 µs in a loop in process; the same call with the caches evicted between
  calls (64 MB swept) took 24 µs. So the per-request cost is the code and data
  a call touches, not the work: every allocation, map and clone in the setup
  counts. Cut from it: norms built for clauses that never score (`FILTER`,
  `MUST_NOT`, a constant score's inner query; 3.8 µs cold), and the points
  rewrite's clone of the whole query when it holds no range (3.4 µs cold);
  24 µs to 17 µs cold.
- **One crossing instead of two.** OpenSearch answers a lone term's total
  from `docFreq`; that was its own JNI call before the search (14 µs in the
  node). `searchDocFreq` (ABI 28) answers both in one.
- **A keyword term with norms never stopped.** A field without frequencies
  gets one impacts level from Lucene, `(freq 1, norm 1)` up to
  `NO_MORE_DOCS`; the port bounded it by the largest norm inverse there is,
  which one-token values never reach, so a full queue never ended the scan.
  With a threshold of 10 (the shortcut's), a 6,457-document term 11.2 µs to
  2.7 µs, the fuzzy query's rewritten term 12.8 µs to 3.8 µs (verified
  against Lucene: `GenDocsOnlyNorms.java`).
- **Per-document collection.** A cached bit set's constant-scored walk counts
  the hits between a full queue and the count threshold by popcount (an
  `exists` with hits 49 µs to 1.7 µs in process), and a block of scored
  documents below the threshold is collected in one batch with the count in a
  register (a 8,635-document term, top 10, counted, 38 µs to 15 µs).
- **The plugin's regexp check** compiled a second `RegexpQuery` per request
  to compare flags (30 µs for `<1-20>`); the reference automaton is now kept
  per pattern.

Still under 1.0× at the last run, and why:

- `fuzzy` on a keyword (0.73×, 74 µs against 54 µs): the rewrite that runs
  before the query phase (Levenshtein automata) leaves both the Java side and
  the native call colder than a plain term's (the call 56 µs against 40 µs in
  the node); the boosted, statistics-carrying term also takes the general
  scorer tree (39 µs cold in process against 28 µs for a lone term's own
  path) -- though counted in instructions that path is only 4% more work
  (see the next round, below).
- A handful of 40-60 µs shapes (`constant_score` over `match_all` 0.84×,
  `regexp` with an interval 0.90×, a lone `exists` counted 0.90×, `exists`
  on a text field 0.93×): the remaining cold setup of one native call against
  Lucene's warm path; these move ±10% between runs (term keyword read 0.83×,
  0.97× and 1.13× across three).
- Real work, 0.5-5 ms: `minimum_should_match` (0.89×), `cross_fields`
  (0.85×), a `regexp` filter (0.87×), `search_after` over `match_all`
  (0.92×), a `post_filter` `exists` behind a paged `bool` (0.88×) -- the next
  round's scorer work.

The next round on the small shapes (2026-09-28) measured before changing
anything, and the premise above did not hold up:

- **Setup is not the small shapes' cost.** A native call for a term no
  segment holds is 0.6-0.7 µs in process (the per-reader caches above are
  what is left of setup). Wall-clock timing on the benchmark VM swings 10x
  between identical runs, so the work was counted instead: instructions per
  call under cachegrind (`valgrind --tool=cachegrind`, the difference of
  two run lengths), deterministic, with the shipped `x86-64-v3` flags -- a
  benchmark crate outside the workspace does not read `.cargo/config.toml`
  and silently measures the 2003 baseline.
- **`fuzzy` on a keyword** (a boosted term with blended statistics, under
  the scorer tree) does 191k instructions a top-10 call against a plain
  term's 184k on the same 6,457 hits: 4%, not worth a second scoring route.
  Both spend them in the same per-hit loop, below. Its `size: 0` count was
  the real gap: the tree walked the query cache's bit set a document at a
  time (156k against 9k). `BooleanWeight.count` passes a lone required term
  through to `TermWeight.count` (its `docFreq`, no deletions), as it did an
  `exists`; and a collector that only counts takes a cached set's hits by
  popcount (`ScoringCollector::add_hits`). Now 10k. The pass-through also
  stopped answering for `minimum_should_match` 1 with no `SHOULD` clause,
  which matches nothing -- `lone_exists` had the same gap.
- **The per-hit loop.** A keyword's hits all score the same, so once the
  queue is full every later one loses until the `totalHits` threshold;
  `TopDocsCollector::collect_many` tested each with a branch. It now tests
  sixteen at a time (none above the worst kept score, or a NaN, which the
  per-hit test does not reject) and falls back to the per-hit test for a
  run that fails. Top-10 on a keyword term: 184k to 100k instructions.

In the node (four alternations of the two builds on the same index, focused
shapes): `bool` with one filter 0.78× of before, `match_all` + filter 0.88×;
`term` and `fuzzy` on a keyword and `match` with fuzziness unchanged within
noise -- the saved instructions do not show against a query phase dominated
by the Java side and a cold call. Three `post_filter` shapes and a `bool` with
a range filter read 4-10% slower while Lucene in the same slots did not move;
the native work for them is identical in instructions (4,839,179 against
4,839,187 for the `post_filter` without a total), so this is layout or noise,
not more work, but it was not shown to be either.

Aggregations finish outside the phase counters, so they are timed over REST
(`REST=1 phase_bench.py 30 single agg`, whole round trips, median per
shape): of 47 aggregation shapes, median 1.08×, 14 under 1.0×. The worst are
real work, not setup: `date_histogram` by month with a `sum` (0.65×, 6.7 ms
against 4.3 ms), `cardinality` (0.68×) and under `terms` at a low precision
(0.76×), `global` beside the main query (0.84×), a float `histogram`
(0.88×), `date_range` (0.89×); the rest between 0.87× and 0.98×. Those are
R7's remaining work.

What closed the gaps the first full run showed (worst 0.28×, 30 shapes
under 1.0×), each measured before and after:

- Fixed cost per request: the plugin's index settings read once per settings
  version; BM25 norm tables once per reader; a field's per-segment norms
  resolved once per reader; postings files validated once per reader; the
  statistics pass's term seek reused by the scorer (`TermStates`); a term no
  segment holds answered without visiting one; a boolean of one term clause
  run as that term (`BooleanQuery.rewrite`); OpenSearch's total-hits shortcut
  for a term (`docFreq` over the segments) counted by the native dictionaries
  rather than Lucene's in the JVM, 25 µs a request; no JNI call at all for a
  `size: 0` total already known.
- Threads: a concurrent search's slices run on the calling thread when their
  work is too small to repay a pool thread's wake-up (fewer than 4,096
  estimated matches, or aggregations answered from index statistics alone):
  a match-all `terms` aggregation over four slices, 134 µs to 18 µs.
- Scorers: a `MUST` + `SHOULD` query no longer walks empty windows past its
  leading clause's last block (32,000 a segment); unscored term disjunctions
  union a block at a time into bit windows; a sort tracking scores scores
  through the bulk scorers, not a second tree advanced per document; a
  two-term sloppy phrase walks two pointers instead of the general matcher
  (checked bit for bit against it); WAND's `scalb` builds its power of two
  from the exponent bits, and `Math.scalb`'s way past the largest normal
  power, not through `powi`; wildcard and prefix automata compiled once per
  pattern; `terminate_after`'s cut segment searched below the cut.
- Aggregations: a multi-valued column decoded only for the matching
  documents; a `size: 0` search's hits counted in the aggregations' own pass
  (ABI 18), with and without `min_score`, as Lucene's `MultiCollector` does.

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
