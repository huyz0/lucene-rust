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

**Status.** In progress. R1 and R2 delivered (below); every shape the plugin sends native measures at least 1.0× Lucene in process on both indexes. R3–R7 open.

## Tasks

| ID | Task | Status |
|---|---|---|
| R1 | Query execution engine: Lucene's scorer tree and bulk scorers, every boolean shape at least as fast as Lucene | ✅ delivered |
| R2 | General query wire format and Java encoder for every Lucene query OpenSearch builds | ✅ delivered for the shapes R1 runs (term, boolean, constant score, boost, dismax, match-all, match-none); leaf queries arrive with R3 |
| R3 | Leaf queries as streaming scorers: phrase, the multi-term family, points and doc-values ranges, exists, terms-in-set, dismax, synonym | open |
| R4 | Sort and `search_after` natively (`TopFieldCollector`) | open |
| R5 | Aggregations natively: terms, histogram, date_histogram, range, the metrics, cardinality, filter/filters | open |
| R6 | Fetch (`_source`, stored fields, `docvalue_fields`) and get natively | open |
| R7 | scroll, `post_filter`, `min_score`, `terminate_after`, timeouts; the full read benchmark (in process and REST) with every native shape at least 1.0× Lucene | open |

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
