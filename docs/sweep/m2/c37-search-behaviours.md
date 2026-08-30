# c37-search-behaviours — the missing search behaviours in `lucene-search`

Five items from `LEDGER.md`'s "Open work, prioritised", all in one crate:
`searchAfter`/`MaxScoreAccumulator` (item 5), `Weight.count` (item 5),
`FieldExistsQuery.count`/`rewrite` (item 6), sloppy phrase reordering (item 7),
and filter-only pruning (item 23). Item 29 (the fixture debt for items 7 and 23)
is closed as a side effect, and is the batch's most transferable finding.

`TwoPhaseIterator`/`matchCost`/`ScorerSupplier.cost()` was **not** started, per
the brief. Neither item 2 nor item 5 turned out to need it: `Weight.count` is a
terms-dictionary read, and filter-only pruning was a one-line guard removal.

Files swept:

- `crates/lucene-search/src/collector.rs` — `TopScoreDocCollector`,
  `MaxScoreAccumulator`, `DocScoreEncoder`
- `crates/lucene-search/src/multi_segment.rs` — the fan-out
- `crates/lucene-search/src/lib.rs` — `try_conjunction_lazy`, the phrase and
  multi-phrase executors
- `crates/lucene-search/src/sloppy_phrase.rs` (**new**) — `SloppyPhraseMatcher`
- `crates/lucene-search/src/weight_count.rs` (**new**) — `Weight.count`
- `crates/lucene-search/src/directory_reader.rs` — `SegmentReader::{num_docs,
  field_exists_leaf}`
- `crates/lucene-search/src/{explain.rs,query.rs,highlighter.rs}` — call sites
  and stale scope notes
- `crates/lucene-ffi/src/query.rs` — `ffi_count_term_query`,
  `ffi_search_term_query_scored_after`
- `fixtures/src/{AppendScoringManifest,AppendCountManifest,
  AppendSearchAfterManifest}.java`, `crates/lucene-search/tests/*`,
  `crates/lucene-search/benches/{term_count,filter_vs_must}.rs`

Java counterparts, all under
`/home/tuong/work/lucene-10.5.0/lucene/core/src/java/org/apache/lucene/`:
`search/TopScoreDocCollector.java`, `search/MaxScoreAccumulator.java`,
`search/DocScoreEncoder.java`, `search/Weight.java`, `search/TermQuery.java`,
`search/MatchAllDocsQuery.java`, `search/TotalHitCountCollector.java`,
`search/IndexSearcher.java`, `search/FieldExistsQuery.java`,
`search/SloppyPhraseMatcher.java`, `search/PhrasePositions.java`,
`search/PhraseQueue.java`, `search/PhraseScorer.java`,
`util/NumericUtils.java`.

---

## Counts

| class | count |
|---|---|
| `CORRECTNESS` | **1** (F-4: reordered sloppy phrases under-matched) |
| `MISSING` | **4** (F-1 `searchAfter`, F-2 `MaxScoreAccumulator`, F-5 `Weight.count`, F-6 `FieldExistsQuery.count`/`rewrite`) |
| `PERF` | **2** (F-7: filter-only pruning, fixed; F-10: the collector's queue, recorded) |
| `INTENTIONAL` | **3** (F-3 `modInterval`, F-8 highlighter enumeration, F-9 `IndexSearcher.count`'s disjunction shortcut) |

All `CORRECTNESS` and `MISSING` findings are fixed with tests, each verified to
fail against the unfixed code.

---

## `crates/lucene-search/src/collector.rs`

Java: `TopScoreDocCollector.java`, `MaxScoreAccumulator.java`,
`DocScoreEncoder.java`, `util/NumericUtils.java`.

| Rust item | Java | Verdict |
|---|---|---|
| `TopDocsCollector::collect` | `TopScoreDocCollector`'s leaf `collect` | was missing the `after` test — now identical |
| `TopDocsCollector::with_after` (**new**) | the `after` constructor parameter | equivalent |
| `TopDocsCollector::with_shared_max_score` (**new**) | the `minScoreAcc` parameter | equivalent |
| `TopDocsCollector::min_competitive_score` | `updateMinCompetitiveScore` + `updateGlobalMinCompetitiveScore` | now both halves |
| `MaxScoreAccumulator` (**new**) | `MaxScoreAccumulator` | equivalent, minus `modInterval` (F-3) |
| `doc_score_encoder` (**new**) | `DocScoreEncoder` + `NumericUtils.sortableFloatBits` | identical |
| `rank_order` | `HitQueue.lessThan` | identical (unchanged) |
| — | `TernaryLongHeap` backing store | PERF, recorded — F-10 |
| — | `pruneLeastCompetitiveHitsTo`, `populateResults`, `topDocsSize` | not-in-Rust: artefacts of the sentinel-filled heap this port does not have |

### F-1 `[MISSING]` `searchAfter`'s `after`

**Java.** `TopScoreDocCollector.getLeafCollector` computes
`afterScore`/`afterDoc = after.doc - context.docBase` and drops any hit that
ranks at or above that pair:

```java
if (after != null && (score > afterScore || (score == afterScore && doc <= afterDoc))) {
  if (totalHitsRelation == TotalHits.Relation.EQUAL_TO) updateMinCompetitiveScore(scorer);
  return;
}
```

**We did.** Nothing. A caller wanting page 2 had to collect a top-`2n` and
discard the prefix, which is `O(n)` extra queue work per page and, more
importantly, gives a *different* answer once the query is large enough that
`total_hits_threshold` pruning kicks in.

**Fixed.** `TopDocsCollector::with_after`, applied after `total_hits` is
incremented (so page 2 reports the same total page 1 did — Java's ordering,
which is easy to get backwards) and before the queue is consulted. The
doc-ID space is the collector's own; the multi-segment fan-out does the
`- docBase` subtraction, in
`multi_segment::merge_multi_segment_scored_after`, and
`search_term_query_multi_segment_after` is the ready-made entry point. C ABI:
`ffi_search_term_query_scored_after`, kept as its own function rather than extra
parameters on `ffi_search_term_query_scored` because there is no `after` value
that honestly means "no after".

**Tests.** Three real-Lucene pages on one segment
(`bm25_scoring_fixtures::search_after_pages_match_real_lucene_bit_for_bit`) and
three across two segments
(`multi_segment_scoring_fixtures::multi_segment_search_after_pages_match_real_lucene_bit_for_bit`),
plus three unit tests on the boundary rule and one FFI test. **The
single-segment fixture is the one that matters for the rule**: `big`'s 300
documents take only four distinct scores, so the page boundary falls inside a
run of score ties and is decided by `doc <= afterDoc`. A score-only comparison
returns page 1 forever, and this catches it.

**What the multi-segment fixture does *not* prove, stated because it looks like
it does.** `multi_segment_scoring_index` has no two documents with equal scores,
so the doc-id half of the rule is never reached there and **removing the
`- docBase` subtraction still passes it**. Checked, not assumed. The subtraction
is pinned by
`multi_segment::tests::the_after_fan_out_translates_the_global_doc_id_into_each_leafs_own_space`,
whose synthetic hits are all ties and which does fail without it. Both facts are
now written on the fixture test.

### F-2 `[MISSING]` `MaxScoreAccumulator`

**Java.** One `LongAccumulator(Math::max, Long.MIN_VALUE)` shared by every
concurrently-searched leaf. Each leaf publishes the `(global doc, score)` of its
own worst kept hit, packed by `DocScoreEncoder` into a `long` whose integer
order *is* `HitQueue`'s ranking order; each leaf may then prune against the best
pair any leaf has published, with a doc-id-aware tie-break:

```java
score = docBase >= DocScoreEncoder.docId(maxMinScore) ? Math.nextUp(score) : score;
```

**We did.** Nothing: each leaf pruned only against its own queue, so a leaf that
had not yet found anything competitive scanned every document even when another
leaf had already proved a far higher bar.

**Fixed.** `collector::MaxScoreAccumulator`, one `AtomicI64` folded with
`fetch_max`, plus `doc_score_encoder` (Java's packing including
`NumericUtils.sortableFloatBits`, which is its own inverse). Wired at the
fan-out, which is where the ledger said it had to live:
`multi_segment::merge_multi_segment_scored_concurrent_shared_max_score`.

**The write gate is Java's, and it matters.** `updateMinCompetitiveScore` only
touches the accumulator `if (localMinScore > minCompetitiveScore)`. The port
keeps the same per-leaf high-water mark (`published_min`), because without it
the `fetch_max` runs once per collected document -- `fetch_max` makes a repeat
write *harmless*, not free, and this is the per-document path. Caught in this
batch's own self-review; the two tests that pin it drive a descending and an
ascending score stream, so the gate cannot pass by suppressing a publish the
accumulator needed.

**The ULP, since it is the one place the two ports' rules differ in spelling.**
Java publishes a threshold and skips when `bound < threshold`; this crate's
scoring loops skip when `bound <= threshold`. Those are the same rule iff this
port's threshold is the immediate predecessor of Java's. `next_down(next_up(x))
== x`, so Java's `(nextUp(score), score)` pair becomes this port's
`(score, next_down(score))` — the same two branches, shifted by exactly the ULP
the two conventions differ by. Written on `threshold_for`.

**Tests.** `the_doc_score_code_orders_exactly_as_the_hit_queue_does` cross-checks
the packing against `rank_order` over every pair drawn from eight
`(doc, score)` values including `-inf`, `f32::MAX` and `i32::MAX` — the claim
the whole single-atomic design rests on. Plus round-tripping, the doc-base
tie-break in both directions, a shared-accumulator collector test, an
exhaustive-collector test (nothing is published, so an exact count stays exact),
a fan-out test asserting the shared and plain concurrent merges return the same
hits, and two tests on the write gate described above.

### F-3 `[INTENTIONAL]` `modInterval` is not modelled

**Java.** `MaxScoreAccumulator.modInterval == 0x3ff`, and the leaf collector
re-reads the accumulator only when `(hitCountSoFar & modInterval) == 0`.

**We do.** Read it whenever a threshold is asked for.

**Why.** The interval exists because Java *pushes* the threshold into the scorer
(`scorer.setMinCompetitiveScore`), which is a per-document code path, so an
atomic read there is worth amortising. This port *pulls* the threshold from
`ScoringCollector::pruning_threshold()` at each block-skip decision — once per
block, not once per document — so the amortisation has nothing to amortise, and
a modelled interval would only make the shared bound go stale for no gain.
Recorded on `min_competitive_score`. Modelling it would also have been dead
code, which the coverage bar would then have to carry.

### F-10 `[PERF, recorded]` the top-`n` queue is still a sorted `Vec`

`TopScoreDocCollector` is backed by a `TernaryLongHeap` over `DocScoreEncoder`
codes -- `O(log n)` per kept hit, and no separate score/doc arrays. This port's
`TopDocsCollector` keeps `hits` fully sorted in a `Vec`, an `O(n)` insert, which
`collector.rs` has documented as a deliberate first cut since task #13.

Recorded here rather than fixed because **the reason it was unschedulable is
now gone and nobody would otherwise notice**: `lucene_util::TernaryLongHeap`
exists (c21 wrote it for HNSW's `NeighborQueue`) and this batch added the other
half, `doc_score_encoder`. Both prerequisites are in the tree. What is *not*
clear is that it is worth taking: the fast reject that handles the overwhelming
majority of documents is one comparison either way, and for the `top_n` values
this port's callers use (20-50) an insert is a memmove of a few hundred bytes.
It wants a measurement before a rewrite, not a rewrite. Noted on the parity row.

### Verdict

Swept clean. Line coverage 99.0%.

---

## `crates/lucene-search/src/weight_count.rs` (new)

Java: `Weight.java`, `TermQuery.java` (`TermWeight.count`),
`MatchAllDocsQuery.java`, `FieldExistsQuery.java`,
`TotalHitCountCollector.java`, `IndexSearcher.count`.

| Rust item | Java | Verdict |
|---|---|---|
| `count_term_query_shortcut` | `TermWeight.count` | identical |
| `count_term_query` | `IndexSearcher.count(TermQuery)` for one leaf | equivalent |
| `count_match_all_docs` | `MatchAllDocsQuery`'s weight's `count` | identical |
| `FieldExistsLeaf` | the `LeafReader` reads inside `count`/`rewrite` | new, a shape Java does not need |
| `count_field_exists_leaf` | `FieldExistsQuery`'s `ConstantScoreWeight.count` | identical |
| `field_exists_rewrites_to_match_all_docs` | `FieldExistsQuery.rewrite`'s loop | identical |
| — | `IndexSearcher.count`'s two-clause disjunction shortcut | MISSING, deliberate — F-9 |
| — | `Weight.count`'s `-1` default for every other query | not-in-Rust: `Option::None`, and the caller runs the query |

### F-5 `[MISSING]` `Weight.count(LeafReaderContext)`

**Java.** `TotalHitCountCollector.getLeafCollector` asks `weight.count(context)`
*before* it collects anything and throws `CollectionTerminatedException` when
the answer is not `-1`. `TermWeight.count` returns `termsEnum.docFreq()`
whenever `context.reader().hasDeletions() == false`, so
`IndexSearcher.count(new TermQuery(...))` on a deletion-free segment opens no
postings file at all.

**We did.** `search_term_query` into a `CountCollector`: the full postings walk,
one `collect` call per matching document, for a number the terms dictionary
already held.

**Fixed.** `count_term_query_shortcut` (the `Option<i64>` standing in for Java's
`-1`) and `count_term_query` (shortcut, else scan). `live_docs:
Option<&FixedBitSet>` **is** this crate's `hasDeletions()` — every search
function here already takes the leaf's live-doc bitset with `None` meaning "no
deletions" — so no second flag was introduced. A missing field or term is
`Some(0)`, Java's "the term cannot be found in the dictionary so the count is
0". C ABI: `ffi_count_term_query`, which only opens `.doc` on the scanning
branch.

**Measured** (`benches/term_count.rs`, criterion, `benchmarks/.corpus/merged`,
~5M documents, one segment):

| query | matches | `Weight.count` | collect every doc | ratio |
|---|---|---|---|---|
| `body:t0` | 4 997 130 | **72.2 ns** | 13.93 ms | **193 000x** |
| `body:tz` | 1 070 709 | **96.7 ns** | 1.43 ms | **14 800x** |

Both are measured because the saving is proportional to `docFreq`, not
constant: a bench on the dense term alone would overstate it by an order of
magnitude. The bench asserts the two agree before timing them, so it cannot
report a speedup for answering a different question.

**Tests.** `weight_count_fixtures.rs` against real `IndexSearcher.count`
recorded by `AppendCountManifest.java`, on `blocktree_index` (no deletions) and
`live_docs_index` (two deleted documents). **The deletions case is the one that
can be silently wrong**: `id:1` has `docFreq == 1` and names a *deleted*
document, so a port that shortcuts unconditionally reports a deleted document as
a hit. The test asserts both that the shortcut declines and that taking it
anyway would have said 1 — so it cannot pass by declining for the wrong reason.
Verified to fail with the deletions gate removed.

### F-6 `[MISSING]` `FieldExistsQuery.count` and `rewrite`'s whole-reader decision

**Java.** `count`'s ladder: `fieldInfo == null` → 0; the norms branch's
`getDocCount(field) == maxDoc() ? numDocs() : -1`; otherwise a raw count from
vectors / the doc-values skipper / points / the terms dictionary, reconciled as
`0 → 0`, `== maxDoc → numDocs()`, `>= 0 && !hasDeletions() → count`, else `-1`.
`rewrite` returns `MatchAllDocsQuery.INSTANCE` iff every leaf says the field is
complete.

**We did.** `field_exists_leaf_is_complete` (c12) — the per-leaf predicate and
nothing above it. The ledger recorded the missing layer as needing "a
reader-level query object this port does not have".

**Fixed, and the recorded blocker did not exist.** No query object is needed:
both rules are pure functions of counts a `LeafReader` already exposes.
`FieldExistsLeaf` is that bundle of counts, `SegmentReader::field_exists_leaf`
gathers it, and `count_field_exists_leaf` /
`field_exists_rewrites_to_match_all_docs` are the two rules. Java's asymmetry —
the norms branch reading `getDocCount(field)`/`maxDoc()` off the **top-level**
reader while the other two read the leaf's, inside the same per-leaf loop — is
reproduced, with the two reader-wide values as explicit parameters so a caller
cannot pass the leaf's by mistake.

One thing the resolver does **not** do: look up vector values. This port's
`FloatVectorValues`/`ByteVectorValues` are opened by the caller
(`vector_query`), not held by the reader, so `vector_size` is a parameter. That
is stated on the method, because a silent `None` there would count a vector
field as absent.

**Tests.** `weight_count_fixtures::field_exists_counts_and_rewrites_match_real_lucene`
over five committed indexes, chosen so that all four `count` arms are reached —
`norms_index` for complete (rewrites to `*:*`) and partial norms,
`doc_values_index` for the arm where *nothing* gives `count` a doc count even
though the field is on every document (easy to get wrong by assuming a dense
doc-values field is always shortcut-able), `doc_values_skip_index` for
`DocValuesSkipper.docCount()` over 36 000 documents, `blocktree_index` for a
normed field on 4 of 8959. Where the shortcut declines, the test *runs the
scan* and checks that against Lucene too, so a shortcut that wrongly declines
cannot pass. Plus seven unit tests. The live-doc arithmetic was verified to fail
with `numDocs` replaced by `maxDoc` in both the norms and the complete-field
branches.

### F-9 `[INTENTIONAL]` `IndexSearcher.count`'s two-clause disjunction shortcut

`IndexSearcher.count` has a second optimisation:
`isTwoClausePureDisjunctionWithTerms` rewrites `a OR b` into three counts and
returns `count(a) + count(b) - count(a AND b)` when the intersection is under
10% of the union. Not ported: it needs `BooleanQuery.rewriteTwoClauseDisjunction
WithTermsForCount`, which is a reader-level `Query`→`Query` rewrite, and it is
an optimisation of an optimisation. Recorded on the module, not in the ledger's
open list, because it changes no answer.

### Verdict

Swept clean. Line coverage 99.22%.

---

## `crates/lucene-search/src/sloppy_phrase.rs` (new)

Java: `SloppyPhraseMatcher.java`, `PhrasePositions.java`, `PhraseQueue.java`,
`PhraseScorer.java`.

| Rust item | Java | Verdict |
|---|---|---|
| `PhraseRepeats::detect` | `repeatingTerms` + `repeatingPPs` + `gatherRptGroups` + `sortRptGroups` | equivalent, one documented divergence |
| `SloppyMatcher::new` | the constructor + `resetPositions` | identical |
| `next_position` / `first_position` | `PhrasePositions.{nextPosition, firstPosition}` | identical |
| `advance_pp` | `advancePP` | identical |
| `tp_pos` / `lesser` / `collide` | `tpPos` / `lesser` / `collide` | identical |
| `pq_add` / `pq_pop` / `pq_key` | `PhraseQueue.lessThan` | equivalent (sorted `Vec`, total order) |
| `init_phrase_positions` / `init_simple` / `init_complex` / `fill_queue` | same names | identical |
| `advance_repeat_groups` / `advance_rpts` | same names | identical |
| `next_match` | `nextMatch` | identical |
| `sloppy_weight` | `sloppyWeight` | identical |
| `sloppy_phrase_matches` | `PhraseScorer.twoPhaseIterator().matches()` | equivalent |
| `sloppy_phrase_freq` | `PhraseScorer.score()`'s freq accumulation | identical |
| — | `startPosition`/`endPosition`/`startOffset`/`endOffset`/`captureLead` | MISSING, no consumer — see F-8 |
| — | `maxFreq()` / `impactsApproximation` | not-in-Rust: sloppy phrases use dummy impacts in Java, and this port's phrase path has no impacts plumbing |

### F-4 `[CORRECTNESS]` sloppy phrase matching was in-order only

**Java.** `PhrasePositions.position = postings.nextPosition() - offset`, where
`offset` is the term's index in the phrase. In that shifted space a match is
simply a window covering one position per slot, `matchLength = end - min`, and
there is **no ordering constraint at all** — which is precisely how Lucene
admits a transposition. `nextMatch` walks the windows with a `PhraseQueue`,
keeping the currently-popped `PhrasePositions` advancing while it stays below
the queue's next element so each emitted `matchLength` is locally minimal;
`PhraseScorer.score()` sums `1 / (1 + matchLength)` over that sequence.

**We did.** Required `p_0 < p_1 < ... < p_{n-1}` strictly increasing *in phrase
order* and charged the summed gap slack. Every transposition was a miss, at
every slop. `pos:"beta alpha"~2` returned nothing where real Lucene returns doc
8555 at `0.22822219`.

**Fixed.** `sloppy_phrase.rs` ports the matcher statement for statement,
including the `rptGroups` machinery that keeps two slots holding the same term
off one raw position (`"a a"~9` must not match a document containing one `a`)
and Java's `hasMultiTermRpts` term-graph union, which only a `MultiPhraseQuery`
can reach. Reproducing the *walk* rather than just the verdict matters for
scoring: `f32` addition is not associative, so the order the `matchLength`s are
emitted in is part of the answer.

**One deliberate divergence**, on `PhraseRepeats`: Java discovers the repeat
groups on the *first candidate document* and, in the single-term case, groups
two `PhrasePositions` whose raw positions coincide **in that document**. For two
slots holding the same term that is the same test as term equality (identical
terms have identical position lists). It differs only when two *different*
repeating terms occupy one position in whichever document the scorer met first —
which makes Java's grouping depend on document order. This port groups by term
identity: stable, and what Java's test is reaching for.

**The recorded blocker was false.** c34 recorded item 7 as "blocked on a
fixture: `GenBlockTree` has no reordered-phrase document". It has had one since
task #55: doc 8558 is `delta`@0 `gamma`@1, added *for the `SpanNearQuery`
`in_order` test*, and it is exactly a reordered document for the phrase
`"gamma delta"`. Any `alpha beta` document is one for `"beta alpha"`. No index
was regenerated and no generator was touched.

**Ground truth**, six new `AppendScoringManifest` entries recorded against the
committed segment (`f32` bits, as the rest of that file):

| key | query | real Lucene | what it pins |
|---|---|---|---|
| `scoring.phrase.reordered.slop1` | `pos:"beta alpha"~1` | *(no hits)* | the budget is real — a port cannot pass by matching everything |
| `scoring.phrase.reordered.slop2` | `pos:"beta alpha"~2` | `8555:0.22822219` | a transposition is a window of width 2, weight `1/3` |
| `scoring.phrase.reordered.slop4` | `pos:"beta alpha"~4` | `8555:0.22822219, 8557:0.14997452` | doc 8557 joins at width 4 and scores **lower** — the window width, not a gap count, feeds `sloppyWeight` |
| `scoring.phrase.reordered.gammadelta` | `pos:"gamma delta"~2` | `8558:0.52346647` | the deliberately-reversed pair, on terms nothing else touches |
| `scoring.phrase.repeat.slop2` | `pos:"alpha alpha"~2` | `8556:0.32424992` | `rptGroups`: the single-`alpha` documents must not match at any budget |
| `scoring.multiphrase.reordered` | `MultiPhraseQuery[beta][alpha]~2` | `8555:0.22822219` | the `MultiPhraseQuery` entry into the same matcher |

All six matched bit for bit on the first run. **Verified to fail against the
unfixed code**: with the in-order matcher restored, both new tests report
`got [], Lucene [(8555, 0.22822219)]`.

Sixteen unit tests in the module cover the transposition boundary, the
three-term window, repeats (including a group that runs out of positions and a
collision created mid-walk, which is the `advanceRpts` path), the multi-term
union, and a brute-force cross-check that slop 0 agrees with exact adjacency for
every pair of positions in `0..8`.

### F-8 `[INTENTIONAL, recorded]` the highlighter's enumeration stays in-order

`highlighter::phrase_match_offsets` walks one greedy in-order alignment per
start. With F-4 fixed it is now the only in-order-only phrase path in the crate,
so a reordered occurrence is **scored but not highlighted** — the two halves of
one query disagree about what matched. Not fixed here: it needs the matcher to
grow `startPosition`/`endPosition`/`startOffset`/`endOffset` (Java has them;
this port skipped them because nothing consumed them), which is a second
contained piece of work rather than a line in this one. Recorded on the function
and as **new ledger item 7b**.

### Verdict

Swept clean against `SloppyPhraseMatcher`'s matching and scoring. Line coverage
99.0%. The `Matches`-API accessors are the open half (7b).

---

## `crates/lucene-search/src/lib.rs`

Java: `BlockMaxConjunctionScorer`, `ConjunctionScorer`, `PhraseScorer`,
`TopScoreDocCollector.updateMinCompetitiveScore`.

| Rust item | Java | Verdict |
|---|---|---|
| `try_conjunction_lazy`'s pruning branch | `BlockMaxConjunctionScorer` + `setMinCompetitiveScore` | now identical (F-7) |
| `search_phrase_query` / `search_phrase_query_scored_with_stats` | `PhraseWeight`/`PhraseScorer` | now dispatch to `sloppy_phrase` |
| `multi_phrase_hits` | `MultiPhraseWeight` | same |
| `phrase_matches_in_doc_sloppy`, `phrase_freq_sloppy` | — | **deleted**: they were the in-order approximation |

### F-7 `[PERF]` a filter-only query could not prune under a top-`n` collector

**Java.** `TopScoreDocCollector.updateMinCompetitiveScore` publishes
`Math.nextUp(topScore)`; for a queue whose bottom is `0f` that is
`Float.MIN_VALUE`, and the block-max scorers skip a block whose maximum is
`< minCompetitiveScore`. A filter-only query's blocks have maximum `0`, so
`0 < 1.4e-45` — **Java prunes it**.

**We did.** c11 switched pruning off for the shape
(`let prunable = legs.iter().any(Leg::scoring)`), reading this port's
`bound <= threshold` as "skipping on a tie, dropping documents Lucene keeps".

**The reasoning was wrong, not merely cautious.** Java's `bound < nextUp(bottom)`
and this port's `bound <= bottom` are the same predicate for every finite
bottom, because `nextUp` is the immediate successor. It is also correct
independently of Java: documents arrive in ascending doc order, `HitQueue` gives
a score tie to the *lower* doc id, and the queue is full — so nothing in a
skipped block could have displaced a kept hit.

**Fixed** by deleting `prunable` and its filter.

**The recorded blocker was half false.** c11 said the fixture segment was too
small to produce a differential test. `body` is: its two-document postings live
in the vint tail, where `current_block_last_doc_id()` is still `-1`, so there is
no decoded block to bound and the guard's removal is invisible — a first attempt
at the unit test on `body` failed for exactly that reason and is now documented
in the test. The *same fixture's* `big` field has 300 documents and fills a
top-20 queue fifteen times over.

**Measured** (`benches/filter_vs_must.rs`, criterion, `benchmarks/.corpus/merged`,
top-50). Both sides re-run in one session, since c11's numbers were taken on a
differently-loaded host:

| shape | before | after |
|---|---|---|
| `#body:t0 #body:t1` | 44.20 ms | **2.00 ms** (**22x**) |
| `+body:t0 +body:t1` (control) | 7.02 ms | 7.02 ms |
| `#body:t0 #body:t1`, pruning forbidden | 41.57 ms | 41.85 ms |

The filter-only form is now the *cheaper* of the two under a top-`n` collector —
which is the relationship c11's own no-pruning row (27% cheaper) predicted and
the top-`n` row contradicted. The exhaustive rows are unchanged, which is the
control: nothing about the matching work moved.

**Tests.** Two real-Lucene entries
(`scoring.boolean.filteronly.big`, `scoring.boolean.filteronly.bigdup` — the
second is the duplicated-clause form `BooleanQuery.rewrite` collapses, so the
port's un-rewritten two-leg execution has ground truth too) asserted bit for
bit, *plus* an assertion that fewer than 300 documents were visited, so the test
cannot pass by not pruning. A unit test compares the pruned run against a
`ScoreMode::COMPLETE` run and asserts the kept hits are identical and the
`TotalHits.Relation` flips. Verified to fail with `prunable` restored.

### Verdict

Swept clean for these paths. Line coverage 96.79%.

---

## `crates/lucene-ffi/src/query.rs`

No Java counterpart (C-ABI glue). Two new exports:

- `ffi_count_term_query` — `IndexSearcher.count(new TermQuery(...))` for one
  segment. Takes the terms-dictionary shortcut and only opens `.doc` when the
  segment has deletions.
- `ffi_search_term_query_scored_after` — `searchAfter`. Its own entry point
  rather than sentinel parameters on the existing one, because a sentinel a
  caller gets subtly wrong is how a paginating caller silently re-reads page 1.

Both wrap in `guard`, validate the out-pointer first, and reject a stale or
mistyped handle — checked by tests, including a `.liv`-attached segment for the
count fallback. Line coverage 97.81%.

---

## Fixtures

Three appenders, all of which **open committed indexes read-only** and rewrite
only their own key prefix. No generator was run, no index regenerated, no
segment id perturbed — the c29 failure mode. `fixtures/segment-ids.txt` is
unchanged.

| program | indexes touched | keys |
|---|---|---|
| `AppendScoringManifest` (extended) | `blocktree_index` | 6 reordered/repeat phrase entries, 2 filter-only entries |
| `AppendCountManifest` (**new**) | `blocktree_index`, `live_docs_index`, `norms_index`, `doc_values_index`, `doc_values_skip_index` | `count.*`, `rewrite.*` |
| `AppendSearchAfterManifest` (**new**) | `blocktree_index`, `multi_segment_scoring_index` | `after.*` |

`AppendCountManifest` needs five indexes because `count` has five arms and no
committed index reaches more than two of them; its class doc says which index
exists for which arm.

---

## What this batch's two false blockers have in common

Items 7 and 23 both carried a "blocked on a fixture" note, and item 29 existed
to schedule building those fixtures. **Neither fixture had to be built.** The
reordered document was added by task #55 for `SpanNearQuery`; the 300-document
field was added for the postings block tests. Both were in the tree, under a
different name, for a different reason, for several batches.

This is c34's "actively misleading entry" class again, with a twist worth
naming: c34's four cases were blockers that *someone removed*. These two were
blockers that were **never quite true** — the entry described the fixture it
wanted in terms of the feature it wanted it for ("a reordered-phrase document",
"a segment large enough"), and nobody re-read the fixture tree asking whether
something already had that shape. The countermeasure is the same as c34's, one
step earlier: when an entry names a *missing artefact* rather than a missing
design, check for the artefact before scheduling its construction.

The third false-blocker was item 6's, and it is a different shape: it named a
missing *abstraction* ("a reader-level query object this port does not have")
where the actual requirement was a struct of six integers. Reading Java's `count`
as a method on a `Weight` made it look like it needed a `Weight`; reading it as
a function of the numbers it touches made it a 60-line module.

---

## What the Tier-2 review changed

The `quality-reviewer` pass found one gating defect and six advisories against
this batch's own diff. All seven are fixed; the two that mattered are worth
recording, because both are the same shape -- **a place where "what the code
does" and "what its comment says Java does" had quietly diverged**, which is the
defect class this whole sweep keeps re-finding.

1. **`count_field_exists_leaf`'s doc-values ladder was presence-based where
   Java's is flag-based, and dropped a `hasDeletions()` gate.** The first draft
   was `skipper.or(points).or(terms)` -- try each proxy, take the first that
   resolves. Java instead *selects* the arm by `FieldInfo` flags
   (`docValuesSkipIndexType() != NONE`, then `getPointDimensionCount() > 0`,
   then `getIndexOptions() != NONE`) and treats a declared-but-unreadable
   structure as a count of **0**, not as "try the next one"; and it gates the
   points and terms proxies on `reader.hasDeletions() == false`, because a
   *doc* count says nothing about which of those documents are live.
   `FieldExistsLeaf` now carries the three flags and `raw_count` is Java's
   ladder verbatim. The answers happened to agree in every case I traced
   (the reconciliation below the ladder absorbs the missing deletions gate), so
   this was a fidelity fix rather than a wrong-answer fix -- but the comment
   asserting flag-based selection over an `.or` chain is exactly how the next
   reader re-derives the wrong Java.

2. **A vector field whose `vector_size` the caller did not supply counted `0`.**
   `raw_count` mapped `None` onto `0`, which the ladder then reads as "no
   document has this field" -- turning a caller's omission into a silently wrong
   count on the one path this module exists to make trustworthy. It is now
   Java's `-1`: go and scan. (Java cannot reach the case at all; it asserts the
   vector values are non-null. This port has to, because
   `SegmentReader::field_exists_leaf` takes the size as a parameter -- the
   reader does not hold the vector values.)

3. **The shared-accumulator fan-out test could not fail.** Its non-vacuity
   assertion (`some leaf saw a threshold`) was satisfied by the *plain* run,
   which shared the same recording buffer -- so an inert `with_shared_max_score`
   still passed. It now records `(leaf, threshold)` pairs into two separate
   buffers and asserts that **leaf 1**, whose own hits top out at 3.0, saw
   `Some(8.0)`, and that it never does without the accumulator. Verified to fail
   with `with_shared_max_score` removed. The first version's flaw is instructive:
   the assertion was true, provable, and about the wrong run.

The other four were smaller: `parity.md` and one test comment still naming the
two deleted functions in the present tense; `points_doc_count` silently reading
a corrupt `.kdm` as "no BKD tree" (kept, with the reason now written down --
an absent proxy costs a scan, which is always a correct answer); a `|`/`&`
precedence in `doc_score_encoder::encode` that was correct but unparenthesised;
and two doc lines left over 100 columns by in-place edits.

**A gate the reviewer proposed and this batch did not build**: for every
`fn`/`struct` a diff *removes*, grep `crates/`, `docs/parity.md` and `PLAN.md`
(excluding `docs/sweep/`, which is an archive) and fail on a hit. That is
precisely the class of finding 4 above, it has no false positives for
`pub(crate)`-and-above symbols, and nothing mechanical catches it today --
`check-parity.py` validates the file *paths* in the Rust column but not
identifiers in the prose. Recorded here rather than added, because it is
repo-wide tooling and belongs with `LEDGER.md` item 27's other two mechanical
gates.

---

## Gate

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`
on both x86_64 and aarch64, `cargo check` on the out-of-workspace benchmark
crate, `scripts/check-arith-allows.py`, `scripts/check-parity.py`,
`scripts/check-java-refs.py`, and `cargo llvm-cov --workspace
--fail-under-lines 95`: **`gate: ok`**, workspace 98.12% lines / 97.58% regions.

Per-file line coverage, every file this batch touched or added:

| file | lines |
|---|---|
| `lucene-search/src/weight_count.rs` | 99.22% |
| `lucene-search/src/sloppy_phrase.rs` | 99.21% |
| `lucene-search/src/collector.rs` | 99.04% |
| `lucene-search/src/highlighter.rs` | 98.57% |
| `lucene-search/src/query.rs` | 98.37% |
| `lucene-ffi/src/query.rs` | 97.81% |
| `lucene-search/src/directory_reader.rs` | 97.81% |
| `lucene-search/src/multi_segment.rs` | 97.22% |
| `lucene-search/src/lib.rs` | 96.79% |
| `lucene-search/src/explain.rs` | 95.50% |

One caveat worth recording, since it is invisible from a green container run:
inside the container `check-parity.py` prints `warning: pinned Lucene tree
missing at $HOME/work/lucene-10.5.0` and **skips its Java-counterpart half**,
because `docker-test.sh` mounts the tree at `/lucene-10.5.0` and that script
looks under `$HOME`. (`check-java-refs.py` looks in both places, so it does run
there — 242 citations verified.) Both were therefore also run on the host, where
the tree is present: `check-parity: ok`, `check-java-refs: ok`. Pre-existing and
not this batch's to fix; noted so the next reader knows which half of
`check-parity` a container gate did not see.

Not committed, per the batch instruction.
