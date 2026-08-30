# b12 — search core

Files swept:

- `crates/lucene-search/src/lib.rs` (7.9k lines)
- `crates/lucene-search/src/query.rs`
- `crates/lucene-search/src/collector.rs`
- `crates/lucene-search/src/similarity.rs`

Java counterparts, all at `/home/tuong/work/lucene/lucene/core/src/java/org/apache/lucene/search/`
unless noted: `IndexSearcher`, `Query`, `Weight`, `Scorer`, `ScorerSupplier`,
`BulkScorer`, `Weight$DefaultBulkScorer`, `TermQuery`, `BooleanQuery`,
`BooleanWeight`, `BooleanScorer`, `Boolean2ScorerSupplier`, `ReqExclScorer`,
`ReqOptSumScorer`, `DisjunctionScorer`, `DisjunctionSumScorer`,
`DisjunctionMaxQuery`, `ConjunctionScorer`, `ConjunctionDISI`, `PhraseQuery`,
`PhraseScorer`, `PhraseMatcher`, `ExactPhraseMatcher`, `SloppyPhraseMatcher`,
`PhraseWeight`, `MultiPhraseQuery`, `ConstantScoreQuery`, `BoostQuery`,
`MatchAllDocsQuery`, `MatchNoDocsQuery`, `TermInSetQuery`, `WANDScorer`,
`MaxScoreCache`, `ImpactsDISI`, `TopScoreDocCollector`, `TopFieldCollector`,
`TopDocs`, `TotalHits`, `HitQueue`, `CollectorManager`, `ScoreMode`,
`DocIdSetIterator`, `TwoPhaseIterator`, and
`search/similarities/{Similarity,BM25Similarity,TFIDFSimilarity,ClassicSimilarity,BooleanSimilarity,LMDirichletSimilarity}.java`,
`search/similarities/IndependenceStandardized.java` (that one lives under
`lucene/misc`, not core).

**The headline result.** Before this batch, *every* scored path in the port was
exactly one ULP away from real Lucene on *every* hit, and no test in the crate
could see it (the one cross-engine scoring test compared to a `1e-4`
tolerance). A new Java-generated fixture (`fixtures/src/AppendScoringManifest.java`,
recording real `IndexSearcher` `TopDocs` scores as `Float.floatToIntBits`) and
a new test suite (`crates/lucene-search/tests/bm25_scoring_fixtures.rs`,
asserting `f32::to_bits` equality) now pin term, boolean, exact-phrase,
sloppy-phrase and multi-phrase scores **bit for bit** against Lucene 10.5.0.
That comparison found four separate defects, one of which (the sloppy phrase
frequency) was a 2x scoring error, not a rounding one. Reviewing the code the
new tests exercised then turned up a fifth, worse one: **a scored phrase query
failed outright — `Corrupted` error, no results — on any segment with
deletions** (F-25).

---

## `crates/lucene-search/src/similarity.rs`

Java counterparts: `search/similarities/BM25Similarity.java` (`idf`,
`avgFieldLength`, `LENGTH_TABLE`, `scorer`, `BM25Scorer.doScore`/`score`/
`explain`/`explainTF`, the constructor validation, `computeQueryTermWeight`),
`util/SmallFloat.java` (`byte4ToInt`), `search/similarities/Similarity.java`.

| Rust fn | Java method | Verdict |
|---|---|---|
| `DEFAULT_K1` / `DEFAULT_B` | `BM25Similarity()` no-arg ctor | identical |
| `Bm25Params` (fields now private) | `BM25Similarity`'s private `k1`/`b` + `getK1()`/`getB()` | identical |
| `Bm25Params::new` (**new**, sole construction path) | that ctor's range checks | now identical (F-6) |
| `Bm25Params::default` | `BM25Similarity()` | identical |
| `decode_norm` | `SmallFloat.byte4ToInt` via `LENGTH_TABLE[i]` | identical |
| `idf` | `BM25Similarity.idf(long, long)` | identical |
| `norm_inverse` (**new**) | `BM25Similarity.scorer`'s `cache[i]` | identical (F-1) |
| `do_score` (**new**) | `BM25Scorer.doScore(float,float)` | identical (F-1) |
| `UNNORMED_NORM_INVERSE` (**new**) | `cache[i]` at the no-norms length | not-in-Java (this port's documented no-norms substitution) |
| `tf_norm` | `BM25Scorer.explainTF`'s tf value, algebraic form | divergent-by-design; scoring no longer routes through it |
| `score` / `score_with_params` | `BM25Scorer.score(freq, encodedNorm)` | now identical (F-1) |
| `max_score_for_impacts` | `MaxScoreCache`'s per-impact max | now identical (F-2) |
| `max_score_for_impacts_unnormed` (**new**) | same, at the no-norms length | not-in-Java |
| `assert_block_pruning_matches_brute_force` | — | test harness, not-in-Java |
| — | `BM25Similarity.avgFieldLength(FieldStats)` | see F-7 (lives in `field_norms.rs`) |
| — | `computeQueryTermWeight` / `k3` | MISSING, F-11 |
| — | `Similarity.computeNorm` / `discountOverlaps` | write side, out of batch |
| — | `TFIDF`/`Classic`/`Boolean`/`LMDirichlet` similarities | MISSING, F-12 |

### F-1 `[CORRECTNESS]` every scored path was one ULP off real Lucene

**Java.** `BM25Scorer.score(freq, encodedNorm)` is
`doScore(freq, cache[norm])` where `doScore` is
`weight - weight / (1f + freq * normInverse)` and
`cache[i] = 1f / (k1 * ((1 - b) + b * LENGTH_TABLE[i] / avgdl))`. Lucene's own
comment says the rewrite exists to guarantee monotonicity in both `freq` and
`norm` without promoting to `double`.

**We did.** `idf(...) * tf_norm(freq, len, avgdl, k1, b)`, i.e.
`weight * freq / (freq + k1 * (1 - b + b*len/avgdl))` — algebraically identical,
and one ULP different in `f32` for essentially every input. Ten call sites in
`lib.rs`, plus both of `explain.rs`'s. (One site, the MAXSCORE term loop, had
already switched to Lucene's form, which is what made the port *internally*
inconsistent as well: the same document scored differently depending on which
entry point found it.)

**Consequence.** Every score this engine returns differs from Lucene's in the
last bit. Score-equality tie-breaks, cross-engine result comparison, and any
consumer diffing OpenSearch scores against a JVM baseline all see it. Measured:
all five initial fixture cases failed bit-comparison by exactly ±1 ULP.

**Fixed.** `similarity::do_score` + `similarity::norm_inverse` added as verbatim
ports; `score`/`score_with_params` now route through them; all eight scoring
sites in `lib.rs` and both in `explain.rs` switched. Boolean clause scoring now
reads `FieldNorms::norm_inverse`'s precomputed `cache[norm]` table instead of
decoding a length and dividing twice — same-or-fewer operations, so this is a
small perf win as well. Proven by
`tests/bm25_scoring_fixtures.rs::{term,boolean,exact_phrase}_..._bit_for_bit`.

### F-2 `[CORRECTNESS]` the MAXSCORE bound and the score it gates used different float expressions

**Java.** `MaxScoreCache` computes block bounds with the *same*
`SimScorer.score` the per-document loop uses.

**We did.** `search_term_query_scored_maxscore_with_stats` scored documents with
`weight - weight/(1+freq*normInverse)` (Lucene's form) but bounded blocks with
`max_score_for_impacts`, which used `idf * tf_norm` (the multiply form). The two
differ by up to one ULP in either direction.

**Consequence.** A latent unsoundness in block-max pruning: whenever the bound
rounded one ULP *below* a real document's score, that document could be skipped
despite belonging in the top-`n`. Never observed in a test — the fixture is too
small for the collector to fill — but it is a wrong-results bug, not a
performance one.

**Fixed** by F-1's unification (`max_score_for_impacts` and
`max_score_for_impacts_unnormed` both call `do_score`), plus a new invariant test
`max_score_for_impacts_is_never_below_a_real_scored_document`.

### F-3 `[PERF]` the `norms == None` impacts bound was open-coded at four call sites

Four separate copies of "when there are no norms, bound with
`UNNORMED_FIELD_LENGTH`, not with the wire norms" in `lib.rs`, each with its own
comment explaining why. The rule is load-bearing (bounding a different formula
than the one scored underestimates the bound), and four copies is four places to
forget it. Folded into `similarity::max_score_for_impacts_unnormed`.

### F-6 `[MISSING]` no BM25 parameter validation

**Java.** `BM25Similarity`'s constructor throws on non-finite or negative `k1`
and on `b` outside `[0, 1]`.

**We did.** `Bm25Params` is a plain struct with public fields; the FFI
`ffi_search_term_query_scored_with_similarity` path could pass anything.

**Consequence.** `b > 1` makes the length-normalization term non-monotonic in
the norm, which invalidates every impacts-derived bound
(`max_score_for_impacts`) — MAXSCORE then silently drops hits. Negative `k1`
can zero the denominator and produce infinities.

**Fixed, and actually closed.** `Bm25Params::new(k1, b) -> Result<Self, String>`
with Lucene's own error messages verbatim, plus
`bm25_params_new_rejects_exactly_what_lucene_rejects`.

The first cut left the fields `pub`, which closed nothing: `Bm25Params { k1, b }`
remained an unchecked back door, and it was the one the FFI entry point — taking
two floats straight off the C ABI from a JVM caller — went through, so the
hazard the constructor documents was still fully open. The fields are now
private with `k1()`/`b()` accessors (`BM25Similarity.getK1()`/`getB()`), making
`new` the only construction path, exactly as Java has a validating constructor
and no setters. `lucene_ffi::query::ffi_search_term_query_scored_with_similarity`
now surfaces a rejection as `FfiStatus::InvalidArgument` carrying Lucene's
message, the way Java throws `IllegalArgumentException`.

### F-11 `[MISSING]` `computeQueryTermWeight` / `k3`

Lucene 10.5 added a `k3` parameter and `Similarity.computeQueryTermWeight(int)`,
used by `BooleanQuery.rewrite`'s `deduplicateClauses` to fold `"a a a"` into one
clause with a recombined boost. Default `k3 = -1` disables it
(`computeQueryTermWeight` returns the raw count). Unported; the only Java caller
is the MUST/SHOULD dedup rewrite this port deliberately does not do (see F-10).
**Recorded, not fixed** — implementing it in isolation would add a knob with no
consumer.

### F-12 `[MISSING]` only one `Similarity` exists

`TFIDFSimilarity`, `ClassicSimilarity`, `BooleanSimilarity`,
`LMDirichletSimilarity`, `IndependenceStandardized` are all unported, and there
is no `Similarity`/`SimScorer` trait to hang them off. The module doc already
declares this ("no speculative polymorphism until a second implementation
exists"), and it is still the right call — but it is a real scope gap, not an
equivalence. **Recorded.** Note that the b12 work makes a future trait easier,
not harder: `do_score`/`norm_inverse`/`idf` are now the three separable pieces a
`SimScorer` would need.

### Verdict

Swept clean for BM25 itself: the formula, the constants, the `SmallFloat` decode
and the 256-entry cache table are now byte-exact against real Lucene output.
Open: F-7 (`avgFieldLength`, owner b13/b15), F-11, F-12.

---

## `crates/lucene-search/src/collector.rs`

Java counterparts: `TopScoreDocCollector.java`, `TopDocsCollector.java`,
`TopFieldCollector.java`, `HitQueue.java`, `TopDocs.java`, `TotalHits.java`,
`ScoreMode.java`, `Collector.java`/`LeafCollector.java`, `Scorable.java`,
`CollectorManager.java`, and (for `CollapsingCollector`) Solr's
`CollapsingQParserPlugin`.

| Rust item | Java | Verdict |
|---|---|---|
| `Collector` | `Collector`+`LeafCollector` (unscored) | divergent-by-design (no per-leaf rebinding; documented) |
| `ScoringCollector` | `LeafCollector` + `Scorable` | divergent-by-design |
| `ScoringCollector::min_competitive_score` | `Scorable.setMinCompetitiveScore` (inverted: pull, not push) | equivalent |
| `ScoringCollector::score_mode` (**new**) | `Collector.scoreMode()` | now present (F-4) |
| `ScoringCollector::pruning_threshold` (**new**) | `createWeight(_, scoreMode, _)`'s decision, made per-call | not-in-Java shape, same effect (F-4) |
| `ScoreMode` (**new**) | `ScoreMode` | identical (5 variants, both predicates) |
| `TotalHits` / `TotalHitsRelation` (**new**) | `TotalHits` / `TotalHits.Relation` | identical incl. `toString` |
| `VecCollector` | — | not-in-Java (test/FFI convenience) |
| `CountCollector` | `TotalHitCountCollector` | equivalent |
| `ScoreDoc` | `ScoreDoc` minus `shardIndex` | divergent-by-design |
| `rank_order` | `HitQueue.lessThan` | identical (score desc, doc id asc) |
| `TopDocsCollector::new` | `TopScoreDocCollectorManager(n, 0)` | equivalent |
| `TopDocsCollector::with_total_hits_threshold` (**new**) | `TopScoreDocCollectorManager(n, threshold)` | identical |
| `TopDocsCollector::total_hits` (**new**) | `TopDocs.totalHits` | identical (F-5) |
| `TopDocsCollector::score_mode` (**new**) | `TopScoreDocCollector.scoreMode()` | identical |
| `TopDocsCollector::collect` | `TopScoreDocCollector`'s leaf `collect` | equivalent (see note) |
| `TopDocsCollector::top_docs` | `TopDocs.scoreDocs` | equivalent, `O(n)` insert (F-13) |
| `TopFieldCollector` | `TopFieldCollector` (one numeric `SortField`) | scoped-down, documented |
| `CollapsingCollector` | Solr `CollapsingTopDocsCollector` | scoped-down, documented |
| — | `ScoreDoc after` / `searchAfter` | MISSING, F-14 |
| — | `MaxScoreAccumulator` (cross-leaf threshold sharing) | MISSING, F-15 |
| — | `CollectorManager.reduce` | lives in `multi_segment.rs` (b13) |

Note on `collect`'s fast reject: Java's is `if (score <= topScore) return;`,
justified by documents arriving in ascending doc-id order. Ours is
`score < worst || (score == worst && doc_id >= worst.doc_id)` — identical for
ascending input and strictly safer otherwise. Left as is.

### F-4 `[MISSING]` `ScoreMode` was not modelled at all

**Java.** `Collector.scoreMode()` returns a `ScoreMode`, which
`IndexSearcher.createWeight` threads into every `Weight`. It decides two things:
`needsScores()` — whether a `PostingsEnum` needs frequencies at all, and
`isExhaustive()` — whether dynamic pruning is legal (it is **not**, in
`COMPLETE`/`COMPLETE_NO_SCORES`, because the collector is promising an exact
`totalHits`).

**We did.** Nothing. Grep for `ScoreMode` across the whole crate returned only
prose in doc comments. Every scored path decoded frequencies unconditionally,
and MAXSCORE block skipping was authorized purely by "is the top-`n` queue
full", with no way for a caller to say "I need every hit counted".

**Consequence.** Two, of different sizes. (a) A caller who wants an exact match
count cannot get one from a scored search — pruning silently makes the count a
lower bound and nothing says so. (b) The `needsScores == false` postings path
does not exist: a filter-only clause decodes every frequency block it will never
read, where `Lucene104PostingsReader` `PForUtil.skip`s them. That half is
blocked on `lucene_codecs::postings` having no `PostingsEnum`-flags plumbing at
all — already a `LEDGER.md` carry-over raised by b5.

**Fixed (the reachable half).** `collector::ScoreMode` with both predicates,
matching Java's `(isExhaustive, needsScores)` table exactly;
`ScoringCollector::score_mode()` defaulting to `Complete` (the conservative
answer, which forbids pruning, so a collector that has not thought about it
cannot accidentally authorize an early exit);
`TopDocsCollector::score_mode()` reproducing Java's
`totalHitsThreshold == Integer.MAX_VALUE ? COMPLETE : TOP_SCORES`.

And — the part that makes it load-bearing rather than decorative —
`ScoringCollector::pruning_threshold()`, which is
`min_competitive_score()` gated on `!score_mode().is_exhaustive()`. All seven
pruning sites in `lib.rs` (the term MAXSCORE loop, the lazy disjunction, the
lazy conjunction, the boolean MAXSCORE loop, and the constant-score early exit)
now ask that instead of `min_competitive_score()` directly. In Java the
decision is made once, at `createWeight(searcher, scoreMode, boost)`, and a
`Weight` built for an exhaustive mode never receives a
`setMinCompetitiveScore` call at all; this port has no `Weight` to make it at,
so it is made in the one accessor every loop goes through. Tests:
`score_mode_predicates_match_javas_enum_table`,
`a_collector_that_says_nothing_gets_the_conservative_default`, and — the one
that matters —
`an_exhaustive_score_mode_collector_disables_every_maxscore_block_skip`, which
uses the existing block-skip counter to prove the gate is live rather than
asserting on results that are identical either way.

The `needs_scores == false` postings path stays open, with its blocker named.

### F-5 `[MISSING]` no `totalHits` and no `TotalHits.Relation`

**Java.** Every `TopDocs` carries `TotalHits(value, relation)`.
`TopScoreDocCollector` increments `totalHits` on every `collect` and flips
`totalHitsRelation` to `GREATER_THAN_OR_EQUAL_TO` the moment it publishes a
min-competitive score, because from then on documents can be skipped uncounted.
`totalHitsThreshold` (default 1000 via `TopScoreDocCollectorManager`) delays
that publication so the count is exact up to the threshold.

**We did.** Nothing: `top_docs().len()` is the *kept* count, capped at `top_n`,
and the collector's own doc comment claimed the port "has no early termination
yet" — untrue since MAXSCORE landed.

**Consequence.** A caller cannot ask how many documents matched, and — worse —
has no way to know that the number it does have is a lower bound. `multi_segment.rs`
already has a comment noting the absence.

**Fixed.** `TotalHits`/`TotalHitsRelation` (including Java's `"1000+ hits"`
`toString`), `TopDocsCollector::total_hits()`, a `total_hits_threshold` with
Java's exact publication rule in `min_competitive_score`, and
`TopDocsCollector::with_total_hits_threshold`. The default threshold stays `0`
(prune as soon as the queue fills), not Java's 1000: that is what every existing
caller already got, and this port's consumers ask for hits rather than counts —
the trade is documented on the constructor. Five tests, including
`a_total_hits_threshold_keeps_the_count_exact_and_delays_pruning` and
`an_infinite_threshold_is_complete_mode_and_never_prunes`.

### F-13 `[PERF]` `HitQueue` is a sorted `Vec`, not a heap

`TopScoreDocCollector` is backed by a `TernaryLongHeap` where this is a
`Vec<ScoreDoc>` kept fully sorted with an `O(n)` insert. Already documented in
the type, and M1.6's finding #O9 measured it: with the fast reject in front, an
insert is rare enough that it does not show in a profile. **Recorded, not
changed** — re-measuring it was not worth the budget against F-1..F-5.

Java also packs `(doc, score)` into one `long` (`DocScoreEncoder`) so the heap is
a `long[]`; our `ScoreDoc` is 8 bytes anyway, so there is nothing to win there.

### F-14 `[MISSING]` no `searchAfter`

`TopScoreDocCollector` takes an `after: ScoreDoc` and skips anything ranking at
or above it — deep pagination. No equivalent here. **Recorded**; it is a
collector-shaped feature with no caller in this port yet.

### F-15 `[MISSING]` no `MaxScoreAccumulator`

Java shares one min-competitive score across concurrently-searched leaves, so a
leaf that starts late inherits a threshold instead of rediscovering it. This
port's concurrent multi-segment search (`multi_segment.rs`, b13) gives each
segment an independent collector. **Recorded**, owner b13 — the mechanism has to
live where the fan-out does.

### Verdict

`ScoreMode` and `TotalHits` now exist and are wired into the one collector that
can act on them. Open: F-13 (measured, not worth fixing), F-14, F-15.

---

## `crates/lucene-search/src/query.rs`

Java counterparts: `TermQuery`, `BooleanQuery` (+`BooleanClause`),
`PhraseQuery`, `MultiPhraseQuery`, `DisjunctionMaxQuery`, `ConstantScoreQuery`,
`BoostQuery`, `MatchAllDocsQuery`, `MatchNoDocsQuery`, `TermInSetQuery`,
`PrefixQuery`, `WildcardQuery`, `FuzzyQuery`, `RegexpQuery`,
`PointRangeQuery`, `queries/spans/Span*Query`.

This file is data definitions plus one algorithm, `BooleanQuery::rewrite`. The
query structs correspond one-to-one with their Java classes (with two
consistently applied scope reductions, both pre-documented: no per-clause
`Occur.FILTER`, and phrase positions implicit rather than explicit).

| Rust item | Java | Verdict |
|---|---|---|
| `Clause` enum (19 variants) | `Query` subclasses reachable as clauses | divergent-by-design (closed enum, not `Query` polymorphism) |
| `MultiPhraseQuery` (**new**) | `MultiPhraseQuery` | now present (F-9) |
| `Clause::rewrite` | `Query.rewrite` recursion | equivalent |
| `BooleanQuery::rewrite` | `BooleanQuery.rewrite(IndexSearcher)` | 4 of 12 rules → now 11 of 12 (F-8) |
| `BooleanQuery::{with_must,with_should,with_must_not,with_minimum_should_match}` | `BooleanQuery.Builder` | equivalent |
| `PhraseQuery::{new,with_slop}` | `PhraseQuery.Builder` | scoped (implicit positions) |
| — | `Occur.FILTER` | MISSING, F-16 |
| — | `BooleanQuery.Builder`'s 1024-clause limit | MISSING, F-17 |

### F-8 `[MISSING]` `BooleanQuery.rewrite` was missing seven rules, including both of Java's `MatchNoDocsQuery` collapses

**Java.** `BooleanQuery.rewrite` runs twelve rules in a fixed order. This port
had: single-clause unwrap, recursion, MUST_NOT dedup. It was missing, among the
ones reachable without `Occur.FILTER`:

0. **Java's own first two rules**: `MatchNoDocsQuery("empty BooleanQuery")` for
   a query with no clauses, and `MatchNoDocsQuery("pure negative BooleanQuery")`
   for one whose every clause is `MUST_NOT`. These were recorded as
   *deliberately* skipped, on the stated grounds that the port had "no separate
   `MatchNoDocsQuery`-equivalent `Clause` variant to rewrite *to*" and that the
   executor reached the same matching outcome anyway. The first half of that
   justification had already expired — `Clause::MatchNoDocs` exists — and the
   rest of this finding's own fixes all `return Clause::MatchNoDocs(..)`, which
   made keeping it indefensible. Both implemented, with Java's two distinct
   reason strings; the two tests that asserted the old structural no-op now
   assert Java's behaviour, and the doc paragraph that carried the dead
   justification is rewritten to say what changed and why.

   One wrinkle worth recording: Java tests these *before* recursing and reaches
   the post-recursion case on `IndexSearcher.rewrite`'s next pass (it loops to a
   fixpoint). This port's `rewrite` is a single bottom-up pass, so the test runs
   after the `MatchNoDocsQuery` drops instead — which reaches Java's fixpoint in
   one go, since a query whose only `SHOULD` clause rewrote to
   `MatchNoDocsQuery` *is* a pure negative query once that clause is gone.
   `rewrite_of_a_query_left_pure_negative_by_its_own_recursion_is_match_no_docs`
   is the test for exactly that.

1. **`MatchNoDocsQuery` clause short-circuit.** A `MUST` clause that is
   `MatchNoDocsQuery` makes the whole query `MatchNoDocsQuery`; a `SHOULD` or
   `MUST_NOT` one is dropped ("the clause can be safely ignored").
2. **Required ∧ excluded.** The same query in `MUST` and `MUST_NOT` →
   `MatchNoDocsQuery("FILTER or MUST clause also in MUST_NOT")`. Likewise
   `MUST_NOT` containing `MatchAllDocsQuery`.
3. **Nested pure-disjunction flattening**, when `minimumNumberShouldMatch <= 1`
   — Java's comment: "this is important for block-max WAND to perform well".
4. **Required-clause inlining**: a `MUST` clause that is itself a `BooleanQuery`
   with no `SHOULD` list has its `MUST`/`MUST_NOT` clauses lifted into the
   parent — "helps run filtered conjunctive queries more efficiently by
   providing all clauses to the block-max AND scorer".
5. **`SHOULD` count vs `minimumNumberShouldMatch`**: fewer shoulds than the
   minimum → `MatchNoDocsQuery`; exactly as many → all shoulds become `MUST`.

**Consequence.** All five are semantics-preserving, so nothing produced wrong
answers — the executor already reaches the same matched set for each. What was
lost is (a) *reportability*: `rewrite()` returned a query that looked
satisfiable when it provably matches nothing, and (b) the structural
simplifications that exist precisely so one conjunction/disjunction scorer sees
every clause. Rules 3 and 4 are the ones Java annotates as pruning-critical, and
this port's `try_conjunction_lazy`/`try_disjunction_lazy` fast paths bail out on
a nested `Clause::Boolean` — so an un-flattened query falls to the slow general
path for exactly the shape the rewrite exists to fix.

**Fixed.** All seven implemented in `BooleanQuery::rewrite`, in Java's order
(the should-count test must run *after* flattening, as Java's own comment
requires). Five existing tests encoded the old behaviour and now assert Java's,
renamed to say what they check.

Implementing Java's rules 1/2 also killed a guard: the required-clause inlining
carried an extra `!inner.must.is_empty()` condition that Java does not have
(Java `assert`s the inner query is not a pure negation, "because the inner
BooleanQuery would have first rewritten to a MatchNoDocsQuery if it only had
prohibited clauses"). That assertion now holds here too, so the guard is
provably dead and was removed rather than left as a comforting no-op.

**Tests.** Ten new unit tests, one per rule plus the negative cases that are
where these rules actually go wrong: the `MUST` short-circuit preserving the
offending clause's own reason string, `SHOULD`/`MUST_NOT` `MatchNoDocs` clauses
being dropped, flattening firing, flattening *declining* for each of the three
ways `isPureDisjunction()` can fail, flattening declining when the outer
minimum exceeds one, both halves of the required-and-excluded rule with their
exact Java reason strings, the near-miss that must *not* trigger it
(`+body:cat -body:dog`), and the all-`SHOULD`-required conversion.

**And a doc claim narrowed.** The function's contract said every rule was
"proven to change neither the matched-doc set nor per-doc scores", citing the
`rewrite_produces_identical_scored_results_*` fixture tests. Those cover rules
1-5 and 8. Rules 6, 7 and 9 *restructure the clause list* `clause_scores` folds
over, and `f32` addition is not associative, so bit-identical scores are not
something they can be argued into — they have to be measured. New fixture test
`rewrite_flattening_and_inlining_preserve_scores_bit_for_bit` measures all
three on the real segment (asserting the rewrite was not a no-op first, so it
cannot pass by comparing a query to itself); the matched set is asserted as
guaranteed, the scores as recorded-and-pinned. They do hold on this fixture.
The doc now says which half is which.

Still not implemented, deliberately: **MUST/SHOULD deduplication**, for the
reason already recorded in the function's doc comment (Java folds duplicates
using `Similarity.computeQueryTermWeight`, which this port has no
`IndexSearcher` in scope to call — a naive dedup would be a different rewrite
wearing the same name). The `FILTER`-specific rules are unreachable (F-16).

### F-9 `[MISSING]` `MultiPhraseQuery` was not ported

**Java.** `MultiPhraseQuery` is the query a synonym filter or a prefix-expanded
last word produces: every position accepts a *set* of terms, merged at match
time by `UnionFullPostingsEnum`. `MultiPhraseWeight` collects `TermStats` for
**every** term of **every** position and hands them all to
`Similarity.scorer`, so the idf is the sum over all of them.

**We did.** Nothing — no type, no `Clause` variant.

**Fixed.** `query::MultiPhraseQuery` + `Clause::MultiPhrase`,
`search_multi_phrase_query`, `search_multi_phrase_query_scored`
(`_with_stats`), arms in `resolve_clause_docs`/`clause_scores`/`explain_clause`.
Matching and scoring run one shared implementation so they cannot disagree about
which documents match — the two-pass defect finding #O15/#O16 hit on the phrase
path. Degenerate shapes follow `MultiPhraseQuery.rewrite`: empty `term_arrays`
matches nothing, a position whose whole alternative set is absent matches
nothing.

**The single-position case is where this bites.** `MultiPhraseQuery.rewrite`
turns a one-position multi-phrase into a `BooleanQuery` of `SHOULD`
`TermQuery`s, which scores each alternative with its **own** idf and its **own**
frequency and sums them — *not* one summed idf against a merged frequency, which
is what a one-slot phrase over the union would give. The first implementation
here got that wrong; the recorded real-Lucene scores
(`scoring.multiphrase.single`) caught it. That is exactly the class of mistake
the differential-testing skill exists for.

**And the merged position list must not be deduplicated**, which is the second
thing here that looks obviously right and is not. `UnionPostingsEnum.freq()`
drains every sub's positions into a `PositionsQueue` and calls `sort()` — there
is no dedup step — so a position reached by two alternatives (the same term
listed twice, or two terms an analyzer stacked at one position, which is the
*normal* synonym case) is yielded twice and the alignment is counted twice. The
first implementation here deduplicated. Real Lucene scores
`pos:"(alpha alpha) beta"` at **0.87906057** on this fixture; the deduplicated
form gives **0.6393168**. Recorded as `scoring.multiphrase.dup` specifically to
pin it.

Five cross-engine tests, all bit-for-bit: `union` (alternatives at one
position), `bothslots` (alternatives at both, exercising the summed idf),
`single` (the boolean rewrite), `dup` (the no-dedup rule), `slop2` (the sloppy
matcher through this path), plus an unscored/scored doc-set agreement test, a
nested-in-`BooleanQuery` test, a degenerate-shapes test, and a deletions test
(F-25).

Scope carried over honestly: implicit positions only (no
`Builder.add(Term[], int position)`), and `slop > 0` inherits the in-order-only
sloppy restriction (F-19).

### F-16 `[MISSING]` no `Occur.FILTER`

Java's `BooleanClause.Occur` has four values; this port has three. `FILTER` is
"required, but does not score", and it drives several rewrite rules (dedup
against `MUST`, `MatchAllDocsQuery` removal, `MUST`→`FILTER` conversion) plus
`BooleanWeight`'s decision to build a non-scoring `Weight`. The nearest thing
here is wrapping a clause in `ConstantScoreQuery`, which is not the same (it
still contributes a score). **Recorded, not fixed**: adding a fourth clause list
touches `matched_boolean_docs`, `clause_scores`, both lazy paths, `explain`,
`query_parser` and `multi_segment`'s stats walk — a milestone-sized change, and
it wants `ScoreMode::CompleteNoScores`'s postings half (F-4) to be worth
anything.

### F-17 `[MISSING]` no `maxClauseCount`

`BooleanQuery.Builder.add` throws `TooManyClauses` past 1024 (also enforced by
the multi-term rewrites). This port has no limit, so a prefix query expanding to
a million terms builds a million-clause query. `stream_constant_score_clause`
mitigates the memory shape for the constant-score family, but the guard itself
is absent. **Recorded** — it is a denial-of-service guard, and the FFI boundary
(b15) is the right place to decide the policy.

### Verdict

`BooleanQuery.rewrite` now implements every rule reachable without
`Occur.FILTER`, including both of Java's `MatchNoDocsQuery` collapses that were
previously skipped on a justification that had expired.
`MultiPhraseQuery` ported and pinned against real Lucene. Open: F-16, F-17.

---

## `crates/lucene-search/src/lib.rs`

Java counterparts: `IndexSearcher`, `Weight`/`Scorer`/`ScorerSupplier`/
`BulkScorer`, `TermWeight`/`TermScorer`, `BooleanWeight`/`BooleanScorer`/
`Boolean2ScorerSupplier`/`ReqExclScorer`/`ReqOptSumScorer`,
`ConjunctionScorer`/`ConjunctionDISI`, `DisjunctionScorer`/
`DisjunctionSumScorer`, `DisjunctionMaxScorer`, `PhraseScorer`/`PhraseMatcher`/
`ExactPhraseMatcher`/`SloppyPhraseMatcher`, `MultiPhraseQuery`,
`ConstantScoreQuery`/`BoostQuery`/`MatchAllDocsQuery`/`MatchNoDocsQuery`/
`TermInSetQuery`, `WANDScorer`/`MaxScoreCache`/`ImpactsDISI`,
`DocIdSetIterator`, `TwoPhaseIterator`.

The structural correspondence is stated once because it governs the whole file:
**this port has no `Weight`/`Scorer`/`ScorerSupplier`/`BulkScorer` hierarchy.**
Each `(query shape, eager|lazy, scored|unscored)` combination is a free function.
That is a deliberate, pre-documented divergence (`INTENTIONAL`), and the M1.6
sweep's measurements support it — the monomorphised lazy loops are where this
port is closest to Lucene's throughput. It does cost the two things the
hierarchy buys: uniform composition (hence the `Clause` enum and its
per-variant arms) and `ScorerSupplier.cost()`-driven clause ordering (F-20).

| Rust fn (selection) | Java | Verdict |
|---|---|---|
| `search_term_query` | `TermWeight.scorer` + `DefaultBulkScorer` | equivalent, eager |
| `search_term_query_scored(_with_stats)` | `TermScorer` + `TopScoreDocCollector` | now bit-exact (F-1) |
| `search_term_query_scored_with_similarity` | same, custom `BM25Similarity` | equivalent |
| `search_term_query_scored_maxscore(_with_stats)` | `TermScorer`+`ImpactsDISI`+`MaxScoreCache` | equivalent; `advanceShallow`, level-0/1 skip, `upTo` invalidation all present |
| `term_doc_ids`/`term_doc_freqs`/`term_doc_scores` | `PostingsEnum` walk | eager, documented |
| `search_boolean_query` | `BooleanWeight.bulkScorer` | equivalent |
| `matched_boolean_docs` | `Boolean2ScorerSupplier.get` | equivalent set algebra |
| `resolve_clause_docs` | per-clause `Weight.scorer` | equivalent |
| `should_match_counts` | `BooleanScorer`'s `minShouldMatch` counting | equivalent |
| `clause_scores` | `BooleanWeight`'s additive combination | now bit-exact (F-1) |
| `try_conjunction_lazy` | `ConjunctionScorer`/`ConjunctionDISI` leapfrog | equivalent |
| `try_disjunction_lazy` | `DisjunctionSumScorer` + `ImpactsDISI` | equivalent; no WAND partition (documented, five measured reverts) |
| `search_boolean_query_scored_maxscore(_with_stats)` | `WANDScorer`, block-max half only | scoped, documented |
| `dismax_scores` | `DisjunctionMaxScorer` | identical formula |
| `phrase_matches_in_doc` / `phrase_freq_exact_impl` (**new**) | `ExactPhraseMatcher.nextMatch` | now the same merge walk (F-21) |
| `phrase_freq_exact` | `PhraseScorer.score`'s exact freq | identical |
| `phrase_matches_in_doc_sloppy` | `SloppyPhraseMatcher.nextMatch` | scoped: in-order only (F-19) |
| `phrase_freq_sloppy` (**new**) | `PhraseScorer.score` + `sloppyWeight()` | now correct (F-18) |
| `search_phrase_query(_scored)(_with_stats)` | `PhraseWeight`/`PhraseScorer` | now bit-exact |
| `search_multi_phrase_query*` (**new**) | `MultiPhraseWeight` | new (F-9) |
| `span_*` | `queries/spans/*` | scoped, flat 1.0 scores, documented |
| `match_all_doc_ids`, `term_in_set_doc_ids`, `points_range_doc_ids`, `prefix/wildcard/fuzzy/regexp_doc_ids` | the corresponding `Weight`s | scoped; constant-score, documented |
| `CollectionStats`/`GlobalStats` | `CollectionStatistics`/`TermStatistics` | equivalent (map, not object) |
| — | `TwoPhaseIterator` | MISSING, F-20 |
| — | `BooleanScorer` (window/bucket bulk OR) | MISSING, F-22 |
| — | `ScorerSupplier.cost()` | MISSING, part of F-20 |
| — | `Weight.count(LeafReaderContext)` | MISSING, F-23 |

### F-18 `[CORRECTNESS]` a sloppy phrase scored every match as frequency 1

**Java.** `PhraseScorer.score()` is
`freq = matcher.sloppyWeight(); while (matcher.nextMatch()) freq += matcher.sloppyWeight();`
and `SloppyPhraseMatcher.sloppyWeight()` is `1f / (1f + matchLength)`, where
`matchLength` is that alignment's total slack. `ExactPhraseMatcher.sloppyWeight()`
returns `1`, which is why the exact case is a plain match count.

**We did.**
`if phrase_matches_in_doc_sloppy(&term_positions, query.slop) { 1 } else { 0 }`
— a boolean, promoted to a frequency of 1.

**Consequence.** Two errors at once. A loosely-matching document scored as
highly as a tightly-matching one, and a document with several sloppy occurrences
scored as if it had one. Measured on the fixture: real Lucene scores
`pos:"alpha beta"~2` at **0.4771918** for doc 8555 (adjacent, `matchLength 0`,
weight 1) and **0.22822219** for doc 8557 (two positions apart, `matchLength 2`,
weight 1/3). This port gave both **0.4771918** — a 2.09x error on doc 8557, and
the wrong relative ranking whenever a query mixes tight and loose matches. It
had gone unnoticed because every sloppy test in the crate asserted *matching*,
and the one scored sloppy test asserted only that explain agreed with search.

**Fixed.** `phrase_freq_sloppy` sums `1/(1 + matchLength)` over every starting
position that admits an in-order alignment, taking that start's minimum
achievable `matchLength` (the greedy scan is optimal for a fixed start). Wired
into `search_phrase_query_scored_with_stats`, `explain_phrase`, and the new
multi-phrase path. For `slop == 0` it provably reduces to `phrase_freq_exact`'s
count, pinned by
`sloppy_phrase_freq_at_slop_zero_equals_the_exact_count`; six unit tests plus
the cross-engine `sloppy_phrase_scores_match_real_lucene_bit_for_bit`.

Residual, stated precisely: real `SloppyPhraseMatcher` enumerates matches by
repeatedly advancing whichever `PhrasePositions` is minimal, which also admits
**reordered** terms. Our enumeration is one match per first-term start position,
in order. Same restriction the matcher already carried and documented (F-19),
now inherited by the frequency.

### F-25 `[CORRECTNESS]` a scored phrase query failed outright on any segment with deletions

**Java.** `PhraseWeight`/`PhraseScorer` iterate the terms' `PostingsEnum`s and
apply `liveDocs` at the collector; deletions never touch the position stream.

**We did.** `search_phrase_query_scored_with_stats`'s "documents first,
positions second" optimisation (finding #O15's fix) built each term's doc and
**frequency** lists with deleted documents removed, then handed that frequency
list to `positions_for_docs`. But `read_positions_for_docs` indexes the wire
position stream by a *running frequency sum* and validates that the total equals
the term's `totalTermFreq` — a deleted document still occupies its slot in that
stream. So the check fires:

```
Err(BlockTree(Postings(Store(Corrupted(
  "sum of per-doc freqs disagrees with total_term_freq")))))
```

**Consequence.** Not a wrong score — a **hard error**. Every scored multi-term
phrase query against a segment with even one deletion in a phrase term's
postings fails. The whole existing phrase test suite passes `live_docs: None`
or exercises the *unscored* path (`term_doc_positions`, which reads all
positions and filters afterwards, and is correct), so nothing caught it. Found
while reviewing `positions_for_docs`'s contract for the new multi-phrase code,
not by a test — the test came after.

**Fixed.** Both the phrase and the new multi-phrase paths now keep each term's
doc and frequency lists unfiltered and apply `live_docs` to the *candidate*
list instead, which is where the deletion belongs: a deleted document is not a
hit, but it still occupies its slot in the term's postings. Pinned by
`scored_phrase_query_with_deletions_still_works`, which deletes a document that
genuinely contributes to `alpha`'s postings and asserts both that the search
succeeds and that it returns the right hit.

### F-19 `[MISSING]` sloppy phrase matching is in-order only

Pre-existing and pre-documented: `SloppyPhraseMatcher` allows term reordering
within the slop budget (`"quick fox"` matching `"fox ... quick"` at high enough
slop) via a priority queue over `PhrasePositions` with repeat handling
(`hasRpts`/`advanceRpts`). This port requires `p_0 < p_1 < ... < p_{n-1}`.
**Recorded, not fixed** — the algorithm is genuinely intricate (repeat-group
detection, `UnionIterator` over repeats, a `captureLead` protocol) and
re-deriving it without a fixture that exercises reordering would be guessing.
The right next step is a fixture: `GenBlockTree` has no reordered-phrase
document, so there is nothing to check an implementation against today. Noted as
the concrete blocker rather than as difficulty.

### F-20 `[MISSING]` no `TwoPhaseIterator`, and therefore no cost model

**Java.** `TwoPhaseIterator` splits a scorer into a cheap `approximation()` and
an expensive `matches()` verification, with `matchCost()` in "estimated number
of simple operations". `ConjunctionDISI` uses it to run all approximations to
agreement *first* and only then verify, cheapest-verification-first;
`PhraseMatcher.getMatchCost` feeds it (`ExactPhraseMatcher` and
`SloppyPhraseMatcher` each compute one from their posting lists);
`PointRangeQuery`, `IndexOrDocValuesQuery` and every doc-values query are built
on it. `ScorerSupplier.cost()` is the sibling that orders clauses by posting-list
length.

**We do.** Nothing of either. `try_conjunction_lazy` leapfrogs the clauses in
the order the query lists them, and phrase/points/doc-values clauses are fully
resolved to a doc-id set before intersection.

**Consequence.** A conjunction of a cheap term and an expensive phrase verifies
the phrase on every candidate the term produces, rather than intersecting the
phrase's *approximation* (its terms' conjunction) first. This port partly
sidesteps it — `search_phrase_query_scored_with_stats` already intersects doc
lists before decoding positions, which is the same idea hard-coded for one query
shape — but there is no general mechanism, so every new expensive clause type
has to rediscover it. It also means clause order is the caller's problem: a
`BooleanQuery` listing its rarest term last leapfrogs from the most common one.

**Recorded, not fixed.** This is a design-level piece: it needs a `Scorer`-shaped
abstraction to hang `approximation()`/`matches()`/`matchCost()` off, which is
precisely the hierarchy this port deliberately does not have. It should be
weighed together with F-16 and F-22 as one "introduce a real scorer abstraction"
milestone, not bolted on. The cheap partial step — ordering conjunction clauses
by `docFreq` ascending, which is `ScorerSupplier.cost()`'s main effect — is
contained and worth doing in b13's `try_conjunction_lazy` work.

### F-21 `[PERF]` exact phrase matching binary-searched where Lucene leapfrogs — **fixed, 5.0-6.8x**

**Java.** `ExactPhraseMatcher.nextMatch` advances all of the phrase's
`PostingsEnum`s together in one merge pass: `O(sum of list lengths)`.

**We did.** `phrase_matches_in_doc`/`phrase_freq_exact` iterated every position
of the first term and `binary_search`ed each subsequent term's list for
`p0 + i`: `O(|p0| · (n-1) · log|list|)`.

The exponent is not the interesting part, and this project has been burned by
assuming it is — M1.6 found `LazyDocsCursor::advance`'s binary search to be the
*cause* of four separate failed pruning attempts, because an unpredictable
branch cost more than the linear scan it replaced. So this was benched rather
than assumed (`crates/lucene-search/benches/phrase_freq.rs`, which implements
both forms side by side and asserts they agree before timing them):

| shape | binary search | leapfrog | |
|---|---|---|---|
| 8 positions, 2 terms | 17.5 ns | 18.7 ns | 0.94x |
| 4096 positions, 2 terms, 50% hit | 26.76 µs | 4.88 µs | **5.49x** |
| 4096 positions, 2 terms, no hit | 28.96 µs | 5.74 µs | **5.04x** |
| 4096 positions, 2 terms, all hit | 29.69 µs | 5.80 µs | **5.12x** |
| 4096 positions, 3 terms, 50% hit | 39.60 µs | 5.80 µs | **6.83x** |

Decisive, and it gets *better* with more phrase terms (the merge pass is
`O(sum)` where the binary-search form multiplies by `n`). The 6.8% loss on an
8-position list is the only regression and is 1.2 ns.

**Fixed.** Both functions now share one `phrase_freq_exact_impl` doing
Lucene's merge walk. The legality argument is that `p0` ascends, so every
target `p0 + i` ascends, so no cursor rewinds; an exhausted cursor breaks the
whole scan rather than just this `p0`, since every later target is strictly
larger. The cursors live in a 32-slot stack array with a heap spill past that —
the walk runs once per candidate *document*, and a `Vec` there would be one
allocation per document, the exact shape finding #O15 spent a milestone
removing from the position stream.

Two new tests beyond the existing phrase suite: the heap-spill branch (a
37-term phrase, which nothing else in the crate reaches), and 800 randomized
shapes compared against an independent `O(n²)` definition — the early exits are
the part most likely to be subtly wrong and no fixture exercises them densely.

The end-to-end effect on `scripts/bench-compare.sh`'s `q16`/`q17` is not
claimed here: those are at 0.31-0.38x for reasons finding #O15 localises to
position *decode*, upstream of this matcher.

### F-22 `[MISSING]` no `BooleanScorer` bulk-OR

**Java.** For a pure disjunction with more than one clause and no
`minShouldMatch`, `BooleanWeight.bulkScorer` picks `BooleanScorer`, not the
document-at-a-time `DisjunctionSumScorer`: it processes a 2048-document window
at a time into 1024 score buckets (`Bucket[] buckets`, `int[] docIDs`), so each
clause is iterated in long runs with no priority-queue churn, then the window's
matches are emitted in doc order. That is a large constant-factor win for
high-cardinality disjunctions — the `or t0 t1 t2 t3` shape which this port's own
benchmark has at 0.26x.

**We do.** `try_disjunction_lazy` is document-at-a-time with a per-leg cursor
array, and the eager path builds a `HashMap<i32, f32>` score map.

**Consequence.** Reasoned rather than measured, because implementing it to
measure it is the whole task: the win is in memory access pattern, not
complexity — same `O(total postings)`, but each clause is walked in a long
contiguous run rather than interleaved, and the "which clause is next" decision
disappears inside a window. M1.6's six failed pruning attempts all concluded
that this port's remaining gap "has to come from not reaching the documents at
all, at a coarser granularity than a document" — a bucket window *is* that
coarser granularity, and it is the one mechanism in Lucene's disjunction path
this port has never tried. **Recorded as the highest-value untried item for the
boolean paths**, ahead of another WAND variant.

### F-23 `[MISSING]` no `Weight.count(LeafReaderContext)`

Java lets a `Weight` answer "how many documents match" without iterating — a
`TermQuery` returns `docFreq` when there are no deletions, `MatchAllDocsQuery`
returns `numDocs`. `IndexSearcher.count` uses it. This port's `CountCollector`
always iterates. Cheap to add for `Term`/`MatchAllDocs`/`MatchNoDocs`;
**recorded**, and it belongs with b13's `multi_segment` count path so the
no-deletions precondition is checked once.

### F-24 `[PERF]` clause scoring allocates a `HashMap` per clause

`clause_scores` returns `HashMap<i32, f32>` per clause and the caller merges
them, where `BooleanScorer`/`ReqOptSumScorer` accumulate into one running score
per document with no map at all. Only the general (non-lazy) path is affected —
the lazy conjunction/disjunction paths, which is where the benchmark's boolean
queries go, already accumulate into a float. **Recorded**: the general path
exists for shapes the lazy paths decline (nested booleans, mixed clause kinds),
and rewriting it into a doc-at-a-time merge is b13-sized work that overlaps with
F-22.

### Verdict

Three `CORRECTNESS` fixes (F-1 scoring form, F-2 bound/score mismatch, F-25
scored phrase queries erroring on any segment with deletions) and two `MISSING`
fixes (F-9 `MultiPhraseQuery`, F-18 sloppy frequency) landed with bit-for-bit
cross-engine tests. One `PERF` fix
landed with a microbenchmark (F-21, 5.0-6.8x on the exact-phrase alignment
walk). Open, in the order they should be taken: F-22 (`BooleanScorer` bulk OR —
the untried mechanism), F-20 (`TwoPhaseIterator` + cost model — needs a scorer
abstraction, take with F-16), F-19 (sloppy reordering — needs a fixture first),
F-23, F-24.

---

## `[INTENTIONAL]` divergences, restated so they are not re-found

### F-26 no `Weight`/`Scorer`/`ScorerSupplier`/`BulkScorer` hierarchy

Each `(query shape, eager|lazy, scored|unscored)` combination is a free
function, monomorphised over the collector. Pre-documented in `lib.rs`'s module
doc, and M1.6's measurements support it — the monomorphised lazy loops are where
this port comes closest to Lucene's throughput, and its conclusion after six
pruning attempts was that per-document costs here are *lower* than Lucene's.
The bill is F-20 (nowhere to hang `TwoPhaseIterator`) and the per-variant
`Clause` arms; both are recorded rather than hidden.

### F-27 `Clause` is a closed enum, not `Query` polymorphism

19 variants and a `match` per operation, where Java composes arbitrary `Query`
subclasses. Correct for a port with a known query set (`rust-performance`'s
"enums where the closed set allows"), and it is what makes the exhaustiveness
checker catch a missing arm — adding `Clause::MultiPhrase` in this batch
surfaced every site that needed one, in three files. The cost is that a caller
cannot add a query type without editing this crate.

### F-28 `Collector`/`ScoringCollector` are two flat traits with no per-leaf rebinding

Java splits `Collector`/`LeafCollector` so a collector can rebind per-segment
state when `IndexSearcher` moves to the next leaf, and pushes the score in via
`setScorer` + `Scorable.score()`. This port has two traits by arity (`fn(i32)`
and `fn(i32, f32)`), scores passed by value, and `min_competitive_score` pulled
rather than pushed. `ScoreDoc` also drops `shardIndex`. All pre-documented in
`collector.rs`'s module doc. The one thing this shape actually costs is F-15
(no cross-leaf `MaxScoreAccumulator`), since there is no per-leaf rebinding
point to install a shared accumulator at.

---

## Cross-batch findings raised here, owned elsewhere

### F-7 `[CORRECTNESS]` `avgFieldLength` is not Java's, at the constructor every caller uses

`field_norms.rs` (b13) has two constructors. `FieldNorms::from_field_stats` is
Java's `avgFieldLength = sumTotalTermFreq / docCount` exactly. `FieldNorms::open`
— the one used by `lucene-ffi`'s query entry point, `explain.rs`, and every test
— sums each **live** document's **decoded** norm and divides by the count of
live documents with a norm. That differs from Java twice:

1. `SmallFloat.byte4ToInt` is lossy above length 24 and one-directional (the
   decode is the floor of a 4-bit-mantissa bucket), so the sum understates
   `sumTotalTermFreq` by up to ~6% on a field of long documents.
2. Java's `docCount` is `CollectionStatistics.docCount()` — documents that have
   the field, **including deleted ones**. Ours excludes deleted documents.

Both shift `avgdl`, and `avgdl` is in every score's denominator. It does not
show in this batch's bit-for-bit tests because the fixture's documents are 1-3
tokens long, where the decode is exact and there are no deletions.

**This was believed fixed and is not.** `docs/parity.md` has claimed since M1.5
that `FieldNorms::open` "is no longer what the search paths use". That is true
of exactly one caller: `benchmarks/rust-runner`, which is why M1.5's benchmark
recall went to zero mismatches and why the claim looked verified.
`from_field_stats` has **no other caller** anywhere in the workspace —
`lucene-ffi/src/query.rs`'s production entry point and `explain.rs` both still
call `FieldNorms::open`. So the formula the benchmark measures and the formula
the FFI serves are different formulas, and only the benchmark's is Java's. M1.5
measured that the wrong average was 0.1–0.6% off and reordered the top-k on 19
of 20 benchmark queries; that error is still live for every FFI-served search.
The parity row is corrected in this batch.

**Owner: b13** for `field_norms.rs` (make `open` compute Java's quantity, or
delete it in favour of `from_field_stats`), **b15** for the FFI call site. Not
fixed here because the call site is two batches away and `open`'s signature has
no `sumTotalTermFreq`/`docCount` to work from — they have to be plumbed from
`blocktree::FieldTerms`, which already carries them.

### F-10 `[MISSING]` `explain` has no per-position breakdown for multi-phrase

`explain.rs` (b14) gained a `Clause::MultiPhrase` arm in this batch because the
enum requires one. It reports the real score — obtained from the scorer itself,
so the two cannot drift — but not `MultiPhraseWeight.explain`'s per-position
detail. **Owner: b14.**

Also noted for b14: `explain.rs`'s `tfNorm` sub-explanation now matches Java's
`explainTF` (`1 - 1/(1 + freq*normInverse)`) rather than the algebraic form, and
two of its tests were relaxed from exact equality to within-one-ULP on the
`idf * tfNorm == score` identity — because that identity is *false* in Lucene
too, which `BM25Scorer.explain` says in its own comment ("not using 'product of'
since the rewrite that we do in score() introduces a small rounding error that
CheckHits complains about").

### For `LEDGER.md`

- F-7 is not a new gap but a **stale fix**: `docs/parity.md`'s M1.5 claim that
  the search paths no longer use `FieldNorms::open` holds only for the
  benchmark runner. Corrected in this batch; the code fix is b13/b15's.
- `ScoreMode::CompleteNoScores`'s other half — a `needs_scores == false`
  postings path — is now blocked only on the `PostingsEnum`-flags carry-over
  b5 already raised. The search-side half landed here.
- `MaxScoreAccumulator` (F-15) and conjunction clause ordering by `docFreq`
  (the contained part of F-20) are b13 work.

---

## Fixture and test additions

- `fixtures/src/AppendScoringManifest.java` (**new**): opens the committed
  `blocktree_index` read-only through a real `DirectoryReader`/`IndexSearcher`
  and records real Lucene's `TopDocs` for nine queries as both decimal and
  `Float.floatToIntBits`. Same non-regenerating technique as
  `AppendDismaxManifest`, for the same reason (regenerating the segment would
  change the random segment ID that `lucene-ffi`'s tests hardcode). Idempotent.
- `crates/lucene-search/tests/bm25_scoring_fixtures.rs` (**new**, 11 tests): the
  bit-for-bit comparison, plus the two deletion regressions from F-25. This is
  the first test in the crate that can see a one-ULP scoring divergence, and it
  found one on its first run in all five original cases.
- 36 new unit tests across `collector.rs` (ScoreMode/TotalHits/pruning gate, 10),
  `similarity.rs` (validation, do_score/norm_inverse, bound soundness, 4),
  `lib.rs` (sloppy frequency, 6; the leapfrog walk's spill branch and a
  randomized brute-force cross-check, 2; exhaustive-mode pruning gate, 1),
  `query.rs` (rewrite rules, 13 new + 5 rewritten), plus
  `tests/boolean_query_fixtures.rs`'s
  `rewrite_flattening_and_inlining_preserve_scores_bit_for_bit`.

## Benchmarks added

`crates/lucene-search/benches/phrase_freq.rs` (criterion): the exact-phrase
alignment walk, binary search vs merge, five shapes. Settled F-21 in favour of
the merge walk by 5.0-6.8x on realistic position-list lengths. This is the
crate's first criterion bench; `Cargo.toml` gained the `[dev-dependencies]`
and `[[bench]]` sections for it.

## Not in this batch, though it is in the same working tree

The working tree also contains a `FuzzyQuery` `TopTermsBlendedFreqScoringRewrite`
port in `lucene-search/src/lib.rs` (`fuzzy_expanded_terms`, `fuzzy_doc_scores`,
`FuzzyExpansion::blended_doc_freq`, and `clause_scores`' `Clause::Fuzzy` arm).
**It is not b12's work.** It was already present in the working tree when this
batch's first read of `lib.rs` was taken, and its subject — fuzzy term expansion
— belongs to `b8-automata-analysis`, which was `running` throughout. It appears
in `git diff` against `HEAD` only because that batch had not committed. Nothing
here reviewed it, and none of the findings above cover it; it needs its own
pass from whoever owns b8. Flagged rather than silently absorbed, because a
sweep report that claimed it would be claiming untested ground.

## Gate

`cargo fmt --all`, `cargo clippy -p lucene-search --all-targets -- -D warnings`,
`cargo test -p lucene-search` — green for this batch's files.

Note for whoever reads this next: b12 ran concurrently with heavy in-flight
edits to `lucene-codecs`, `lucene-index` and (unexpectedly) six other
`lucene-search` files, so the crate gate went red and green repeatedly for
reasons unrelated to this work — a `FixedBitSet` derive removal, a `NormsEntry`
becoming `Copy`, a `Fragment` gaining fields, a `scratch_probe.rs` left in
`tests/`. Every red was in a file outside this batch and none was touched here;
the greens quoted above are with those files in a compiling state.
