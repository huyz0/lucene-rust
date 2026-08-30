# c11 — `BooleanClause.Occur.FILTER`

Follow-up batch closing b12's **F-16** (`[MISSING]` no `Occur.FILTER`), which
b12 had bundled with F-20 (`TwoPhaseIterator`) as "one milestone, needs a scorer
abstraction". c6 re-assessed and disagreed on the `FILTER` half; c6 was right.

Files swept:

- `crates/lucene-search/src/query.rs` — `BooleanQuery`, `BooleanQuery::rewrite`
- `crates/lucene-search/src/lib.rs` — `matched_boolean_docs`, `clause_scores`,
  `try_conjunction_lazy`, `try_disjunction_lazy`,
  `search_boolean_query_scored_with_stats`,
  `search_boolean_query_scored_maxscore_with_stats`
- `crates/lucene-search/src/explain.rs` — `explain_boolean`, `describe_clause`
- `crates/lucene-search/src/multi_segment.rs` — the global-term-stats walk
- `crates/lucene-search/src/query_parser.rs` — `BooleanQuery` construction
- `fixtures/src/AppendScoringManifest.java`,
  `crates/lucene-search/tests/bm25_scoring_fixtures.rs`
- `crates/lucene-search/benches/filter_vs_must.rs` (**new**)

Java counterparts, all at
`/home/tuong/work/lucene/lucene/core/src/java/org/apache/lucene/search/`:
`BooleanClause.java`, `BooleanQuery.java`, `BooleanWeight.java`,
`BooleanScorerSupplier.java`, `ConjunctionScorer.java`, `FilterScorer.java`,
`TermQuery.java` (`TermWeight`'s `needsScores` branch),
`queryparser/.../QueryParserBase.java`.

`TwoPhaseIterator`/the cost model (F-20) was **not** started, per the brief and
c6's assessment: it needs every clause expressed as
`(approximation, matches(), matchCost())`, i.e. b12's per-shape free functions
turned into a scorer enum. That is a milestone.

---

## `crates/lucene-search/src/query.rs`

Java: `BooleanClause.java`, `BooleanQuery.java`.

| Rust item | Java | Verdict |
|---|---|---|
| `BooleanQuery.filter` (**new**) | `clauseSets.get(Occur.FILTER)` | now present |
| `BooleanQuery::with_filter` (**new**) | `Builder.add(q, Occur.FILTER)` | equivalent |
| `BooleanQuery::rewrite` | `BooleanQuery.rewrite(IndexSearcher)` | 9 of 14 rules → **13 of 14** |
| `dedup_clauses` (**new**, extracted) | `clauseSets`' `HashSet` for FILTER/MUST_NOT | equivalent |
| — | `deduplicateClauses` (MUST/SHOULD) | still MISSING, deliberate (b12, unchanged) |
| — | `rewriteNoScoring` / `ConstantScoreQuery.rewrite` recursion | MISSING, F-2 below |
| — | "Inline SHOULD clauses from the only MUST clause" | MISSING, F-3 below |

### F-1 `[MISSING]` `Occur.FILTER` — the whole feature (b12's F-16)

**Java.** `BooleanClause.Occur` has four values. `FILTER` is "like `MUST` except
that these clauses do not participate in scoring": `BooleanClause.isRequired()`
is `MUST || FILTER`, `isScoring()` is `MUST || SHOULD`. `BooleanWeight`'s
constructor builds a non-scoring clause's `Weight` with
`ScoreMode.COMPLETE_NO_SCORES`; `BooleanScorerSupplier.req` puts filter scorers
in `required` but not in `requiredScoring`, and `ConjunctionScorer.score()`
iterates `scorers` (the scoring subset), never `required`.

**We did.** Three clause lists. `BooleanQuery`'s doc comment said a fourth would
be "a distinction without a difference here".

**Consequence.** No way to express the single most common production query shape
(a filter context), and — because the port's nearest equivalent was wrapping a
clause in `ConstantScoreQuery` — the nearest available workaround *adds* a score
where Lucene adds none.

**Fixed.** `BooleanQuery.filter: Vec<Clause>` + `with_filter`, threaded through
every reachable path:

- `matched_boolean_docs`: `has_required = !must.is_empty() || !filter.is_empty()`
  drives both the base-set choice and whether `should` participates in matching;
  the required conjunction is `must` ∪ `filter`. The "matches nothing" guard now
  tests all three positive buckets, so a **filter-only query matches** — Java's
  pure-negative test is `clauses.size() == clauseSets.get(MUST_NOT).size()`,
  which a filter clause fails.
- `clause_scores` / `search_boolean_query_scored_with_stats`: unchanged, and
  that is the point. The sum is over `must.iter().chain(should.iter())`, so a
  filter clause is structurally incapable of entering it — including inside a
  nested `Clause::Boolean`, whose matched set now includes its own filters while
  its score sum still does not.
- `try_conjunction_lazy`: filter clauses become non-scoring legs (see F-6).
- `try_disjunction_lazy` and the boolean MAXSCORE entry point: both are
  pure-`SHOULD` fast paths, so both now decline a query with filters rather than
  silently ignoring them.
- The single-scoring-clause fast path additionally requires `filter.is_empty()`.
- `multi_segment`'s global-term-stats walk skips `filter`: Java builds a filter
  clause's `TermWeight` with `COMPLETE_NO_SCORES`, which takes the
  "we do not need the actual stats, use fake stats" branch and never calls
  `searcher.termStatistics`. Collecting reader-wide statistics for a term only a
  filter mentions is a cross-segment seek per term for a number nothing reads.
- `query_parser` is **unchanged, deliberately**: `QueryParserBase.addClause`'s
  only outcomes are `MUST`/`SHOULD`/`MUST_NOT`. Classic query syntax has no
  filter operator, so producing one here would be a divergence, not a feature.
  Recorded as a comment at the construction site so the next reader does not
  "fix" it.

**`minimumNumberShouldMatch`.** A filter clause does not count toward it —
`BooleanWeight` increments `shouldMatchCount` only for `Occur.SHOULD`. Verified
against real Lucene (`scoring.boolean.filter.minshouldmatch`: doc 2 matches the
`cat` filter and neither optional clause, and drops out) and unit-tested in both
directions, including "the same query without the threshold brings doc 2 back",
so the test cannot pass because the filter excluded it for some other reason.

**Tests.** 7 real-Lucene bit-exact cases (see the fixtures section), 13 unit
tests in `lib.rs` covering matching, zero score contribution, the
`minimumNumberShouldMatch` interaction in both directions, a filter-only query
with a positive minimum (matches nothing), filter + `must_not`, a nested
`BooleanQuery` as a filter clause, and a filter clause that matches nothing.

### F-2 `[MISSING]` the five `FILTER`-specific `rewrite` rules

**Java**, in `BooleanQuery.rewrite`'s order:

1. *(single-clause)* a lone `FILTER` clause →
   `new BoostQuery(new ConstantScoreQuery(query), 0)` — "no scoring clauses, so
   return a score of 0".
2. *(dedup)* duplicate `FILTER` clauses collapse, because `clauseSets` stores
   `FILTER` and `MUST_NOT` in a `HashSet`.
3. *(remove redundant filters)*
   `if (filters.size() > 1 || clauseSets.get(MUST).isEmpty() == false)
   modified = filters.remove(MatchAllDocsQuery.INSTANCE);
   modified |= filters.removeAll(clauseSets.get(Occur.MUST));`
4. *(promote)* a clause that is both `FILTER` and `SHOULD` becomes `MUST`, the
   `FILTER` copy is dropped, and `minimumNumberShouldMatch` is decremented,
   floored at 0.
5. *(constant-score collapse)* a single `MUST` `MatchAllDocsQuery` alongside
   filters →
   `ConstantScoreQuery(BooleanQuery(FILTER.., MUST_NOT..))`, carrying the
   `MatchAllDocsQuery`'s boost, with the `SHOULD` clauses re-attached.

Plus two existing rules that change behaviour once `FILTER` exists: a
`MatchNoDocsQuery` in `FILTER` collapses the whole query (Java's
`case MUST: case FILTER: return rewritten;`), and required-clause inlining
**re-labels an inner `MUST` as a parent `FILTER`** when the outer clause was a
filter (`assert outerClause.occur() == Occur.FILTER && innerOccur ==
Occur.MUST; // ... change the occur of the inner query from MUST to FILTER`).

**We did.** None of them — they were unreachable.

**Fixed.** All seven, in the port's existing single-bottom-up-pass order. Two
notes on faithfulness:

- The guard on rule 3's `MatchAllDocsQuery` half is not an optimisation detail.
  Dropping the *only* filter of a filter-only query turns "every document,
  scored 0" into "no positive clauses at all", i.e. `MatchNoDocsQuery`. Tested
  in all three directions (dropped because a `MUST` exists, dropped because
  another filter remains, kept because neither holds).
- Rules 4 and 5 return a query that Java's `IndexSearcher.rewrite` fixpoint loop
  would simplify again on its next pass. This port is a single pass, so it takes
  the fixpoint directly (rule 4's promoted `SHOULD` meets rule 9 in the same
  pass; rule 5 returns the `ConstantScoreQuery` unwrapped when there are no
  `SHOULD` clauses). Both are called out in the tests, with what Java returns on
  which pass.

### F-3 `[CORRECTNESS]` the required-and-excluded reason string was not Java's

**Java.** `new MatchNoDocsQuery("FILTER or MUST clause also in MUST_NOT")`, and
the predicate is `clauseSets.get(MUST)::contains` **or**
`clauseSets.get(FILTER)::contains`.

**We did.** `"MUST clause also in MUST_NOT"` — b12 deliberately shortened it
because the `FILTER` half was unreachable.

**Consequence.** b14 made every `Explanation`/`MatchNoDocsQuery` description
string Java-verbatim; this one was the exception, and a caller comparing reason
strings across engines would have seen it.

**Fixed.** Java's string verbatim, and the predicate now tests `MUST_NOT`
against `must` ∪ `filter`. The existing test that asserted the shortened string
now asserts Java's, and gained the `FILTER` half of the predicate as a second
case.

### F-4 `[MISSING]` `ConstantScoreQuery.rewrite`'s non-scoring simplification, and "Inline SHOULD clauses from the only MUST clause"

**Java.** `BooleanQuery.rewrite`'s recursion step wraps every `FILTER`/`MUST_NOT`
clause in a `ConstantScoreQuery` before rewriting it — "clauses that are not
involved in scoring can get some extra simplifications" — which reaches
`BooleanQuery.rewriteNoScoring` (converting inner `MUST` to `FILTER` and dropping
purely-optional clauses). And a separate final rule inlines the `SHOULD` clauses
of a query whose only `MUST` clause is a pure disjunction.

**We did / do.** Neither. `Clause::rewrite`'s `ConstantScore` arm recurses into
the inner clause but implements none of `ConstantScoreQuery.rewrite`'s own rules.

**Consequence.** Both are structural simplifications, not semantic ones — the
matched set and the scores are the same either way. What is lost is the same
thing b12 recorded for its rules 3 and 4: a query that could have been handed to
one conjunction/disjunction scorer as a flat clause list arrives nested.

**Recorded, not fixed.** `rewriteNoScoring` is a `ConstantScoreQuery` concern
with its own rule set and its own fixpoint interaction; "inline SHOULD from the
only MUST" is not `FILTER`-specific and was already absent before this batch.
Both are in `LEDGER.md`.

### Verdict

`BooleanQuery` has all four of Java's `Occur` values. `rewrite` implements 13 of
Java's 14 rules; the two open ones are recorded above and the MUST/SHOULD dedup
remains deliberately unimplemented for b12's unchanged reason. Line coverage
98.27%.

---

## `crates/lucene-search/src/lib.rs`

Java: `BooleanWeight`, `BooleanScorerSupplier`, `ConjunctionScorer`,
`FilterScorer`.

| Rust item | Java | Verdict |
|---|---|---|
| `matched_boolean_docs` | `BooleanScorerSupplier.get` (matching half) | now handles `FILTER` |
| `clause_scores` | `ConjunctionScorer.score()` / `ReqOptSumScorer` | unchanged — correct by construction |
| `try_conjunction_lazy` | `ConjunctionScorer` + `BlockMaxConjunctionScorer` | now handles `FILTER` legs |
| `try_disjunction_lazy` | `WANDScorer`/`DisjunctionSumScorer` | declines queries with filters |
| `search_boolean_query_scored_maxscore_with_stats` | `BooleanScorerSupplier` (TOP_SCORES) | declines; **body is dead**, F-7 |

### F-5 `[CORRECTNESS-risk, prevented]` the `f32` summation order

**Java.** `ConjunctionScorer.score()` sums `scorers`, which is
`requiredScoring` — the `MUST` scorers, in the order `BooleanScorerSupplier`
collected them. Filters are in `required`, never in `scorers`, so they cannot
appear in the sum *or* shift the position of anything that does.

**We do.** The same, structurally: `clause_scores` folds
`must.iter().chain(should.iter())`, and `try_conjunction_lazy` sums
`legs.iter().filter(|l| l.scoring)`. The lazy path sorts legs rarest-first, but
`sort_by_key` is stable and inserting filter legs into that sort cannot reorder
the scoring legs relative to each other.

This is asserted, not argued: `scoring.boolean.filter.dupmust`
(`+body:cat +body:dog #body:dog`) is recorded from real Lucene as
**bit-identical** to `scoring.boolean.must` (`+body:cat +body:dog`), and
`scoring.boolean.filter.nested` (`+(+body:cat #body:dog) body:dog`) is too. A
unit test additionally asserts `+body:cat #body:dog` scores doc 0 bit-for-bit
the same as `+body:cat` alone, so a filter contributing even one ULP fails.

### F-6 `[PERF]` a filter-only conjunction was 15x *dearer* than the MUST form

**Found by the measurement this batch was asked to take**, not by reading.

`try_conjunction_lazy` initially required at least one `MUST` leg, so
`#body:t0 #body:t1` fell to the general path, which materialises every clause's
whole doc list (`resolve_clause_docs` → `Vec<i32>`) before intersecting. On the
5M-document benchmark corpus that is ~500k doc IDs per clause.

**Measured** (`benches/filter_vs_must.rs`, `benchmarks/.corpus/merged`,
criterion, top-50):

| query | before | after |
|---|---|---|
| `#body:t0 #body:t1` | 129.5 ms | **58.7 ms** |

**Fixed**: the lazy leapfrog now accepts a filter-only conjunction. Block-max
pruning is switched off for that shape (`prunable`), because with no scoring
clause the summed bound is 0 and `bound <= threshold` would authorise a skip on
a tie — see the open item below.

### F-7 `[PERF]` `search_boolean_query_scored_maxscore_with_stats`' body is unreachable

Not a `FILTER` finding; found while closing the coverage gap, since it is the
single largest uncovered block in the file (~97 lines).

The function calls `try_disjunction_lazy` first and returns if it succeeded.
Every shape the body below can handle — pure `SHOULD`, all `Clause::Term`,
`doc_in` present, `minimumNumberShouldMatch <= 1`, no pulsed term —
`try_disjunction_lazy` has already handled; and every shape
`try_disjunction_lazy` declines (non-`Term` clause, `must`/`filter`/`must_not`
present, minimum > 1, no `doc_in`, a pulsed `docFreq <= 1` term) the body also
declines, falling back. The two are exactly complementary, so the body never
runs. Its own doc comment already records that it is 4-5x slower than the lazy
union and says "prefer the plain scored entry point"; the block-skip counter its
invariant test reads is recorded by `try_disjunction_lazy`, not by it.

**Recorded, not fixed** — deleting ~180 lines of a deliberate alternative
implementation is not a `FILTER` batch's call. In `LEDGER.md`.

### Verdict

`FILTER` is threaded through every matching and scoring path. Line coverage
**95.20%**, up from 90.52% — see the coverage section.

---

## `crates/lucene-search/src/explain.rs`

Java: `BooleanWeight.explain`, `BooleanQuery.toString`.

### F-8 `[MISSING]` no `FILTER` arm in `explain`, and no `#` in `toString`

**Java.**

```java
subs.add(Explanation.match(0f, "match on required clause, product of:",
    Explanation.match(0f, Occur.FILTER + " clause"), e));
```

`Occur.FILTER.toString()` is `"#"`, so the inner description is exactly
`"# clause"` and both values are `0f`. A *failing* filter clause routes through
`c.isRequired()` and gets `MUST`'s own
`"no match on required clause (...)"`. `matchCount` counts a matching filter
(Java increments for every non-prohibited match); `shouldMatchCount` does not.

**Fixed.** The arm verbatim, emitted between `must` and `should` (the port
groups by occur — Java's own `Occur` declaration order — rather than by
insertion order, which is stated in `BooleanQuery`'s doc comment). The wrapper's
value is `0`, so it is in `details` for diagnosis but contributes nothing to the
node's total. `describe_clause` prefixes a filter clause with `#`.

**Tests.** Five: the matching arm (asserting the two nested descriptions, both
zero values, the clause's own explanation surviving as a child, and the total
being the `MUST` clause's value *bit-for-bit*), the failing arm, a filter-only
query explaining as a match of 0, the `minimumNumberShouldMatch` interaction
(`matched: 0` for a doc that matched only the filter), and `toString`'s `#`.

### Verdict

Swept clean. Line coverage 95.49%.

---

## Verification against real Lucene

`fixtures/src/AppendScoringManifest.java` gained seven `Occur.FILTER` entries,
recorded from a real `DirectoryReader`/`IndexSearcher` over the committed
`fixtures/data/blocktree_index/` segment as `Float.floatToIntBits`.
`crates/lucene-search/tests/bm25_scoring_fixtures.rs` re-runs each through this
port and asserts `f32::to_bits` equality. Corpus: `cat`={0,2}, `dog`={0,1},
`bird`={1,4}.

| key | query | real Lucene | what it pins |
|---|---|---|---|
| `scoring.boolean.filter` | `+body:cat #body:dog` | `0:0.39608413` | the filter's contribution is exactly zero — `scoring.boolean.must` is `0:0.67334306` |
| `scoring.boolean.filteronly` | `#body:cat #body:dog` | `0:0.0` | a query with no scoring clause still matches |
| `scoring.boolean.filter.single` | `#body:cat` | `0:0.0, 2:0.0` | `rewrite`'s `BoostQuery(ConstantScoreQuery(q), 0)` |
| `scoring.boolean.filter.should` | `#body:cat body:dog` | `0:0.2772589, 2:0.0` | the filter fixes the set; the optional clause only adds score |
| `scoring.boolean.filter.minshouldmatch` | `(#body:cat body:dog body:bird)~1` | `0:0.2772589` | filters do not count toward the threshold — doc 2 drops |
| `scoring.boolean.filter.dupmust` | `+body:cat +body:dog #body:dog` | `0:0.67334306` | bit-identical to `scoring.boolean.must` |
| `scoring.boolean.filter.nested` | `+(+body:cat #body:dog) body:dog` | `0:0.67334306` | a filter inside a nested `BooleanQuery`, same sum in the same order |

All seven passed on the first run against the implementation. The lone-filter
case additionally asserts that `rewrite()` produces exactly
`BoostQuery(ConstantScoreQuery(TermQuery), 0)` and that *that* clause scores the
same recorded hits.

Regenerated with `AppendScoringManifest` alone (it opens the committed index
read-only and replaces its own `scoring.*` lines), **not** with a full
`gen-fixtures.sh` run, which would perturb the committed random segment ID other
suites hardcode. The manifest diff is 21 added lines, no deletions.

---

## Measurement: is a `FILTER` clause actually cheaper?

`crates/lucene-search/benches/filter_vs_must.rs` (new), criterion, against
`benchmarks/.corpus/merged` (5M documents, one segment), top-50. The bench
skips itself when that corpus is absent.

| shape | time | vs. the `MUST` form |
|---|---|---|
| `+body:t0 +body:t1` | 9.97 ms | — |
| `+body:t0 #body:t1` | **5.58 ms** | **44% cheaper** |
| `+body:t0 +body:tz` | 5.86 ms | — |
| `+body:t0 #body:tz` | **5.50 ms** | 6% cheaper |
| `+body:t0 +body:t1` (no pruning) | 76.8 ms | — |
| `#body:t0 #body:t1` (no pruning) | **56.4 ms** | **27% cheaper** |
| `#body:t0 #body:t1` (top-50) | 54.0 ms | see below |

So yes: with the second clause as a filter, the query is 44% cheaper on a dense
conjunction and 6% cheaper when the second clause is selective enough that
scoring is a small share of the total. The saving is concretely the frequency
block decode (`freq()`), the norms cursor (and, on a sparse field, its
`IndexedDISI` walk), and the `BM25Scorer.score` call — all of which
`try_conjunction_lazy` now skips for a non-scoring leg.

**The one place it is not cheaper, with numbers.** Under a top-`n` collector,
`#body:t0 #body:t1` costs 54.0 ms against 10.0 ms for `+body:t0 +body:t1`. That
is not the filtering: it is that a scoring conjunction *prunes* (its block-max
bound beats the queue's bottom and whole spans are skipped) and a filter-only
one has no score to bound. The last two rows above re-run both with pruning
forbidden on either side — the like-for-like comparison of the matching work —
and there the filter form is 27% cheaper, as expected. Java would prune the
filter-only shape (`FilterScorer`'s `getMaxScore() == 0` against
`TopScoreDocCollector`'s `Math.nextUp(bottom)`, which is algebraically this
port's `bound <= threshold`); this port deliberately does not, because a zero
bound would authorise a skip on a tie and the fixture segment is too small to
produce a differential test for it. Recorded in `LEDGER.md`.

---

## Coverage: closing `lib.rs`' pre-existing gap (raised by c6)

`cargo llvm-cov -p lucene-search --summary-only`, after
`cargo llvm-cov clean --workspace` (a stale profile from the concurrent batches
otherwise reports every touched file 15-20 points low, as c6 noted):

| file | before | after |
|---|---|---|
| `lib.rs` | 90.52% | **95.20%** |
| `query.rs` | 92.04%* | **98.27%** |
| `explain.rs` | 94.27%* | **95.49%** |

\* measured mid-batch, after this batch's code was added but before its tests.

Every file in `lucene-search` is now above the 95% bar. The crate's suite went
from 854 to 899 tests.

**Which paths were uncovered — the answer c6 asked for.** Two clusters, and
neither was random:

1. **The single-scoring-clause fast path in
   `search_boolean_query_scored_with_stats`, and the
   `stream_constant_score_clause` / `expanded_terms` streaming union it is the
   only caller of** (~145 lines). Nothing reached them because every
   wildcard-family fixture suite calls `search_prefix_query_scored` and friends
   *directly*, never through a `BooleanQuery` — so the shortcut, the streaming
   union, its `total_doc_freq < terms * BLOCK_SIZE` decline, and its
   `pruning_threshold() >= 1.0` early exit were all untested. Six new tests, on
   the fixture's `big` field (one term, 300 documents — enough to stream) and
   `many` field (400 terms at one document each — the shape that must *decline*
   to stream). One of them documents something worth knowing: the fast path's
   `Clause::Term` arm is reachable **only** for a pulsed term (`docFreq <= 1`),
   because `try_conjunction_lazy` intercepts every other single-term
   conjunction.
2. **Both lazy paths' block-max pruning branches** (~100 lines). Unreachable
   from the fixture's 5-document `body` field for a structural reason: the
   collector's queue has to *fill* before `pruning_threshold()` returns
   anything at all, and no `body` query returns enough hits. Two new tests use
   the 300-document `big` and 8,250-document `l1` fields, each asserting both
   that the pruned run's top-`n` scores are bit-identical to a
   `ScoreMode::Complete` run *and* that fewer documents were visited — so
   neither can pass by failing to prune.

**What is still uncovered in `lib.rs`, and why it should stay that way for now**:
~97 lines are `search_boolean_query_scored_maxscore_with_stats`' dead body
(F-7); ~21 are `search_term_query_scored_maxscore`'s impacts skip loop, which
needs a corpus larger than the fixture for the same queue-filling reason;
the rest are single `?` error arms and `unreachable!` guards.

---

## Gate

- `cargo fmt --all` — clean (`--check` passes).
- `cargo clippy -p lucene-search --all-targets -- -D warnings` — **clean, exit
  0**. (`lucene-index`'s `check_index.rs` lints, which c6 hit, are gone —
  c9 fixed them.)
- `cargo test -p lucene-search` — **899 passed, 0 failed**, 22 targets.
- `cargo build --workspace` — clean. `cargo test -p lucene-ffi` — 441 passed,
  0 failed (`lucene-ffi` compiles again since c7 landed; the new `filter` field
  did not break it, since it builds `BooleanQuery` through the builders).
- `docs/parity.md` updated in the same change (rows for `search/BooleanQuery` +
  `search/BooleanClause` and `search/BooleanQuery.rewrite`), per invariant #7.

Not committed, per the brief.
