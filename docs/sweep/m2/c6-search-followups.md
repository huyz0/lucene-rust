# c6 — search follow-ups

A follow-up batch, not a file sweep: it holds the whole of
`crates/lucene-search` (swept by b12/b13/b14) and closes the five items those
batches had to hand off, plus one gap found on the way. Eight findings: five
`CORRECTNESS` (all fixed), two `MISSING` (both fixed and measured), one `PERF`
(fixed and measured); plus one `INTENTIONAL` divergence recorded inside F-6.
One of the five (F-7) is a hang this batch introduced and its own Tier-2 review
caught before it shipped. Java counterparts are
named per item; the per-file method-correspondence tables live in the b12/b13/b14
reports and are not repeated.

Files touched:

- `crates/lucene-search/src/{lib,field_norms,multi_segment,directory_reader,docid_set,explain}.rs`
- `crates/lucene-search/tests/multi_segment_scoring_fixtures.rs` (**new**)
- `crates/lucene-search/benches/{field_norms_sparse,boolean_bulk_or}.rs` (**new**)
- `fixtures/src/GenMultiSegmentScoring.java` (**new**) + `fixtures/data/multi_segment_scoring_index/`
- `crates/lucene-ffi/src/directory_reader.rs` (minimal, mechanical — the crate is unowned)
- `docs/sweep/m2/LEDGER.md` (carry-overs ticked)
- `docs/parity.md` (BM25/similarity, multi-segment and `docid_set` rows, plus a
  **new `search/BooleanScorer` row**), `docs/sweep/findings.md`,
  `fixtures/README.md`

---

## 1. `lib.rs::search_term_query_scored_maxscore_with_stats`

Java: `search/IndexSearcher.termStats`/`fieldStats` +
`search/similarities/BM25Similarity.idf`; the fallback structure has no Java
counterpart (Lucene's `MaxScoreBulkScorer` never falls back to a different
`Weight`, it just scores without pruning).

### F-1 `[CORRECTNESS]` reader-wide statistics dropped on all three fallbacks

**Java.** `IndexSearcher` computes `TermStats`/`FieldStats` once for the whole
reader and hands the same numbers to every leaf. There is no path on which a
leaf reverts to its own counters.

**We did.** The MAXSCORE entry point takes a reader-wide `CollectionStats` and
then, on each of its three fallback `return`s — no `.doc` input, `docFreq <= 1`,
and an index option `LazyDocsCursor` rejects — called the **no-stats**
`search_term_query_scored`. Found by b13 (its F-5), handed to b12, which had
already reported; nobody fixed it.

**Consequence.** On b13's two-segment scenario, idf 0.288 (leaf A) / 1.204
(leaf B) / 0.876 (reader-wide): the merged top-k fills from whichever leaf makes
the term look rarest.

**Fixed.** `term_doc_scores` split so the map lookup and the scoring loop are
separate; new private
`search_term_query_scored_with_collection_stats` takes the already-resolved
`CollectionStats` (a `TermQuery` mentions exactly one term, so a `GlobalStats`
map was the wrong shape for the fallback) and all three `return`s forward
`global` through it.

**b13's workaround removed.** `multi_segment::maxscore_keeps_global_stats` — a
caller-side guard that re-derived another function's private control flow from
the outside — and its helper `single_term_global_stats` are deleted; both
term-query fan-outs (sequential and concurrent) now call the MAXSCORE entry point
unconditionally, as they read before b13 had to work around it.

**Test.** b13's tripwire is kept and **inverted**:
`every_maxscore_fallback_honours_the_global_stats_it_is_handed` (in
`multi_segment.rs`) hands the entry point a `global` nothing like the segment's
own counters and asserts each reachable fallback — no `.doc` input, and
`docFreq <= 1` — scores with it **bit for bit**, with the pruned path as a
control so "both wrong the same way" cannot pass. The third fallback
(`LazyDocsCursor` rejecting `IndexOptions::None`) is unreachable for a scored
term query: that index option carries no postings to score, which the test's doc
comment records.

### Verdict

Closed. The carry-over in `LEDGER.md` can be ticked.

---

## 2. `field_norms.rs` — `FieldNorms`'s sparse `Vec<i32>`

Java: `index/NumericDocValues` (per-leaf, per-scorer) obtained from
`LeafReader.getNormValues(field)`, over `codecs/lucene90/IndexedDISI`; the shared
object is `index/SegmentCoreReaders`' entry.

### F-2 `[PERF]` an eager doc-id `Vec` where Lucene has a cursor

**Java.** `Lucene90NormsProducer.getNorms` returns a fresh `NumericDocValues`
per scorer, holding an `IndexedDISI` positioned in the region. Nothing is
decoded up front and nothing is allocated per document.

**We did.** `FieldNorms` held `sparse_doc_ids: Option<Vec<i32>>` — *every*
doc id with a norm, decoded in the constructor — and each lookup binary-searched
it. That was b13's fix for `norms::norm_value`'s then-quadratic sparse branch;
c2 removed the reason for it by making `IndexedDISI` a real incremental cursor,
and handed the `Vec` over as a "two-line swap".

**Why it was not two lines.** `DisiCursor::advance_exact` takes `&mut self`,
`FieldNorms::sparse_norm` took `&self`, and `multi_segment.rs` passes
`&FieldNorms<'_>` into `rayon` `par_iter` closures, so `FieldNorms` must stay
`Sync`. A `RefCell` would have made it non-`Sync` and broken the concurrent
fan-out — a fix that serialises search is a regression. `LEDGER.md` records the
analysis and names three designs.

**Fixed, design 1 (the one Lucene uses).** The cursor belongs to the caller:

- `FieldNorms` keeps `sparse_region: Option<&'a [u8]>` (a slice, no allocation,
  no decode) and stays immutable + `Sync` — the `SegmentCoreReaders` entry.
- New `FieldNormsCursor<'n, 'a>` = `&FieldNorms` + a `DisiCursor`. Created by
  `FieldNorms::cursor()`, which is Lucene's `getNormValues`: allocates nothing,
  reads no bytes.
- Forward-only underneath but **tolerant**: a target behind the last one rewinds
  the cursor rather than panicking, so a non-monotonic caller stays correct.
  Correct, but not free — see "One caller needed restructuring" below.
- Every document-at-a-time loop in `lib.rs` now takes one cursor per scan —
  the two `Leg` structs and the boolean `ClauseCursor` carry theirs per clause,
  exactly as each `TermScorer` carries its own `NumericDocValues`.
- `FieldNorms::{norm_inverse, field_length}` survive as `&self` one-shots
  (`self.cursor().…`) for single-document callers — `explain.rs` and the FFI
  need no change at all, which is less ripple than the ledger predicted.
- `FieldNorms::open` also runs off a single cursor now: its `0..max_doc` scan is
  monotonic, and it used to call `norms::norm_value` per document, i.e. a fresh
  block walk each time.

**A second defect fixed by the same change.** For a sparse field the old code
treated its `Vec` lookup as advisory: a `None` (document genuinely absent) fell
through to `norms::norm_value`, which walks the region's block headers from the
start again only to reach the same answer. The cursor is authoritative in both
directions, so an absent document now costs nothing extra.

**One caller needed restructuring, not just a cursor.** Every scoring loop in
`lib.rs` walks documents in ascending order — except `fuzzy_doc_scores`, which
loops over the expanded terms and walks *each one's* postings, restarting near
doc 0 every time. One cursor across that outer loop rewinds once per expansion,
and on a sparse field a rewind is a fresh `IndexedDISI` walk: O(expansions ×
region) where the `Vec` was O(region) once. At the measured 677 µs per
100,000-document walk, a 50-term expansion would have spent ~34 ms in norms
alone — a real regression the first cut of this batch shipped and presented as an
unambiguous win. (Tier-2 review; dense norms were never affected, since their
lookup is an array index.)

**Fixed** by resolving every expanded term's live postings into one
`Vec<(doc, freq, boost)>` first and scoring in a single pass. The sort that makes
that pass ascending is **stable**, which is what keeps the result bit-for-bit
identical: a document's contributions keep their expansion-term order, so the
sequence of `+=` into the score map — and therefore every float bit — is exactly
what the per-term loop produced. `f32` addition is not associative, so that is a
requirement, not a nicety. The sort is skipped entirely when it cannot pay for
itself, gated on a new `FieldNorms::prefers_ascending_lookups()` — true only for
a sparse field, where a lookup is order-sensitive; false for the ordinary dense
one-byte field, whose lookup is an array index.

**Measured** (`benches/field_norms_sparse.rs`, eager `Vec` vs cursor, same
data, side by side, four groups because the change *moves* cost rather than only
removing it):

| documents with the field | | eager `Vec` | cursor | |
|---|---|---|---|---|
| 1,000 | construct | 760 ns | 486 ns | 1.6x |
| 1,000 | whole query's scan (construct + every doc) | 9.03 µs | 8.19 µs | 1.10x |
| 1,000 | one isolated lookup | **6.0 ns** | 301 ns | **0.02x** |
| 1,000 | one forward step of a warm cursor | — | 9.6 ns | |
| 100,000 | construct | 140 µs | 493 ns | **283x** |
| 100,000 | whole query's scan (construct + every doc) | 1.74 ms | 799 µs | **2.2x** |
| 100,000 | one isolated lookup | **12.3 ns** | 171 ns | **0.07x** |
| 100,000 | one forward step of a warm cursor | — | 9.8 ns | |

Read the shape, not just the sizes. **Construction goes from linear to
constant** — 140 µs to 493 ns at 100,000 documents with the field, and flat
where the old path grew — which is what matters, because a `FieldNorms` is built
per query per leaf. **The isolated random lookup gets dramatically slower**: a
one-shot cursor walks the region's block headers from the start (171 ns) where a
binary search over a prebuilt `Vec` is a handful of cache-resident comparisons
(12 ns). A *warm* cursor's forward step is 9.8 ns, i.e. slightly cheaper than the
binary search and flat in cardinality where the search is logarithmic — which is
why the scan, the shape that actually matters, is 2.2x faster end to end even
though the isolated lookup is 14x slower.

The only caller doing isolated lookups is `explain`, one document at a time,
where 160 ns is invisible next to the construction that just got 283x cheaper.

**The first cut of this bench claimed 1,183x on that lookup, and it was wrong.**
The Tier-2 review caught it: the eager arm rebuilt its `Vec` inside `b.iter()`
while the cursor arm reused a `FieldNorms` built outside, so it measured
construction and reported it as lookup. Both arms now build *outside* the timed
loop for `lookup_only` and *inside* it for `per_query_scan`, where a per-query
construction genuinely is part of the cost, and a fourth group measures the warm
step the scan actually pays. The conclusion held; the number quoted to support it
did not, and a benchmark that flatters its own change by a factor of a thousand
is worse than no benchmark.

**Memory**: `4 × numDocsWithField` bytes → **0**. At 100,000 documents with the
field that is 400 KB per `FieldNorms`, i.e. per query per leaf; on a 15-segment
reader, 6 MB per query.

**Concurrency confirmed, not assumed.**
`field_norms::tests::a_shared_field_norms_still_fans_out_concurrently` asserts
`FieldNorms: Sync` and `FieldNormsCursor: Send` statically, then runs two rayon
tasks that each take their own cursor over one shared `&FieldNorms` and neither
of which may finish until both have started. Serialised execution would spin to
the timeout and fail the assert.

**Other tests.** `the_cursor_agrees_with_the_one_shot_lookup_forwards_and_backwards`
(10,000 documents of a DENSE `IndexedDISI` block, ascending, descending, and
jumbled — the descending pass is what would catch a forward-only cursor that
failed to rewind); `a_negative_doc_is_an_error_not_a_panic_on_either_shape`
(`advance_exact` asserts on a negative doc, so the cursor must raise
`DocOutOfRange` before reaching it, same error the general path raises);
`the_cursor_carries_its_fields_avg_field_length`. The existing sparse/dense
agreement tests are unchanged and still pass.

### Verdict

Closed, with the concurrent fan-out proven still concurrent.

---

## 3. `directory_reader.rs` / `multi_segment.rs` — reader-wide `avgFieldLength`

Java: `search/IndexSearcher.fieldStats(String)` (sums
`Terms.getSumTotalTermFreq()`/`getDocCount()` over `reader.leaves()`) feeding
`search/similarities/BM25Similarity.avgFieldLength(FieldStats)` =
`(float) (sumTotalTermFreq / (double) docCount)`.

### F-3 `[CORRECTNESS]` `avgdl` was per leaf where Java's is reader-wide (b13's F-26)

**Java.** All three `CollectionStatistics` numbers are reader-wide. `avgdl` is
baked into `BM25Similarity.scorer`'s 256-entry `cache[]`, one table for the whole
reader.

**We did.** `SegmentReader::field_norms` read *this segment's* `.tmd` counters,
so each leaf got a different length-normalization curve for the same term — the
same class of divergence as the idf one b13 fixed, and the half it left open.

**Fixed.**

- `field_norms::avg_field_length(stf, doc_count)` — Java's formula, `f64`
  intermediate, narrowed once — and `FieldNorms::with_avg_field_length`, the
  constructor a multi-segment search wants.
- `SegmentReader::field_stats(field) -> Option<(i64, i32)>` (Java's two `Terms`
  getters) and `field_norms_with_avg_field_length`.
- `DirectoryReader::avg_field_length(field)` — `IndexSearcher.fieldStats`, summed
  over every leaf — plus `DirectoryReader::field_norms(field)` and
  `field_norms_by_field(&[String])`, which build every leaf's `FieldNorms` from
  the one value and are the `Vec` the multi-segment search functions take.
- `multi_segment::global_avg_field_length(segments, field)`, the sibling of
  `global_term_stats` b13 asked for, for callers holding `OpenSegment`s rather
  than a `DirectoryReader`.

`avgdl` cannot be threaded through the search functions the way
`global_term_stats` is, because it is consumed when the `FieldNorms` is
*constructed*, before any search function sees it. That is why the fix lives on
the reader.

**Consistency with b15.** b15 moved the FFI's *single-segment* path from
`FieldNorms::open` (an average of lossy `SmallFloat`-decoded norms) to
`from_field_stats` (`sumTotalTermFreq/docCount`). This is the same formula, with
the sum taken across leaves; `from_field_stats` is now a thin wrapper over
`with_avg_field_length` so the two cannot drift.

**New fixture.** `fixtures/src/GenMultiSegmentScoring.java` builds a genuine
**two-segment** real-Lucene index (`NoMergePolicy` + a `commit()` between the two
batches, so two segments survive) with deliberately lopsided statistics: four
1–3-term documents in segment 0 and four 40-term documents in segment 1, so the
two leaves' own `avgdl` are **1.75** and **40.0** against a reader-wide
**20.875**, and `fox` has `docFreq` 1-of-4 in one leaf and 3-of-4 in the other.
It records real `IndexSearcher` `TopDocs` as `Float.floatToIntBits`, in global
doc-id space. Every earlier scoring fixture in this tree is one segment, where
per-leaf and reader-wide are the same number by construction — which is exactly
why F-26 survived two batches.

**Tests** (`tests/multi_segment_scoring_fixtures.rs`, 5):
term query, boolean disjunction and the concurrent fan-out all match real Lucene
**bit for bit**; `the_two_segments_own_avgdl_values_are_far_apart_and_neither_is_the_readers`
pins the fixture's premise so a regenerated corpus that evens out fails loudly;
and `per_leaf_avgdl_does_not_reproduce_lucenes_scores` is the negative control —
scoring each leaf with its own `avgdl` (what `SegmentReader::field_norms` alone
can compute) must **not** reproduce Lucene's scores, so the positive tests
cannot pass for the wrong reason.

### F-4 `[MISSING]` the multi-segment FFI entry points passed no norms at all

Found while wiring F-3. All six multi-segment FFI functions
(`ffi_search_{term,boolean}_query_multi_segment{,_concurrent}`,
`…_maxscore{,_concurrent}`) passed `vec![None; segments.len()]` for norms, so
**every multi-segment FFI search scored unnormed** — real Lucene always applies
the field's real per-document lengths. The module doc justified it with "task
#45's `DirectoryReader` carries no `.nvm`/`.nvd` data per segment", which b13
made untrue.

**Fixed** in `lucene-ffi/src/directory_reader.rs`: term queries take
`reader.field_norms(&query.field)`, boolean queries take
`reader.field_norms_by_field(boolean_query_term_fields(&query))` (a local helper
mirroring `query.rs`'s single-segment field-name collection). Minimal and
mechanical, as the batch brief requires for that crate; the module doc is
corrected rather than left describing the old behaviour.

**Test.** `term_query_multi_segment_scores_with_real_norms_like_lucene` asserts
the FFI multi-segment term query returns real Lucene's own recorded
`scoring.term.cat` `TopDocs` **bit for bit** — the same
`AppendScoringManifest.java` ground truth `bm25_scoring_fixtures.rs` uses — and
then re-runs the same query with `vec![None; n]` and asserts that answer is
*different*, so the test cannot pass by the norms silently going missing again.
That none of the 439 existing FFI tests changed behaviour is itself the finding:
every one of them asserted doc ids and score ordering, never a score value.

### Verdict

Closed, including the F-7 remainder (`explain.rs` and the FFI both take
`from_field_stats`-derived norms now; `FieldNorms::open` is left only for callers
that hold a `.nvd`/`.nvm` and nothing else). `docs/parity.md`'s BM25 and
multi-segment rows updated.

---

## 4. `docid_set.rs` — `BooleanScorer`'s window/bucket bulk OR

Java: `search/BooleanScorer` (`SHIFT = 12`, `SIZE = 4096`),
`search/BooleanScorerSupplier.{bulkScorer, booleanScorer, optionalBulkScorer}`.

### F-5 `[CORRECTNESS of the premise]` `BooleanScorer` is **not** what Lucene uses for a top-k disjunction

b12's F-22 records `BooleanScorer` as "the highest-value untried mechanism" for
this port's boolean queries and cites the benchmark's `or t0 t1 t2 t3` shape at
0.26x. Reading `BooleanScorerSupplier.optionalBulkScorer` in 10.5.0 shows the
premise is half wrong:

```java
if (scoreMode == ScoreMode.TOP_SCORES) {
  if (minShouldMatch > 1) { return null; }          // -> BS2/WANDScorer
  return new MaxScoreBulkScorer(maxDoc, optionalScorers, null);
}
…
return new BooleanScorer(optional, Math.max(1, minShouldMatch), scoreMode.needsScores());
```

`searcher.search(query, n)` uses a `TopScoreDocCollectorManager`, whose
`scoreMode()` is `TOP_SCORES`, so real Lucene answers that benchmark query with
`MaxScoreBulkScorer` (WAND) — which this port already approximates in
`search_boolean_query_scored_maxscore`. Lucene's own comment says `BooleanScorer`
"does not consult score upper bounds and would score every doc in the 2048-doc
window, defeating top-K pruning" (that comment is stale in Lucene itself —
`SHIFT = 12` makes the window 4,096, as `BooleanScorer`'s own class javadoc
says; b12's F-22 inherited the 2048 from it). **Porting `BooleanScorer` onto the scored
top-k path would therefore have been a regression, not the missing mechanism.**
Recorded here so the next reader does not re-derive it from F-22.

Where `BooleanScorer` genuinely is Lucene's answer: `ScoreMode.COMPLETE` and
`COMPLETE_NO_SCORES` — a disjunction whose hits are all collected or merely
counted. In this port that is `search_boolean_query` and everything downstream
of `matched_boolean_docs`.

### F-6 `[MISSING]` no window/bucket bulk OR (b12's F-22, for the paths it applies to)

**Java.** `BooleanScorer` walks one 4,096-document window at a time. Each clause
pours its whole run of doc ids into a `FixedBitSet` (`scoreWindowIntoBitSetAndReplay`),
plus a parallel `Bucket[]` carrying the clause count when `minShouldMatch > 1`;
the window is then replayed by walking the bitset's words with
`Long.numberOfTrailingZeros`. Each clause is read in one long contiguous run and
the per-document "which clause is next" decision disappears inside the window.

**We did.** `docid_set::Disjunction`: an `O(clauses)` min-scan **per emitted
document**, each step a `peek()`/`next()` through a `Box<dyn Iterator>`. And for
`minimum_should_match > 1`, `should_match_counts` built a `HashMap<i32, usize>`
tally over **the whole segment** first.

**Ported** as `docid_set::WindowedDisjunction`, over this port's
already-materialized per-clause lists and pull-shaped so it composes with
`Excluding` exactly as `Disjunction` does. `lib.rs::matched_boolean_docs`
chooses it the way `BooleanScorerSupplier.booleanScorer` does — no required
clauses and more than one optional clause (`BooleanScorer`'s own constructor
refuses fewer than two) — and the prohibited (`must_not`) union takes it too,
being the same pure OR. `should_match_counts`' whole-segment `HashMap` becomes a
`u16` at a window-relative index.

**Two deliberate divergences**, both recorded in the type's doc comment:

1. Only non-empty windows are visited (`scoreWindow`'s `top.doc & ~MASK` jump).
2. Only the word range a clause actually touched is cleared and replayed, where
   Java pays a fixed 64-word clear per window. `[INTENTIONAL]`

**Java's density gate (`costThreshold = maxDoc / 3` when `minShouldMatch > 1`)
is deliberately not ported, and the first version of this report got the reason
wrong.** It claimed the two divergences above removed Java's premise. They do
not: Java's per-window cost is *also* proportional to that window's postings
(`scoreWindowIntoBitSetAndReplay` pours only `leads`, and
`scoreWindowMultipleScorers` skips a window outright when
`maxFreq < minShouldMatch`), so the 64-word clear is fixed overhead, not the
premise. Java's stated premise — with a minimum-should-match there is no way to
know whether the clauses intersect inside a window, so its postings may be
poured in for nothing — applies to `WindowedDisjunction` **verbatim**.

What actually fails to transfer is the *choice* the gate expresses. Java is
choosing between `BooleanScorer`, which reads every posting of every clause, and
`MinShouldMatchSumScorer` (via BS2), which **leapfrogs** — it can advance past
postings without reading them, so on a sparse query it does strictly less I/O
and the gate picks the cheaper of two real options. This port's only alternative
is `should_match_counts`, which reads every posting of every clause *and* hashes
each one into a whole-segment `HashMap<i32, usize>` before a per-document
min-scan: worse on every axis at every density, so there is no trade-off left to
gate on. **Restoring the gate — with Java's threshold and Java's reason —
becomes necessary the moment a leapfrogging min-should-match scorer exists
here**, and that is written on the type. (Caught by the Tier-2 review: right
conclusion, wrong reason, which is the kind of comment that misleads the next
reader precisely because the code around it is correct.) `[INTENTIONAL]`

**Measured** (`benches/boolean_bulk_or.rs`, 1,000,000-document segment):

| shape | min-scan | windowed | |
|---|---|---|---|
| `or4_dense` (4 clauses, ~1 doc in 10) | 3.54 ms | 1.13 ms | **3.1x** |
| `or4_sparse` (4 clauses, ~1 doc in 1,000) | 36.9 µs | 34.5 µs | 1.07x |
| `or16_dense` (16 clauses, ~1 doc in 10) | 23.86 ms | 2.55 ms | **9.4x** |
| `or4_msm2` (4 clauses, `minimum_should_match = 2`) | 27.80 ms | 1.12 ms | **24.8x** |

The scaling is the point: the min-scan is `O(clauses)` per emitted document, so
it gets worse as clauses are added, while the window's per-clause contiguous run
does not. `or4_sparse` is the case a window-at-a-time scorer could plausibly
lose and does not, which is what the two divergences above buy. `or4_msm2`'s
24.8x is a different effect entirely: it is the whole-segment
`HashMap<i32, usize>` tally disappearing, replaced by a `u16` at a
window-relative index.

### F-7 `[CORRECTNESS]` the window loop hung at the top of the doc-id space

Found by the Tier-2 review, in the code this batch had just written.

`let max = base.saturating_add(WINDOW_SIZE as i32)`. For `min_doc == i32::MAX`,
`base` is `0x7FFF_F000` and `base + 4096` is `0x8000_0000` — an `i32` overflow,
saturated to `i32::MAX`. `doc >= max` was then true for the very document that
had *selected* the window, so no clause advanced, `ready` stayed empty, and
`fill_next_window`'s loop re-derived the same `min_doc` **forever**. The
`Disjunction` it replaces terminates on that input.

Java never reaches it because `windowMax = Math.min(max, windowBase + SIZE)` is
bounded by the caller's `max` and therefore by `maxDoc`; this port is pull-shaped
and has no `max` argument, so it dropped the bound without replacing the
guarantee — and then hid the consequence behind a `saturating_add` that looked
like a safety measure.

**Fixed**: the window's exclusive end is computed in `i64`, which removes the
cliff rather than clamping at it. A hang is the worst failure mode to ship, and
reachability is not the argument — the input is a legal doc-id list.

**Tests.** `the_windowed_or_matches_the_min_scan_disjunction_on_every_shape`
runs **eleven** shapes × four `minimum_should_match` values against an
independently computed brute-force answer *and* against the implementation being
replaced — documents exactly on a window boundary (4095/4096), a clause spanning
four windows against one confined to the third, whole skipped windows, doc id 0,
an empty clause, heavy overlap, and three shapes at the top of the doc-id space
(`i32::MAX` alone, `i32::MAX - 1` with `i32::MAX`, and that window entered from
below). Those last three run under a **30-second deadline** on a worker thread
(`run_before`), because the defect is non-termination and an assertion in the
same thread cannot observe it — it would wedge the test binary instead of failing
it. Verified to fail (by timeout, not by wrong answer) against the
`saturating_add` version.

`window_counts_reset_between_windows` pins that a document in window 1 cannot
inherit window 0's tally, in both directions.
`a_single_clause_is_not_the_windowed_shape` pins the selection rule that lives in
`lib.rs`. A `debug_assert!(doc >= base, …)` now names the ascending/duplicate-free
clause contract at the point it would be violated, where before a violation
surfaced as a bare slice-index panic. The `u16` freq bucket's implied
65,535-clause ceiling (`maxClauseCount` defaults to 1,024, and
`matched_boolean_docs` is the only caller — but `WindowedDisjunction::new` is
`pub` and enforces nothing) is written down on the field rather than assumed.

The existing real-Lucene `boolean_query_fixtures.rs` (15 tests) is unchanged and
still passes, which is the cross-engine check on the new path.

### Verdict

Ported for the paths Lucene uses it on, measured, and F-22's premise corrected.
Open, and both are the same shape of follow-on:

- the *scored* `COMPLETE` disjunction (`clause_scores`' `HashMap<i32, f32>` per
  clause, b12's F-24) could use the same window with an `f32` score bucket —
  that is literally Java's `needsScores` branch of
  `scoreWindowIntoBitSetAndReplay`. Left out here because it means reworking
  `clause_scores`' return shape, which is a batch of its own.
- `must`-plus-`minimum_should_match` still builds `should_match_counts`' whole-
  segment `HashMap`, because there the count is a *filter over a conjunction*
  rather than the match set itself. Java has the same asymmetry
  (`BooleanScorerSupplier` returns `null` for that shape and falls back to BS2),
  so this is not a divergence — but the `u16` window bucket would serve it too.
- a **leapfrogging min-should-match scorer** (`MinShouldMatchSumScorer`) is the
  thing whose absence makes Java's density gate moot here. Adding it means
  restoring the gate at the same time; the two are one item, not two.

---

## 5. `explain.rs` — invalid `dense_rank_power: 0`

Java: `codecs/lucene90/IndexedDISI`'s constructor validation — the only legal
`denseRankPower` values are `-1` (the byte `0xFF`, "no rank table") and `7..=15`.

### F-8 `[CORRECTNESS]` test metadata no writer can produce

`explain.rs`'s `synthetic_norms` helper built a `NormsEntry` with
`dense_rank_power: 0`. Unreachable — the entry is dense, so nothing reads the
field — but `dense_rank_bytes(0)` rejects it, so any DENSE `IndexedDISI` block
reached with it fails to decode. Almost-right test input of exactly the kind
that hides a real decode bug. **Fixed**: `0xFF`, with the reason in a comment
pointing at `field_norms.rs`'s `NO_RANK` constant.

`field_norms.rs`'s own literal `0` is b13's deliberate rejection test
(`an_illegal_dense_rank_power_is_rejected_rather_than_guessed`) and is left
alone; that test now also asserts the `field_length` path errors, not just
`norm_inverse`.

---

## Reachability of F-20 / F-16 (asked for, not started)

- **`Occur.FILTER` (b12's F-16) is reachable on its own**, without the scorer
  abstraction. A `filter` clause is a `must` clause whose score contribution is
  dropped; nothing about that needs `TwoPhaseIterator`. Touch points, all
  visited by this batch or adjacent to it: a `filter` list on `BooleanQuery`,
  `matched_boolean_docs`, the two lazy paths (`try_conjunction_lazy`,
  `try_disjunction_lazy`), `clause_scores`, the two boolean MAXSCORE paths,
  `explain_boolean`, the query parser's `+`/`AND` handling, and the FFI clause
  lists. Estimate: **one focused batch**, with a real-Lucene fixture for
  `FILTER` vs `MUST` score equality (the whole point is that they match the same
  documents and score differently).
- **`TwoPhaseIterator` + a cost model (F-20) is not reachable.** It needs every
  clause to be expressible as `(approximation, matches(), matchCost())` and a
  `cost()` on each, which means turning b12's F-26 design — one monomorphised
  free function per `(shape, eager|lazy, scored|unscored)` — into a scorer enum.
  That is a milestone, not a batch, and b12 was right to say so.

---

## Gate

`cargo fmt --all && cargo clippy -p lucene-search -p lucene-ffi --all-targets --
-D warnings && cargo test -p lucene-search -p lucene-ffi` — **green, exit 0**.
**1,295 tests** (854 across 22 `lucene-search` targets, 441 in `lucene-ffi`),
zero failures, including every test added or changed here.

The hang test (F-7) was verified to *fail* — by 30-second timeout, not by a wrong
answer — against the `saturating_add` version it guards, so it is a real tripwire
and not a passing assertion.

**Cross-batch turbulence, recorded because it shaped how long the gate took.**
This batch ran alongside c1/c4/c5 in one shared working tree, and the gate was
red for hours at a time on other people's files:

- `lucene-codecs/src/{vectors,hnsw,postings,stored_fields}.rs` and
  `lucene-index/src/check_index.rs` were each mid-rewrite at various points,
  which blocks *compilation* of everything downstream and therefore this batch's
  tests.
- Clippy lints path dependencies as well as the named packages, so two
  `doc_lazy_continuation` lints (doc-comment indentation) in
  `lucene-index/src/check_index.rs` failed `-D warnings` for `lucene-search` too.
  Left alone rather than fixed — that file was being written *four seconds*
  before one of the checks — and green once its owner landed.
- `lucene-index`'s `IndexWriter` delete/update API was replaced mid-batch with a
  `DeleteQueue`/`Term` one (`update_document(Term, Document)`,
  `delete_documents_by_term(&[Term])`), leaving `lucene-ffi` uncompilable until
  its `writer.rs` call sites were migrated. That migration deletes
  `open_all_segment_sources`/`build_delete_sources` (~150 lines of eager delete
  resolution) and changes *when* a delete takes effect, so it was left to the
  batch that made the API change rather than guessed at here. They landed it;
  this batch's own `lucene-ffi` change — the six multi-segment entry points and
  `boolean_query_term_fields` in `directory_reader.rs` — came through untouched,
  and `term_query_multi_segment_scores_with_real_norms_like_lucene` passes.

None of those were this batch's files and none were touched here.

Per-file line coverage (`cargo llvm-cov -p lucene-search -p lucene-ffi
--summary-only`, after `cargo llvm-cov clean --workspace` — a stale profile from
the concurrent batches otherwise reports every file this batch touched as
"mismatched data" and 15-20 points low):

| file | lines |
|---|---|
| `docid_set.rs` | 99.73% |
| `directory_reader.rs` | 99.00% |
| `field_norms.rs` | 98.95% |
| `lucene-ffi/directory_reader.rs` | 95.84% |
| `multi_segment.rs` | 95.86% |
| `explain.rs` | 95.31% |
| `lib.rs` | **90.52%** |

All above the 95% bar except `lib.rs`, which is a pre-existing gap in b12's
8,650-region file rather than anything this batch introduced (every line added
here is exercised: the new fallback path by the inverted tripwire, the per-scan
cursors by every scoring test, the windowed-OR wiring by the boolean suites).
Worth its own item.

Tier-2 (`quality-reviewer`) review was launched over this batch's changes and
was still running when the batch reported; its findings are not folded in here.

Workspace `clippy`/`llvm-cov` could not be run to completion: c1, c4 and c5 were
mid-edit in `lucene-codecs` and `lucene-index` throughout, and their errors are
not this batch's. One `cargo test` run also failed with
`extern location for lucene_codecs does not exist` — a concurrent rebuild
deleting the rlib mid-run, green on retry.
