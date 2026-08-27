# M1 end-to-end — final verdict

**Date:** 2026-08-27 · **Corpus:** 5,000,000 docs, Lucene 10.5.0, merged ·
**Raw:** [results](results-merged-m1e2e.txt) ·
**Prior:** [M1](verdict.md) · [M1.5](verdict-m1.5.md)

---

## Verdict: the gate **cannot be passed** by optimising this architecture

Search is now **7.2× faster at the median** than when M1 first measured it
(max 21.1×), with **exact scoring agreement** against Java Lucene. That is real
and it is worth having. It is also not close to the bar.

| Gate criterion | Required | M1 | Now |
|---|---|---|---|
| ≥1.5× throughput on ≥80% of the mix | ≥80% | 0% | **0%** |
| No workload slower than Java | 0 | 20/20 slower | **20/20 slower** |
| Identical hit sets and top-1 scores | 0 mismatches | 19/20 mismatched | **0/20** ✅ |
| FFI overhead <1µs/call | <1000 ns | ≈0 ns ✅ | ≈0 ns ✅ |

**One query now beats Lucene** — `keyword:t0` at **3.71×**, after a global score
bound closed a hole where fields without frequencies were never pruned at all.
It is 1 of 20; the bar is 80%. Every other query remains slower, the median by
15×.

Median total gain versus M1: **7.9×**, max **2,597×**.

---

## The whole journey, per query

| | shape | M1 | M1.5 | now | total | vs Java |
|---|---|---|---|---|---|---|
| q18 | `term title t0` | 27.5 | 215.2 | **580.4** | **21.1×** | 0.19× |
| q01 | `term t0` | 13.8 | 107.0 | **283.5** | **20.5×** | 0.28× |
| q02 | `term t1` | 13.7 | 99.1 | **276.1** | **20.2×** | 0.22× |
| q07 | `and t0 tz` | 1.3 | 21.2 | **23.6** | **18.2×** | 0.02× |
| q10 | `or t0 t1` | 0.7 | 6.2 | **6.6** | **9.4×** | 0.02× |
| q06 | `and t0 t1` | 0.7 | 5.3 | **6.1** | **8.7×** | 0.02× |
| q03 | `term tz` | 70.9 | 327.9 | **520.5** | **7.3×** | 0.16× |
| q20 | `and title t0 t1` | 2.6 | 16.9 | **18.1** | **7.0×** | 0.12× |

Median total gain **7.2×**; median ratio versus Java improved 0.010 → 0.050.

Untouched, and honestly so: `q13`–`q15` (the MAXSCORE entry point),
`q16`–`q17` (phrase queries, still materializing), `q19` (a marginal 0.8×
regression on an `omitNorms` field).

---

## What produced the gains

1. **BM25 `avgdl` corrected** — the recall cross-check went 19 mismatches → 0,
   which is what made every later change verifiable.
2. **Routing to the pruned term path** — 3.3×–7.9×. The pruned function already
   existed; nothing called it.
3. **Leapfrog conjunctions, lazy disjunctions** — 6.3×–16.3× and 7.1×–12.5×.
4. **Caching the impact bound per block** — `max_score_for_impacts` was 25.2% of
   runtime, recomputed per document instead of per block.
5. **Level-1 span skipping** — 2.13×, the single largest late win.

Items 4 and 5 came directly from reading Lucene's `MaxScoreCache` and
`ImpactsDISI`. Neither was visible from the Rust side alone; both were obvious
within minutes of opening the reference.

---

## One experiment that failed, recorded because it is informative

Profiling suggested the remaining cost was decoding block bodies that are then
skipped, so I added `skip_uncompetitive_blocks` — walk forward on block headers
alone, deciding competitiveness before any `ForUtil` unpack. Lucene's
`ImpactsDISI.advanceTarget` does exactly this.

**It made things 41% slower** (276.8 → 163.3 qps) and was reverted.

The reason is the part I had not thought through: walking *level-0* headers one
block at a time competes with the level-1 span skip that had just won 2.13×.
Reading 32 block headers costs more than reading one level-1 entry. Lucene does
not have this problem because `getSkipLevel` picks the **highest** level whose
bound is under the threshold — the shallow walk and the multi-level skip are one
mechanism there, not two fighting each other.

Copying a piece of a design without its selection logic made it worse.

---

## Why the remaining gap is not incremental

This was checked rather than assumed, and the check changed my answer twice.

Early on I concluded the ceiling was near, reasoning from profile percentages
that at most 1.4x remained. That was wrong: `keyword:t0` then went **3,186x**
faster from a single missing mechanism, because percentages assume the amount of
work is fixed and a missing mechanism changes the amount of work. Every
subsequent win -- block-max conjunction, block-max disjunction, the maxscore
routing -- came from asking "is a mechanism absent?" rather than "is this code
slow?".

That question is now exhausted. The final profile of `term t0`, on an idle
machine, is **flat**:

```
  14.78%  LazyDocsCursor::advance
  14.55%  search_term_query_scored_maxscore
  11.43%  for_util::split_ints
   7.85%  search_term_query_scored_maxscore::{closure}
   5.99%  postings::decode_full_block_body
   5.21%  postings::decode_impacts
   4.60%  TopDocsCollector::collect
   3.02%  for_util::for_decode
```

No item exceeds 15%. Every earlier profile had one at 25% or more, and each of
those was a missing mechanism. Block decode totals 20.4% -- removing *all* of it
yields 1.26x. The search loop is 22.4% -- removing it entirely yields 1.29x. The
gate needs **5.2x** on this query and **10x** at the median.

A flat profile after the mechanisms are in place means the remaining difference
is not one thing. It is that this port does uniformly more work per unit of
progress than Lucene does, across decode, iteration, and collection. Closing
that is not an optimisation; it is a rewrite of the execution layer against a
decade of tuning.

Specifically what is left, and why none of it is enough alone:

1. **SIMD `ForUtil`** -- `PLAN.md` §3.5 asked for it and it was never built. But
   decode is 20.4%, so even an infinitely fast decoder gives 1.26x.
2. **WAND's essential/non-essential clause partitioning** -- the flat unions
   (`or tz t2s`, `or x4`) do not move under a summed bound, because with a
   common high-frequency term that bound stays above the threshold nearly
   everywhere. This is the largest single remaining item.
3. **Phrase queries** -- still materializing, 11-17 s per query, ~50x off.
4. **Per-document cost across the board** -- collection, norms reads and cursor
   advance together are ~25% and have no single fix.

## Recommendation

**The M1 gate should be treated as answered: no.** Not "not yet" — the question
M1 posed was whether this port is decisively faster than Java Lucene, and after
a 7.2× median improvement it remains 20× slower at the median. `PLAN.md`'s own
instruction applies.

What the work does establish, and it is worth stating plainly:

- The port's **decoders are not the problem**. They are correct, fixture-verified
  against real Lucene across a dozen formats, and now scoring-exact.
- The gap was **execution strategy**, and it responded to being fixed — 7.2×
  from primitives the repository already had.
- The **FFI design is sound** and never appeared in any profile.

So the honest framing is not "Rust cannot be faster". It is that this port is a
correct Lucene *reader* whose query execution is roughly a decade behind
Lucene's, and closing that is a research-and-engineering programme rather than a
milestone. The read-only library is a genuinely valuable artefact today. M2–M6
as scoped in the roadmap are not fundable on this evidence.
