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

**Best single query: 0.28× of Java. The bar is 1.50×.** The best-case query
needs another **5.4×**; the median query needs **30×**.

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

`term t0` matches 4,997,130 of 5,000,000 documents and this port already skips
98.4% of its blocks. The query costs ~3,500 µs; Java's costs ~1,000 µs. After
five rounds of optimisation the profile has no single dominant cost left — the
largest remaining item is ForUtil block decode at ~13–19%, and the search loop
itself is ~17%. Removing *all* decode would yield well under 1.5×.

Reaching the bar means closing 5.4× on the best query and 30× at the median.
Nothing in the current profile offers that. It requires work that is
architectural rather than incremental:

1. **SIMD `ForUtil`.** Java's decode is Panama-vectorised; this port's is scalar.
   `PLAN.md` §3.5 called for "SIMD from the start"; it was never done.
2. **Block-max WAND** for disjunctions, which still visit every doc in the union
   — and which would also fix the MAXSCORE path that is currently *slower* than
   a plain lazy union.
3. **Phrase queries**, still materializing, at 11–17 s per query.
4. **Multi-level skip selection** (`getSkipLevel`), the missing piece that made
   the failed experiment fail.

Each is a milestone-sized piece of work. Together they might close the gap;
individually none of them will.

---

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
