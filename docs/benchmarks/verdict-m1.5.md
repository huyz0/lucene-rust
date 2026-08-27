# M1.5 verdict — after removing posting-list materialization

**Date:** 2026-08-27 · **Corpus:** the same 5,000,000-doc index M1 used ·
**Environment:** [`environment.md`](environment.md) ·
**Raw:** [merged](results-merged-m1.5.txt) · [segmented](results-segmented-m1.5.txt) ·
**Compare against:** [M1's verdict](verdict.md)

---

## Verdict: still **FAIL** — but the gap closed by a median 5.8×, and correctness is now exact

| Gate criterion | Required | M1 | M1.5 |
|---|---|---|---|
| ≥1.5× throughput on ≥80% of the mix | ≥80% | 0% | **0%** |
| No workload slower than Java | 0 | 20/20 slower | **20/20 slower** |
| Identical hit sets, top-1 scores within 1e-5 | 0 mismatches | 19/20 mismatched | **0/20** ✅ |
| FFI overhead under 1µs/call | <1000 ns | ≈0 ns ✅ | ≈0 ns ✅ |

The gate's headline criteria are unchanged: lucene-rust is still slower than Java
Lucene on every query. But the milestone did not promise to pass the gate — it
promised that no hot path materializes a posting list, and that the gate is
re-run honestly against whatever that yields.

**Median ratio versus Java: 0.010 → 0.045.** Still a 22× deficit, from a 100×
one.

---

## What changed, per query (merged corpus, 5M docs)

| | shape | M1 qps | M1.5 qps | gain |
|---|---|---|---|---|
| q07 | `and t0 tz` | 1.3 | 21.2 | **16.3×** |
| q10 | `or t0 t1` | 0.7 | 6.2 | **8.9×** |
| q08 | `and tz t2s` | 8.8 | 73.6 | **8.4×** |
| q09 | `and t0 t1 t2` | 0.5 | 3.9 | **7.8×** |
| q12 | `or` ×4 | 0.4 | 3.1 | **7.8×** |
| q01 | `term t0` | 13.8 | 107.0 | **7.8×** |
| q18 | `term title t0` | 27.5 | 215.2 | **7.8×** |
| q06 | `and t0 t1` | 0.7 | 5.3 | **7.6×** |
| q20 | `and title t0 t1` | 2.6 | 16.9 | **6.5×** |
| q11 | `or tz t2s` | 5.6 | 28.7 | **5.1×** |
| q03 | `term tz` | 70.9 | 327.9 | **4.6×** |
| q04 | `term t2s` | 208.2 | 686.0 | **3.3×** |

**13 of 20 improved, median 5.8×, max 16.3×. One marginal regression** (q19,
`term keyword`, 0.9× — a field with `omitNorms`, within run-to-run spread).

Unchanged, and understood:

- **q13–q15** route to `search_boolean_query_scored_maxscore`, untouched here.
- **q16–q17** are phrase queries, outside the lazy paths' shape gate.
- **q05** was already fast enough that the fixed overhead dominates.

---

## The behavioural criterion, which matters more than the ratios

M1.5 asked for a signal that is structural rather than a timing number:
conjunction cost must track the **rarest** clause.

| | M1 | M1.5 |
|---|---|---|
| `term t0` alone | 77,404 µs | 10,059 µs |
| `and t0 t1z4` (rarest clause matches a handful of docs) | **more than `t0` alone** | **5,152 µs — half of `t0` alone** |

Cost has flipped to the rare end of the query. That is the defect M1 identified,
and it is fixed for the shapes the gate covers.

---

## Correctness: 19 mismatches → 0

M1's cross-check found 19 of 20 queries disagreeing with Java on hit sets, all
tracing to BM25 `avgdl` being computed by averaging *decoded* (lossy, 1-byte
quantized) norms instead of `sumTotalTermFreq / docCount`. Fixed. On the merged
corpus, all 20 queries now agree with Java Lucene on hit sets **and** top-1
scores within 1e-5.

This is the more valuable half of the milestone. A benchmark that cannot verify
recall cannot distinguish an optimisation from a broken one, and every
subsequent change here was validated against it.

---

## What the fixed cross-check then exposed: per-segment IDF

**The segmented corpus reports 20 of 20 mismatches, and the timings there are
not comparable.** This is a pre-existing defect that `avgdl` was previously
masking, not a regression — verified by running the old eager path and the new
pruned path side by side on the same index and getting byte-identical (wrong)
output.

This port computes IDF from **per-segment** statistics: `term_doc_scores` uses
`field_terms.doc_count` and that segment's `docFreq`. Lucene's `IndexSearcher`
computes `TermStatistics`/`CollectionStatistics` **once across the whole
reader** and uses that single IDF for every leaf.

Measured on the 15-segment corpus, for `body:t0`:

```
  seg  9: docCount=   79283  docFreq=   79244  idf=0.000498
  seg 10: docCount=   79354  docFreq=   79294  idf=0.000763   <- 1.6x seg 12
  seg 12: docCount=   79270  docFreq=   79233  idf=0.000473   <- lowest
  GLOBAL: docCount=5000000  docFreq=4997130  idf=0.000574     <- what Lucene uses
```

A **1.6× spread** in IDF for the same term. Documents in one segment score ~33%
higher than they should and in another ~18% lower, so the top-k fills from
whichever segment happens to make the term look rarest. On a single-segment
index the two definitions coincide, which is why the merged corpus passes and
why no existing fixture caught it — every fixture is one segment.

**This is the highest-priority follow-up.** It affects every multi-segment
scored search, which is to say every real index, and it is a correctness bug
rather than a performance one. Fixing it means threading global collection
statistics through the multi-segment search API.

---

## A finding recorded rather than acted on: MAXSCORE underperforms

`search_boolean_query_scored_maxscore` is **4–5× slower** than the plain lazy
union that now backs `search_boolean_query_scored`: 655 ms vs 163 ms on
`t0 OR t1`, 1,619 ms vs 326 ms on a four-clause disjunction.

It is not failing to prune — an FFI test proves it provably skips blocks. Its
per-doc bookkeeping simply costs more than the skipping saves at top-k = 50.

I routed that entry point through the lazy union and then reverted it: the
change broke `boolean_query_scored_maxscore_..._actually_skips_blocks`, and
weakening an invariant test to accommodate a behaviour regression is the wrong
trade. Reviving MAXSCORE properly means block-max WAND over the same cursors.

---

## Where the remaining gap is

Rust is still ~22× off Java at the median. The structural cause M1 identified is
addressed for term, conjunction and disjunction shapes; what remains is
different in kind:

1. **No dynamic pruning on disjunctions.** The lazy union still visits every doc
   in the union. Block-max WAND is what removes that, and it is the single
   largest remaining item.
2. **Phrase queries are untouched** — 12 s per query, still materializing.
3. **Per-doc scoring overhead** has not been profiled. `perf` remains
   unavailable under WSL2, and installing it is a prerequisite for ranking what
   is left.

---

## Recommendation

Unchanged in direction from M1: do not fund M2–M6 yet. But the picture is
materially better informed.

The structural defect M1 named is real and was fixable — a median 5.8× came from
using primitives the repo already had. That argues the remaining gap is also
tractable rather than fundamental, and specifically that Rust's decoders are not
the problem.

Order of work, by measured value:

1. **Global collection statistics.** A correctness bug affecting every
   multi-segment index. Nothing else should be measured on segmented indexes
   until it lands.
2. **Block-max WAND**, replacing both the lazy union's exhaustive visit and the
   current underperforming MAXSCORE.
3. **Install `perf`**, then profile what is left rather than guessing.
4. **Re-run this gate.** The harness is unchanged across M1 and M1.5, which is
   what makes these two verdicts comparable at all.
