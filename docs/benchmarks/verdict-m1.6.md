# M1.6 verdict — Lucene source sweep

**Date:** 2026-08-28
**Machine/method:** unchanged from M1 — see [`environment.md`](environment.md).
CPU-pinned to a P-core pair, load-guarded (`BENCH_MAX_LOAD`), time-boxed warmup
and measurement, Java run with `--add-modules jdk.incubator.vector`.

---

## What was measured

Two kinds of number, and the difference between them is the point of this
milestone.

**Component**, via `scripts/bench-micro.sh` — the same operation run on both
engines over identical bytes, joined and divided. New in M1.6; M1 had no such
measurement.

**End to end**, via `scripts/bench-compare.sh` — the M1 query set, unchanged, so
the numbers are directly comparable to `verdict.md` and `verdict-m1-e2e.md`.

---

## Component results

Ratios are `java_ns / rust_ns`: **above 1.0 means this port is faster than
Lucene**.

| component | before | after |
|---|---|---|
| `ForUtil.decode`, median over `bitsPerValue` 1..=31 | 0.75x | **2.36x** |
| posting-list `nextDoc()`, median over 4 terms | 0.20x | **1.91x** |
| `DirectReader.get`, median over 14 widths | 0.82x | **1.86x** |
| sparse `IndexedDISI` lookup, 100k present docs | 326 us | **178 ns** |
| reader open, merged corpus | 551.9 ms | **52.7 ms** (10.5x; still 155x Lucene) |
| RSS after reader open | 1,690 MB | **70 MB** (24x) |

Re-measured with the noise-aware harness (`--reps 3`, interleaved engines,
measured noise floor 1.03x-1.09x), not with the single-run instrument the
earlier figures came from.

The three decode kernels are now faster than Lucene's — the postings decode
against Lucene's Panama-vectorised `MemorySegmentPostingDecodingUtil`, with no
SIMD of our own.

Reader open is the opposite result and is left in the table because it is the
most important number in this milestone: 560 ms against Lucene's 4.2 ms. See
"What the remaining gap is" below.

## End-to-end results

Merged corpus (5M documents, one segment), against `verdict-m1-e2e.md`'s numbers:

| query | | M1-e2e | M1.6 |
|---|---|---|---|
| q01 | `term body:t0` | 0.53x | **0.84x** |
| q02 | `term body:t1` | 0.37x | 0.54x |
| q03 | `term body:tz` | 0.29x | 0.37x |
| q04 | `term body:t2s` | 0.39x | 0.59x |
| q05 | `term body:t1z4` | 0.48x | 0.71x |
| q06 | `and t0 t1` | 0.22x | 0.31x |
| q07 | `and t0 tz` | 0.11x | 0.15x |
| q08 | `and tz t2s` | 0.17x | 0.18x |
| q09 | `and t0 t1 t2` | 0.14x | 0.22x |
| q10 | `or t0 t1` | 0.27x | 0.37x |
| q11 | `or tz t2s` | 0.08x | 0.16x |
| q12 | `or t0 t1 t2 t3` | 0.11x | 0.27x |
| q13 | `or_maxscore t0 t1` | 0.26x | 0.38x |
| q14 | `or_maxscore tz t2s` | 0.08x | 0.16x |
| q15 | `or_maxscore t0..t3` | 0.11x | 0.27x |
| q16 | `phrase t0 t1` | 0.04x | **0.31x** |
| q17 | `phrase t1 t2` | 0.04x | **0.38x** |
| q18 | `term title:t0` | 0.34x | 0.45x |
| q19 | `term keyword:t0` | 4.53x | 4.46x |
| q20 | `and title t0 t1` | 0.15x | 0.17x |

**Recall mismatches: 0 of 20 on both corpus variants.** The 15-segment corpus
reached zero for the first time; it stood at 13 when M1.6 opened.

## Gate

`>= 1.5x on >= 80% of queries` — **FAIL**, 1 of 20, unchanged from M1.

Median ratio moved 0.15x -> 0.34x. Every query improved or held; none regressed.

---

## Reading these two tables together

This is the useful part of the milestone, and it is a lesson about method rather
than about any one optimisation.

The component numbers say both decode kernels are now *faster than Lucene's*.
The end-to-end numbers say queries are still 3x-6x slower. Both are true, and
neither could have been derived from the other.

M1 profiled end to end, got a flat profile — largest single item 14.78% — and
concluded there was no single mechanism left to fix. That conclusion was wrong,
and the reason it was wrong is instructive: a profile tells you where time goes,
not whether that is the *right amount* of time. `LazyDocsCursor::next_doc` spent
14.5 ns per document. Nothing about that number is alarming until you put
Lucene's 2.9 ns next to it. Then it is a 5x defect, and it turned out to be a
binary search computing an offset that is always 1.

Six of the fifteen findings in [`../sweep/findings.md`](../sweep/findings.md)
were of that shape: not slow code, but code doing a reasonable thing that Lucene
does not do at all — a loop nest in the order that prevents vectorisation, a
scratch buffer in the wrong scope, an impact bound evaluated per document where
`ImpactsDISI` evaluates it per block, a `ln()` recomputed per document where
`BM25Similarity` computes it once per term.

None of them is exotic. All of them were found by reading the two implementations
side by side and then measuring the same operation on both.

## What the remaining gap is

Named, measured, and not fixed here:

0. **The term dictionary is materialized when a segment is opened.**
   *Partly closed, and the original diagnosis was wrong.* Most of the 560 ms was
   `DirectoryReader` copying every mmap'd postings file onto the heap -- 1.57 GB
   of memcpy per open -- which is now fixed: 52.7 ms and 70 MB, from 551.9 ms
   and 1,690 MB. The residue, 52.7 ms against Lucene's 0.34 ms, *is* the eager
   materialization: `blocktree::FieldTerms` holds a
   `Vec<(Vec<u8>, TermStats, TermMetadata)>` of every term, one allocation each,
   where `SegmentTermsEnum` walks the `.tip` FST to a block and scans it in
   place. A profile of what is left puts ~28% in allocation, sorting 579,255
   heap-keyed tuples, and dropping them, and ~19% in the block decode itself.
   Still milestone-sized, and it still blocks M2/M5 independently of query
   speed, because a search engine reopens readers on every refresh.

1. **Phrase matching materializes every position of every document.** About 50%
   of a phrase query is `malloc`/`free`/`memcpy`: `term_doc_positions` builds a
   `Vec<Vec<i32>>`, one allocation per matching document — roughly five million
   per query on `body:t0`. This is exactly what M1 diagnosed and M1.5 fixed for
   the doc stream, never done for the position stream. It needs a
   `LazyPositionsCursor` in `lucene-codecs` and a phrase matcher rewritten
   against it.
2. **`tf_norm` is two divisions where Lucene's `BM25Scorer.doScore` is one.**
   *Attempted and reverted.* Switching the two boolean scoring loops to
   `weight - weight / (1 + freq * normInverse)` breaks two cross-path
   consistency tests immediately, because `clause_scores` -> `term_doc_scores`
   -- the path a boolean query takes when its shape does not fit a lazy one --
   still uses the multiply form, and the two must agree. The change is therefore
   not local: it means moving *every* scoring path onto Lucene's expression at
   once and re-deriving every exact-score fixture against
   `IndexSearcher.explain()` rather than against this port's own previous
   output. Worth doing as its own task with its own fixtures; the saving is one
   division per document per clause.
   14% of a disjunction's profile. The term path already uses Lucene's form; the
   boolean paths cannot adopt it without re-deriving their expected values
   against `IndexSearcher.explain()`, since it is not bit-identical to what they
   produce today.
3. **WAND essential/non-essential clause partitioning.** The remaining structural
   difference from `WANDScorer`, and the one the disjunction's own comment
   already admits to.

Each is milestone-sized. The sweep's job was to find them, measure them, and say
what the fix looks like — not to do them.


---

## Addendum: after the counting instrument

The verdict above was written before both engines were instrumented to count
work. Doing that found the divergence the timings could not, and fixing it moved
the end-to-end numbers again.

### Final end-to-end, merged corpus

| | median ratio | queries slower than Java | recall mismatches |
|---|---|---|---|
| M1 (milestone opened) | 0.15x | 19 of 20 | 13 (segmented) |
| M1.6 first verdict | 0.34x | 19 | 0 |
| after `advanceShallow` | 0.46x | 16 | 0 |
| after the linear-scan `advance` and the score-map removal | **0.53x** | **16** | **0** |

`q01 term body:t0` 1.16x, `q02` 1.04x, `q18 term title:t0` 1.09x -- faster than
Lucene on their own queries. The M1 gate is still FAIL at 1/20.

### Decode utilisation: documents unpacked that were actually scored

| query | before | after |
|---|---|---|
| `and t0 tz` | 1.3% | 14% |
| `or t0 t1` | 1.4% | 28% |
| `term body:t0` | 5.8% | 56% |
| `or tz t2s` | 92% | 92% |

### The one remaining divergence, precisely stated

Documents *scored*, against Lucene, on the final tree:

| query | this port | Lucene | ratio |
|---|---|---|---|
| `or t0 t1 t2 t3` | 4,121,444 | 1,625 | 2536x |
| `and t0 t1 t2` | 1,151,317 | 1,451 | 793x |
| `or tz t2s` | 1,334,994 | 11,505 | 116x |
| `term body:t0` | 82,564 | 1,425 | 58x |

Unchanged by this work, because `advanceShallow` removes wasted *decoding*, not
wasted *scoring*. Every query still slower than Lucene is slower for this reason
and no other -- the per-document costs are all now faster than Lucene's, and the
decode waste is largely gone.

The fix is a per-span clause partition. A static-maxima partition was built and
measured and is a net regression; see `docs/sweep/findings.md`. That is the next
milestone, and it now has both a target and a baseline.
