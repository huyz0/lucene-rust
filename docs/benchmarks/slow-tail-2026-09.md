# Why a quarter of the M1 mix is still under 1.5x (September 2026)

M1's bar is >=1.5x Lucene on 80% of `benchmarks/queries.tsv`. On the M6 build
it held on 69-75% ([`perf-gate.md`](perf-gate.md)). This is where the rest
goes, measured rather than guessed, and what was fixed.

## Method

- **Machine:** 4-core Xeon @ 2.8 GHz (AVX-512 capable; the build targets
  x86-64-v3), 15 GB, cloud VM. Not the development machine of
  [`environment.md`](environment.md), so absolute ratios differ from the
  committed baselines; both engines ran on the same machine, pinned to cores
  2-3, load average under 1.0.
- **Corpora:** `scripts/bench-corpus.sh --docs 1000000`, as the nightly job
  builds them (`merged`, and `segmented` at 6 segments).
- **Ratios:** `scripts/bench-compare.sh`, 2 s warm-up, 3 s measured per query.
- **Work counts:** both runners report the documents their collector
  received (`scored`), in one untimed run. Equal counts mean the gap is speed
  per document; unequal ones mean the engines did different work.
- **Profiles:** `perf record -F 2000` on the Rust runner, one query at a time.

## Before

`merged`: 60/87 at >=1.5x (69%), median 1.80x. `segmented`: 61/87 (70%),
median 2.20x. Nothing slower than Lucene; no recall mismatch. Every miss was
between 1.02x and 1.49x -- no outlier pointing at a missing algorithm.

## What the counts and profiles say, by group

| group | queries | scored, Rust vs Lucene | where Rust's time goes | verdict |
|---|---|---|---|---|
| exact phrases | q16, q17, q56-q60, q63 | identical (q16: 269,713 vs 269,731) | merging positions (`phrase_freq_exact_impl`, `score_phrase`) 67%, `next_position` 15% | same work; 0.52 vs 0.81 us per document already. Lucene does not skip here either, so its merged-impacts `ImpactsDISI` (which this port does not have, `exec/phrase.rs`) would not change these queries |
| term / AND / OR top-k | q02, q03, q06, q07, q10, q12, q13, q20 | Rust collects **34-60%** of Lucene's | per-block work: `expand_bitset` 24%, block bounds (`level0_max` 14-17%, `decode_impacts_into` 6-16%, `advance_shallow` 4-13%), norms 6-12% | Rust prunes at least as well; the cost is per decoded or bounded block |
| point range | q28, q71 | identical | **libc `memcmp` 22%**, doc-id sort 17% | fixed (below) |
| regexp | q32, q34 | Rust 50 vs 2,000-3,600 | automaton x term-dictionary walk 33%, tail-block postings reads 22% | term expansion, not scoring |
| sorted | q74-q80 | not comparable: the sorted path does not bump the Rust counter | points-based competitive pruning 19%, `memmove` 6% | not investigated further |

## Fixed

1. **Point comparisons called `memcmp`.** `RangeVisitor` compared each
   4- or 8-byte value as a `[u8]` slice: two libc calls per point. Lucene's
   visitor uses `ArrayUtil.getUnsignedComparator`, one unsigned big-endian
   integer compare; `points::FixedWidthRange` does the same for
   one-dimensional `int`/`float`/`long`/`double` points.
   `a_fixed_width_range_agrees_with_the_byte_wise_comparison` pins it (and
   fails with a signed compare).
2. **`expand_bitset` branched on empty bytes.** At a dense block's density
   about one byte in eight is empty -- too often to predict. The branch is
   gone; an empty byte writes nothing the count keeps.

| query | merged before -> after | segmented before -> after |
|---|---|---|
| q03 `term tz` | 1.36x -> 1.59x | 1.44x -> 1.65x |
| q07 `t0 AND tz` | 1.21x -> 1.42x | 1.14x -> 1.48x |
| q20 `title: t0 AND t1` | 1.40x -> 1.55x | 1.34x -> 1.46x |
| q71 `num:[1000 TO 2000]` | 1.32x -> 1.42x | 1.56x -> 1.75x |
| q28 points `num` 0-1000 | 2.48x -> 3.05x | 1.77x -> 2.23x |

Interleaved A/B of the Rust side alone (same binary otherwise): q71 +20%,
q28 +17%, q03 +23%, q07 +25%, q20 +16%.

## After

`merged`: 60/87 at >=1.5x (69%), median 1.81x. `segmented`: 64/87 (74%),
median 2.21x. Nothing slower than Lucene; no recall mismatch.

The merged count did not move although every targeted query did: ten
queries sit between 1.40x and 1.50x, and a run moves each by about +-9%
([`perf-gate.md`](perf-gate.md)), so some crossed down as others crossed up
(q02 1.52x -> 1.49x, q76 1.37x -> 1.15x, neither touched by the change). A
count at a threshold is a noisy measure of a distribution sitting on it.

## Tried and rejected

- **Memoising a block's bound by its encoded impacts** (for the OR group,
  where bounds are 30-40% of the time). Measured first: a frequent term
  almost never repeats an impacts list -- `t0 OR t1` hit a 16-entry memo once
  in 3,843 blocks. Reverted.
- **An uninitialised staging buffer in `expand_bitset`** (skipping a 1,280-byte
  zero fill). No difference beyond noise in three interleaved runs; not worth
  an `unsafe` block.

## What is left, and why

- **Term/boolean top-k** does the same block-level work as Lucene -- the same
  `MaxScoreCache`-style bound per block, the same batch decode, norms and
  scoring -- and is 1.3-1.6x faster at it. Closing the rest needs cheaper
  blocks, not fewer: the bound over a block's impacts (one BM25 evaluation per
  `(freq, norm)` pair), the varint impacts decode, and the norms gather.
- **Phrases** are bounded by decoding positions, which both engines do for
  the same documents.
- **Regexp and sorted** spend their time outside scoring (term expansion,
  points pruning) and need their own profiles against Lucene's.
