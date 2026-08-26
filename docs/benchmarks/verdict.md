# M1 performance gate — verdict

**Date:** 2026-08-27 · **Corpus:** 5,000,000 docs, Lucene 10.5.0 ·
**Environment:** [`environment.md`](environment.md) ·
**Raw results:** [merged](results-merged-2026-08-27.txt) · [segmented](results-segmented-2026-08-27.txt)

---

## Verdict: **FAIL**

lucene-rust is **not** faster than Java Lucene. It is between **6× and 1000×
slower**, on every query shape measured, on both corpus variants.

| Gate criterion | Required | Measured |
|---|---|---|
| ≥1.5× throughput on ≥80% of the mix | ≥80% | **0%** (0 of 20 queries) |
| No workload slower than Java | 0 | **20 of 20 slower** |
| Identical hit sets and top-1 scores within 1e-5 | 0 mismatches | **19 of 20 mismatch** |
| FFI overhead under 1µs/call | <1000 ns | **≈0 ns** ✅ |

Ratio distribution over the query mix:

| Corpus variant | segments | ≥1.5× | slower than Java | min | median | max |
|---|---|---|---|---|---|---|
| force-merged | 1 | 0/20 | **20/20** | 0.00 | 0.01 | 0.16 |
| segmented | 15 | 0/20 | **20/20** | 0.00 | 0.01 | 0.28 |

Both variants agree, so this is not an artefact of segment layout. The single
best result across either variant is still a **3.6× loss**; the median is a
**100× loss**.

`PLAN.md` §4 states the consequence plainly: *"if Rust search isn't decisively
faster there, stop before paying for the write path."* It is not faster. The
gate has done its job.

---

## The result reverses at scale, and that is the most important finding

An early smoke run on a 50,000-document corpus showed lucene-rust **winning**
term queries by 1.6×–11×. That conclusion was entirely an artefact of scale.

| `term t0` (high-frequency) | 50k docs | 5M docs |
|---|---|---|
| lucene-rust | 612 µs | **77,404 µs** |
| Java Lucene | 2,180 µs | **1,352 µs** |
| ratio | 1.60× (Rust wins) | **0.01× (Rust loses 57×)** |

Rust's cost grew ~126× for 100× the documents — linear in the posting list.
Java's cost *fell* in absolute terms on 100× the data, because Lucene uses
block-max impacts to skip blocks that cannot enter the top-k. Java is not
scanning the postings at all; this port is scanning all of them.

The gap is therefore **algorithmic, not a constant factor**, and it is invisible
on a small corpus. Had the gate been run against a toy index it would have
returned a confident PASS.

---

## Why: three compounding causes

### 1. Clause resolution materializes entire posting lists

`lucene-search`'s `resolve_clause_docs` returns `Result<Vec<i32>>`. Every clause
of every boolean query builds a complete `Vec` of its matching doc IDs before
any intersection happens. There is no leapfrogging and no skip-list use.

The signature is visible in a scaling experiment (50k corpus, so that a single
term query is still fast):

| Query | Rust | Java | Ratio |
|---|---|---|---|
| `term t0` (high-frequency) | 652 µs | 2,180 µs | 1.60× |
| `term t1z4` (ultra-rare) | 5 µs | 161 µs | 7.38× |
| `and t0 t1z4` | **2,744 µs** | 1,005 µs | **0.05×** |

A conjunction whose rarest clause matches a handful of documents should cost
about what that clause costs — roughly 5 µs. This port pays 2,744 µs, *more than
scanning `t0` by itself*. Its cost tracks the **commonest** clause; Lucene's
`ConjunctionDISI` advances on the **rarest**.

### 2. No impacts, so no dynamic pruning on the collection path

Even a single-term top-50 query scores every matching document. Lucene skips
whole blocks using per-block max impacts. `docs/parity.md` records MAXSCORE as
"partially ported, scoped narrowly on both the single-term and `BooleanQuery`
sides", and the measurement shows that narrow scope does not reach the paths
that matter. This is why `term t0` alone loses 57×.

### 3. Per-call setup is repeated

`ffi_search_term_query_multi_segment` calls `open_segments()` inside its own
per-call guard — 278 ns per search on a one-segment index, scaling with segment
count. Small against the above, but it is pure repeated work.

---

## What passed: the FFI boundary

The C-ABI crossing costs **≈0 ns per call** — below measurement noise, against a
1 µs budget. Measured through the exported symbol, decomposed so that
`open_segments()` setup is not misattributed to the boundary.

The design decision in `PLAN.md` §0.4 — opaque handles, batch result buffers, no
per-doc crossings — is sound and is **not** where the problem lies. Nothing in
this verdict argues against the FFI architecture.

---

## A separate correctness finding: BM25 `avgdl` is wrong

19 of 20 queries failed the recall cross-check, and nearly all trace to one
cause: top-1 scores differ from Lucene's by a systematic ~0.1–0.6%
(`1.314393` vs `1.317220`; `3.869550` vs `3.895010`). Not float noise.

`FieldNorms::open` computes average field length by **decoding each doc's norm
and averaging** — but norms are lossy 1-byte quantized values. Real Lucene's
`BM25Similarity` uses `avgdl = sumTotalTermFreq / docCount`, computed from exact
counters. The lossy average is close but not equal, and the difference cascades
into different top-50 membership at the score boundary.

This is a genuine scoring-fidelity bug, independent of performance. The correct
input is already decoded and available: `blocktree.rs`'s
`pub sum_total_term_freq: i64`. It should be fixed regardless of what happens to
the rest of this roadmap, and it needs a fixture-verified test of its own.

---

## T1.6 — the `Weight`/`Scorer` question, answered

M1 said to revisit the deferred `Weight`/`Scorer`/`BulkScorer` hierarchy **only
if the profile implicated it**. The profile implicates it decisively.

`PLAN.md` §3.5 point 2 argued for monomorphised per-doc loops over Java's
megamorphic `DocIdSetIterator.nextDoc()`, and that argument is still right about
*dispatch*. But it was applied to the wrong layer: what was actually dropped is
not virtual dispatch, it is the **lazy iterator contract** — `advance(target)`,
skip lists, and block-max impacts. Those are not an abstraction tax. They are
the algorithm, and Lucene's speed comes from them, not from its object model.

**Decision: adopt a lazy, skippable iterator abstraction.** Monomorphisation via
enums or generics can be kept — the two concerns are orthogonal. What cannot be
kept is materializing posting lists into `Vec<i32>`.

---

## Recommendation

The plan's own instruction is to stop before funding M2–M6. That instruction
was written for the case "Rust is not decisively faster". This is a stronger
case: Rust is decisively **slower**, by two orders of magnitude, for a
structural reason that is understood.

But the honest reading is that the premise was never tested, not that it was
disproved. Nothing here shows Rust *cannot* be faster. It shows this port has
not implemented the algorithm that makes Lucene fast. The port's decoders are
correct and fixture-verified against real Lucene across a dozen formats; that
work stands. What is missing sits above them.

Three options, in the order I would rank them:

1. **Fix the algorithm, then re-run this gate.** Implement a skippable iterator
   with `advance()`, wire block-max impacts into collection, and stop
   materializing clause doc lists. This harness re-runs unchanged and answers
   the question again. This is a real milestone, not a patch — but it is the
   only path that makes M2–M6 worth funding.
2. **Ship the read path as a library** and stop. The decoders are the genuinely
   valuable, hard-won artefact, and they are correct.
3. **Proceed to M2 anyway.** Not defensible on this data. An OpenSearch
   integration that is 100× slower than the engine it replaces has no adoption
   path.

The recommendation is **option 1**, with the gate re-run as its exit criterion.

---

## Limitations of this measurement

Recorded so the number can be argued with:

- **Synthetic corpus.** Zipf-distributed generated tokens, not natural language.
  Term-frequency distribution is realistic; phrase co-occurrence is not, so
  phrase selectivity is less realistic than real prose. This does not affect the
  conclusion — the gap is 25×–1000× and structural.
- **Scores differ**, so 19 of 20 queries technically compare slightly different
  result sets. The differences are at the top-50 boundary and cannot account for
  a 100× timing gap.
- **Single machine, WSL2**, frequency governor not exposed. Runs are pinned to
  P-cores with `taskset`. The second-machine reproduction criterion is not met.
- **No flamegraphs.** T1.4 specified profiling with `perf`/`cargo flamegraph`
  against Java's async-profiler. `perf` is not installed, needs root, and is
  limited under WSL2. The cause was instead established two other ways, which
  for this defect are stronger than a profile would have been: the code path is
  explicit (`resolve_clause_docs -> Result<Vec<i32>>` materializes by
  construction), and a scaling experiment confirms the behavioural signature —
  conjunction cost tracks the commonest clause rather than the rarest. A
  flamegraph would show time inside those same functions and add nothing to the
  diagnosis. It *would* be needed to rank costs once the algorithm is fixed, so
  installing `perf` is a prerequisite for re-running this gate.
- **Java given every advantage**: Panama Vector API enabled (the runner refuses
  to start without it), time-boxed JIT warmup per query, warm page cache. Any
  residual measurement bias favours Java, and Rust still lost by 100×.
