# M1.5 — Stop materializing: lazy iteration on the search hot paths

> **Goal:** close the structural gap [M1](m1-performance-gate.md) measured — no
> top-k search path materializes a full posting list — and re-run the gate.

| | |
|---|---|
| **Effort** | M–L |
| **Depends on** | [M1](m1-performance-gate.md) (the measurement that justifies it) |
| **Unblocks** | re-running the M1 gate, and therefore M2–M6 |
| **Status** | in progress |

---

## Why this milestone exists

M1's verdict: lucene-rust is 6×–1000× slower than Java Lucene, because
`resolve_clause_docs` returns `Vec<i32>` — every clause materializes its whole
matching doc list, and every match is scored, for a query that wants 50 results.

What M1 did **not** establish, and what changes the shape of this work: the
primitives to fix it already exist in this repo.

- `lucene_codecs::postings::LazyDocsCursor` is genuinely lazy — it decodes one
  block at a time and skips whole `BLOCK_SIZE` blocks without running
  `ForUtil`/`PForUtil` decode, using the level-0 header's `docDelta`.
- `level0_impacts()`/`level1_impacts()` are decoded and available.
- `search_term_query_scored_maxscore` already uses both.

The defect is that the **multi-segment API and the FFI do not call those
paths**. `search_term_query_multi_segment` routes to
`search_term_query_scored`, which materializes.

Measured directly on the M1 corpus (5M docs, merged), the existing pruned path
against the existing materializing path:

| term | materializing | pruned | speedup |
|---|---|---|---|
| `t0` (high-frequency) | 77,111 µs | **10,059 µs** | **7.7×** |
| `tz` (mid) | 15,243 µs | **3,383 µs** | **4.7×** |
| `t2s` (low) | 5,500 µs | **1,781 µs** | **3.4×** |

So a large fraction of M1's gap is recoverable by wiring, not by inventing. The
remainder is real work.

---

## Scope

### In scope

- Routing the multi-segment and FFI search entry points onto the pruned paths.
- Replacing clause materialization with lazy leapfrog iteration for
  conjunctions and disjunctions.
- Fixing the BM25 `avgdl` computation M1 found, because the gate cannot verify
  recall while scores disagree.
- Re-running the M1 gate and recording a new verdict.

### Out of scope

- **Reaching parity with Java.** That is open-ended and is not what this
  milestone promises. The promise is that no hot path materializes a full
  posting list, and that the gate is re-run honestly against whatever that
  yields.
- New query types, new codec formats, write-path changes.
- The `Weight`/`Scorer` *type hierarchy* as an object model. M1 decided the
  needed thing is the lazy iterator contract, not Java's class graph.

---

## Tasks

### T1.5.1 — Fix BM25 `avgdl`

`FieldNorms::open` averages **decoded** norms — lossy 1-byte quantized values —
to get average field length. Lucene's `BM25Similarity` uses
`avgdl = sumTotalTermFreq / docCount`, from exact counters. Scores differ
systematically by 0.1–0.6%, which is why 19–20 of 20 queries failed M1's recall
cross-check.

`blocktree.rs` already exposes `pub sum_total_term_freq: i64`. This is first
because until it lands, no performance change can be validated: the gate cannot
tell a correct optimisation from a broken one while every query mismatches.

### T1.5.2 — Route multi-segment term search onto the pruned path

`search_term_query_multi_segment` calls `search_term_query_scored`. Route it to
the impacts-pruned equivalent, and do the same for the FFI entry point.
Expected: the 3.4×–7.7× above, on the real API rather than a probe.

### T1.5.3 — Leapfrog conjunctions instead of materializing

`resolve_clause_docs` builds a `Vec<i32>` per clause. Replace the conjunction
path with lazy iteration that advances on the rarest clause, using
`LazyDocsCursor::advance`.

The acceptance signal is behavioural, not just a timing number: conjunction cost
must track the **rarest** clause. M1's scaling experiment is the test —
`and t0 t1z4` must approach the cost of `t1z4` alone, not exceed the cost of
`t0` alone.

### T1.5.4 — Lazy disjunctions

Same treatment for `should` clauses, merged through a priority queue over
cursors rather than by unioning materialized vectors.

### T1.5.5 — Re-run the gate

`scripts/bench-compare.sh` re-runs unchanged — that was the point of building it
in M1. Record a new verdict alongside the old one; do not overwrite it. The
comparison between the two verdicts is the evidence this milestone worked.

---

## Acceptance criteria

- [ ] BM25 scores match Java Lucene within 1e-5 on the M1 query mix, so the
      recall cross-check reports **0 mismatches**.
- [ ] No top-k search path materializes a full posting list — checked in code,
      not inferred from timings.
- [ ] Conjunction cost tracks the **rarest** clause: `and t0 t1z4` costs less
      than `term t0` alone, on the 5M corpus.
- [ ] Measurable improvement over M1's recorded baseline on every query shape,
      with the numbers recorded next to M1's.
- [ ] A new verdict exists in `docs/benchmarks/`, alongside — not replacing —
      M1's.
- [ ] The full gate stays green: fmt, clippy, ≥95% coverage, fixtures, and the
      write-path verifiers.
- [ ] `docs/parity.md` reflects the changed search-execution model.

---

## Risks

- **Scoring changes are silent.** Any change to iteration order or pruning can
  alter which docs enter the top-k. T1.5.1 lands first precisely so that the
  recall cross-check becomes a working tripwire for the rest.
- **Pruning correctness is subtle.** Block-max pruning is only safe if the
  impact bound is a true upper bound. A too-aggressive bound silently drops
  results, and the differential tests are what must catch it.
- **The remaining gap may not close.** Wiring recovers 3.4×–7.7×; Java is
  another ~9.6× beyond that on term queries. This milestone does not promise to
  close that, and the honest outcome may be a second FAIL with a much smaller
  margin. That is still a better-informed decision than today's.
