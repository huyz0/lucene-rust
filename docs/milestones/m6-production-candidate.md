# M6 — Production candidate

> **Goal:** a build a team would actually deploy, with a documented rollback
> path.

| | |
|---|---|
| **Effort** | M |
| **Depends on** | [M5](m5-engine-integration.md) |
| **Unblocks** | shipping |
| **Status** | not started |

---

## Why this milestone exists

Everything before this proves individual properties: the bytes are right, the
engine is faster, the shard works. This milestone proves the properties hold
*together*, *over time*, and that there is a way back if they do not.

`PLAN.md`'s Phase 6 exit criteria first asked for a multi-day soak test with
random restarts. That was replaced (see [T6.1](#t61--resource-bounds-instead-of-a-soak)):
a soak cannot run in CI, takes days per iteration, and only says *that*
something grew. Tight-loop tests on each thing that can accumulate say *what*
grew, on the iteration it grew, in seconds, on every push.

The second half — the rollback path — matters because backward-codecs are out
of scope project-wide. A cluster that moves onto the Rust engine and cannot
move back is a cluster nobody will move in the first place.

---

## Scope

### In scope

- Resource-bound tests on everything that can accumulate across repeated
  operations (in place of a soak).
- Continuous performance regression tracking.
- Completing the parity ledger and publishing an operator-facing feature
  matrix.
- The documented, tested upgrade and rollback procedure.
- Licensing and attribution audit.

### Out of scope

- New features of any kind.
- Anything on the deferred list. If it is deferred, it stays deferred and gets
  documented as such.

---

## Tasks

### T6.1 — Resource bounds instead of a soak

A multi-day soak was built and run (a three-node cluster under mixed load
with random restarts and SIGKILLs). It was dropped as a gate: it cannot run in
CI, a leak costs days per iteration to see and to confirm fixed, and it
reports growth, not its cause. Its three trial runs found only defects in the
soak harness itself. The CI step that ran `concurrent_soak` for two minutes
went too: its correctness half is covered by the concurrent writer's unit
tests and the interop matrix's threaded direction. The example remains a
manual tool.

In their place, one test per thing that can accumulate. Each drives one
operation in a tight loop and fails on the iteration a bound is broken. Each
was seen to fail against the defect it targets (the negative control).

| What can accumulate | Test | Bound | Negative control |
|---|---|---|---|
| The writer's per-segment caches (`segment_versions`, the deleter's file sets) and files on disk: the two leaks M4's soak found | `index_writer::tests::per_segment_state_and_files_track_the_live_segments_over_many_rounds`: 400 rounds of adds, deletes, flushes, commits, 50+ merges | entries ⊆ live segments; no file of a dead segment; one `segments_N` | M4's pruning removed: fails at round 3 |
| The same under the concurrent writer, whose merges and flushes stamp no commit | `concurrent_writer::tests::merges_between_commits_keep_per_segment_state_to_the_view`: 300 rounds, a commit every 50 | versions ⊆ view; file sets and `.si` on disk ≤ view + last commit | view-change pruning removed: fails at round 40 |
| The NRT refresh loop: reader handles, their `mmap`s, descriptors, commit holds, old commits | `engine_writer::tests::the_nrt_refresh_loop_releases_every_file_mapping_and_hold`: 150 refreshes as OpenSearch drives them, then `CheckIndex` | no fd under the index; no mapping of a deleted file; no file neither the commit nor the hold names; nothing mapped after the last close | reader not closed: round 16; hold not released: round 8 |
| Heap behind the C ABI: handles, result buffers, error strings, reopen chains | `crates/lucene-ffi/tests/resource_bounds.rs`: a counting global allocator, 2 × 2,000 calls per path | ≤ 1 KiB growth per window (measured: 0); fd and thread counts unchanged | freed error buffer, results handle, previous reader each skipped: 218 KB, 179 KB, 5.2 MB |
| Process-wide pattern caches | `wildcard::…stays_bounded_under_distinct_patterns`, `regexp::…stays_bounded_under_distinct_patterns`: 3 × the bound in distinct patterns | ≤ 512 / ≤ 64 entries | eviction removed: 513 / 65 |

Already bounded by existing tests: the per-segment native query cache
(`exec::cache` eviction tests) and handle-slot reuse (any slot not reused
grows the heap in the allocator test).

**What these cannot catch:** growth that only a mix of operations across
threads produces, which none of the loops runs; memory a cache legitimately
fills during a warm-up and then holds; the JVM side's own heap; disk growth
from soft-deleted history a retention policy keeps (that is policy, not a
leak).

**What they found.** Four defects. Two are bounded lags, fixed:
`segment_versions` kept a merge's sources until the next `segments_N` (under
the concurrent writer, until the next commit); the deleter forgot a merged
segment's file set only at the next checkpoint. One is a false failure in this
port's `CheckIndex`, fixed: a soft delete applied as a doc-values update was
counted from the base column, so a segment real Lucene's `CheckIndex` passes
failed ours. One is queued: the concurrent writer merges only committed
segments, so between commits its segment count grows with every flush. Nothing
in production uses that writer yet.

### T6.2 — Continuous performance regression tracking

[M1](m1-performance-gate.md)'s harness measured once. This makes it a
standing gate.

- A nightly CI job running `scripts/bench-compare.sh` against a fixed corpus.
- Results recorded over time so a regression is visible as a trend, not
  discovered during a release.
- An alert threshold tied to M1's bar: if the ratio drops below it, that is a
  build failure, not a note. The gate that justified the project should not be
  allowed to quietly stop holding.

### T6.3 — Complete the parity ledger and publish the feature matrix

Two different documents for two different audiences:

- **`docs/parity.md`** — per-Java-file, for contributors. Complete for the
  supported matrix, with every remaining `partial` or `deferred` entry stating
  precisely what is missing. `AGENTS.md` invariant #7 requires it to be current
  anyway; this is the final sweep.
- **An operator-facing feature matrix** — what works natively, what falls back
  to Java Lucene, what is unsupported. Written in OpenSearch's vocabulary
  (query types, field types, APIs), not Lucene class names. This is the
  document that lets someone decide whether the plugin fits their workload, and
  it should carry the fallback-rate instrumentation from
  [M2](m2-opensearch-read-path.md) so the answer is measurable rather than
  aspirational.

### T6.4 — Upgrade and rollback, tested

Backward-codecs are out of scope, which shapes both directions:

- **Forward**: how a running cluster adopts the Rust engine. Rolling restart,
  per-index opt-in, or per-node — pick one and document it. Existing segments
  written by an older Lucene remain readable only by the Java engine, so the
  procedure must account for them.
- **Backward**: how a cluster moves off. Force-merge is the escape hatch —
  after a force-merge under the Java engine, segments are in a codec the Java
  engine fully owns.
- **Test it.** Execute a real rollback on a test cluster carrying real data,
  and record the procedure and timings. An untested rollback procedure is a
  hypothesis.

### T6.5 — Licensing and attribution audit

`PLAN.md` §3 requires it: this is a derivative work of Apache Lucene, so
Apache-2.0 with NOTICE attribution.

- Confirm `LICENSE` and `NOTICE` are present, correct, and attribute Lucene.
- Audit every third-party crate's licence for compatibility — `memmap2`,
  `zstd`, `lz4_flex`, `crc32fast`, `unicode-segmentation`, `rayon`, `jni`,
  `thiserror`, `miniz_oxide`, `proptest`, `criterion`.
- Confirm no code was transliterated from Tantivy. `PLAN.md` §1 is explicit
  that it is prior art to study, not to depend on.

### T6.6 — Operational documentation

What someone needs to run this who did not build it:

- Installation, configuration, and the supported platform matrix.
- What the stats and logs mean, and which ones matter.
- Failure modes and their signatures — especially what a shard failure from the
  Rust engine looks like in the logs versus a Java one.
- Known limitations, linked to the feature matrix.

---

## Acceptance criteria

- [x] **Resource bounds** (replacing the 7-day soak): every structure that can
      accumulate across repeated writes, commits, merges, refreshes, searches
      and FFI calls has a tight-loop test holding it to a bound, each seen to
      fail against its defect ([T6.1](#t61--resource-bounds-instead-of-a-soak)).
- [x] The index the NRT refresh loop leaves passes `CheckIndex` (and real
      Lucene's, checked by hand on the kept directory).
- [ ] [M1](m1-performance-gate.md)'s performance bar is still met on the final
      build, measured by the nightly job rather than by hand.
- [ ] The nightly performance job fails the build when the ratio drops below
      the M1 bar (verified with a deliberate negative control).
- [ ] A **rollback from the Rust engine to the Java engine** is executed
      successfully on a test cluster with real data, and the procedure is
      documented with timings.
- [ ] `docs/parity.md` is complete for the supported matrix, with every
      remaining gap stated precisely.
- [ ] An operator-facing feature matrix is published, in OpenSearch's
      vocabulary.
- [ ] `LICENSE` and `NOTICE` are correct; every dependency licence is audited
      and recorded.

---

## Risks and unknowns

- **A bound test only covers the loop it runs.** The resource tests each drive
  one operation; growth that only an interleaving produces is outside them.
  New state that can accumulate needs its own test, with its own negative
  control.
- **"No memory growth" needs a definition.** Caches legitimately grow to a
  steady state. The tests hold each structure to a stated bound, and measure
  heap only after a warm-up.
- **The rollback path may reveal a one-way door.** If some Rust-written state
  cannot be consumed by the Java engine even after force-merge, that is a
  release blocker discovered late. Sanity-check the rollback direction during
  [M4](m4-write-path-hardened.md)'s interoperability matrix rather than
  first meeting it here.
- **Scope pressure.** This milestone sits between a working system and a
  shipped one, which is exactly when feature requests arrive. The out-of-scope
  section is binding.

---

## Exit artifacts

- Resource-bound tests, each with a recorded negative control
- A nightly performance regression job and its historical record
- A complete `docs/parity.md`
- An operator-facing feature matrix
- A tested, documented upgrade and rollback procedure
- `LICENSE`, `NOTICE`, and a dependency licence audit
- Operational documentation
