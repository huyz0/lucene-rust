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
*together*, *for days*, and that there is a way back if they do not.

`PLAN.md`'s Phase 6 exit criteria name it directly:

> multi-day soak test with random restarts, no index corruption.

The second half — the rollback path — matters because backward-codecs are out
of scope project-wide. A cluster that moves onto the Rust engine and cannot
move back is a cluster nobody will move in the first place.

---

## Scope

### In scope

- Sustained soak testing under realistic mixed load.
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

### T6.1 — Multi-day soak

- ≥7 days, mixed index and search load, on a multi-node cluster.
- Random restarts and random node kills throughout — not a quiet steady-state
  run.
- Merges running continuously, so the merge scheduler and file lifecycle are
  exercised rather than idle.
- Monitor for the failure modes that only appear over time: memory growth,
  file-handle growth, mmap exhaustion, disk growth from undeleted segments,
  slow degradation in query latency.
- At the end, run real Lucene's `CheckIndex` over every shard.

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

- [ ] **7-day soak** completes with: no index corruption, no memory growth, no
      file-handle growth, no unexplained shard failures.
- [ ] Real Lucene's `CheckIndex` passes on every shard after the soak.
- [ ] Query latency at the end of the soak is within noise of the start — no
      slow degradation.
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

- **Soak failures are expensive to diagnose.** A leak that manifests on day
  five costs five days per iteration. Instrument memory, handles and segment
  counts continuously from the start of the run so the diagnosis does not
  require a re-run.
- **"No memory growth" needs a definition.** Caches legitimately grow to a
  steady state. Define the criterion as a bounded plateau, and record what the
  plateau is, rather than as a flat line.
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

- Soak test results, with continuous instrumentation traces
- A nightly performance regression job and its historical record
- A complete `docs/parity.md`
- An operator-facing feature matrix
- A tested, documented upgrade and rollback procedure
- `LICENSE`, `NOTICE`, and a dependency licence audit
- Operational documentation
