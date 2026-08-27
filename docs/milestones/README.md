# Milestones

One file per milestone. Each is a self-contained work order: goal, scope
boundaries, task breakdown, objectively checkable acceptance criteria, risks,
and the artifacts that must exist before it can be called done.

[`../roadmap.md`](../roadmap.md) is the executive summary of the same plan.
[`../../PLAN.md`](../../PLAN.md) is the architectural source of truth.
[`../parity.md`](../parity.md) is the per-Java-file status ledger.

| | Milestone | Goal | Effort | Status |
|---|---|---|---|---|
| M0 | [Green tree, real CI](m0-ci-and-green-tree.md) | Every gate runs automatically, and HEAD passes them | S | ✅ **complete** |
| M1 | [The performance gate](m1-performance-gate.md) | Decide go/no-go with data: is Rust search decisively faster? | M | ✅ **answered: NO** — 7.2x median improvement, still 20x slower than Java ([final verdict](../benchmarks/verdict-m1-e2e.md)) |
| M1.5 | [Lazy iteration on the hot paths](m1-5-lazy-iteration.md) | Stop materializing posting lists; re-run the gate | M–L | ✅ **delivered — 5.8× median, gate still FAIL** |
| M1.6 | [Lucene source sweep](m1-6-lucene-sweep.md) | Read the port file by file against Lucene 10.5.0; find parity gaps and un-done optimisations, and measure each component against Lucene's own number | M–L | 🔶 **partly delivered — decode kernels now 1.7×–2.3× *faster* than Lucene, segmented recall 13→0 mismatches, end-to-end median 0.15×→0.27×, gate still FAIL. Positions/blocktree/doc-values not yet swept** |
| M2 | [OpenSearch read path](m2-opensearch-read-path.md) | A node answers `_search` from Rust over JNI/FFM | M–L | not started |
| M3 | [Write path proven](m3-write-path-proven.md) | Real Lucene reads a full Rust-written index | L | not started |
| M4 | [Write path hardened](m4-write-path-hardened.md) | Crash-safe, concurrent, interoperable both directions | L | not started |
| M5 | [Engine integration](m5-engine-integration.md) | A shard fully served by Rust — indexing and search | XL | not started |
| M6 | [Production candidate](m6-production-candidate.md) | Soak-proven, perf-held, rollback-documented | M | not started |

## Dependency structure

```
                         ┌──────────────── pass ────────────────┐
                         │                                      │
M0 ──────▶ M1 (GATE) ────┤          ┌──▶ M2  OpenSearch read ───┼──▶ M5 ──▶ M6
CI green   benchmark     │          │                           │   engine   prod
                         └──────────┴──▶ M3  write proof ──▶ M4 ┘
                         │                                 harden
                         └─ fail ─▶ stop, or ship as a standalone library
```

- **M0 unblocks everything.** The tree does not currently pass its own
  pre-commit gate, so nothing can land cleanly until it does.
- **M1 is the only branch point.** It decided whether M2–M6 are funded at all,
  and it cost a benchmark rather than a Java plugin to answer. **It returned
  FAIL**: lucene-rust is 6×–1000× slower than Java Lucene, for a structural
  reason (posting lists are materialized instead of skipped). See
  [`docs/benchmarks/verdict.md`](../benchmarks/verdict.md). M2–M6 are on hold
  pending the algorithmic fix the verdict recommends.
- **M2 and M3 are independent.** One is Java-writes/Rust-reads through
  OpenSearch; the other is Rust-writes/Java-reads at the format level. Work
  them in parallel or in either order. M5 needs both.

## Conventions used in these files

- **Task IDs** are `T<milestone>.<n>` — stable handles for commit messages and
  cross-references.
- **Acceptance criteria** are written as checkboxes and phrased so that a
  reader can determine pass/fail without judgement. "Fast enough" is not an
  acceptance criterion; "≥1.5× throughput on ≥80% of the query mix" is.
- **Out of scope** sections are binding. Moving an item out of them is a scope
  change that belongs in a commit, not a decision made mid-task.
- Every milestone ends with **exit artifacts**: the files that must exist, so
  "done" is checkable by `ls` rather than by memory.
