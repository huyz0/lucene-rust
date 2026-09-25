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
| M1.6 | [Lucene source sweep](m1-6-lucene-sweep.md) | Read the port file by file against Lucene 10.5.0; find parity gaps and un-done optimisations, and measure each component against Lucene's own number | M–L | ✅ **delivered — median 0.15×→0.585×, 19→16 queries slower than Java, recall 13→0 mismatches, reader open 552→33 ms and 1,690→60 MB, phrase 0.04×→0.58×. Decode kernels 1.9×–2.4× *faster* than Lucene. M1 gate still FAIL; one characterised divergence remains (documents scored)** |
| M2 | [OpenSearch read path](m2-opensearch-read-path.md) | A node answers `_search` from Rust over JNI/FFM | M–L | ✅ **delivered 2026-09-25** — an OpenSearch 3.8.0 node with the plugin serves the query phase from Rust per query, falling back to Lucene; OpenSearch's own REST YAML suites fail identically with and without it (501 tests), a 43-shape native-vs-Lucene matrix matches exactly, SIGKILL and force-merge release verified, JNI crossing under 10 ns; shapes M1 never measured (boosts, `must_not`, mixed booleans) are correct natively but 4–8× slower, so they are routed to Lucene |
| M3 | [Write path proven](m3-write-path-proven.md) | Real Lucene reads a full Rust-written index | L | ✅ **delivered 2026-09-25** — real Lucene opens a 120k-doc, 7-segment Rust-written index, `CheckIndex` clean, 57 queries return identical top-50s (scores within 2.4e-7); the term dictionary is now a real block tree whose block structure matches Lucene's own writer |
| M4 | [Write path hardened](m4-write-path-hardened.md) | Crash-safe, concurrent, interoperable both directions | L | ✅ **delivered** — crash campaign (power loss, `kill -9`, concurrent power loss under load) always recovers to the last or in-flight commit and passes Lucene's `CheckIndex`; 1000-seed op streams agree with Java's `IndexWriter`; 11-direction interop matrix passes; multi-threaded indexing 1.70× Lucene on adds, 0.81× on updates (cause written up) |
| M5 | [Engine integration](m5-engine-integration.md) | A shard fully served by Rust — indexing and search | XL | ✅ **delivered 2026-09-25** — `RustEngine` is `InternalEngine` with the Rust writer: OpenSearch's own `InternalEngineTests` pass on it (117, 34 skipped with reasons); it agrees with OpenSearch's engine operation for operation, through restarts, SIGKILL, a writer panic (one shard fails) and a tripped breaker; a 3-node cluster does document replication, peer recovery, failover and relocation between Rust and Java nodes. A Rust segment-replication *primary* is impossible on OpenSearch 3.8 (`EngineBackedIndexer`) |
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
