---
name: port-workflow
description: "WHAT: The order every ported area goes through — closest-to-Java port, then a benchmark against Lucene, then optimisation — and the rule that an area is not done until all three are. USE WHEN: starting to port any Java class, format or algorithm (a new file under crates/*/src/, a new writer/reader/query in lucene-codecs/lucene-index/lucene-search), or deciding whether an area is finished and the next may start."
---

# Port → benchmark → optimise

Every area is worked in three stages, **in order**, and the next area does not
start until the current one has finished stage 3.

1. **Port, as close to Java as possible.** Same algorithm, same control flow,
   same data structures, same constants and thresholds, same method
   boundaries — Java method names cited in doc comments (`// Java:
   Class.method`). No redesign yet, however obvious it looks. Rust-only
   changes are limited to what the language forces (ownership, `Result`
   instead of exceptions, arithmetic under `docs/arithmetic-gate.md`), and
   each one is named in the module doc. This stage is the correctness
   baseline: it must pass the differential tests (`differential-testing`)
   before stage 2 begins.
2. **Benchmark against Lucene 10.5.0** on the same bytes / same inputs: a
   component case in `benchmarks/rust-runner/src/micro.rs` with its Java twin
   in `benchmarks/micro/java/`, run through `scripts/bench-micro.sh --bench
   <area>` (queries: `scripts/bench-compare.sh`). Record the ratio.
3. **Optimise until not slower than Lucene** (and not larger in memory),
   now applying the Rust-first design rules in `rust-performance` — each
   change driven by what the stage-2 profile shows, re-verified by the same
   differential tests, re-measured by the same benchmark. A case left below
   1.0 is written up with its cause (see `docs/benchmarks/sweep-2026-09.md`,
   "Below 1.0, and why each is left").

Why this order: a faithful port is the cheapest way to be *right*, and a
measured baseline is the only way to know which redesign is worth its risk.
Redesigning before either exists has repeatedly produced code that was both
wrong and not measurably faster.

## Enforced by

- Nothing mechanical checks the order or that a benchmark exists — say so
  rather than pretend. The `quality-reviewer` agent and the `code-review`
  checklist ask for the stage-2 ratio and stage-3 result on any diff that
  finishes an area; `docs/parity.md` rows record them (`parity-tracking`).
- Stage 1 correctness is enforced by the differential and unit tests
  (`scripts/gate.sh`, `scripts/verify-write-path.sh`).

## Deep dive

[`docs/porting-workflow.md`](../../../docs/porting-workflow.md) — what "as
close as possible" allows, how to add a benchmark pair, what counts as done.
