# Porting workflow: port → benchmark → optimise

The process every ported area follows. The rule and its trigger live in the
`port-workflow` skill; this file is the detail behind it.

An **area** is one unit of Lucene behaviour that can be tested and measured on
its own: a file format's reader or writer (`Lucene103BlockTreeTermsWriter`), an
algorithm (`CompetitiveImpactAccumulator`), a query type, an analysis
component. A milestone is several areas; each goes through all three stages
before the next begins.

## Stage 1 — port as close to Java as possible

Goal: a correct baseline whose every line can be checked against the Java
source it came from.

- **Keep** Java's algorithm, control flow, method decomposition, data
  structures, constants, thresholds and tie-breaks. If Java keeps a
  prefix-coded byte buffer, so does the port; if Java runs a greedy segmenter
  with a known weakness (a TODO in the Java source), so does the port.
- **Cite** the Java method each Rust function ports in its doc comment
  (`` `TrieBuilder.freezeNode` ``), so a reviewer can put the two side by side.
- **Change only what Rust forces**: ownership and borrowing, `Result` for
  exceptions, `u8` for Java's signed-byte-masked-`& 0xFF`, arithmetic written
  under [`arithmetic-gate.md`](arithmetic-gate.md). Name each such change in the
  module doc.
- **Do not** apply `rust-performance`'s redesign rules yet. They are stage-3
  tools, used when a measurement asks for them.
- **Exit:** the differential tests pass (Java-written fixtures read correctly,
  or for a writer, real Lucene reads the Rust bytes —
  `scripts/verify-write-path.sh`), plus unit tests and the coverage bar.

Byte-identity with Java's output is welcome where the stage-1 port gives it for
free (same algorithm, same inputs); it is not a requirement where Lucene's own
reader accepts any valid encoding.

## Stage 2 — benchmark against Lucene 10.5.0

Goal: a number, `java_time / rust_time`, on identical work.

- Add a case to `benchmarks/rust-runner/src/micro.rs` and its twin to
  `benchmarks/micro/java/` (`SweepMicro.java`, or a dedicated `*Micro.java`
  for a write-side or whole-component bench, e.g. `IndexMicro.java`). Both sides
  generate the same inputs from the same seed, or read the same corpus
  directory, and emit `case<TAB>ns_per_unit<TAB>units`.
- Run `scripts/bench-micro.sh --bench <area>`; for query-level work,
  `scripts/bench-compare.sh` (which cross-checks recall before comparing
  time). Read the noise column: a ratio inside it is not a result.
- Record the ratio (and memory, where the area holds any) in the commit
  message and in the area's `docs/parity.md` row.

## Stage 3 — optimise until not slower than Lucene

Goal: ratio ≥ 1.0 and memory no larger, without losing stage 1's correctness.

- Profile first (`perf`, flamegraphs); change what the profile implicates.
- Now the `rust-performance` rules apply: monomorphised loops, zero-copy,
  SIMD, struct-of-arrays, dropping Java's GC-avoidance machinery.
- Every change re-runs the stage-1 differential tests and the stage-2
  benchmark. Keep the stage-1 structure recognisable where the redesign
  allows, and move the Java-method citations to wherever the logic now lives.
- **Exit:** every case at or above 1.0, or each one below it written up with
  its measured cause and why it stays (the model is
  [`benchmarks/sweep-2026-09.md`](benchmarks/sweep-2026-09.md), "Below 1.0,
  and why each is left").

## Done means

An area is done when stage 3 has exited. Only then does the next area start,
even inside one milestone — measuring as you go is what stops a milestone
ending with a pile of unmeasured, slow ports and no idea which one to fix
first.

## What this cannot catch

Nothing mechanical checks the order or that a benchmark pair exists. The
`quality-reviewer` agent and the `code-review` checklist ask for it; a diff
that finishes an area without a stage-2 number is a review failure, not a gate
failure.
