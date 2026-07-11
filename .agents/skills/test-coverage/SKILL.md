---
name: test-coverage
description: "WHAT: The ≥90%-per-file line coverage bar and how tests are layered (differential + unit). USE WHEN: writing or changing any parser/decoder, adding a module, or before declaring a change done."
---

# Test coverage: two layers, ≥90% per file

Coverage is a floor, not the goal — but the floor is real and enforced from
day one, not bolted on later. Every ported module carries **two distinct test
layers**, and both are required, not either/or:

## The two layers

1. **Differential tests** (`crates/*/tests/*_fixtures.rs`, see the
   `differential-testing` skill): prove format fidelity against real
   Java-written bytes. These naturally only exercise the happy path a real
   Lucene writer produces.
2. **Unit tests** (`#[cfg(test)] mod tests` inside the module itself): prove
   the decoder's own boundary and error handling — truncated input, corrupted
   magic/checksum, illegal enum markers, negative counts, empty/singleton/many
   collections. A real Lucene file never exercises these; only a hand-built
   or property-driven one does.

## Rules

- **Mine Lucene's own JUnit tests for edge cases** (`TestCodecUtil`,
  `TestSegmentInfos`, `TestFixedBitSet`, etc. under
  `/home/tuong/work/lucene/lucene/core/src/test/`) — they encode decades of
  "here's what actually breaks this format." Do not transliterate their
  JUnit structure; port the *scenario*, write it as an idiomatic Rust test
  (a `proptest` round-trip property beats a dozen hand-picked JUnit cases
  where the property holds for all inputs, not just the ones Java's test
  author thought of).
- **A test-only encoder is legitimate**, not a shortcut. To hit a decoder's
  error path you often need to hand-build bytes no real writer would ever
  produce (illegal marker byte, negative count). Write a small local
  builder/encoder in the test module (see `codec_util.rs`, `segment_info.rs`,
  `segment_infos.rs` for the pattern) — this is different from *dumbly*
  porting Java's writer; keep it minimal and test-only (`#[cfg(test)]`).
- **≥90% line coverage per file**, not just workspace-average — a 100%-file
  hiding a 40%-file is a gap. Check `cargo llvm-cov --workspace
  --summary-only` per-file, not just the `TOTAL` row.
- **Property tests for anything with a decode/encode symmetry** (vint/vlong/
  zigzag round-trips, base36) — `proptest`, not a handful of examples.
- **No coverage theater.** A test that calls a function without asserting
  its result doesn't count toward quality even if the tool counts it toward
  coverage — see the `code-review` skill's test-meaningfulness check.

## Enforced by

- `cargo llvm-cov --workspace --fail-under-lines 90` — part of
  `.githooks/pre-commit`, blocks the commit below threshold.
- `cargo llvm-cov --workspace --summary-only` — per-file breakdown; read it,
  don't just check the aggregate passed.

## Deep dive

No separate deep-dive doc; this file plus the `differential-testing` skill
are the whole testing policy. `cargo llvm-cov --open` for the annotated HTML
report when you need to see exactly which lines are uncovered.
