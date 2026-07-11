---
name: differential-testing
description: "WHAT: Java-fixture-driven differential testing — the correctness backbone of the port. USE WHEN: implementing any decoder/parser for a Lucene file format, or adding/changing anything under fixtures/."
---

# Differential testing against real Lucene

A ported decoder is not done until it agrees with Java Lucene on real,
Java-written bytes. Byte-level intuition is not enough — every field-order or
endianness assumption gets checked against ground truth (see the numSegments
BE-vs-LE bug in `segment_infos.rs`'s history: it looked plausible and was
still wrong).

## Rules

- **Every format decoder ships with a `fixtures/src/Gen*.java` generator**
  pinned to the workspace's Lucene version (currently 10.5.0 — must match
  OpenSearch's `gradle/libs.versions.toml`). The generator writes real bytes
  via the actual Lucene codec class, round-trips them through Java Lucene
  itself before shipping, and emits a plain-text manifest of expected values
  so the Rust test asserts without parsing Java.
- **Prefer a real `IndexWriter` session over hand-built bytes** when the
  format depends on writer-assigned state (segment names, generations,
  counters) — see `GenSegmentInfos.java`. Hand-built fixtures are fine for
  isolated formats (`GenCodecUtil.java`).
- **Include at least one corruption/negative fixture** per format (wrong id,
  truncated footer, flipped byte) — a decoder that only ever sees valid input
  will happily mis-parse invalid input.
- **When a decode loop disagrees with Java, don't guess-and-check.** Write a
  throwaway field-by-field debug print against the fixture bytes (see
  `segment_infos.rs`'s development), diff against Java's source for that
  field's exact read call, fix, then delete the scratch tool.
- **`fixtures/data/` is checked in.** Regenerate and re-commit it whenever the
  pinned Lucene version changes; `cargo test` must never require a JVM.

## Enforced by

- `cargo test --workspace` — every differential test lives in
  `crates/*/tests/*_fixtures.rs` and reads from `fixtures/data/`.
- Nothing mechanical yet checks "every new decoder has a fixture" — that's a
  code-review item (see the `code-review` skill).

## Deep dive

[fixtures/README.md](../../../fixtures/README.md) (regeneration recipe, list of
generators).
