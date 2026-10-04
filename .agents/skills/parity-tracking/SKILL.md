---
name: parity-tracking
description: "WHAT: Keeping the parity ledger (docs/parity.md + docs/parity/*.md) the source of truth for what's ported. USE WHEN: finishing a decoder/encoder for a Java file/format, deciding something is intentionally unsupported, or adding/editing a ledger row."
---

# Parity ledger maintenance

The ledger answers "is X ported yet?" without reading every crate. It is
`docs/parity.md` -- a short index: how to read a row, the status vocabulary,
the area table -- plus one table per area in `docs/parity/`. It decays fast
if not updated in the same change as the code, and it bloats fast if rows
become changelogs (it reached 1.25 MB in one file before the split).

## Rules

- **Update the ledger in the same commit** that ports, partially ports, or
  deliberately defers a Java file/format. One row per Java class or format
  concept, pointing at the Rust module that owns it.
- **Where a row goes**: the area file of the crate and module the Rust code
  lives in (a `lucene-search` collector goes in `search-collectors.md`), next
  to the rows for the same Java class. The M9/M10 feature files (geo,
  spatial, joins, grouping, functions) and `engine.md` group by feature. A
  new area file must be linked from the index's area table, with its row
  count; bump the count when you add a row.
- **Row format** -- `| Java | Rust | Status |`:
  - Java: classes relative to `org/apache/lucene/`, or `--` for Rust-only.
  - Rust: `` `crate/src/file.rs::{items}` `` (no `crates/` prefix). Every
    path and item must exist.
  - Status: a word (**ported**, **partial**, **rust-only**, **not-needed**,
    **generated**) and milestone, then current facts only: **Differs:**
    every deviation from Java and why; **Gap:** what is missing; **Tests:**
    the fixture generator and test file or named tests; **Bench:** the
    current ratio against Lucene.
- **Current facts, not history.** No dates, batch names, "previously",
  test-count logs, "cargo test passes", or before/after numbers. When a
  change supersedes a sentence, rewrite the sentence; do not append an
  update. Rationale and history belong in `git log` and `docs/sweep/`; link
  them.
- **Distinguish "not started" from "unsupported by design".** If a feature is
  intentionally out of scope, say so and name the typed error/return path a
  caller sees.
- **Pin the Lucene version** at the top of `docs/parity.md` and update it
  (plus `fixtures/`, see the `differential-testing` skill) together.

## Enforced by

- `scripts/check-parity.py` (in `scripts/gate.sh` and CI): every Rust path
  and `::item` in a Rust column exists; every `crates/*/src/*.rs` (bar
  `lib.rs`, `error.rs` and a reasoned `EXEMPT` list) is named by some row.
  What it cannot catch: [`docs/mechanical-gates.md`](../../../docs/mechanical-gates.md).
- Code review checks that a PR touching a new format also touches the ledger,
  and that a row's prose still matches the code.

## Deep dive

[docs/parity.md](../../../docs/parity.md), [PLAN.md](../../../PLAN.md) §2
(phase-by-phase scope).
