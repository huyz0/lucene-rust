---
name: parity-tracking
description: "WHAT: Keeping docs/parity.md the source of truth for what's ported. USE WHEN: finishing a decoder/encoder for a Java file/format, or deciding something is intentionally unsupported/deferred."
---

# Parity matrix maintenance

`docs/parity.md` is the only place that answers "is X ported yet?" without
reading every crate. It decays fast if not updated in the same change as the
code.

## Rules

- **Update `docs/parity.md` in the same commit** that ports, partially ports,
  or deliberately defers a Java file/format. One row per Java class or format
  concept, pointing at the Rust module that owns it.
- **Distinguish "not started" from "unsupported by design".** If a feature is
  intentionally out of scope (e.g. index-sorted segments, backward-codecs),
  say so and name the typed error/return path a caller sees, rather than
  leaving it silently unimplemented.
- **Pin the Lucene version at the top of the file** and update it (plus
  `fixtures/`, see the `differential-testing` skill) together whenever the
  pinned version changes — never let the two drift independently.
- **Link forward, not just back.** A parity row for a reader should note the
  writer's phase (e.g. "write side deferred to Phase 5") so it's clear the gap
  is planned, not forgotten.

## Enforced by

- Nothing mechanical — this is a discipline, not a lint. Code review checks
  that a PR touching a new format also touches `docs/parity.md`.

## Deep dive

[docs/parity.md](../../../docs/parity.md), [PLAN.md](../../../PLAN.md) §2
(phase-by-phase scope).
