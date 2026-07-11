---
name: code-review
description: "WHAT: The self-review checklist run before declaring a change done. USE WHEN: finishing a unit of work, before committing, or reviewing a diff."
---

# Code review / self-review

Development is LLM-driven; the implementing agent self-reviews before
declaring work done. A change that cannot truthfully tick these boxes is not
ready.

## Checklist

- **Correctness vs Java semantics**: decoded values match Lucene's actual wire
  format (endianness, vint vs BE-int — don't assume, verify against a fixture;
  see the `differential-testing` skill). Float/scoring math matches Java's
  order of operations where applicable (see `rust-performance`).
- **Structure**: dependency direction intact (see `architecture`); no `unsafe`
  outside `lucene-util`/`lucene-ffi` (see `ffi-safety`); one Java
  format/concept per module.
- **Not a dumb port**: the in-memory design was considered on its own merits,
  not transliterated field-for-field from the Java class (see
  `rust-performance`).
- **Tests**: a new decoder ships with a fixture generator and a differential
  test (see `differential-testing`); at least one negative/corruption case;
  round-trip or cross-module consistency checks where two modules should
  agree (e.g. `segments_N`'s doc count vs the segment's own `.si`).
- **Docs**: `docs/parity.md` updated in the same change (see
  `parity-tracking`); module-level doc comments describe the wire format,
  not "what the code does."
- **No stray scratch files**: throwaway debug binaries/examples used to
  diagnose a decode mismatch are deleted before the change is done.

## Enforced by

- `cargo fmt --all --check`, `cargo clippy --workspace -- -D warnings`,
  `cargo test --workspace` (see `git-workflow`).
- Nothing mechanical yet checks "new decoder has a fixture" or "parity.md
  updated" — self-review + the `quality-reviewer` subagent (`/quality-review`)
  cover it until this repo has an `xtask`-style gate worth building.

## Deep dive

No separate deep-dive doc yet; this file is the whole checklist.
