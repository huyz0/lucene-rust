---
name: git-workflow
description: "WHAT: Commit message format, the gate-before-done rule, and hooks. USE WHEN: committing, writing a commit message, or finishing a unit of work."
---

# Git workflow

Single developer, work directly on `main` (no feature branches). Every commit
leaves the tree green.

## Rules

- **Conventional commit subject**: `type(scope): lowercase description` where
  `type ∈ {feat, fix, docs, test, chore, refactor, perf, build, ci}`. Example:
  `feat(index): parse segments_N commit files`.
- **End with the trailer** `Co-Authored-By: Claude <noreply@anthropic.com>`
  (adjust the name to whichever model authored the change).
- **Gate before done.** `cargo fmt --all --check`, `cargo clippy --workspace
  -- -D warnings`, and `cargo llvm-cov --workspace --fail-under-lines 95` must
  all pass before a task is considered complete (see `test-coverage`).
  Commit/push only when the user asks.
- **Update docs/skills in the same commit** as the code they describe — see
  the `parity-tracking` and `manage-skills` skills. Drift is a bug.

## Enforced by

- `.githooks/commit-msg` — validates the subject format and trailer presence.
- `.githooks/pre-commit` — runs fmt/clippy/coverage-gated tests, blocks on
  failure.
- Install once: `git config core.hooksPath .githooks` (or run
  `scripts/setup-hooks.sh`).

## Deep dive

None yet — this repo has no `docs/10-review-process.md` equivalent; the rules
above are the whole policy.
