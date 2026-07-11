# AGENTS.md

Orientation for AI agents on **lucene-rust**. A router + invariants list; it
does **not** repeat the detail in [`.agents/skills/`](.agents/skills/),
[`PLAN.md`](PLAN.md), or [`docs/`](docs/). On conflict, the skill/doc wins for
its topic — fix the drift.

## What this is

A Rust port of Apache Lucene, exposed over an FFI (JNI / Panama FFM) boundary
so OpenSearch (JVM) can use it as a native search engine. Pinned Lucene
version: **10.5.0** (matches OpenSearch's `gradle/libs.versions.toml` — see
`docs/parity.md`). The Java source of truth lives at `/home/tuong/work/lucene`;
the OpenSearch checkout at `/home/tuong/work/OpenSearch`.

Phase 1 (foundations: `lucene-util`/`lucene-store`) is in progress. The full
plan — phases, crate layout, verification strategy, effort estimates — is
[`PLAN.md`](PLAN.md).

## Invariants (don't break)

1. **Downward-only crate deps.** `util ← store ← codecs ← index ← search ←
   core ← ffi`. Siblings never depend on each other. See the **architecture**
   skill.
2. **Port by on-disk format, not by class hierarchy.** The Java class graph is
   not the target; the byte-level wire format is. See **architecture** and
   **rust-performance**.
3. **A "faithful" port that's slower than Java is a bug.** Redesign the
   in-memory shape for Rust (ownership, monomorphization, zero-copy, SIMD) —
   don't transliterate. See **rust-performance**.
4. **`unsafe` only in `lucene-util`, `lucene-store`, and `lucene-ffi`.** Every
   other crate is `#![forbid(unsafe_code)]`. See **ffi-safety**.
5. **A Rust panic must never cross the FFI boundary into the JVM.** Every
   exported `lucene-ffi` function wraps in `catch_unwind`. See **ffi-safety**.
6. **No decoder ships without a Java-fixture differential test.** Byte-level
   assumptions get it wrong more often than intuition predicts — verify
   against real Lucene output, not just plausibility. See
   **differential-testing**.
7. **`docs/parity.md` updates in the same commit** as any format that gets
   ported, partially ported, or deliberately deferred. See
   **parity-tracking**.
8. **≥95% line coverage, per file, from day one.** Differential fixture tests
   prove format fidelity; unit tests (inspired by Lucene's own JUnit tests,
   not transliterated from them) prove the decoder's own boundary/error
   handling. See **test-coverage**.
9. **Keep the gates green** — `cargo fmt --check`, `cargo clippy -- -D
   warnings`, `cargo llvm-cov --fail-under-lines 95` must pass before a task
   is done.

## Commands

`.githooks/pre-commit` (install via `scripts/setup-hooks.sh`) runs the gate
and blocks on failure. Run it before calling a task done:

| Step | Command |
|------|---------|
| Format | `cargo fmt --all --check` |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| Tests + coverage gate | `cargo llvm-cov --workspace --fail-under-lines 95` |
| Coverage report, per file | `cargo llvm-cov --workspace --summary-only` |
| Regenerate Java fixtures | see [fixtures/README.md](fixtures/README.md) |

**Commits**: `commit-msg` allows only `feat|fix|docs|test|chore|refactor|
perf|build|ci` + optional `(scope)` + lowercase description, and requires a
`Co-Authored-By:` trailer. Single dev — work directly on `main`. See the
**git-workflow** skill.

## Where to look (task → skill)

Skills are the process source of truth; `PLAN.md`/`docs/` are the deep-dives.

| Task | Skill |
|------|-------|
| Crates / module boundaries / where code belongs | `architecture` |
| New decoder for a Lucene file format | `differential-testing` |
| In-memory design for a ported module | `rust-performance` |
| Anything in `lucene-ffi`, any `unsafe` block | `ffi-safety` |
| Finished a format, need to record it | `parity-tracking` |
| Committing / finishing a unit of work | `git-workflow`, `code-review` |
| Writing tests for a new/changed module | `test-coverage` |
| Editing skills | `manage-skills` |

## Workflow

- **Read the matching skill before acting** — it encodes the rule and names
  the gate that enforces it.
- **Fixture-first for new decoders**: write the `Gen*.java` generator, run it,
  write the Rust parser against real bytes, write the differential test —
  don't hand-roll expected bytes from reading the Java source alone.
- **Before declaring work done**, after the gate is green, run the Tier 2
  semantic review: spawn the `quality-reviewer` subagent or run
  `/quality-review`.
- **Update `PLAN.md`/`docs/parity.md`/skills in the same change** — drift is a
  bug.
- **Roadmap**: build in phase order, [`PLAN.md`](PLAN.md) §2.
