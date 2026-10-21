---
name: architecture
description: "WHAT: Crate layout and the strict downward dependency graph. USE WHEN: adding or moving a crate/module/file, changing Cargo dependencies, or deciding where new code belongs."
---

# Architecture & structure

lucene-rust is a workspace of crates mirroring Lucene's own module DAG, with a
**strictly downward dependency graph**: `util ← store ← codecs ← index ←
search ← core ← ffi`, with `util ← analysis ← index` beside it and the
language modules hanging off `analysis`: `util, analysis ←
analysis-<m>` (`phonetic`, `stempel`, `morfologik`, `kuromoji`, `nori`, ...;
M12). A crate only depends on crates to its left; siblings never depend on
each other.

## Rules

- **Dependency direction is downward only.** `lucene-util` depends on nothing
  in the workspace. `lucene-ffi` is the only crate allowed to depend on
  everything (it is the boundary). See PLAN.md §1 for the full table.
- **Nothing depends on a `lucene-analysis-<m>` crate** (PLAN.md §1): each
  language module depends on `lucene-util` and `lucene-analysis`, and is a
  leaf -- not another module, not `lucene-search` or above (the
  out-of-workspace `benchmarks/rust-runner` is the one consumer). Shared
  code a second module needs moves down into `lucene-analysis`. One edge
  to `lucene-index` is allowed, the one Lucene's `analysis-icu` has to
  `lucene-core`: `ICUCollationDocValuesField` is a `SortedDocValuesField`
  (Java: a `Field`), so `lucene-analysis-icu` depends on `lucene-index` for
  that type alone (M12, Architecture).
- **`lucene-test-framework` is a dev-dependency only** (PLAN.md §1, M14.h).
  A crate may name it under `[dev-dependencies]`, never `[dependencies]` or
  `[build-dependencies]`, so it never reaches `lucene-ffi`'s library: `cargo
  tree -p lucene-ffi -e normal,build` must not list it. It depends on the
  crates it tests (`lucene-util` up), which a normal edge back would turn
  into a cycle.
- **Port by format, not by class hierarchy.** A Java class name is not a
  license to create a matching Rust type — port the on-disk contract; design
  the in-memory shape for Rust (see the `rust-performance` skill).
- **`unsafe` is scoped, not sprinkled.** Only `lucene-util` (SIMD),
  `lucene-store` (mmap), and `lucene-ffi` (C ABI) may contain `unsafe`. Every
  other crate carries
  `#![forbid(unsafe_code)]`. See the `ffi-safety` skill.
- **No `util`/`misc`/`common` dumping ground inside a crate.** One module, one
  Java-format concept (e.g. `codec_util.rs`, `segment_info.rs`,
  `segment_infos.rs` — each maps to one Java file or format).
- **Read path before write path.** Don't add write-side (encode) support for a
  format until its phase (PLAN.md §2) says so — flag it in the parity ledger (`docs/parity/`) as
  deferred instead of half-implementing it.

## Enforced by

- `cargo build --workspace` (a downward-only Cargo.toml graph fails loudly on
  an accidental sibling dependency -- but only on a cycle: a leaf rule such
  as "nothing depends on `lucene-analysis-<m>`" is not mechanically checked;
  `grep -E '^lucene-analysis-' crates/*/Cargo.toml` shows any offender).
- `cargo clippy --workspace` — `forbid(unsafe_code)` turns stray `unsafe` into
  a hard compile error outside `lucene-util`/`lucene-store`/`lucene-ffi`.
- `check-port-invariants.py`'s `test-framework-dev-only` rule fails on a
  `lucene-test-framework` edge outside a `dev-dependencies` section. The
  rule is per package: `cargo build --workspace` still unifies the test
  framework's `test-support` feature on `lucene-util` into that build, so
  the shipped artifact is the `-p lucene-ffi` build (`cargo tree -p
  lucene-ffi -e normal,build` does not list it).
- Code review (no automated dep-direction linter yet — a good candidate for a
  future `xtask` if the workspace grows past ~10 crates).

## Deep dive

[PLAN.md](../../../PLAN.md) §1 (crate table) and §3.5 (Rust-first design
principles).
