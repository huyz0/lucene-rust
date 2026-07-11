---
name: architecture
description: "WHAT: Crate layout and the strict downward dependency graph. USE WHEN: adding or moving a crate/module/file, changing Cargo dependencies, or deciding where new code belongs."
---

# Architecture & structure

lucene-rust is a workspace of crates mirroring Lucene's own module DAG, with a
**strictly downward dependency graph**: `util ← store ← codecs ← index ←
search ← core ← ffi`. A crate only depends on crates to its left; siblings
never depend on each other.

## Rules

- **Dependency direction is downward only.** `lucene-util` depends on nothing
  in the workspace. `lucene-ffi` is the only crate allowed to depend on
  everything (it is the boundary). See PLAN.md §1 for the full table.
- **Port by format, not by class hierarchy.** A Java class name is not a
  license to create a matching Rust type — port the on-disk contract; design
  the in-memory shape for Rust (see the `rust-performance` skill).
- **`unsafe` is scoped, not sprinkled.** Only `lucene-util` (SIMD, mmap) and
  `lucene-ffi` (C ABI) may contain `unsafe`. Every other crate carries
  `#![forbid(unsafe_code)]`. See the `ffi-safety` skill.
- **No `util`/`misc`/`common` dumping ground inside a crate.** One module, one
  Java-format concept (e.g. `codec_util.rs`, `segment_info.rs`,
  `segment_infos.rs` — each maps to one Java file or format).
- **Read path before write path.** Don't add write-side (encode) support for a
  format until its phase (PLAN.md §2) says so — flag it in `docs/parity.md` as
  deferred instead of half-implementing it.

## Enforced by

- `cargo build --workspace` (a downward-only Cargo.toml graph fails loudly on
  an accidental sibling dependency).
- `cargo clippy --workspace` — `forbid(unsafe_code)` turns stray `unsafe` into
  a hard compile error outside `lucene-util`/`lucene-ffi`.
- Code review (no automated dep-direction linter yet — a good candidate for a
  future `xtask` if the workspace grows past ~10 crates).

## Deep dive

[PLAN.md](../../../PLAN.md) §1 (crate table) and §3.5 (Rust-first design
principles).
