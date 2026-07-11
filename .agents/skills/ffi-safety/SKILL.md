---
name: ffi-safety
description: "WHAT: The C-ABI/JNI boundary contract — handles, panics, unsafe scope. USE WHEN: touching crates/lucene-ffi, adding an exported function, or writing any `unsafe` block anywhere in the workspace."
---

# FFI safety (the JVM-facing boundary)

A bug in `lucene-ffi` can crash the whole OpenSearch node, not just fail a
test. This boundary gets more scrutiny than anything else in the workspace.

## Rules

- **Opaque handles only.** No Rust pointers, references, or types cross the
  boundary — `u64` generation-tagged slotmap handles for `Directory`,
  `IndexReader`, `IndexSearcher`, `Query`, result buffers.
- **A panic must never unwind into the JVM.** Every exported function wraps
  its body in `catch_unwind`; a caught panic becomes an error code plus a
  last-error message in a TLS slot, never a propagated unwind.
- **All exported calls return a status code**, results via out-buffers/handles
  — no exceptions-as-control-flow across the boundary.
- **`unsafe` lives only in `lucene-util` (SIMD/mmap) and `lucene-ffi` (C ABI).**
  Every other crate keeps `#![forbid(unsafe_code)]`. An `unsafe` block outside
  those two crates is a design smell — the abstraction leaked; fix the
  boundary instead of adding more `unsafe`.
- **Validate handles before use.** A stale/unknown handle returns an error
  code, never a dereference — the slotmap's generation tag exists precisely
  to catch use-after-free/close races from the Java side.
- **No callbacks from Rust into Java in v1.** Collectors run entirely in Rust;
  keep the boundary one-directional until there's a concrete need otherwise.

## Enforced by

- `cargo clippy --workspace` (`forbid(unsafe_code)` outside the two allowed
  crates fails the build).
- Miri on `lucene-util`'s `unsafe` blocks (`cargo +nightly miri test -p
  lucene-util`) — run before landing any SIMD/mmap change.
- Code review: no exported `lucene-ffi` function without a `catch_unwind`
  wrapper and a handle-validation check.

## Deep dive

[PLAN.md](../../../PLAN.md) §2 Phase 4 (FFI layer design), risk #3 in §4
(JNI crash blast radius).
