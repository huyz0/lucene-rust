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
- **`unsafe` lives only in `lucene-util` (SIMD), `lucene-store` (mmap), and
  `lucene-ffi` (C ABI).** Every other crate keeps `#![forbid(unsafe_code)]`.
  An `unsafe` block outside those three crates is a design smell — fix the
  boundary instead of adding more `unsafe`.
- **Validate handles before use.** A stale/unknown handle returns an error
  code, never a dereference — the slotmap's generation tag exists precisely
  to catch use-after-free/close races from the Java side.
- **`c_char` signedness is target-dependent — never `as`-cast it.** It is `i8`
  on `x86_64-unknown-linux-gnu` and `u8` on `aarch64-unknown-linux-gnu`, both
  of which this port supports. `ptr as *const u8` on a `*const c_char` is a
  genuine cast on x86_64 and a no-op on aarch64, so clippy's
  `unnecessary_cast` fails the build on exactly one architecture. Convert once,
  centrally, with `.cast::<u8>()` — a method call, not an `as` expression, so
  it is correct on both. `raw.rs`'s `str_from_raw`/`bytes_from_raw` are that
  central place; call sites pass their `*const c_char` through untouched.
- **No callbacks from Rust into Java in v1.** Collectors run entirely in Rust;
  keep the boundary one-directional until there's a concrete need otherwise.

## Enforced by

- `cargo clippy --workspace` (`forbid(unsafe_code)` outside the three allowed
  crates fails the build), run by CI on **both** `x86_64` and `aarch64` —
  target-dependent defects like `c_char` signedness are invisible on one
  architecture alone. Reproduce locally before touching this crate with
  `cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu
  -- -D warnings` — a check-only build, so no cross C compiler is needed.
- Miri on `lucene-util`/`lucene-store`'s `unsafe` blocks (`cargo +nightly miri
  test -p lucene-util -p lucene-store`) — run before landing any SIMD/mmap
  change.
- Code review: no exported `lucene-ffi` function without a `catch_unwind`
  wrapper and a handle-validation check.

## Deep dive

[PLAN.md](../../../PLAN.md) §2 Phase 4 (FFI layer design), risk #3 in §4
(JNI crash blast radius).
