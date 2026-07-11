---
name: rust-performance
description: "WHAT: Rust-first design rules that keep this a fast port, not a transliteration. USE WHEN: designing a new module's in-memory representation, touching a hot decode/search loop, or reviewing for idiom vs Java-isms."
---

# Rust-first design (not a dumb port)

The on-disk **format** is the compatibility contract; the in-memory **design**
is ours. A "faithful" port that is slower than Java is a bug, not a milestone —
see PLAN.md's exit-criteria note in §3.5.

## Rules

- **No GC-shaped object graphs.** Skip Java's allocation-avoidance machinery
  (ByteBlockPool, parallel arrays, AttributeSource reuse) — Rust gets
  deterministic memory for free. Use arenas (`bumpalo`) per-DWPT/per-query
  where lifetimes are scoped; choose struct-of-arrays for cache behavior, not
  to dodge a garbage collector that doesn't exist here.
- **Monomorphize per-doc loops.** `dyn` only at Query/Weight level; scorers
  and `DocIdSetIterator`s are enums or generics so `collect()` inner loops
  inline with zero virtual calls.
- **Zero-copy end-to-end.** `IndexInput` over mmap yields `&[u8]` views
  (`SliceInput` is `Copy`-cheap — that's how `IndexInput.clone()` maps here);
  copy only at true ownership boundaries.
- **SIMD from the start** for decode-heavy kernels (PFOR, bitset ops, BKD
  compares), `std::simd` + runtime dispatch, scalar fallback kept for
  correctness testing. Java's generated code (`ForUtil`, Panama vector code)
  is a spec to read, not a source to transliterate.
- **Ownership over locks.** Immutable segment readers (`Arc`, lock-free), one
  owner per DWPT, rayon leaf-slices for query concurrency. `Mutex` only on
  control-plane state (commits, merge scheduling).
- **Async-free core.** No async runtime in the library — CPU-bound work uses
  blocking + rayon; FFI callers get plain blocking calls.
- **Skip Java's abstraction taxes wholesale**: AttributeSource reflection
  (plain token structs), boxed autoboxing in collectors (doesn't exist),
  `ThreadLocal` pools (scoped ownership), finalizers (`Drop`).

## Enforced by

- Nothing mechanical (perf regressions don't fail `cargo test`). Each PLAN.md
  phase's exit criteria include profiling the new component against Java on
  the same workload before moving to the next phase — treat that as the gate.
- Code review: flag any module whose design is a line-by-line Java
  transliteration rather than a redesign around the format.

## Deep dive

[PLAN.md](../../../PLAN.md) §3.5 (full rationale), §7 (dedicated SIMD/perf
hardening phase).
