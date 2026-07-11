# lucene-rust

A Rust port of Apache Lucene with an FFI (JNI / Panama FFM) boundary for use as a
native search engine inside OpenSearch.

- **[PLAN.md](PLAN.md)** — the detailed porting plan (phases, crate layout, verification strategy).
- `crates/` — Cargo workspace, one crate per Lucene module boundary.
- `opensearch-plugin/` — Java-side OpenSearch engine plugin + JNI bindings (Phase 4+).
- `docs/` — porting conventions and the Java→Rust parity matrix.

Status: Phase 0 — scaffold only. See PLAN.md §2 for milestones.

License: Apache-2.0 (derivative work of Apache Lucene).
