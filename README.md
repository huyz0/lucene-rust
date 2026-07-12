# lucene-rust

A Rust port of Apache Lucene with an FFI (JNI / Panama FFM) boundary for use as a
native search engine inside OpenSearch.

- **[PLAN.md](PLAN.md)** — the detailed porting plan (phases, crate layout, verification strategy).
- `crates/` — Cargo workspace, one crate per Lucene module boundary.
- `opensearch-plugin/` — Java-side OpenSearch engine plugin + JNI bindings (Phase 4+).
- `docs/` — porting conventions and the Java→Rust parity matrix.

Status: Phase 0 — scaffold only. See PLAN.md §2 for milestones.

## Benchmarks

`criterion` microbenchmarks cover the read-path hot loops that are fully
ported so far (see `docs/parity.md` for what that includes):

- `lucene-store` (`benches/varint_decode.rs`): `DataInput::read_vint`/
  `read_vlong`/`read_zlong`/`read_group_vints` — the per-value decode cost
  paid by nearly every format that hasn't moved to a bulk-decode path yet.
- `lucene-util` (`benches/util_ops.rs`): zigzag encode/decode, and
  `FixedBitSet::get`/`cardinality` over a 16384-bit block (Lucene's typical
  doc-values/postings block granularity).
- `lucene-codecs` (`benches/hot_paths.rs`): `direct_monotonic::get`
  (bit-unpacked monotonic sequence lookup), `StoredFieldsReader::document`
  (per-doc LZ4 chunk decompress), `PointsReader::decode_all_points` (BKD
  leaf decode), and `doc_values::numeric_value` (per-doc numeric lookup).

All of these reuse the same real, Java-Lucene-produced bytes under
`fixtures/data/` that the differential tests verify against, rather than
synthetic input — so a regression here tracks something representative of
real segments.

Run all of them:

```sh
cargo bench --workspace
```

Or a single crate's suite:

```sh
cargo bench -p lucene-store
cargo bench -p lucene-util
cargo bench -p lucene-codecs
```

**These are performance regression-tracking tools, not correctness
tests** — correctness is the differential tests' job (`crates/*/tests/`,
see the `differential-testing` skill). Benchmarks are not part of
`.githooks/pre-commit`; they're opt-in, run manually when you want to check
whether a change moved the needle on a hot path. `cargo llvm-cov` doesn't
compile or run `benches/` targets (it only builds lib/bin/test targets
unless passed `--all-targets`), so no coverage exclusion for `benches/` is
needed — nothing there shows up in the coverage report today.

License: Apache-2.0 (derivative work of Apache Lucene).
