# Contributing

The repository is <https://github.com/huyz0/lucene-rust>. Pull requests run a short x64 pipeline;
the heavy proofs (ARM, fuzzing, full OpenSearch matrices) are run by hand before a release. See
[CI](contributing/gates/ci.md).

## The ground rules

1. **Port by on-disk format.** The Lucene index format is the contract; the Java class graph is not.
2. **Port, benchmark, then optimise**, one area at a time. A port left slower than Lucene is a bug.
3. **Every decoder has a differential test against real Lucene output.** Fixtures are written by
   Java programs under `fixtures/` and read by the Rust tests.
4. **`unsafe` lives in three crates only** (`lucene-util`, `lucene-store`, `lucene-ffi`) and a Rust
   panic never crosses the FFI boundary.
5. **Dependencies point downward**, enforced by the build.
6. **Stock compatibility.** New formats stay readable by stock Lucene; behaviour matches Lucene and
   OpenSearch unless deviating measurably helps speed, memory, I/O or disk, and then the deviation is
   recorded in [Compatibility](reference/compatibility.md).
7. **Gates stay green.** Format, clippy (including the arithmetic gate), port invariants, parity
   ledger, port inventory, licences, rustdoc, coverage (95% line coverage), file-size and binary-size
   ratchets. See [Mechanical gates](contributing/gates/mechanical-gates.md).

## Day to day

```sh
scripts/setup-hooks.sh        # pre-commit runs the gate
scripts/lr test quick         # fast loop
```

Commit messages are `type(scope): description` with one of `feat`, `fix`, `docs`, `test`, `chore`,
`refactor`, `perf`, `build`, `ci`; the commit hook checks it. AGENTS.md at the repository root is the
short orientation for contributors and coding agents, and `.agents/skills/` holds the detailed
process rules.

## Where things are

| Path | Contents |
|---|---|
| `crates/` | the Cargo workspace, one crate per Lucene module boundary plus the engine and FFI crates |
| `opensearch-plugin/` | the Java plugin, its tests and end-to-end drivers |
| `fixtures/` | Java generators and the real-Lucene bytes the tests read |
| `benchmarks/` | benchmark harnesses and corpus builders |
| `scripts/` | gates, benches, verification and the `lr` entry point |
| `docs/` | reference documentation, the parity ledger and the Lucene class inventory |
| `site/` | this documentation site |

The [parity ledger](https://github.com/huyz0/lucene-rust/blob/main/docs/parity.md) says, file by
file, what of Lucene is ported, partial or deliberately not ported.
