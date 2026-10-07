#!/usr/bin/env bash
# The gate, in one place.
#
# AGENTS.md's "Commands" table, `.githooks/pre-commit` and
# `scripts/docker-test.sh gate` all defer to this script, so the definition
# cannot drift between them. CI runs the same steps natively
# (.github/workflows/ci.yml) because its runners are already isolated.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "gate: cargo fmt --check"
cargo fmt --all --check

echo "gate: cargo clippy -D warnings (includes the arithmetic gate)"
cargo clippy --workspace --all-targets -- -D warnings

echo "gate: cargo clippy for aarch64 (c_char signedness differs by target)"
cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings

# `benchmarks/rust-runner` is deliberately outside the workspace (it depends on
# `test-support` features the shipped crates must not carry), so
# `clippy --workspace` above never compiles it. Twice now it has been left
# broken for several batches by an API reshape in a crate it consumes, while a
# stale binary under `target-docker/release/` kept producing plausible numbers
# from pre-change code -- which is worse than a red build, because a benchmark
# nobody can compile is at least obviously untrustworthy. `check`, not `build`:
# this is about the crate still type-checking against the current APIs, and a
# release build with fat LTO would cost minutes.
echo "gate: cargo check (benchmarks/rust-runner, outside the workspace)"
cargo check --manifest-path benchmarks/rust-runner/Cargo.toml --all-targets

echo "gate: check-arith-allows (every #[allow] carries an // ARITH: proof)"
python3 scripts/check-arith-allows.py

echo "gate: check-port-invariants (the defect shapes clippy cannot see)"
python3 scripts/check-port-invariants.py

echo "gate: check-parity (the parity ledger: paths, items, index links, budgets)"
python3 scripts/check-parity.py

echo "gate: check-java-refs (comments cite Java that exists in the *pinned* tree)"
python3 scripts/check-java-refs.py

echo "gate: check-licences (every shipped dependency is under an allowed licence)"
python3 scripts/check-licences.py

echo "gate: check-vendored-licences (every embedded third-party file's licence ships in LICENSE)"
python3 scripts/check-vendored-licences.py

echo "gate: check-port-inventory (every lucene-core, lucene-backward-codecs, lucene-spatial3d, lucene-spatial-extras, lucene-join, lucene-grouping, lucene-queries, lucene-analysis-common and M12 analysis module class has a status; every 'ported' location exists)"
python3 scripts/check-port-inventory.py --require-jar
python3 scripts/check-port-inventory.py --module backward-codecs --require-jar
python3 scripts/check-port-inventory.py --module spatial3d --require-jar
python3 scripts/check-port-inventory.py --module spatial-extras --require-jar
python3 scripts/check-port-inventory.py --module join --require-jar
python3 scripts/check-port-inventory.py --module grouping --require-jar
python3 scripts/check-port-inventory.py --module queries --require-jar
python3 scripts/check-port-inventory.py --module analysis-common --require-jar
for m in icu kuromoji nori smartcn stempel morfologik phonetic opennlp; do
  python3 scripts/check-port-inventory.py --module "analysis-$m" --require-jar
done

# rustdoc's link lints are warn-by-default and are reported by none of `fmt`,
# `clippy`, `test` or `llvm-cov`, so a broken doc link ships through a fully
# green gate -- c4 did exactly that. `--document-private-items` is deliberate:
# the port's wire-format knowledge lives in the doc comments of private
# decoders, and a link that only breaks there is still a broken link.
# `private_intra_doc_links` is allowed (166 sites): a public module doc here
# routinely points at the private helper that implements the thing it is
# describing, and that is the documentation working, not failing. See
# `docs/mechanical-gates.md#rustdoc`.
echo "gate: cargo doc (rustdoc link lints)"
RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" \
  cargo doc --workspace --no-deps --document-private-items

# Through scripts/coverage.sh, not `cargo llvm-cov` directly: it drops
# lucene-ffi's never-loaded cdylib from the report, whose phantom copy of every
# `#[no_mangle]` entry point otherwise hides the executed one. See
# docs/mechanical-gates.md#coverage-objects.
echo "gate: cargo llvm-cov (tests + >=95% line coverage)"
scripts/coverage.sh --workspace -- --fail-under-lines 95

# The plugin ships the release build, and the optimiser may change what debug
# arithmetic promised -- a NaN's bits, for one. lucene-search's fixtures compare
# doubles bit for bit, so they run in release too. See
# `docs/mechanical-gates.md#release-profile`.
echo "gate: lucene-search tests in release"
cargo test --release -p lucene-search

echo "gate: ok"
