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

echo "gate: check-parity (docs/parity.md rows point at files that exist)"
python3 scripts/check-parity.py

echo "gate: check-java-refs (comments cite Java that exists in the *pinned* tree)"
python3 scripts/check-java-refs.py

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

echo "gate: cargo llvm-cov (tests + >=95% line coverage)"
cargo llvm-cov --workspace --fail-under-lines 95

echo "gate: ok"
