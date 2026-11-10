#!/usr/bin/env bash
# Runs crates' tests built for aarch64 under qemu user emulation, so the NEON
# kernels can be checked without an ARM machine or a CI run.
#
#   scripts/test-aarch64-qemu.sh -p lucene-util --lib simd
#
# Needs: rustup target aarch64-unknown-linux-gnu, gcc-aarch64-linux-gnu and
# qemu-user (apt-get install gcc-aarch64-linux-gnu qemu-user). Release mode,
# since debug builds are slow under emulation. Timings mean nothing here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rustup target add aarch64-unknown-linux-gnu >/dev/null
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
exec cargo test --release --target aarch64-unknown-linux-gnu "$@"
