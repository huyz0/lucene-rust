#!/usr/bin/env bash
# Profile-guided build of the plugin's native library, liblucene_ffi.so.
#
#   1. builds an instrumented library (the `dist` profile plus
#      -Cprofile-generate) into target/pgo-gen;
#   2. runs the training command with LUCENE_FFI_LIB set to that library, so
#      every process that loads it writes a .profraw file when it exits;
#   3. merges the profiles and rebuilds with -Cprofile-use into target/pgo.
#
# The training command decides what gets faster. LLVM optimizes functions the
# profile never saw for size, so code the training skips runs slower than in a
# plain `dist` build (7-10% on the untrained micro suites measured). Train on
# the workload the cluster runs: indexing and the query shapes it serves.
#
# Usage:
#   scripts/pgo-build.sh [--train CMD]
# The default CMD times the plugin's indexing path in process:
#   gradle --no-daemon -q -p opensearch-plugin indexingBench -Pmodes=rust \
#       -PnativeLib="$LUCENE_FFI_LIB"
# Then build the zip around the result:
#   gradle -p opensearch-plugin bundlePlugin -PnativeLib=target/pgo/dist/liblucene_ffi.so
#
# Needs llvm-profdata from the toolchain's llvm-tools component
# (`rustup component add llvm-tools`).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TRAIN='gradle --no-daemon -q -p opensearch-plugin indexingBench -Pmodes=rust -PnativeLib="$LUCENE_FFI_LIB"'
while [ $# -gt 0 ]; do
  case "$1" in
    --train) TRAIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "pgo-build: unknown argument: $1" >&2; exit 2 ;;
  esac
done

sysroot=$(rustc --print sysroot)
host=$(rustc -vV | sed -n 's/^host: //p')
PROFDATA="$sysroot/lib/rustlib/$host/bin/llvm-profdata"
[ -x "$PROFDATA" ] || { echo "pgo-build: no $PROFDATA; run rustup component add llvm-tools" >&2; exit 1; }

# RUSTFLAGS replaces .cargo/config.toml's rustflags, so restate its target CPU.
base_flags=""
[ "$host" = x86_64-unknown-linux-gnu ] && base_flags="-C target-cpu=x86-64-v3"

data="$PWD/target/pgo-data"
rm -rf "$data"; mkdir -p "$data"

echo "pgo-build: instrumented build" >&2
CARGO_TARGET_DIR=target/pgo-gen RUSTFLAGS="$base_flags -Cprofile-generate=$data" \
  cargo build --profile dist -p lucene-ffi --features mimalloc

# RUSTFLAGS instruments the build scripts too, and they wrote profiles while
# the build ran: keep only the training's.
rm -f "$data"/*.profraw

echo "pgo-build: training: $TRAIN" >&2
LUCENE_FFI_LIB="$PWD/target/pgo-gen/dist/liblucene_ffi.so" LLVM_PROFILE_FILE="$data/%p-%m.profraw" \
  bash -c "$TRAIN"
shopt -s nullglob
raw=("$data"/*.profraw)
[ ${#raw[@]} -gt 0 ] || { echo "pgo-build: the training wrote no profile" >&2; exit 1; }
"$PROFDATA" merge -o "$data/merged.profdata" "${raw[@]}"

echo "pgo-build: optimized build" >&2
CARGO_TARGET_DIR=target/pgo RUSTFLAGS="$base_flags -Cprofile-use=$data/merged.profdata" \
  cargo build --profile dist -p lucene-ffi --features mimalloc
echo "pgo-build: ok (target/pgo/dist/liblucene_ffi.so, from ${#raw[@]} profiles)"
