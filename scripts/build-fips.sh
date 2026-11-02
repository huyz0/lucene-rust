#!/usr/bin/env bash
# The FIPS build of the encryption layer: AES-256-GCM and the DRBG from
# AWS-LC's FIPS 140-3 validated module (aws-lc-rs, feature `fips`), with the
# default RustCrypto backend left out (`--no-default-features`), so no
# non-validated AES code is linked.
#
#   scripts/build-fips.sh          # build lucene-store-crypto (release)
#   scripts/build-fips.sh --test   # and run its tests, which assert FIPS mode
#
# Needs CMake, Go, Perl and a C compiler (aws-lc-fips-sys builds the module
# from source, as its validation requires). Uses its own target directory so
# the two backends never share build artifacts.
#
# FIPS compliance is a property of the deployed module, its version and the
# platform: check AWS-LC's current CMVP certificate for the aws-lc-fips-sys
# version Cargo resolved (`cargo tree -p lucene-store-crypto --no-default-features
# --features fips -i aws-lc-fips-sys`) against your operational environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/fips}"
args=(-p lucene-store-crypto --no-default-features --features fips)

for tool in cmake go cc; do
    command -v "$tool" >/dev/null || { echo "build-fips: $tool is required" >&2; exit 1; }
done

cargo build --release "${args[@]}"
if [[ "${1:-}" == "--test" ]]; then
    cargo test "${args[@]}"
fi
cargo tree "${args[@]}" -i aws-lc-fips-sys --depth 0
echo "build-fips: ok (${CARGO_TARGET_DIR})"
