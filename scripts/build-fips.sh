#!/usr/bin/env bash
# The FIPS build of the encryption layer: AES-256-GCM and the DRBG from
# AWS-LC's FIPS 140-3 validated module (aws-lc-rs, feature `fips`), with the
# default non-validated AWS-LC build left out (`--no-default-features`), so no
# non-validated AES code is linked.
#
#   scripts/build-fips.sh          # build lucene-store-crypto (release)
#   scripts/build-fips.sh --test   # and run its tests, which assert FIPS mode
#   scripts/build-fips.sh --ffi    # build the plugin's library, lucene-ffi, with
#                                  # the FIPS module (target/fips/<profile>/
#                                  # liblucene_ffi.so); `gradle -Pfips` runs this
#   scripts/build-fips.sh --ffi --profile dist   # with a Cargo profile other
#                                  # than release
#   scripts/build-fips.sh --ffi --test   # and its encrypted-store tests
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
ffi=false
tests=false
profile=release
while [ $# -gt 0 ]; do
    case "$1" in
        --ffi) ffi=true ;;
        --test) tests=true ;;
        --profile) profile="$2"; shift ;;
        *) echo "build-fips: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
if $ffi; then
    # lucene-ffi's default features, with `fips` in place of `crypto`, and
    # the plugin build's `mimalloc`.
    args=(-p lucene-ffi --no-default-features --features
        fips,analysis-icu,analysis-kuromoji,analysis-nori,analysis-smartcn,analysis-stempel,analysis-morfologik,analysis-phonetic,mimalloc)
fi

for tool in cmake go cc; do
    command -v "$tool" >/dev/null || { echo "build-fips: $tool is required" >&2; exit 1; }
done

cargo build --profile "$profile" "${args[@]}"
if $tests; then
    if $ffi; then
        cargo test "${args[@]}" --lib store::
    else
        cargo test "${args[@]}"
    fi
fi
cargo tree "${args[@]}" -i aws-lc-fips-sys --depth 0
echo "build-fips: ok (${CARGO_TARGET_DIR})"
