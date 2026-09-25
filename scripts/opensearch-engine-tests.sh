#!/usr/bin/env bash
# OpenSearch's own engine tests (InternalEngineTests, 3.8.0), run on the Rust
# engine: opensearch-plugin/tools/derive_engine_tests.py rewrote them to build
# RustEngine, and every test not listed in its SKIPPED map must pass.
#
#   scripts/opensearch-engine-tests.sh [-Dtests.method=testFoo] [-Dtests.seed=...]
#
# The test framework refuses to run as root, so a root shell runs the suite as
# `nobody` from a staged classpath. Needs Maven Central for the test framework.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

gradle --no-daemon -q -p opensearch-plugin engineTestClasspath
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -r opensearch-plugin/build/engine-test-classpath "$stage/cp"
mkdir -p "$stage/tmp"
chmod -R a+rwX "$stage"
as=()
[[ $(id -u) == 0 ]] && as=(runuser -u nobody --)
cd "$stage/tmp"
"${as[@]}" java -ea -Xmx2g --add-opens=java.base/java.nio=ALL-UNNAMED \
    -Dtests.security.manager=false -Djava.io.tmpdir="$stage/tmp" \
    -Dlucene_rust.library.path="$stage/cp/liblucene_ffi.so" \
    "$@" -cp "$stage/cp:$stage/cp/*" \
    org.junit.runner.JUnitCore org.lucenerust.opensearch.engine.RustEngineTests
