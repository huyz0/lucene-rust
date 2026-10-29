#!/usr/bin/env bash
# M14.h h4's acceptance run: crates/lucene-test-framework/examples/random_index.rs
# over 300 seeds. A RandomIndexWriter under a random IndexWriterConfig runs a
# random stream of adds, updates, deletes, doc-values updates, commits,
# flushes and forced merges; every NRT reader it opens must hold exactly the
# live ids (and `n` values) a model holds, and survive a walk through
# AssertingLeafReader. The closed index must pass this port's CheckIndex; the
# last 50 seeds run real Lucene's org.apache.lucene.index.CheckIndex too.
#
# Usage: scripts/test-framework-index.sh [--jars DIR]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "test-framework-index: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

cargo build --quiet --release -p lucene-test-framework --example random_index
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/random_index"

# Seed ranges run as parallel slices: see scripts/lib-seed-shards.sh.
source "$(dirname "$0")/lib-seed-shards.sh"
run() {
  local first="$1" end="$2"
  shift 2
  run_seed_shards "$first" "$end" -- "$BIN" "$@" || exit 1
}
run 0 250
run 250 300 --java-cp "$CP"
echo "test-framework-index: ok"
