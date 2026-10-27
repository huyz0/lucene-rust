#!/usr/bin/env bash
# M14.h's fault run: crates/lucene-test-framework/examples/io_faults.rs over
# 300 seeds. An IndexWriter indexes through a MockDirectoryWrapper failing
# 10% of its writes and opens; each round must not panic, must see only
# injected errors, and must leave (after the mock directory's crash on close)
# an index holding a document count some commit could have left, which this
# port's CheckIndex passes and a new writer appends to. The last 50 seeds run
# real Lucene's org.apache.lucene.index.CheckIndex over the index too.
#
# Usage: scripts/test-framework-faults.sh [--jars DIR]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "test-framework-faults: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

cargo build --quiet --release -p lucene-test-framework --example io_faults
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/io_faults"

run() {
  local out
  if ! out=$("$BIN" "$@"); then
    echo "$out" | tail -3
    exit 1
  fi
  echo "$out" | tail -1
}
run --seeds 0..250
run --seeds 250..300 --java-cp "$CP"
echo "test-framework-faults: ok"
