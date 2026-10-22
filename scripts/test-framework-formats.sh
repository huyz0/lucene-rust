#!/usr/bin/env bash
# M14.h h5's acceptance run: crates/lucene-test-framework/examples/format_suites.rs
# over 30 seeds per format suite (postings, stored fields, term vectors, doc
# values). Each seed's index -- the suite's addRandomFields through a
# RandomIndexWriter under a random IndexWriterConfig, with deletes, commits
# and sometimes a forced merge -- must pass this port's CheckIndex and real
# Lucene's org.apache.lucene.index.CheckIndex, which decodes every posting,
# stored document, term vector and doc value of every segment.
#
# Usage: scripts/test-framework-formats.sh [--jars DIR]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "test-framework-formats: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

cargo build --quiet --release -p lucene-test-framework --example format_suites
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/format_suites"

if ! out=$("$BIN" --seeds 0..30 --java-cp "$CP"); then
  echo "$out" | tail -3
  exit 1
fi
echo "$out" | tail -1
# Gate rule #10: 4 suites x 30 seeds, every one checked by Lucene.
echo "$out" | grep -q '^format_suites: 120 index(es), .* checked by Lucene too' || {
  echo "test-framework-formats: the run did not check all 120 indexes with Lucene" >&2
  exit 1
}
echo "test-framework-formats: ok"
