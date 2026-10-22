#!/usr/bin/env bash
# M14.h h8's acceptance run: crates/lucene-test-framework/examples/
# threaded_stress.rs over 200 seeds of ThreadedIndexingAndSearchingTestCase
# (indexing threads adding, updating and deleting documents and blocks while
# search threads open and search near-real-time readers; seeds alternate a
# fresh NRT reader per search with a shared reader reopened when changed).
# Every run must pass the test's own checks and the port's CheckIndex. The
# multiplier sizes each run (10 x it iterations a thread); at 2 the writer
# rarely commits enough segments to merge, so the default is 5 (seen: over
# --seeds 0..3 --multiplier 2 the merge check below fails).
#
# Usage: scripts/test-framework-stress.sh [--seeds A..B] [--multiplier M]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
SEEDS=0..200
MULTIPLIER=5

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --multiplier) MULTIPLIER="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "test-framework-stress: unknown argument: $1" >&2; exit 2 ;;
  esac
done

cargo build --quiet --release -p lucene-test-framework --example threaded_stress
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/threaded_stress"

if ! out=$("$BIN" --seeds "$SEEDS" --multiplier "$MULTIPLIER"); then
  echo "$out" | tail -3
  exit 1
fi
echo "$out" | tail -1
# Gate rule #10: every seed ran, and the runs deleted, searched, merged and
# warmed what they merged.
want=$(( ${SEEDS#*..} - ${SEEDS%..*} ))
summary=$(echo "$out" | tail -1)
echo "$summary" | grep -q "^threaded_stress: $want run(s), " || {
  echo "test-framework-stress: expected $want runs" >&2
  exit 1
}
for what in deletes searches "merged segment(s) seen" warmed; do
  echo "$summary" | grep -Eq "[1-9][0-9]* $(printf '%s' "$what" | sed 's/[()]/\\&/g')" || {
    echo "test-framework-stress: no $what across the runs" >&2
    exit 1
  }
done
echo "test-framework-stress: ok"
