#!/usr/bin/env bash
# M4's crash fuzzing (T4.4): crates/lucene-search/examples/crash_fuzz.rs over
# a spread of seeds, in both crash models, with real Lucene's CheckIndex on a
# share of them.
#
#   power loss  150 seeds: the writer runs over CrashingDirectory, which fails
#               a uniformly chosen directory operation and then leaves what a
#               power loss could (unsynced files damaged, a prefix of the
#               unpublished creates/renames/deletes kept)
#   + Lucene     25 more seeds with org.apache.lucene.index.CheckIndex too
#   kill -9      40 seeds: a child process killed at a random moment
#
# Each must open, hold exactly the last durable commit (or the one in flight),
# pass CheckIndex, and recover under a new writer. A failure prints its seed
# and the command that replays it.
#
# Usage: scripts/crash-fuzz.sh [--jars DIR] [--duration SECS]
#   --duration SECS  instead of the fixed seed ranges, run power-loss rounds
#                    (Lucene's CheckIndex on every one) until SECS have passed
#                    -- the 24-hour soak is --duration 86400
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"
DURATION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "crash-fuzz: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

cargo build --quiet --release -p lucene-search --example crash_fuzz
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/crash_fuzz"

if [ -n "$DURATION" ]; then
  "$BIN" --duration "$DURATION" --seed 0 --java-cp "$CP" | tail -3
  exit "${PIPESTATUS[0]}"
fi

run() {
  local out
  if ! out=$("$BIN" "$@"); then
    echo "$out" | tail -3
    exit 1
  fi
  echo "$out" | tail -1
}
run --seeds 0..150
run --seeds 1000..1025 --java-cp "$CP"
run --kill --seeds 0..40
echo "crash-fuzz: ok"
