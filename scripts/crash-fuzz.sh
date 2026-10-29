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
#   concurrent   40 seeds: power loss under a ConcurrentIndexWriter, its merge
#                thread and a committer; what survives must be a clean prefix
#                of the operations by sequence number
#
# Each must open, hold exactly the last durable commit (or the one in flight),
# pass CheckIndex, and recover under a new writer. A failure prints its seed
# and the command that replays it.
#
# Usage: scripts/crash-fuzz.sh [--jars DIR] [--duration SECS]
#   --duration SECS  instead of the fixed seed ranges, run power-loss rounds
#                    (Lucene's CheckIndex on every one) until SECS have passed.
#                    For a long campaign prefer scripts/crash-storm.sh, which
#                    spends the time on varied conditions in parallel.
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

# The four runs are independent, so they run at once (wall time: the longest,
# the power-loss run). Each stays one process over its whole seed range:
# crash_fuzz only demands its every-window evidence of a power-loss run of 100
# rounds or more and a kill run of 20 or more, so slicing them would drop it.
WORK=$(mktemp -d)
pids=()
names=()
start() {
  local name="$1"
  shift
  "$BIN" "$@" >"$WORK/$name.log" 2>&1 &
  pids+=($!); names+=("$name")
}
start power-loss --seeds 0..150
start java --seeds 1000..1025 --java-cp "$CP"
start kill --kill --seeds 0..40
start concurrent --concurrent --seeds 0..40 --ops 600
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    tail -1 "$WORK/${names[$i]}.log"
  else
    tail -3 "$WORK/${names[$i]}.log"
    rc=1
  fi
done
rm -rf "$WORK"
[ "$rc" -eq 0 ] || exit 1
echo "crash-fuzz: ok"
