#!/usr/bin/env bash
# M4's crash campaign (T4.4): crates/lucene-search/examples/crash_fuzz.rs
# driven hard instead of long. Where the fixed scripts/crash-fuzz.sh run is a
# regression check, this spends a time budget on as many *different*
# conditions as it can:
#
#   - every core busy: one worker per job, each on its own seed range;
#   - every crash model in rotation: power loss (single writer), concurrent
#     power loss (ConcurrentIndexWriter + merge thread + committer), kill -9;
#   - stream lengths from short to long (crashes early, after many merges);
#   - each seed draws its own operation mix, buffer size and merge policy
#     (see crash_fuzz.rs), so seeds do not repeat one condition;
#   - real Lucene's CheckIndex on one batch in five;
#   - with --load, CPU burners and an fsync-heavy disk writer beside it, so
#     thread interleavings and sync latencies are not the quiet machine's.
#
# Usage: scripts/crash-storm.sh [--duration SECS] [--jobs N] [--load]
#                               [--seed-base S] [--jars DIR]
#   defaults: --duration 1800 --jobs $(nproc) --seed-base 5000000
# Any failure stops the run and prints the command that replays its seed.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
JARS="$PWD/fixtures/.jars"
DURATION=1800
JOBS=$(nproc)
LOAD=0
SEED_BASE=5000000

while [ $# -gt 0 ]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --load) LOAD=1; shift ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "crash-storm: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

cargo build --quiet --release -p lucene-search --example crash_fuzz
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/crash_fuzz"

WORK=$(mktemp -d)
pids=()
stressors=()
cleanup() {
  for p in "${stressors[@]}" "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

if [ "$LOAD" -eq 1 ]; then
  for _ in $(seq 1 $(( (JOBS + 1) / 2 ))); do
    ( while :; do :; done ) & stressors+=($!)
  done
  ( while :; do dd if=/dev/zero of="$WORK/io.bin" bs=1M count=64 conv=fsync status=none; done ) &
  stressors+=($!)
fi

# One worker: batches of 10 seeds, cycling mode x stream length, until the
# deadline. Seeds never overlap between workers or batches.
worker() {
  local job="$1" deadline="$2" batch=0
  local log="$WORK/job$job.log"
  local modes=("" "--concurrent" "" "--kill" "--concurrent" "")
  local lengths=(120 400 1500 700)
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local mode="${modes[$((batch % ${#modes[@]}))]}"
    local ops="${lengths[$((batch / ${#modes[@]} % ${#lengths[@]}))]}"
    local first=$(( SEED_BASE + job * 1000000 + batch * 10 ))
    local java=()
    [ $((batch % 5)) -eq 4 ] && java=(--java-cp "$CP")
    # shellcheck disable=SC2086  # an empty $mode must vanish
    if ! "$BIN" $mode --seeds "$first..$((first + 10))" --ops "$ops" \
         --dir "$WORK/dir$job" "${java[@]}" >>"$log" 2>&1; then
      echo "FAILED batch: $BIN $mode --seeds $first..$((first + 10)) --ops $ops" >>"$log"
      return 1
    fi
    batch=$((batch + 1))
  done
}

deadline=$(( $(date +%s) + DURATION ))
echo "crash-storm: $JOBS jobs for ${DURATION}s$([ "$LOAD" -eq 1 ] && echo ", under load")"
for job in $(seq 0 $((JOBS - 1))); do
  worker "$job" "$deadline" & pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
pids=()

count() { cat "$WORK"/job*.log | grep -c "$1" || true; }
power=$(count "round(s) passed" )
echo "crash-storm: rounds passed -- power loss: $(cat "$WORK"/job*.log | grep -E '^crash_fuzz: [0-9]+ power-loss' | awk '{s+=$2} END {print s+0}')," \
  "concurrent: $(cat "$WORK"/job*.log | grep -E '^crash_fuzz: [0-9]+ concurrent' | awk '{s+=$2} END {print s+0}')," \
  "kill -9: $(cat "$WORK"/job*.log | grep -E '^crash_fuzz: [0-9]+ kill' | awk '{s+=$2} END {print s+0}')" \
  "($power batches)"
echo "crash-storm: visible as the in-flight commit: $(count 'visible = in-flight'), as the last commit: $(count 'visible = last commit')"
if [ "$rc" -ne 0 ]; then
  grep -h -B2 "FAIL" "$WORK"/job*.log | head -20
  echo "crash-storm: FAILED; logs kept in $WORK"
  trap - EXIT
  cleanup
  exit 1
fi
rm -rf "$WORK"
echo "crash-storm: ok"
