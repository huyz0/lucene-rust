#!/usr/bin/env bash
# M4's differential operation-stream fuzzer (T4.5): the same seeded stream of
# adds, updates, deletes (by id and by body word), numeric doc-values updates,
# flushes and commits, applied to real Lucene 10.5.0's IndexWriter
# (fixtures/src/OpStreamFuzz.java) and to this port's
# (crates/lucene-search/examples/op_stream_fuzz.rs), each index then dumped
# semantically -- every live document with its doc values and points, the live
# documents every body term and a set of phrases match, a point range -- and
# the two dumps compared seed by seed. Segment counts, merge timing and file
# layouts legitimately differ and are not in the dump.
#
# Usage: scripts/op-stream-fuzz.sh [--jars DIR] [--seeds A..B] [--ops N] [--jobs J]
#   defaults: --seeds 0..1000 --ops 200 --jobs 4
# A failing seed replays with --seeds S..S+1; the dumps are kept on failure.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"
FIRST=0
END=1000
OPS=200
JOBS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --seeds) FIRST="${2%%..*}"; END="${2##*..}"; shift 2 ;;
    --ops) OPS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "op-stream-fuzz: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "$JOBS" -lt 1 ] || [ "$END" -le "$FIRST" ]; then
  echo "op-stream-fuzz: need --jobs >= 1 and a non-empty --seeds range" >&2
  exit 2
fi

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core lucene-analysis-common)

WORK=$(mktemp -d)
CLASSES="$WORK/classes"
OUT="$WORK/dumps"
mkdir -p "$CLASSES" "$OUT"

javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES/src/OpStreamFuzz.java"
cargo build --quiet --release -p lucene-search --example op_stream_fuzz
RUST="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/op_stream_fuzz"

# Seeds split into $JOBS contiguous chunks, each engine's chunks in parallel,
# each job logging to its own file so a failure can say why.
pids=()
logs=()
step=$(( (END - FIRST + JOBS - 1) / JOBS ))
for (( a = FIRST; a < END; a += step )); do
  b=$(( a + step < END ? a + step : END ))
  java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" OpStreamFuzz "$OUT" "$a" "$b" "$OPS" \
    >"$WORK/log.java.$a" 2>&1 &
  pids+=($!); logs+=("$WORK/log.java.$a")
  "$RUST" "$OUT" "$a" "$b" "$OPS" >"$WORK/log.rust.$a" 2>&1 &
  pids+=($!); logs+=("$WORK/log.rust.$a")
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "op-stream-fuzz: a job failed -- ${logs[$i]}:"
    grep -v '^Picked up JAVA_TOOL_OPTIONS' "${logs[$i]}" | tail -15 | sed 's/^/    /'
    rc=1
  fi
done
if [ "$rc" -ne 0 ]; then
  echo "op-stream-fuzz: dumps and logs kept in $WORK"
  exit 1
fi

failed=0
for (( seed = FIRST; seed < END; seed++ )); do
  if ! cmp -s "$OUT/$seed.java.txt" "$OUT/$seed.rust.txt"; then
    if [ "$failed" -lt 3 ]; then
      echo "seed $seed: the engines disagree (< Lucene, > this port)"
      diff "$OUT/$seed.java.txt" "$OUT/$seed.rust.txt" | head -12 | sed 's/^/    /'
    fi
    failed=$((failed+1))
  fi
done

if [ "$failed" -eq 0 ]; then
  docs=$(cat "$OUT"/*.java.txt | grep -c '^doc ' || true)
  # Evidence that merges ran on each side: segments left at the end, and
  # documents a merge reclaimed (added, but no longer counted in maxDoc).
  summary() { awk '{s += $1; m += $2} END {print s " segments, maxDoc " m}' "$OUT"/*."$1".meta; }
  echo "op-stream-fuzz: ok -- $((END - FIRST)) seeds x $OPS ops agree ($docs live documents compared)"
  echo "    Lucene: $(summary java); this port: $(summary rust)"
  rm -rf "$WORK"
else
  echo "op-stream-fuzz: $failed of $((END - FIRST)) seeds disagree; dumps kept in $OUT"
  exit 1
fi
