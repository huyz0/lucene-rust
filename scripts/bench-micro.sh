#!/usr/bin/env bash
# Component microbenchmarks: run the Rust and Java harnesses over the same
# generated inputs, join on case name, and report the per-case ratio.
#
# Where scripts/bench-compare.sh answers "is a query faster", this answers
# "is this decode kernel faster". M1-e2e's profile came out flat -- largest
# single item 14.78% -- and a flat profile is precisely what an end-to-end
# benchmark cannot diagnose: it cannot tell you a kernel sitting at 9% of the
# profile is 3x off Lucene's. That is what this measures.
#
# Java runs with --add-modules jdk.incubator.vector so Lucene's Panama
# vectorized decode is live. Without it Lucene silently falls back to the
# scalar DefaultVectorizationProvider, and every ratio here flatters Rust.
#
# Usage: scripts/bench-micro.sh [--bench NAME] [--warmup-ms N] [--measure-ms N] [--pin CPUS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BENCH=for_decode
INDEX_ARG=""
# Repetitions of the whole A/B pair. Three is the minimum that gives a median
# and a spread; the spread is what decides whether a difference is reportable.
REPS=3
WARMUP=1500
MEASURE=2000
JARS="$PWD/fixtures/.jars"
# Same pinning rationale as bench-compare.sh: this is a hybrid P/E-core part,
# and a mid-measurement migration onto an E-core dominates the variance.
PIN="0,1"

while [ $# -gt 0 ]; do
  case "$1" in
    --bench)      BENCH="$2";   shift 2 ;;
    --index)      INDEX_ARG="$2"; shift 2 ;;
    --reps)       REPS="$2";    shift 2 ;;
    --warmup-ms)  WARMUP="$2";  shift 2 ;;
    --measure-ms) MEASURE="$2"; shift 2 ;;
    --pin)        PIN="$2";     shift 2 ;;
    --jars)       JARS="$2";    shift 2 ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "bench-micro: unknown argument: $1" >&2; exit 2 ;;
  esac
done

INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/merged}"
NEEDS_INDEX=""
case "$BENCH" in
  for_decode)
    MAIN=org.apache.lucene.codecs.lucene104.ForUtilMicro
    SRC=benchmarks/micro/java/org/apache/lucene/codecs/lucene104/ForUtilMicro.java ;;
  postings_iter)
    MAIN=PostingsIterMicro
    SRC=benchmarks/micro/java/PostingsIterMicro.java
    NEEDS_INDEX=1 ;;
  direct_reader)
    MAIN=DirectReaderMicro
    SRC=benchmarks/micro/java/DirectReaderMicro.java ;;
  stored_fields)
    MAIN=StoredFieldsMicro
    SRC=benchmarks/micro/java/StoredFieldsMicro.java
    NEEDS_INDEX=1 ;;
  reader_open)
    MAIN=ReaderOpenMicro
    SRC=benchmarks/micro/java/ReaderOpenMicro.java
    NEEDS_INDEX=1 ;;
  *) echo "bench-micro: no Java counterpart for $BENCH" >&2; exit 2 ;;
esac

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core)

OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

# Refuse to measure on a busy machine -- see bench-compare.sh's own note; two
# M1 measurement rounds were thrown away to background load before that guard.
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAXLOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAXLOAD)}"; then
  echo "bench-micro: refusing to measure -- 1-minute load average is $LOAD (limit $MAXLOAD)." >&2
  ps -eo pcpu,comm --sort=-pcpu | head -4 | sed 's/^/    /' >&2
  echo "  Wait for the machine to settle, or override with BENCH_MAX_LOAD=<n>." >&2
  exit 3
fi

echo "bench-micro: building" >&2
( cd benchmarks/rust-runner && cargo build --release --quiet )
javac -nowarn -cp "$CP" -d "$OUT/classes" "$SRC"

PINCMD=(taskset -c "$PIN")
command -v taskset >/dev/null || PINCMD=()

# Interleave the two engines rather than running all of one then all of the
# other. A run takes minutes and this machine drifts over that: whatever the
# drift is, alternating makes it fall on both sides equally instead of biasing
# whichever went second.
for rep in $(seq 1 "$REPS"); do
  echo "bench-micro: rep $rep/$REPS rust ($BENCH)" >&2
  MICRO_WARMUP_MS="$WARMUP" MICRO_MEASURE_MS="$MEASURE" \
    "${PINCMD[@]}" benchmarks/rust-runner/target/release/micro "$BENCH" ${NEEDS_INDEX:+"$INDEX"} \
    > "$OUT/rust.$rep.tsv"

  echo "bench-micro: rep $rep/$REPS java ($BENCH)" >&2
  "${PINCMD[@]}" java --add-modules jdk.incubator.vector \
    -DwarmupMs="$WARMUP" -DmeasureMs="$MEASURE" \
    -cp "$CP:$OUT/classes" "$MAIN" ${NEEDS_INDEX:+"$INDEX"} > "$OUT/java.$rep.tsv"
done

python3 scripts/bench-micro-report.py "$OUT" "$REPS"
