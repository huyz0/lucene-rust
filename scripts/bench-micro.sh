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

echo "bench-micro: rust ($BENCH)" >&2
MICRO_WARMUP_MS="$WARMUP" MICRO_MEASURE_MS="$MEASURE" \
  "${PINCMD[@]}" benchmarks/rust-runner/target/release/micro "$BENCH" ${NEEDS_INDEX:+"$INDEX"} \
  > "$OUT/rust.tsv"

echo "bench-micro: java ($BENCH)" >&2
"${PINCMD[@]}" java --add-modules jdk.incubator.vector \
  -DwarmupMs="$WARMUP" -DmeasureMs="$MEASURE" \
  -cp "$CP:$OUT/classes" "$MAIN" ${NEEDS_INDEX:+"$INDEX"} > "$OUT/java.tsv"

join -t $'\t' <(sort "$OUT/rust.tsv") <(sort "$OUT/java.tsv") > "$OUT/joined.tsv"
if [ ! -s "$OUT/joined.tsv" ]; then
  echo "bench-micro: no cases joined -- the harnesses disagree on case names" >&2
  exit 1
fi

echo
printf '%-10s %12s %12s %8s\n' case rust_ns java_ns ratio
printf '%-10s %12s %12s %8s\n' ---- ------- ------- -----
awk -F'\t' '{printf "%-10s %12.3f %12.3f %7.2fx\n", $1, $2, $4, ($2>0)?$4/$2:0}' "$OUT/joined.tsv"

awk -F'\t' '{print ($2>0)?$4/$2:0}' "$OUT/joined.tsv" | sort -g > "$OUT/ratios"
awk -v n="$(wc -l < "$OUT/ratios")" '
  { sum+=$1; r[NR]=$1 }
  END {
    med = (n%2) ? r[int(n/2)+1] : (r[n/2]+r[n/2+1])/2
    printf "\n%d cases   mean %.2fx   median %.2fx\n", n, sum/n, med
    print "ratio > 1 means Rust is faster than Lucene on this case."
  }' "$OUT/ratios"
