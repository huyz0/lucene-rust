#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-bulk-scorer-wrapper.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-750}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "bulk-scorer-wrapper benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example bulk_scorer_wrapper_micro
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/BulkScorerWrapperMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/bulk_scorer_wrapper_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m \
    -cp "$WORK/classes:$CP" BulkScorerWrapperMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
CHECKS=()
for density in dense sparse; do
  for mode in next advance; do CHECKS+=(--require-check "bulk_wrapper_${density}_${mode}"); done
done
python3 scripts/bench-micro-report.py "$WORK" 3 "${CHECKS[@]}"
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'Each batch traverses the actual wrapper over identical 8192-document bulk sources, dense gap1 or sparse gap17, buffer64; advances seek current document+17.'
echo 'Source/wrapper/iterator construction, buffer allocation, checks and teardown are outside timing. Refill, score collection, binary search and consumption checksum are timed.'
echo 'Java GC may reclaim untimed setup allocations inside a later timed interval; its fixed 128 MiB heap and Rust deterministic teardown are part of this bounded observation.'
echo 'Sources return the window maximum lower bound, deliberately exercising empty windows. Every batch verifies all returned doc IDs and raw score bits by digest and count.'
echo 'The 512-byte metric is only the two logical 64-entry buffer payloads; object headers, pointers, padding, allocator metadata and source/input arrays are excluded.'
echo 'This measures wrapper traversal, not indexed-query performance. Actual indexed bulk scorers are covered by Java differential tests; Rust intrinsic arbitrary-query cost remains caller supplied.'
