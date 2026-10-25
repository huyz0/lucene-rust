#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-cranky-points.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_points_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CrankyPointsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_points_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -Xms128m -Xmx128m -cp "$OUT/classes:$CP" \
    org.apache.lucene.tests.codecs.cranky.CrankyPointsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check cranky_points_read --require-check cranky_points_write \
  --require-check cranky_points_merge_native --require-check cranky_points_merge_wrapped
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Owned bytes: format wrapper, concrete format and one shared RNG graph, counted once.'
echo 'Excludes readers/writers, BKD buffers, input/output directories, metadata, allocator metadata, runtime and RSS.'
echo 'Each batch checks full metadata/visited values or all output bytes plus RNG tails; 16 operations per batch.'
echo 'Read includes opening, integrity and close. Write includes ordinary tree addition, finish and close.'
echo 'Native merge uses the specialized 1D path; wrapped-reader merge uses the generic visitor fallback.'
