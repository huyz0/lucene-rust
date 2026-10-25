#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-cranky-norms.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_norms_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CrankyNormsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_norms_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -Xms128m -Xmx128m -cp "$OUT/classes:$CP" \
    org.apache.lucene.tests.codecs.cranky.CrankyNormsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check cranky_norms_read --require-check cranky_norms_lifecycle
for engine in rust java; do
  for rep in 1 2 3; do
    sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"
  done
done
echo 'Owned bytes: format wrapper, concrete delegate and one shared RNG graph, counted once.'
echo 'Excludes consumer/producer buffers, outputs, directories, segment metadata, allocator metadata, runtime and RSS.'
echo 'Read timing includes producer opening and ten document lookups; lifecycle includes two field additions, close and all output-byte consumption.'
