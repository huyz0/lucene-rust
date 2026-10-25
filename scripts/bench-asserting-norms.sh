#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-asserting-norms.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example asserting_norms_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/AssertingNormsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_norms_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" \
    AssertingNormsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check asserting_norms_validation --require-check asserting_norms_read --require-check asserting_norms_lifecycle
for engine in rust java; do
  for rep in 1 2 3; do
    sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"
  done
done
echo 'Owned bytes: format wrapper and concrete delegate, counted once.'
echo 'Excludes validation delegate, input producers/counters, consumer/producer buffers, outputs, directories, segment metadata, allocator metadata, runtime and RSS.'
echo 'Validation: 4096 docs, assertion pass plus original-producer delegate pass, all acquisition/next/value callbacks counted. Read: 65-doc dense/sparse producer opening and iteration. Lifecycle: fresh directory, two field additions, close and every output byte. Reflection, format/default-codec construction and saved read state are outside timing.'
