#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-asserting-knn-vectors.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example asserting_knn_vectors_micro
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" fixtures/src/testframework/GenAssertingKnnVectors.java benchmarks/micro/java/AssertingKnnVectorsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_knn_vectors_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingKnnVectorsMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$WORK" 3 \
  --require-check asserting_knn_vectors_values --require-check asserting_knn_vectors_read \
  --require-check asserting_knn_vectors_search --require-check asserting_knn_vectors_merge_lifecycle
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'values acquires/asserts real native values; read opens, verifies CRCs, visits all vectors and graph neighbors, and double closes.'
echo 'search runs all twelve float/byte/query/filter cases with scores and visits; merge_lifecycle acquires values and finishes an actual merge reader.'
echo 'Java uses ByteBuffersDirectory; Rust native readers also check data CRCs during open, while Java defers data CRCs to checkIntegrity.'
echo 'Reader memory includes wrapper state and its two counters (Rust includes Arc allocation header); delegate/FieldInfos/bytes/parsed metadata and all temporary scratch excluded.'
echo 'AcceptDocs memory is its asserting wrapper only, excluding delegate/filter storage; neither figure claims peak or total native memory.'
echo 'Native writer/merge construction and Codec/IndexWriter routing are absent; no write/merge-throughput or byte-output claim.'
