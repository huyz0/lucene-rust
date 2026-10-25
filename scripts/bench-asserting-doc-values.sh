#!/usr/bin/env bash
# Use scripts/docker-test.sh bash scripts/bench-asserting-doc-values.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example asserting_doc_values_micro
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" fixtures/src/testframework/GenCrankyDocValues.java fixtures/src/testframework/GenAssertingDocValues.java benchmarks/micro/java/AssertingDocValuesMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_doc_values_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingDocValuesMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$WORK" 3 \
  --require-check asserting_doc_values_validate --require-check asserting_doc_values_read \
  --require-check asserting_doc_values_write --require-check asserting_doc_values_merge_lifecycle
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'validate checks all five input types then retains the original producer at the delegate boundary.'
echo 'read opens, checks integrity, consumes every value/dictionary/skip entry, and double closes; write checks all three complete outputs.'
echo 'merge_lifecycle acquires a merge producer, reads all values and skips, and double closes; no inherited merge-throughput claim.'
echo 'Native Rust eagerly owns columns and buffers publication; Java native writer acquisition passes/error timing are not reproduced.'
echo 'Wrapper memory excludes delegates, field metadata, columns, files, temporary bitsets, allocator metadata, runtime and RSS.'
echo 'Java shallow wrapper includes object headers; Rust reports additional wrapper state beyond its embedded delegate.'