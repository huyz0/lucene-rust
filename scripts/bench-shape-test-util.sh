#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-shape-test-util.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework lucene-misc):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"; MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example shape_test_util_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/ShapeTestUtilMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/shape_test_util_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java -ea -Dtests.nightly=false -Xms128m -Xmx128m -cp "$OUT/classes:$CP" ShapeTestUtilMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3; do
 if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
 else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 --require-check shape_biased_floats --require-check shape_primitives --require-check shape_polygon_line --require-check shape_regular_polygon --require-check shape_contains_slowly
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Owned bytes: one RNG graph, or one five-vertex polygon and its arrays; excludes runtime, allocator metadata, temporary arrays, unrelated shapes and RSS.'
echo 'Every batch checks all raw float bits or all oracle results and RNG continuation. Java randomized context construction is outside timing; actual context lookups remain timed.'
