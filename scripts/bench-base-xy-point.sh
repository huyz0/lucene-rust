#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-base-xy-point.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework lucene-misc):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"; MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example base_xy_point_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/BaseXYPointMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/base_xy_point_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Dtests.seed=0 -Dtests.multiplier=1 -Dtests.nightly=false -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.geo.BaseXYPointMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3; do
 if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
 else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
 --require-check base_xy_random_tiny --require-check base_xy_random_medium \
 --require-check base_xy_same_point --require-check base_xy_low_cardinality \
 --require-check base_xy_all_x --require-check base_xy_all_y \
 --require-check base_xy_shape_hooks --require-check base_xy_oracles
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Every generator batch uses seed 42, multiplier 1 and four capture callbacks that digest every raw coordinate bit plus callback order; RNG tails are included.'
echo 'Hooks batch consumes sixteen X/Y/box/polygon/geometry sequences with raw coordinates and types. Oracle batch consumes 1,024 rectangle and distance pairs, including result bits.'
echo 'Java randomized context and capture construction are outside timing; actual inherited context lookups, arrays, geometry creation, capture walks and seed reset remain timed. No indexing/query verifier work is measured.'
echo 'Owned bytes: capture shallow state; two representative 1,000-float coordinate arrays (Rust vector headers/payload, Java array headers/payload); one five-vertex polygon/arrays. Excludes allocator metadata, unrelated shapes, temporary publication/copies, runtime and RSS; no peak or whole-suite memory claim.'
