#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-asserting-points.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
cargo build --release -p lucene-test-framework --example asserting_points_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/AssertingPointsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_points_micro"
PIN=(taskset -c "${BENCH_PIN:-0,1}")
for REP in 1 2 3; do
  if [ "$((REP % 2))" = 1 ]; then
    "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
    "${PIN[@]}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.index.AssertingPointsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
  else
    "${PIN[@]}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.index.AssertingPointsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
    "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
  fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
 --require-check asserting_points_read \
 --require-check asserting_points_write \
 --require-check asserting_points_merge_native \
 --require-check asserting_points_merge_wrapped \
 --require-check asserting_points_validation
for ENGINE in rust java; do sed -n '/_bytes\t/p' "$OUT/$ENGINE.1.tsv" | sed "s/^/$ENGINE /"; done
echo 'Assertions enabled. Every warmup/timed batch verifies the complete result digest, required across both engines.'
echo 'Read opens real1D+2D files, integrity, all metadata/values, doc IDs and original/cloned trees. Write hashes every output byte. Both merge paths map/delete two sources and hash every output byte.'
echo 'Fixtures, fields, source readers, Java reflection and merge states are prepared outside timing. Validation creates a1D visitor, grow/compare and consumes13 points. Rust merge descriptors are stack-built in timing.'
echo 'Memory is one visitor state plus its three owned arrays, excluding delegate, inputs, allocator metadata, runtime, reader/tree/output state and RSS; Rust uses actual Vec capacities, Java shallow arrays include their object headers.'
