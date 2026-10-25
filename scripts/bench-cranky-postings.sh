#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-cranky-postings.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_postings_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CrankyPostingsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_postings_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
 taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.codecs.cranky.CrankyPostingsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
 if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
 else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 --require-check cranky_postings_read --require-check cranky_postings_write --require-check cranky_postings_merge_sequential --require-check cranky_postings_merge_sorted
for engine in rust java; do
 for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Owned bytes: shallow format wrapper, concrete native format and one shared RNG graph, counted once.'
echo 'Excludes fields, postings, reader/writer state, all input/output buffers, allocator metadata, runtime and RSS.'
echo 'Every batch consumes full posting values or seven full output files plus lengths and RNG tails; 16 operations.'
echo 'Read includes opening, full integrity, enumeration and close. Native Rust reader eagerly materializes.'
echo 'Write and inherited merge include native encoding and close; Rust exposes buffers, Java uses ByteBuffersDirectory.'
echo 'Merge includes two-source mapped Fields construction and integrity, sequential or sorted maps with deletions.'
