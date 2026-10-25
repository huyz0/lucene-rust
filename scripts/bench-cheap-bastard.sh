#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-cheap-bastard.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cheap_bastard_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CheapBastardMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cheap_bastard_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
 taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.codecs.cheapbastard.CheapBastardMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
 if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
 else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 --require-check cheap_bastard_read --require-check cheap_bastard_write --require-check cheap_bastard_merge_sequential --require-check cheap_bastard_merge_sorted --require-check cheap_bastard_seeks
for engine in rust java; do
 for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Owned bytes: shallow codec and selected native format; delegate component graphs excluded.'
echo 'Excludes fields, postings, reader/writer state, all input/output buffers, allocator metadata, runtime and RSS.'
echo 'Every batch consumes full posting values or seven full output files plus lengths; 16 operations.'
echo 'Read includes opening, full integrity, enumeration and close. Native Rust reader eagerly materializes.'
echo 'Write and native mapped merge include encoding and close; Rust exposes buffers, Java uses ByteBuffersDirectory.'
echo 'Merge includes two-source borrowed mapping and integrity, sequential or sorted maps with deletions; inherited callback/error order is preserved.'


echo 'Seeks use preopened selected-format dictionaries in both engines and fresh TermsEnum per target; exclude opening and field data.'
