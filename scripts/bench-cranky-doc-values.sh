#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-cranky-doc-values.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_doc_values_micro
mkdir -p "$OUT/source"
{ printf 'package org.apache.lucene.tests.codecs.cranky;\n'; cat fixtures/src/testframework/GenCrankyDocValues.java; } > "$OUT/source/GenCrankyDocValues.java"
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" "$OUT/source/GenCrankyDocValues.java" benchmarks/micro/java/CrankyDocValuesMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_doc_values_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" org.apache.lucene.tests.codecs.cranky.CrankyDocValuesMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3; do
 if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
 else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 --require-check cranky_doc_values_read --require-check cranky_doc_values_write --require-check cranky_doc_values_merge_sequential --require-check cranky_doc_values_merge_sorted --require-check cranky_doc_values_empty_close
for engine in rust java; do
 for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Owned bytes: shallow native format + wrapper and one shared RNG graph; excludes readers, consumers, all column/file/snapshot buffers, allocator metadata, JVM runtime and RSS.'
echo 'Every batch consumes 16 full value/dictionary/skip digests or three full output files with lengths and RNG tails. Read includes open, integrity, all five types/dictionaries/skip data and close.'
echo 'Native Rust owns decoded columns; Java iterators decode on demand. Both consume all values without textual formatting.'
echo 'Write/merge/empty-close include native encoding and close; Rust exposes full buffers, Java ByteBuffersDirectory publication is included.'
