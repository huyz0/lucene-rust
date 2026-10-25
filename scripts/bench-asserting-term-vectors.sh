#!/usr/bin/env bash
# Official capped runner: scripts/docker-test.sh bash scripts/bench-asserting-term-vectors.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/AssertingTermVectorsMicro.java
cargo build --release -p lucene-test-framework --example asserting_term_vectors_micro
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_term_vectors_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingTermVectorsMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
printf 'Rust '; sed -n '/^#memory/p' "$WORK/rust.1.tsv"
printf 'Java '; sed -n '/^#memory/p' "$WORK/java.1.tsv"
python3 scripts/bench-micro-report.py "$WORK" 3 \
  --require-check asserting_term_vectors_validation --require-check asserting_term_vectors_merge \
  --require-check asserting_term_vectors_write --require-check asserting_term_vectors_read
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'Memory is shallow asserting writer state only; excludes delegate storage, private buffers, decoded models, runtime and allocator metadata.'
echo 'Validation consumes all callback values and inherited proximity, including explicit empty payload; finish/double-close are included per batch.'
echo 'Merge includes fresh wrapped writer, sorted deleted/remapped sources, all callbacks and double close. Write hashes every SimpleText output byte.'
echo 'Read includes clone, integrity, full occurrence consumption and double close; native owned model allocation remains timed, without Java lazy asserting iterator APIs.'
echo 'Reflection, source-reader opening, field metadata and format construction are outside timing. Java merge state is saved; Rust borrowed stack descriptors are timed.'
echo 'Java assertions stay enabled with a fixed 128 MiB heap.'
