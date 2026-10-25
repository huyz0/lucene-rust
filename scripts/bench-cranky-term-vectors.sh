#!/usr/bin/env bash
# Official capped runner: scripts/docker-test.sh bash scripts/bench-cranky-term-vectors.sh
# Actual Cranky wrapper; nested addProx and sorted superclass merge callbacks,
# real SimpleText writer output and cached-reader full occurrence consumption.
# Setup, reflection, reader opening and merge metadata are outside timed batches.
# Each result includes complete callback/output digest and shared RNG continuation.
# Memory: format/writer wrapper overhead + shared RNG once; delegates excluded.
# Each language uses 1000ms warmup and 1500ms measurement per case/repetition.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/CrankyTermVectorsMicro.java
cargo build --release -p lucene-test-framework --example cranky_term_vectors_micro
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_term_vectors_micro"
REPS=3
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) = 1 ]; then
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" CrankyTermVectorsMicro > "$WORK/java.$rep.tsv"
  else
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" CrankyTermVectorsMicro > "$WORK/java.$rep.tsv"
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
  fi
done
printf 'Rust '; sed -n '/^#memory/p' "$WORK/rust.1.tsv"
printf 'Java '; sed -n '/^#memory/p' "$WORK/java.1.tsv"
python3 scripts/bench-micro-report.py "$WORK" "$REPS" --require-check cranky_term_vectors_prox --require-check cranky_term_vectors_merge --require-check cranky_term_vectors_write --require-check cranky_term_vectors_read
