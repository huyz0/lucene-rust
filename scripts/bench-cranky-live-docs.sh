#!/usr/bin/env bash
# Capped runner: scripts/docker-test.sh bash scripts/bench-cranky-live-docs.sh
# Both cases use actual CrankyLiveDocsFormat, replaying seed 42 per 256 writes.
# guard: RNG, failure construction/catching, counting delegate; write: actual
# .liv bytes through a fresh ByteBuffersDirectory, consumed into a digest.
# Memory is wrapper + shared RNG once; it excludes delegate/input/transients.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/CrankyLiveDocsMicro.java
cargo build --release -p lucene-test-framework --example cranky_live_docs_micro
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_live_docs_micro"
REPS=3
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) = 1 ]; then
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" CrankyLiveDocsMicro > "$WORK/java.$rep.tsv"
  else
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" CrankyLiveDocsMicro > "$WORK/java.$rep.tsv"
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
  fi
done
printf 'Rust '; sed -n '/^#memory/p' "$WORK/rust.1.tsv"
printf 'Java '; sed -n '/^#memory/p' "$WORK/java.1.tsv"
python3 scripts/bench-micro-report.py "$WORK" "$REPS" --require-check cranky_live_docs_guard --require-check cranky_live_docs_write
