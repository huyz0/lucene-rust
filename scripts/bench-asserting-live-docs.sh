#!/usr/bin/env bash
# Run capped: scripts/docker-test.sh bash scripts/bench-asserting-live-docs.sh
# Real Java asserting wrapper with -ea. Guard checks 4096 bits then counts a
# delegate call; write checks 65 bits and emits real Lucene90 .liv bytes.
# Both consume equal digests. Memory covers format wrapper only.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/AssertingLiveDocsMicro.java
cargo build --release -p lucene-test-framework --example asserting_live_docs_micro
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_live_docs_micro"
REPS=3
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) = 1 ]; then
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
    taskset -c 0,1 java -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingLiveDocsMicro > "$WORK/java.$rep.tsv"
  else
    taskset -c 0,1 java -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingLiveDocsMicro > "$WORK/java.$rep.tsv"
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
  fi
done
printf 'Rust '; sed -n '/^#memory/p' "$WORK/rust.1.tsv"
printf 'Java '; sed -n '/^#memory/p' "$WORK/java.1.tsv"
python3 scripts/bench-micro-report.py "$WORK" "$REPS" --require-check asserting_live_docs_guard --require-check asserting_live_docs_write
