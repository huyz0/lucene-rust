#!/usr/bin/env bash
# Test-only benchmark: its Rust example avoids a production dependency on
# lucene-test-framework. Run through scripts/docker-test.sh bash <this-script>.
# Each unit opens and consumes every byte of a resource (8 or 256 repetitions
# of the same Unicode string); paired result digests must agree before timing
# ratios are reported. This includes Java's UTF-16-to-UTF-8 transcoding, which
# Rust's String already provides, and each language's resource-consumption API.
# *_retained_bytes and *_open_bytes report owned storage excluding allocator
# metadata: Rust struct + capacity, Java shallow objects + backing arrays.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-test-framework)"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/StringMockResourceLoaderMicro.java
cargo build --release -p lucene-test-framework --example string_resource_loader_micro
RUST="${CARGO_TARGET_DIR:-target}/release/examples/string_resource_loader_micro"
REPS=3
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) = 1 ]; then
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" StringMockResourceLoaderMicro > "$WORK/java.$rep.tsv"
  else
    taskset -c 0,1 java -Xms128m -Xmx128m -cp "$WORK/classes:$CP" StringMockResourceLoaderMicro > "$WORK/java.$rep.tsv"
    taskset -c 0,1 "$RUST" > "$WORK/rust.$rep.tsv"
  fi
done
python3 scripts/bench-micro-report.py "$WORK" "$REPS" \
  --require-check string_resource_loader_8 --require-check string_resource_loader_256
