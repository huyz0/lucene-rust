#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-fst-tester-lists.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework lucene-misc):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example fst_tester_lists_micro
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" benchmarks/micro/java/FstTesterListsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/fst_tester_lists_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java -ea -Dtests.multiplier=1 -Xms128m -Xmx128m -cp "$WORK/classes:$CP" FstTesterListsMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$WORK" 3 \
  --require-check fst_tester_lists_utf8 --require-check fst_tester_lists_utf32
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'Each operation copies all input/pair/list containers, constructs a seeded tester and builds and verifies its complete FST.'
echo 'Checks include complete saved bytes, labels/list values, node/arc counts and RNG tails; all assertions remain enabled.'
echo 'Java uses a reused ByteBuffersDirectory and real offheap/save outputs; Rust models those paths as owned save/read byte round trips.'
echo 'Memory includes shallow tester state and its complete input-container/scalar graph, counted once; Java cached Long objects count once.'
echo 'Memory excludes RNG, directory, output algebra, compiled FST, verification/compiler scratch, allocator metadata and runtime; no peak-memory claim.'
