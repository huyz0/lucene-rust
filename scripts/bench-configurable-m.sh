#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-configurable-m.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-750}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example configurable_m_micro
javac -nowarn -proc:none --add-modules jdk.incubator.vector -cp "$CP" -d "$WORK/classes" \
  fixtures/src/testframework/GenConfigurableM.java benchmarks/micro/java/ConfigurableMMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/configurable_m_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m \
    -cp "$WORK/classes:$CP" ConfigurableMMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
CHECKS=(--require-check configurable_m_construct)
for label in default_647 default_648 4_647 4_648; do
  for kind in read search_unfiltered search_filtered write; do
    CHECKS+=(--require-check "configurable_m_${label}_${kind}")
  done
done
python3 scripts/bench-micro-report.py "$WORK" 3 "${CHECKS[@]}"
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'Construction units contain four actual codecs (M=128/default,4,1,512), selected-format accessors, names and stable identity; Rust black_box and Java published handles prevent elimination.'
echo 'Construction includes Java default Lucene104 component graph creation; Rust default delegation is a routing descriptor only. This partial API does not establish equivalent whole-codec assembly cost.'
echo 'Every other unit covers all six real fields: dense float/byte counts 1294/1296, sparse counts 647/648 at the graph threshold, empty float/byte.'
echo 'All four output files must equal the actual fixture bytes before timing. Every timed result consumes complete file bytes, values/arcs, or hit score bits and visits; all 17 checks are required in every repetition.'
echo 'Read includes opening, full integrity, every ordinal value/doc and graph arc, then close. Rust checks data CRCs on open as well as integrity; Java defers data CRCs to integrity.'
echo 'Rust read adapters return owned vector/neighbor buffers; Java reuses vector scratch and a graph cursor. This ownership cost is included.'
echo 'Search uses preopened native readers, precomputed masks, all six fields and three queries, separately filtered/unfiltered; fresh collectors and query vectors are created inside each unit.'
echo 'Build/write uses precomputed identical source vectors, a fresh owned Rust batch clone versus Java addValue buffering, unsorted graph build with seed 42 and all four encoded files. Java global seed is restored in finally.'
echo 'Rust flat ordinal replay occurs at flush; Java replays when the threshold is reached during addValue. Encoding/output is Rust owned buffers versus Java ByteBuffersDirectory plus copying its four files into byte arrays.'
echo 'Memory metadata reports serialized file payload lengths only. It excludes input/output capacities, graph/reader/writer allocations, transient buffers, allocator/runtime costs and RSS; no peak-memory or graph-size claim.'
echo 'No streaming, sorted flush, merge or Codec/SPI/IndexWriter installation throughput is claimed.'
