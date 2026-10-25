#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-cranky-codec.sh.
# Explicit inert probes: no native default codec graph or I/O workload.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "cranky-codec benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_codec_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/CrankyCodecMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_codec_micro"
run_rust() {
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m -cp "$CP:$OUT" CrankyCodecMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check cranky_codec_components --require-check cranky_codec_knn \
  --require-check cranky_codec_shared_guards --require-check cranky_codec_display
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Per operation: components=two wrappers from each of ten factories; KNN=two accessor calls; shared guards=one write; display=one fresh string.'
echo 'Components escape Java volatile Object sinks and Rust black_box; Java allocates identity-bearing wrappers, Rust constructs owned values. Includes sink/identity/count/digest overhead.'
echo 'Shallow adapter bytes exclude the delegate probe (inline delegate size subtracted in Rust), allocator overhead, returned component wrappers and transient strings.'
echo 'Owned name bytes count the active Rust name payload: zero for inline names up to 15 UTF-8 bytes, boxed byte length otherwise; enum storage remains in the shallow adapter row. Java adds zero because its name reference shares the delegate name.'
echo 'Shared RNG bytes count one Rc/RefCell/JavaRandom allocation in Rust and Random plus AtomicLong in Java; exclude the probe graph, runtime, heap/GC and RSS.'
echo 'Names and RNG are separate memory rows, not a total graph-memory comparison. Java heap fixed at -Xms128m -Xmx128m.'
