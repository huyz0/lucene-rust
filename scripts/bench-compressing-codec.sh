#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-compressing-codec.sh.
# Selected configuration/composition only: no constructor or format I/O timing.
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
  echo "compressing-codec benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example compressing_codec_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/CompressingCodecMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/compressing_codec_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m -cp "$CP:$OUT" CompressingCodecMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check compressing_cached \
  --require-check compressing_stored_display --require-check compressing_vectors_display \
  --require-check compressing_codec_display --require-check compressing_delegation \
  --require-check compressing_delegate_errors
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Per operation: cached=two SF and two TV accessors plus codec name; displays=one fresh string; delegation=two calls to each of nine factories; error=one failing factory.'
echo '23 preconstructed configurations cover six default/custom/suffix classes and five base cases, including null mode. Construction, reflection and delegate injection occur outside timing.'
echo 'Java values escape volatile Object sinks; Rust values escape black_box. Timing includes identity assertions, callback counters/traces and complete result-byte digests.'
echo 'Selected memory rows count codec/SF/TV shallow state, one shared settings node, custom name descriptors/payloads and one preset-mode allocation. Standard names/suffixes are derived static strings; custom Rust names/suffixes are owned once. Static modes add no per-codec allocation.'
echo 'Default delegate graphs, probe graphs, allocator metadata, returned components, transient strings, runtime and RSS are excluded. Rust default delegate is a zero-size routing descriptor.'
echo 'No complete-codec memory, constructor/random-factory performance, compression or stored-field/term-vector I/O parity is claimed.'
