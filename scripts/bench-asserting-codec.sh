#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-asserting-codec.sh.
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
  echo "asserting-codec benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example asserting_codec_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/AssertingCodecMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_codec_micro"
run_rust() {
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Dtests.seed=0 -Dtests.multiplier=1 -Dtests.nightly=false -Xms128m -Xmx128m -cp "$CP:$OUT" AssertingCodecMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check asserting_codec_cached --require-check asserting_codec_suppressed \
  --require-check asserting_codec_mixed --require-check asserting_codec_selectors \
  --require-check asserting_codec_knn_bypass --require-check asserting_codec_display \
  --require-check asserting_codec_errors
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Per step: cached/suppressed/mixed=all eleven accessors twice; selectors/KNN bypass=three selectors x four fields x direct/internal/public paths; display=one fresh string for each of three suppression masks; errors=all eleven accessors once for each mask.'
echo 'Actual asserting wrappers and cached routers; explicit stable provider/delegate handles prepared outside timing. Java uses actual AssertingCodec and native inner defaults; Rust defaults are opaque handles except the existing live-docs format. No constructor, native graph, component I/O or SPI performance claim.'
echo 'Stable delegate points handles deliberately exclude native Lucene104 fresh-points construction. Includes returned-handle consumption, identity checks, counters, digest and Java volatile sink/Rust black_box overhead. Errors reuse one precreated exception or Rc error per probe; no timed stack creation.'
echo 'Memory uses a plain DefaultSelectors codec, not the instrumented hooks/alternates. Selected sum counts shallow codec, eight cached asserting wrappers, separately allocated cached routers, shared selector state and suppression EnumSet exactly once. Java shallow sizes include object headers; Rust Rc nodes include two counters, alignment and padding using the pinned standard-library repr(C) allocation layout.'
echo 'Memory excludes allocator metadata, all native default/delegate inner graphs, provider/probe instrumentation, expected-reference arrays, alternate hook formats, static names, transient strings, runtime/GC and RSS. Rust suppression bits and three typed router handles are inline in codec shallow size; cached_routers_bytes is zero because no separate router nodes exist. Router logical identity is router type plus shared Rc selector state; selected format Rc identity is unchanged. Java bindings are already in shallow codec/router references; no extra selector allocation. Selected adapter scope only; JVM heap fixed at -Xms128m -Xmx128m.'
