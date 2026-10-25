#!/usr/bin/env bash
# Actual GeoTestUtil.containsSlowly; run through scripts/docker-test.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
check_load() {
  local load
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk "BEGIN{exit !($load > 1.5)}"; then
    echo "geo-contains-slowly benchmark: load average $load exceeds 1.5" >&2
    exit 3
  fi
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example geo_contains_slowly_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/GeoContainsSlowlyMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/geo_contains_slowly_micro"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" GeoContainsSlowlyMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check geo_contains_small_full_scan --require-check geo_contains_large_full_scan \
  --require-check geo_contains_large_bbox_rejection --require-check geo_contains_large_boundary_early_exit
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Polygon/query setup is outside timing. Timed calls use the actual utility, consume every boolean in a checked digest and include per-call coordinate copies when present. Temporary payload and retained polygon bytes have separate scopes; neither measures RSS, allocator headers or GC.'
