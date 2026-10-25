#!/usr/bin/env bash
# Run through scripts/docker-test.sh; includes real codec and directory costs.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "cranky-segment-info benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
PIN=(taskset -c "${BENCH_PIN:-0,1}")
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_segment_info_micro --quiet
javac -nowarn -cp "$CP" -d "$OUT" benchmarks/micro/java/CrankySegmentInfoMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_segment_info_micro"
run_timed() {
  python3 -c 'import resource, subprocess, sys; subprocess.run(sys.argv[2:], check=True); open(sys.argv[1], "w").write(str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))' "$@"
}
run_java() {
  run_timed "$OUT/java.$REP.rss" "${PIN[@]}" java -Xms128m -Xmx128m -cp "$CP:$OUT" CrankySegmentInfoMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
}
run_rust() {
  run_timed "$OUT/rust.$REP.rss" "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
}
for REP in 1 2 3; do
  if [ "$((REP % 2))" = 1 ]; then run_java; run_rust; else run_rust; run_java; fi
  printf 'rep=%s java_peak_rss_kib=%s rust_peak_rss_kib=%s\n' "$REP" "$(cat "$OUT/java.$REP.rss")" "$(cat "$OUT/rust.$REP.rss")"
done
python3 scripts/bench-micro-report.py "$OUT" 3 --require-check cranky_segment_info_read --require-check cranky_segment_info_write
echo 'Peak RSS includes process runtime, JVM heap/GC, directories and transient buffers; it is not per-wrapper owned memory.'
echo 'Owned bytes count one wrapper, its concrete delegate, and one shared RNG allocation (Rc header/RefCell in Rust, Random/AtomicLong in Java); exclude directories and transient buffers.'
echo 'Java benchmark heap is fixed at -Xms128m -Xmx128m.'
