#!/usr/bin/env bash
# Run through scripts/docker-test.sh; this never starts an uncapped build.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "dummy-hit benchmark: load average $LOAD exceeds $MAX_LOAD; wait for the machine to settle" >&2
  exit 3
fi
PIN=(taskset -c "${BENCH_PIN:-0,1}")
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example dummy_hit_bench --quiet
javac -nowarn -cp "$CP" -d "$OUT" benchmarks/micro/java/DummyTotalHitCountMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/dummy_hit_bench"
run_timed() {
  python3 -c 'import resource, subprocess, sys; subprocess.run(sys.argv[2:], check=True); open(sys.argv[1], "w").write(str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))' "$@"
}
run_java() {
  run_timed "$OUT/java.rss" "${PIN[@]}" java -cp "$CP:$OUT" DummyTotalHitCountMicro "$WARMUP" "$MEASURE" > "$OUT/java"
}
run_rust() {
  run_timed "$OUT/rust.rss" "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust"
}
for REP in 1 2 3; do
  if [ "$((REP % 2))" = 1 ]; then run_java; run_rust; else run_rust; run_java; fi
  paste "$OUT/java" "$OUT/rust" | awk -v rep="$REP" '{
    rows++
    if (NF != 8 || $1 != "dummy_hit_callbacks" || $5 != $1 || $4 != $8 ||
        $2 !~ /^[0-9]+([.][0-9]+)?$/ || $6 !~ /^[0-9]+([.][0-9]+)?$/ ||
        $3 !~ /^[0-9]+$/ || $7 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ ||
        $2 <= 0 || $6 <= 0 || $3 <= 0 || $7 <= 0) {
      bad = 1
      next
    }
    result = sprintf("rep=%s case=%s digest=%s java_ns=%s rust_ns=%s java/rust=%.3f",rep,$1,$4,$2,$6,$2/$6)
  } END {
    if (bad || rows != 1) {
      print "dummy-hit benchmark: expected one matching nonempty result per engine" > "/dev/stderr"
      exit 1
    }
    print result
  }'
  printf 'rep=%s java_peak_rss_kib=%s rust_peak_rss_kib=%s\n' "$REP" "$(cat "$OUT/java.rss")" "$(cat "$OUT/rust.rss")"
done
echo 'RSS includes runtime startup, JVM heap/GC and benchmark input; it is not per-collector owned memory.'
echo 'Counter payload is 2 x 4-byte signed integers in each engine; Java object and leaf headers are additional.'
