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
  echo "cranky-stored-fields benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
PIN=(taskset -c "${BENCH_PIN:-0,1}")
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_stored_fields_micro --quiet
javac -nowarn -cp "$CP" -d "$OUT" benchmarks/micro/java/CrankyStoredFieldsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_stored_fields_micro"
run_timed() {
  python3 -c 'import resource, subprocess, sys; subprocess.run(sys.argv[2:], check=True); open(sys.argv[1], "w").write(str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))' "$@"
}
run_java() {
  run_timed "$OUT/java.$REP.rss" "${PIN[@]}" java -Xms128m -Xmx128m -cp "$CP:$OUT" CrankyStoredFieldsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
}
run_rust() {
  run_timed "$OUT/rust.$REP.rss" "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
}
for REP in 1 2 3; do
  if [ "$((REP % 2))" = 1 ]; then run_java; run_rust; else run_rust; run_java; fi
  printf 'rep=%s java_peak_rss_kib=%s rust_peak_rss_kib=%s\n' "$REP" "$(cat "$OUT/java.$REP.rss")" "$(cat "$OUT/rust.$REP.rss")"
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check cranky_stored_fields_read_speed \
  --require-check cranky_stored_fields_write_speed \
  --require-check cranky_stored_fields_read_compression \
  --require-check cranky_stored_fields_write_compression \
  --require-check cranky_stored_fields_merge_sequential \
  --require-check cranky_stored_fields_merge_sorted \
  --require-check cranky_stored_fields_faults
echo 'Peak RSS includes process runtime, JVM heap/GC, directories and transient buffers; it is not per-wrapper owned memory.'
echo 'Owned bytes count one wrapper, its concrete delegate, and one shared RNG allocation (Rc header/RefCell in Rust, Random/AtomicLong in Java); exclude directories, reader/writer instances, input values, metadata and transient codec buffers.'
echo 'Java benchmark heap is fixed at -Xms128m -Xmx128m.'

echo 'Read includes open, integrity, every field value via visitor and close. Write includes all seven field overloads, finish, close, complete output hashing and deletion.'
echo 'Merge uses real fixture readers, all value types, deletions and field-number remaps through the cranky base visitor merge, in sequential and sorted order.'
echo 'Faults use actual wrapper guards with precomputed seeds, untouched DataInput, delegated close count, exact messages and RNG tails. Reflection and fixtures are loaded before timing.'
