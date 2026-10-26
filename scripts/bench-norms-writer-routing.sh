#!/usr/bin/env bash
# Operational same-wire asserting writers; run through scripts/docker-test.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
check_load() {
  local load attempt
  for attempt in $(seq 0 60); do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk "BEGIN{exit !($load <= 1.5)}"; then return; fi
    if [ "$attempt" = 0 ]; then
      echo "norms routing benchmark: cooling down; load average $load exceeds 1.5" >&2
    fi
    if [ "$attempt" = 60 ]; then
      echo "norms routing benchmark: load average $load still exceeds 1.5 after 300 seconds" >&2
      exit 3
    fi
    sleep 5
  done
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example norms_writer_routing_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/NormsWriterRoutingMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/norms_writer_routing_micro"
# Prove the asserted Rust writer remains readable by Lucene's native readers.
"$RUST" --write-verify "$OUT/rust-index"
java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" NormsWriterRoutingMicro --verify-dir "$OUT/rust-index"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" NormsWriterRoutingMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3 4 5 6; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 6 \
  --require-check norms_routing_default_flush_commit \
  --require-check norms_routing_asserting_flush_commit \
  --require-check norms_routing_default_merge_commit \
  --require-check norms_routing_asserting_merge_commit
for engine in rust java; do
  for rep in 1 2 3 4 5 6; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Whole flush commit or forceMerge+commit intervals are timed, including native/asserting norms close/output, sibling formats and publication. Corpus creation, merge source preparation, deletion queuing and decoded-result checks are outside each interval. Memory rows have separate scopes; neither whole-writer retention nor RSS is measured.'
