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
      echo "live-docs routing benchmark: cooling down; load average $load exceeds 1.5" >&2
    fi
    if [ "$attempt" = 60 ]; then
      echo "live-docs routing benchmark: load average $load still exceeds 1.5 after 300 seconds" >&2
      exit 3
    fi
    sleep 5
  done
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example live_docs_writer_routing_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/LiveDocsWriterRoutingMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/live_docs_writer_routing_micro"
# Prove the asserted Rust writer remains readable by Lucene's native readers.
"$RUST" --write-verify "$OUT/rust-index"
java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" LiveDocsWriterRoutingMicro --verify-dir "$OUT/rust-index"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" LiveDocsWriterRoutingMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check liv_routing_default_first_commit \
  --require-check liv_routing_asserting_first_commit \
  --require-check liv_routing_default_subsequent_commit \
  --require-check liv_routing_asserting_subsequent_commit
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'Only the specified write intervals are timed. Corpus setup, prior-generation preparation, deletion queuing and decoded-result checks are outside each interval. Memory rows have separate scopes; neither whole-writer retention nor RSS is measured.'
