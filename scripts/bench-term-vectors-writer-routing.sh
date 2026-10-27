#!/usr/bin/env bash
# Installed asserting term-vector writers; run through scripts/docker-test.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
for value in "$WARMUP" "$MEASURE"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then echo 'durations must be positive integers' >&2; exit 2; fi
done
check_load() {
  local load attempt
  for attempt in $(seq 0 60); do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk "BEGIN{exit !($load <= 1.5)}"; then return; fi
    if [ "$attempt" = 0 ]; then echo "term vectors routing benchmark: cooling down; load $load exceeds 1.5" >&2; fi
    if [ "$attempt" = 60 ]; then echo 'term vectors routing benchmark: load remains above 1.5' >&2; exit 3; fi
    sleep 5
  done
}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example term_vectors_writer_routing_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" fixtures/src/testframework/TermVectorsRoutingSupport.java benchmarks/micro/java/TermVectorsWriterRoutingBench.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/term_vectors_writer_routing_micro"
# Verify every asserted Rust operation with Lucene's native readers as well.
for mode in 0 1 2; do
  "$RUST" --write-verify "$OUT/rust-index-$mode" "$mode" > "$OUT/rust-verify-$mode.tsv"
  java --add-modules jdk.incubator.vector -ea -Xms256m -Xmx256m -cp "$OUT/classes:$CP" TermVectorsWriterRoutingBench --verify-dir "$OUT/rust-index-$mode" "$mode" > "$OUT/java-verify-$mode.tsv"
  diff <(grep '^#check' "$OUT/rust-verify-$mode.tsv") <(grep '^#check' "$OUT/java-verify-$mode.tsv")
done
run_rust() { check_load; taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { check_load; taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms256m -Xmx256m -cp "$OUT/classes:$CP" TermVectorsWriterRoutingBench "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3 4 5 6; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
checks=()
for mode in flush merge sorted_merge; do for engine in default asserting; do checks+=(--require-check "term_vectors_routing_${engine}_${mode}_commit"); done; done
python3 scripts/bench-micro-report.py "$OUT" 6 "${checks[@]}"
for engine in rust java; do for rep in 1 2 3 4 5 6; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Flush: prepared-document insertion + commit timed; merges: forceMerge + commit timed. Writer setup, document preparation, accounting samples, verification and destruction excluded. Writer accounted bytes use different Rust and Java accounting scopes. Encoded output lengths and logical payload bytes are distinct from peak process RSS.'
