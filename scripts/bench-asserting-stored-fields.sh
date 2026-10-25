#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-asserting-stored-fields.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example asserting_stored_fields_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/AssertingStoredFieldsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_stored_fields_micro"
PIN=(taskset -c "${BENCH_PIN:-0,1}")
for REP in 1 2 3; do
  if [ "$((REP % 2))" = 1 ]; then
    "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
    "${PIN[@]}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" AssertingStoredFieldsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
  else
    "${PIN[@]}" java -ea -Xms128m -Xmx128m -cp "$OUT/classes:$CP" AssertingStoredFieldsMicro "$WARMUP" "$MEASURE" > "$OUT/java.$REP.tsv"
    "${PIN[@]}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$REP.tsv"
  fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check asserting_stored_fields_validation \
  --require-check asserting_stored_fields_read_speed \
  --require-check asserting_stored_fields_write_speed \
  --require-check asserting_stored_fields_read_compression \
  --require-check asserting_stored_fields_write_compression \
  --require-check asserting_stored_fields_merge_sequential \
  --require-check asserting_stored_fields_merge_sorted
for engine in rust java; do sed -n '/_bytes\t/p' "$OUT/$engine.1.tsv" | sed "s/^/$engine /"; done
echo 'Assertions stay enabled in both engines. Every batch verifies the complete callback/output digest.'
echo 'Validation includes format/wrapper creation, all seven field overloads, two documents, finish, double-close delegation and all values consumed by the counting delegate.'
echo 'Read opens real files, checks integrity, consumes all values through a visitor and closes. Write creates a fresh directory and hashes/deletes every output byte. Merge uses real readers, deletions and sorted/sequential maps through the asserting base visitor merge.'
echo 'Inputs, reflection lookup, formats, saved readers and merge metadata are prepared before timing. Both measured merge readers use ordinary decoding. Java merge-instance chunk caching is a separate unbenchmarked native API gap.'

echo 'Owned bytes count one format plus concrete delegate and writer wrapper state only. Excludes delegate graph, reader/writer codec buffers, outputs, directories, payloads, allocator metadata and process runtime; no per-iterator or RSS claim.'
