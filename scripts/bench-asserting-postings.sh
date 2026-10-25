#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-asserting-postings.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example asserting_postings_micro
javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" \
  fixtures/src/testframework/GenCrankyPostings.java \
  fixtures/src/testframework/GenAssertingPostings.java \
  benchmarks/micro/java/AssertingPostingsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_postings_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms128m -Xmx128m -cp "$WORK/classes:$CP" AssertingPostingsMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$WORK" 3 \
  --require-check asserting_postings_read --require-check asserting_postings_write \
  --require-check asserting_postings_merge_sequential --require-check asserting_postings_merge_sorted \
  --require-check asserting_postings_validation
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'All assertions stay enabled. Every batch checks full posting values/statistics or all seven file lengths/bytes; 16 operations.'
echo 'Validation-only re-enumerates every source enum through a no-op delegate, including close; its digest is a completion marker.'
echo 'Source readers, field infos, norms, maps, format configuration and reflected merge/consumer construction are outside timing.'
echo 'Read includes open, full integrity, checked statistics/enumeration and close. Rust eagerly decodes; acquired Terms are snapshots.'
echo 'Write and merge include delegate encoding followed by a fresh assertion walk and double close. Rust buffers files; Java uses ByteBuffersDirectory.'
echo 'Memory is shallow validation consumer wrapper plus stateless delegate, counted once. Excludes referenced SegmentWriteState/FieldInfos, enums, reader/writer state, buffers, allocator and runtime; no retained/peak/RSS claim.'
