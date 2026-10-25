#!/usr/bin/env bash
# Invoke only through scripts/docker-test.sh bash scripts/bench-block-score-query.sh.
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
  echo "block-score-query benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example block_score_query_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" fixtures/src/testframework/GenBlockScoreQueryWrapper.java benchmarks/micro/java/BlockScoreQueryMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/block_score_query_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$CP:$OUT" BlockScoreQueryMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check block_buffer_1 --require-check block_bounds_1 \
  --require-check block_buffer_2 --require-check block_bounds_2 \
  --require-check block_buffer_3 --require-check block_bounds_3 \
  --require-check block_buffer_7 --require-check block_bounds_7
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done
done
echo 'buffer: construct and consume one wrapper scorer over eight exact constant-scored documents; bounds: eight shallow advances and sixteen maxima on a prebuilt scorer.'
echo 'Index construction and weight/query preparation are excluded. Bounds scorer construction and destruction are excluded; buffer includes scorer construction and destruction.'
echo 'Both engines hash every returned document, score and block maximum each batch. Four block lengths run on the same documents/scores. No whole-search speedup is claimed.'
