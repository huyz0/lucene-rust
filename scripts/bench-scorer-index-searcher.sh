#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-scorer-index-searcher.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "scorer-index-searcher benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example scorer_index_searcher_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" fixtures/src/testframework/GenScorerIndexSearcher.java benchmarks/micro/java/ScorerIndexSearcherMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/scorer_index_searcher_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"; }
run_java() { taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m -cp "$CP:$OUT" ScorerIndexSearcherMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check scorer_searcher_complete --require-check scorer_searcher_no_scores \
  --require-check disabling_query_complete --require-check disabling_query_no_scores
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Each step searches the same nine queries across three committed leaves of 768 documents, 192 deletions, under COMPLETE or COMPLETE_NO_SCORES. Both engines hash every global doc id and score bit, counts and query order; four mandatory checks must agree.'
echo 'Actual ScorerIndexSearcher and DisablingBulkScorerQuery APIs; both deliberately suppress specialized bulk scorers and top-level supplier optimization. Preparation/rewriting, weight/scorer creation and digest collectors are timed; index construction, reader opening and query construction are not.'
echo 'Memory: wrapper shallow increment over the same underlying IndexSearcher only; Rust owns its searcher by value without a size increment; Java subclass has no added fields; neither wrapper adds a heap allocation. Excludes complete reader/searcher graphs, per-search temporary allocations, query-wrapper memory, allocator/GC/RSS. Field sorts/facets and leaf subranges are not benchmarked.'
