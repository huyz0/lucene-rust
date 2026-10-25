#!/usr/bin/env bash
# Run through scripts/docker-test.sh bash scripts/bench-random-codec.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-500}"
MEASURE="${2:-750}"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAX_LOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAX_LOAD)}"; then
  echo "random-codec benchmark: load average $LOAD exceeds $MAX_LOAD" >&2
  exit 3
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cargo build --release -p lucene-test-framework --example random_codec_micro
javac -nowarn -proc:none --add-modules jdk.incubator.vector -cp "$CP" -d "$WORK/classes" \
  benchmarks/micro/java/RandomCodecMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/random_codec_micro"
run_rust() { taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$WORK/rust.$1.tsv"; }
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m \
    -Dtests.seed=0 -Dtests.nightly=false -Dtests.multiplier=1 \
    -cp "$WORK/classes:$CP" RandomCodecMicro "$WARMUP" "$MEASURE" > "$WORK/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
CHECKS=()
for kind in postings doc_values; do
  for mode in cached first_lookup; do
    CHECKS+=(--require-check "random_codec_${kind}_${mode}")
  done
done
python3 scripts/bench-micro-report.py "$WORK" 3 "${CHECKS[@]}"
for engine in rust java; do
  for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$WORK/$engine.$rep.tsv"; done
done
echo 'Comparative cases time actual RandomCodec field routing only, using precomputed identical 4096 Java UTF16 keys including collisions, supplementary pairs and unpaired surrogates.'
echo 'Cached cases prepopulate the selected cache outside timing and perform 16384 lookups per batch. First-lookup cases time 4096 misses, including key insertion and map growth, below the assertion-enabled 10000 limit.'
echo 'Every first-lookup batch constructs its codec and candidate identity/settings table before the timer, and releases its graph/cache references after the timer. Warmup and measurement accumulate routing intervals only.'
echo 'Java constructs a full retained native format graph; Rust constructs descriptors. Constructor, teardown, key construction and identity/settings table preparation are excluded; no whole-codec or point-format construction performance is claimed.'
echo 'Java input String.hashCode caches are prewarmed outside timing. Rust hashes borrowed UTF16 keys for cache lookup and computes Java hash on misses, then copies key units into cache-owned boxed slices.'
echo 'Equivalent timed loops identify each returned candidate by <=4/3 reference comparisons and digest key index, exact instance ordinal and precomputed complete name/settings token. Rust Rc clone/drop versus Java returned references is included.'
echo 'Every warmup/measured batch checks its digest and populated key count; all four routing checks carry the constructor/settings/input digest in every engine repetition.'
echo 'Memory metadata separates external logical UTF16 units from per-cache owned text payload: Rust reports 2*units requested live copied payload bytes and inline boxed-slice key-record bytes; Java owns no additional text/key Box records because its map aliases the exact external Strings.'
echo 'These are scoped logical/requested payload and key-record counts, not total cache size. HashMap tables/buckets/nodes/references, capacities/padding, Rc graphs, input String physical encodings, allocator/runtime overhead and RSS are excluded.'
echo 'Java GC can run inside a measured routing interval, including reclamation pressure from untimed setup; the fixed heap and these lifecycle differences are part of this bounded observation.'
echo 'No native postings/doc-values dispatch, suffix grouping, BKD split, SPI or IndexWriter throughput is measured.'
