#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-corrupt-unknown-files.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
check_load() {
  local load
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk "BEGIN{exit !($load > 1.5)}"; then
    echo "corrupt-unknown-files benchmark: load average $load exceeds 1.5" >&2
    exit 3
  fi
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
# The fixed Java-produced empty commit retains the original three commits.
# Every original codec file is therefore protected solely by older commits.
python3 - "$OUT/snapshot" <<'PY'
from pathlib import Path
import shutil
import sys
source = Path('fixtures/data/merge_policies/commits_index')
target = Path(sys.argv[1])
shutil.copytree(source, target)
rows = [line.split('\t') for line in Path(
    'fixtures/testframework/data/corrupt_unknown_files.tsv').read_text().splitlines()
    if line.startswith('snapshot\t')]
if len(rows) != 1 or len(rows[0]) != 3 or rows[0][1] != 'segments_4':
    raise SystemExit('expected one Java-produced segments_4 snapshot row')
commit = target / rows[0][1]
if commit.exists():
    raise SystemExit('snapshot commit already exists')
commit.write_bytes(bytes.fromhex(rows[0][2]))
if len(list(target.glob('segments_*'))) != 4:
    raise SystemExit('benchmark requires all four retained commits')
PY
cargo build --release -p lucene-test-framework --example corrupt_unknown_files_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/CorruptUnknownFilesMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/corrupt_unknown_files_micro"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$OUT/snapshot" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m -cp "$CP:$OUT" CorruptUnknownFilesMicro "$OUT/snapshot" "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check corrupt_unknown_small --require-check corrupt_unknown_large
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Actual MockDirectoryWrapper.corruptUnknownFiles selection: four retained Java-produced commits, with every original codec file referenced only by older commits; 8 or 2048 deterministic orphan codec/pending-segments files, plus ignored README/write.lock names.'
echo 'Directory listing, all commit parsing, referenced-file union, selection and callback name normalization/hash are timed. Snapshot copying, wrapper construction and expected-result construction are excluded. Every iteration checks the exact victim digest and count; every repetition must match across engines. Damage is replaced by a recording callback in both engines.'
echo 'Memory: no retained fields are added by this method. Transient scan sets and callback scratch allocations are unmeasured; this does not establish whole-wrapper owned-memory, peak-allocation, GC or RSS parity.'
