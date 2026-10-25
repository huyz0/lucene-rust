#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-cranky-field-infos.sh.
# Construction plus 256 explicit reads/writes per batch. The counting delegate
# consumes the same suffix lengths; this does not measure codec or directory I/O.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_field_infos_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CrankyFieldInfosMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_field_infos_micro"
run_rust() {
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  taskset -c "${BENCH_PIN:-0,1}" java -Xms128m -Xmx128m -cp "$OUT/classes:$CP" \
    org.apache.lucene.tests.codecs.cranky.CrankyFieldInfosMicro "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check cranky_field_infos_read --require-check cranky_field_infos_write
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for engine in ('rust', 'java'):
    sizes = []
    for rep in range(1, 4):
        lines = (root / f'{engine}.{rep}.tsv').read_text().splitlines()
        for line in lines:
            if not line.startswith('#memory\t'):
                print(f'{engine} rep={rep} {line}')
        rows = [line.split('\t') for line in lines
                if line.startswith('#memory\t')]
        if (len(rows) != 1 or len(rows[0]) != 3 or rows[0][1] != 'cranky_field_infos'
                or not rows[0][2].isdigit() or int(rows[0][2]) <= 0):
            raise SystemExit(f'invalid {engine} owned-memory result')
        sizes.append(int(rows[0][2]))
    if len(set(sizes)) != 1:
        raise SystemExit(f'inconsistent {engine} owned-memory results: {sizes}')
    print(f'{engine}_owned_bytes={sizes[0]}')
print('Owned bytes include wrapper, counting delegate and one RNG with its backing state; shared RNG is counted once.')
print('Excludes allocator metadata, directories, empty field infos, static input strings, runtime and RSS.')
PY
