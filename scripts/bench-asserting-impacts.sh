#!/usr/bin/env bash
# Actual AssertingLeafReader.AssertingImpactsEnum; run through docker-test.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
MODE="${3:-wrapped}"
EXTRA=()
REQUIRED=(--require-check asserting_impacts_single --require-check asserting_impacts_multilevel --require-check asserting_impacts_positions)
if [ "$MODE" = raw ]; then
  EXTRA=(raw)
  REQUIRED=(--require-check asserting_impacts_single_raw)
elif [ "$MODE" != wrapped ]; then
  echo "asserting-impacts benchmark: mode must be wrapped or raw" >&2
  exit 2
fi
check_load() {
  local load
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk "BEGIN{exit !($load > 1.5)}"; then
    echo "asserting-impacts benchmark: load average $load exceeds 1.5" >&2
    exit 3
  fi
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
mask = (1 << 64) - 1
# Identical higher-level lists dominate lower lists. Bounds are valid fixed
# synthetic blocks, not codec decode benchmarks. Every synthetic document has
# norm 1 and frequency <=4; the first pair dominates them. Later norms straddle
# signed i64 in the multilevel and positions cases.
for name, count, gap, levels, pairs, positions in [
    ('single', 4096, 1, 1, 1, False),
    ('multilevel', 4096, 3, 3, 32, False),
    ('positions', 2048, 4, 2, 8, True),
]:
    rows = []
    docs = [i * gap + 1 for i in range(count)]
    freqs = [i % 4 + 1 for i in range(count)]
    for doc, freq in zip(docs, freqs):
        rows.append(f'D\t{doc}\t{freq}')
    impact_lists = []
    for level in range(levels):
        values = [(4 + i, i + 1 if i < max(1, pairs // 2) else (1 << 63) + i + 1)
                  for i in range(pairs)]
        impact_lists.append(((level + 1) * 1_000_000, values))
        for freq, norm in values:
            rows.append(f'L\t{level}\t{(level + 1) * 1_000_000}\t{freq}\t{norm}')
    digest = 0
    operations = 0
    def op(kind, arg=0, values=()):
        global digest, operations
        rows.append(f'T\t{kind}\t{arg}')
        for value in (kind, *values):
            digest = (digest * 31 + value) & mask
        operations += 1
    def impacts():
        values = [levels]
        for bound, pairs_ in impact_lists:
            values.extend((bound, len(pairs_)))
            for freq, norm in pairs_:
                values.extend((freq, norm))
        op(3, values=values)
    op(9, values=[count])
    previous_index = -1
    for index, doc in enumerate(docs):
        # Alternate advance and nextDoc in the mixed case; multilevel always
        # uses advance with a target in the gap, so returned doc may exceed it.
        use_advance = name == 'multilevel' or (positions and index % 2 == 1)
        target = doc - 1 if use_advance else (docs[previous_index] + 1 if previous_index >= 0 else 0)
        op(0, target)
        if index % 8 == 0:
            impacts()  # legal before moving: shallow already positioned it
        op(2 if use_advance else 1, target if use_advance else 0, [doc])
        op(8, values=[doc])
        op(4, values=[freqs[index]])
        impacts()
        if index % 8 == 0:
            impacts()  # repeated view while doc/shallow state stays unchanged
        if positions:
            for position in range(freqs[index]):
                value = 1 + 3 * position
                op(5, values=[value])
                op(6, values=[value * 5, value * 5 + 2])
                op(7, values=[2, doc & 255, position + 1] if position % 2 == 0 else [-1])
        previous_index = index
    op(1, values=[(1 << 31) - 1])
    op(8, values=[(1 << 31) - 1])
    op(9, values=[count])
    rows.append(f'Z\t{digest}')
    (root / f'{name}.tsv').write_text('\n'.join(rows) + '\n')
    print(f'#input\tasserting_impacts_{name}\t{digest}\t{operations}\t{count}')
PY
cargo build --release -p lucene-test-framework --example asserting_impacts_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/AssertingImpactsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/asserting_impacts_micro"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$OUT" "$WARMUP" "$MEASURE" "${EXTRA[@]}" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea -Xms128m -Xmx128m -cp "$CP:$OUT" org.apache.lucene.tests.index.AssertingImpactsMicro "$OUT" "$WARMUP" "$MEASURE" "${EXTRA[@]}" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 "${REQUIRED[@]}"
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done

if [ "$MODE" = raw ]; then
  echo 'Unwrapped single-pair control: fresh processes use the same delegate, trace, digest and timing boundaries with no asserting wrapper. Rust retains its owned-vector ImpactsEnum contract; Java retains borrowed impacts buffers. This compares aggregate wrapped and unwrapped traversal costs; removing wrappers also changes dispatch and compiler profiles, so subtraction cannot isolate individual wrapper, allocator or dispatch costs.'
  exit 0
fi
echo 'Actual AssertingImpactsEnum over shared precomputed synthetic docs/frequencies, impact levels and explicit operation traces. Single: 4096 dense docs, one level/pair; multilevel: 4096 gap-3 docs, three levels of 32 unsigned-norm pairs; positions: 2048 gap-4 docs, two levels of eight pairs, every position/offset/payload. All include legal shallow calls, document traversal, getImpacts validation, repeated immediate views, cost and terminal state.'
echo 'Timed: complete trace dispatch, actual wrapper checks, delegate traversal, impact materialization/validation, view consumption and digest. Input parsing, data construction, delegate/wrapper construction, final digest assertion and final wrapper destruction are outside timing. Java -ea; Rust release assertions. Each iteration must match a Python-generated expected-value digest; every engine/repetition must agree.'
echo 'Ownership: Rust ImpactsEnum returns owned cloned vectors while Java returns borrowed buffers and an asserting view; both representations and Java validation scratch copies are included. The shallow wrapper byte row counts Rust inline AssertingImpactsEnum (including its inline postings checks), and Java AssertingImpactsEnum plus its owned AssertingPostingsEnum. Both exclude the delegate, shared input/trace, allocator headers, alignment padding outside objects and transient impacts/views/validation copies. Fixed synthetic blocks are not codec I/O or search benchmarks. Transient allocations, peak allocation, GC and RSS are unmeasured; shallow bytes do not establish whole-graph memory parity.'
