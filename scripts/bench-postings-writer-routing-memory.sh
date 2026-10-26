#!/usr/bin/env bash
# Fresh Linux child per case. Run through scripts/docker-test.sh.
# Usage: [iterations_per_case [repetitions]], default 8 and 3.
set -euo pipefail
cd "$(dirname "$0")/.."
ITERATIONS="${1:-8}"
REPS="${2:-3}"
if [ "$#" -gt 2 ]; then echo 'expected iterations and repetitions' >&2; exit 2; fi
for value in "$ITERATIONS" "$REPS"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then echo 'iterations and repetitions must be positive integers' >&2; exit 2; fi
done
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example postings_writer_routing_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/PostingsWriterRoutingBench.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/postings_writer_routing_micro"
peak_rss() {
  python3 - "$@" <<'PY'
from pathlib import Path
import resource
import subprocess
import sys
engine, case, repetition, output = sys.argv[1:5]
if not sys.platform.startswith('linux'):
    raise SystemExit('run in Linux test container; RSS units are KiB')
result = subprocess.run(sys.argv[5:], check=True, text=True, stdout=subprocess.PIPE)
Path(output).write_text(result.stdout, encoding='utf-8')
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if peak <= 0:
    raise SystemExit('missing positive child peak RSS')
print(f'{engine}\t{case}\t{repetition}\t{peak}')
PY
}
run_engine() {
  local engine="$1" case="$2" rep="$3"
  if [ "$engine" = rust ]; then
    peak_rss rust "$case" "$rep" "$OUT/$engine.$case.$rep.tsv" taskset -c "${BENCH_PIN:-0,1}" "$RUST" --memory-case "$case" "$ITERATIONS"
  else
    peak_rss java "$case" "$rep" "$OUT/$engine.$case.$rep.tsv" taskset -c "${BENCH_PIN:-0,1}" java --add-modules jdk.incubator.vector -ea -Xms256m -Xmx256m -cp "$OUT/classes:$CP" PostingsWriterRoutingBench --memory-case "$case" "$ITERATIONS"
  fi
}
: > "$OUT/rss.tsv"
for mode in flush merge sorted_merge; do
  for codec in default asserting; do
    case="postings_routing_${codec}_${mode}_commit"
    for rep in $(seq 1 "$REPS"); do
      if [ $((rep % 2)) = 1 ]; then run_engine rust "$case" "$rep" >> "$OUT/rss.tsv"; run_engine java "$case" "$rep" >> "$OUT/rss.tsv";
      else run_engine java "$case" "$rep" >> "$OUT/rss.tsv"; run_engine rust "$case" "$rep" >> "$OUT/rss.tsv"; fi
    done
  done
done
python3 - "$OUT" "$REPS" "$ITERATIONS" <<'PY'
from pathlib import Path
import math
import re
import statistics
import sys
root, repetitions, iterations = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cases = {f'postings_routing_{codec}_{mode}_commit' for codec in ('default', 'asserting') for mode in ('flush', 'merge', 'sorted_merge')}
checks, samples, accounting = {}, {}, {}
for line in (root/'rss.tsv').read_text().splitlines():
    parts = line.split('\t')
    if len(parts) != 4 or parts[0] not in ('rust','java') or parts[1] not in cases or not all(re.fullmatch('[0-9]+', x) for x in parts[2:]):
        raise SystemExit(f'invalid RSS row: {line}')
    engine, case, rep, peak = parts[0], parts[1], int(parts[2]), int(parts[3])
    key = engine, case, rep
    if key in samples or not 1 <= rep <= repetitions or peak <= 0:
        raise SystemExit(f'duplicate or invalid RSS row: {line}')
    samples[key] = peak
    rows = (root/f'{engine}.{case}.{rep}.tsv').read_text().splitlines()
    check_rows = [row for row in rows if row.startswith('#check\t')]
    if len(check_rows) != 1 or check_rows[0].split('\t')[1] != case:
        raise SystemExit(f'missing semantic check: {key}')
    check = check_rows[0].split('\t')[2:]
    if len(check) != 7 or not all(re.fullmatch('[0-9]+', x) for x in check):
        raise SystemExit(f'invalid semantic check: {key}')
    if case in checks and checks[case] != check:
        raise SystemExit(f'semantic mismatch: {key}')
    checks[case] = check
    timing = [row.split('\t') for row in rows if row.startswith(case+'\t')]
    if (len(timing) != 1 or len(timing[0]) != 3
            or not math.isfinite(float(timing[0][1])) or float(timing[0][1]) <= 0
            or int(timing[0][2]) != iterations):
        raise SystemExit(f'invalid timing or iteration count: {key}')
    memory = {}
    for suffix in ('encoded_bytes', 'writer_accounted_bytes', 'logical_payload_bytes'):
        matches = [row.split('\t') for row in rows if row.startswith(case+'_'+suffix+'\t')]
        if (len(matches) != 1 or len(matches[0]) != 2
                or not re.fullmatch('[0-9]+', matches[0][1])):
            raise SystemExit(f'invalid {suffix}: {key}')
        memory[suffix] = int(matches[0][1])
    accounting[key] = memory
expected = {(engine,case,rep) for engine in ('rust','java') for case in cases for rep in range(1,repetitions+1)}
if samples.keys() != expected:
    raise SystemExit('missing RSS samples')
print(f'# iterations={iterations} repetitions={repetitions} java_heap=256MiB')
print('engine\tcase\tmedian_peak_process_rss_kib\tmin\tmax')
for case in sorted(cases):
    print('#check\t'+case+'\t'+'\t'.join(checks[case]))
    for engine in ('rust','java'):
        values=[samples[engine,case,rep] for rep in range(1,repetitions+1)]
        print(f'{engine}\t{case}\t{statistics.median(values):g}\t{min(values)}\t{max(values)}')
for key,peak in sorted(samples.items()):
    print('#rss\t'+'\t'.join(map(str,key))+f'\t{peak}')
    for scope, value in accounting[key].items():
        print('#accounting\t'+'\t'.join(map(str,key))+f'\t{scope}\t{value}')
print('Whole process peak includes setup/source buffers, verification, runtime/JIT, GC/allocator retention and directory data. It does not isolate adapter overhead or prove total memory parity.')
PY
