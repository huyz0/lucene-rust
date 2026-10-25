#!/usr/bin/env bash
# Fresh-process Linux RSS for both real norms cases. Run via docker-test.sh.
# Usage: bash scripts/bench-cranky-norms-memory.sh [warmup_ms [measure_ms [reps]]]
# Uses the timing benchmark's exact arguments and fixed 128 MiB JVM heap.
# Peak RSS covers both cases together, including consumer/producer buffers,
# directories, runtime/JIT, allocator/GC retention and startup. It cannot assign
# a peak to one case or isolate the transient buffers' incremental footprint.
set -euo pipefail
cd "$(dirname "$0")/.."
WARMUP="${1:-500}"
MEASURE="${2:-1000}"
REPS="${3:-3}"
if [ "$#" -gt 3 ]; then
  echo 'cranky-norms-memory: expected at most warmup_ms, measure_ms and reps' >&2
  exit 2
fi
for value in "$WARMUP" "$MEASURE" "$REPS"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
    echo 'cranky-norms-memory: durations and repetitions must be positive integers' >&2
    exit 2
  fi
done
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
cargo build --release -p lucene-test-framework --example cranky_norms_micro
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" benchmarks/micro/java/CrankyNormsMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/cranky_norms_micro"

# A new Python parent for every child prevents cumulative RUSAGE_CHILDREN peaks.
# Compilation and javac never enter the measured child's resource usage.
peak_rss() {
  python3 - "$@" <<'PY'
from pathlib import Path
import resource
import subprocess
import sys

engine, repetition, output = sys.argv[1:4]
if not sys.platform.startswith('linux'):
    raise SystemExit('cranky-norms-memory: run in Linux test container; RSS units are KiB')
result = subprocess.run(sys.argv[4:], check=True, text=True, stdout=subprocess.PIPE)
Path(output).write_text(result.stdout)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if peak <= 0:
    raise SystemExit('cranky-norms-memory: missing positive child peak RSS')
print(f'{engine}\t{repetition}\t{peak}')
PY
}
run_engine() {
  local engine="$1" repetition="$2"
  if [ "$engine" = rust ]; then
    peak_rss rust "$repetition" "$OUT/rust.$repetition.tsv" \
      taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$WARMUP" "$MEASURE"
  else
    peak_rss java "$repetition" "$OUT/java.$repetition.tsv" \
      taskset -c "${BENCH_PIN:-0,1}" java -Xms128m -Xmx128m -cp "$OUT/classes:$CP" \
      org.apache.lucene.tests.codecs.cranky.CrankyNormsMicro "$WARMUP" "$MEASURE"
  fi
}
: > "$OUT/rss.tsv"
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) = 1 ]; then
    run_engine rust "$rep" >> "$OUT/rss.tsv"
    run_engine java "$rep" >> "$OUT/rss.tsv"
  else
    run_engine java "$rep" >> "$OUT/rss.tsv"
    run_engine rust "$rep" >> "$OUT/rss.tsv"
  fi
done
# The shared reporter enforces paired timings/checks and finite positive times.
python3 scripts/bench-micro-report.py "$OUT" "$REPS" \
  --require-check cranky_norms_read --require-check cranky_norms_lifecycle
python3 - "$OUT" "$REPS" "$WARMUP" "$MEASURE" <<'PY'
from pathlib import Path
import re
import statistics
import sys

root, repetitions = Path(sys.argv[1]), int(sys.argv[2])
cases = {'cranky_norms_read', 'cranky_norms_lifecycle'}
memory_case = 'cranky_norms_format_owned_bytes'
reference = None
owned = {}
for engine in ('rust', 'java'):
    sizes = []
    for rep in range(1, repetitions + 1):
        checks, timings = {}, set()
        size = None
        for line in (root / f'{engine}.{rep}.tsv').read_text().splitlines():
            if not line:
                continue
            fields = line.split('\t')
            if fields[0] == '#check':
                if (len(fields) != 4 or fields[1] not in cases or fields[1] in checks
                        or not re.fullmatch(r'[0-9]+', fields[2])
                        or int(fields[2]) >= 1 << 64 or fields[3] != '256'):
                    raise SystemExit(f'invalid {engine} rep {rep} check: {line}')
                checks[fields[1]] = fields[2]
            elif fields[0] == memory_case:
                if len(fields) != 2 or size is not None or not re.fullmatch(r'[0-9]+', fields[1]):
                    raise SystemExit(f'invalid {engine} rep {rep} owned bytes: {line}')
                size = int(fields[1])
            elif fields[0] in cases:
                if (len(fields) != 3 or fields[0] in timings
                        or not re.fullmatch(r'[0-9]+', fields[2])
                        or int(fields[2]) < 256 or int(fields[2]) % 256):
                    raise SystemExit(f'invalid {engine} rep {rep} timing units: {line}')
                timings.add(fields[0])
            else:
                raise SystemExit(f'unexpected {engine} rep {rep} row: {line}')
        if checks.keys() != cases or timings != cases or size is None:
            raise SystemExit(f'incomplete {engine} rep {rep} benchmark output')
        if reference is None:
            reference = checks
        elif checks != reference:
            raise SystemExit(f'{engine} rep {rep}: deterministic digests differ')
        sizes.append(size)
    if len(set(sizes)) != 1:
        raise SystemExit(f'{engine}: inconsistent owned byte estimates')
    owned[engine] = sizes[0]

samples = {}
for line in (root / 'rss.tsv').read_text().splitlines():
    fields = line.split('\t')
    if (len(fields) != 3 or fields[0] not in ('rust', 'java')
            or not all(re.fullmatch(r'[0-9]+', value) for value in fields[1:])):
        raise SystemExit(f'invalid RSS row: {line}')
    key = fields[0], int(fields[1])
    if key in samples or not 1 <= key[1] <= repetitions or int(fields[2]) <= 0:
        raise SystemExit(f'duplicate or invalid RSS sample: {line}')
    samples[key] = int(fields[2])
expected = {(engine, rep) for engine in ('rust', 'java') for rep in range(1, repetitions + 1)}
if samples.keys() != expected:
    raise SystemExit('missing RSS samples')

print(f'# warmup_ms={sys.argv[3]} measure_ms={sys.argv[4]} repetitions={repetitions} java_heap=128MiB')
for case, digest in sorted(reference.items()):
    print(f'#check\t{case}\t{digest}\t256')
print('engine\trepetition\tpeak_process_rss_kib')
for (engine, rep), peak in sorted(samples.items()):
    print(f'{engine}\t{rep}\t{peak}')
print('engine\tformat_owned_bytes\tmedian_peak_rss_kib\tmin_peak_rss_kib\tmax_peak_rss_kib')
medians = {}
for engine in ('rust', 'java'):
    values = [samples[engine, rep] for rep in range(1, repetitions + 1)]
    medians[engine] = statistics.median(values)
    print(f'{engine}\t{owned[engine]}\t{medians[engine]:g}\t{min(values)}\t{max(values)}')
print(f'java/rust_peak_rss\t{medians["java"] / medians["rust"]:.3f}')
print('Format owned bytes: wrapper, concrete delegate and one shared RNG graph, counted once.')
print('Owned scope excludes consumer/producer buffers, directory/output state, allocator metadata, runtime and RSS.')
print('Peak RSS covers both real cases together, including those buffers, runtime/JIT, JVM heap/GC, startup and allocator retention.')
print('RSS cannot isolate the transient-buffer increment, attribute a peak to one case or prove the retained-format memory ratio.')
PY
