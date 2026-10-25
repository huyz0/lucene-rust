#!/usr/bin/env bash
# Peak Linux process RSS for the paired integer FST enumeration cases.
# Run AFTER scripts/bench-micro.sh --bench ints_ref_fst_enum has built the runner:
#   scripts/docker-test.sh bash scripts/bench-ints-fst-memory.sh
# Optional: --reps N --warmup-ms N --measure-ms N --jars DIR
#
# This measures whole-process high-water RSS, including runtime/JIT, parsed trace,
# shared FST bytes, enum allocations and allocator/GC retention. It does not
# isolate owned enum state or prove the per-enum retained-memory ratio. Both
# engines run the identical fixture and four operations; all digests must match.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JARS="${JARS:-$PWD/fixtures/.jars}"
REPS=1
WARMUP=3000
MEASURE=3000
while [ "$#" -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --warmup-ms) WARMUP="$2"; shift 2 ;;
    --measure-ms) MEASURE="$2"; shift 2 ;;
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "ints-fst-memory: unknown argument: $1" >&2; exit 2 ;;
  esac
done
for value in "$REPS" "$WARMUP" "$MEASURE"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
    echo 'ints-fst-memory: repetitions and durations must be positive integers' >&2
    exit 2
  fi
done
RUST_TARGET_DIR="${CARGO_TARGET_DIR:-benchmarks/rust-runner/target}"
if [ ! -x "$RUST_TARGET_DIR/release/micro" ]; then
  echo 'ints-fst-memory: build the current runner through bench-micro.sh first' >&2
  exit 2
fi
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
javac -nowarn -cp "$CP" -d "$OUT/classes" \
  benchmarks/micro/java/IntsRefFstEnumMicro.java benchmarks/micro/java/SweepMicro.java

# A fresh Python parent per invocation keeps RUSAGE_CHILDREN from carrying
# one process's peak into the next. javac and shell setup are outside the sample.
peak_rss() {
  python3 - "$@" <<'PY'
import pathlib
import resource
import subprocess
import sys

engine, repetition, output = sys.argv[1:4]
if not sys.platform.startswith('linux'):
    raise SystemExit('ints-fst-memory: run in the Linux test container (RSS units are KiB)')
completed = subprocess.run(sys.argv[4:], check=True, text=True, stdout=subprocess.PIPE)
pathlib.Path(output).write_text(completed.stdout)
print(f'{engine}\t{repetition}\t{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}')
PY
}
run_engine() {
  local engine="$1" repetition="$2"
  if [ "$engine" = rust ]; then
    peak_rss rust "$repetition" "$OUT/rust.$repetition.tsv" \
      env -u MICRO_CASE MICRO_WARMUP_MS="$WARMUP" MICRO_MEASURE_MS="$MEASURE" \
      "$RUST_TARGET_DIR/release/micro" ints_ref_fst_enum
  else
    peak_rss java "$repetition" "$OUT/java.$repetition.tsv" \
      java --add-modules jdk.incubator.vector -DwarmupMs="$WARMUP" -DmeasureMs="$MEASURE" \
      -cp "$CP:$OUT/classes" IntsRefFstEnumMicro
  fi
}
: > "$OUT/rss.tsv"
for repetition in $(seq 1 "$REPS"); do
  if (( repetition % 2 == 1 )); then
    run_engine rust "$repetition" >> "$OUT/rss.tsv"
    run_engine java "$repetition" >> "$OUT/rss.tsv"
  else
    run_engine java "$repetition" >> "$OUT/rss.tsv"
    run_engine rust "$repetition" >> "$OUT/rss.tsv"
  fi
done
python3 - "$OUT" "$REPS" <<'PY'
import hashlib
import pathlib
import re
import statistics
import sys

output = pathlib.Path(sys.argv[1])
repetitions = int(sys.argv[2])
expected = {'ints_enum_next', 'ints_enum_exact', 'ints_enum_ceil', 'ints_enum_floor'}
reference = None
for repetition in range(1, repetitions + 1):
    for engine in ('rust', 'java'):
        checks = {}
        timings = set()
        for line in (output / f'{engine}.{repetition}.tsv').read_text().splitlines():
            if line.split(maxsplit=1)[:1] == ['#check']:
                _, case, digest = line.split()
                if case in checks or not re.fullmatch('[0-9a-f]{16}', digest):
                    raise SystemExit(f'invalid or duplicate digest: {line}')
                checks[case] = digest
            elif line and not line.startswith('#'):
                timings.add(line.split('\t')[0])
        if checks.keys() != expected or timings != expected:
            raise SystemExit(f'{engine} repetition {repetition}: incomplete benchmark results')
        if reference is None:
            reference = checks
        elif checks != reference:
            raise SystemExit(f'{engine} repetition {repetition}: result digests disagree')

rss = {'rust': [], 'java': []}
for line in (output / 'rss.tsv').read_text().splitlines():
    engine, repetition, peak = line.split('\t')
    rss[engine].append(int(peak))
if any(len(values) != repetitions for values in rss.values()):
    raise SystemExit('incomplete RSS samples')
fixture = pathlib.Path('fixtures/data/ints_ref_fst_enum/continuous.fst')
print(f'# fixture_sha256 {hashlib.sha256(fixture.read_bytes()).hexdigest()}')
print('# Peak Linux process RSS includes runtime/JIT, fixture/trace, enum allocation and allocator/GC retention.')
print('# This is not an isolated measurement of retained enum state; JVM heap settings affect its peak.')
for case, digest in sorted(reference.items()):
    print(f'#check {case} {digest}')
print('engine\tmedian_peak_process_rss_kib\tmin_kib\tmax_kib\tsamples')
for engine, values in rss.items():
    print(f'{engine}\t{statistics.median(values):g}\t{min(values)}\t{max(values)}\t{len(values)}')
print(f'java/rust\t{statistics.median(rss["java"]) / statistics.median(rss["rust"]):.3f}')
PY
