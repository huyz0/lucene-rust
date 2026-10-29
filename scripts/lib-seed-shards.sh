#!/usr/bin/env bash
# Seed-range sharding for the seeded acceptance runs (crash-fuzz, the
# test-framework fault, index, format and stress runs), sourced by them.
#
#   run_seed_shards A B -- BIN [ARGS...]
#
# runs `BIN ARGS... --seeds a..b` over up to SEED_JOBS disjoint slices of A..B
# at once and prints one summary line: the slices' last lines summed. No slice
# is shorter than SEED_MIN_SLICE (20) seeds: io_faults and random_index only
# demand their every-outcome evidence of a run of 20 rounds or more, and a
# shorter slice would skip that check. Every seed
# is still run exactly once, by the same binary and arguments as a serial
# `--seeds A..B`; each seed builds its own state from its seed, and every
# scratch directory these binaries make carries the process id, so the slices
# share nothing. On a failing slice it prints that slice's last lines (the
# binaries print the seed and a replay command there) and returns 1.
#
# Why: the runs are serial loops over independent seeds whose threads spend
# much of their time in Lucene's own `Thread.sleep`s, so one process leaves the
# machine mostly idle: 20 stress seeds took 72s serially and 26s as 4 slices
# on a 4-core machine, with the same adds, deletes and searches. SEED_JOBS defaults to the CPU count; SEED_JOBS=1 is
# the serial run.
#
# The slices' summary lines have one shape (`name: 12 round(s) passed in 5.6s
# (34 injected errors, ...)`), so the merge adds up the numbers in matching
# positions, except the elapsed `in Xs`, where it takes the largest: the
# slowest slice's wall time is the step's.

SEED_JOBS="${SEED_JOBS:-$(nproc 2>/dev/null || echo 2)}"

run_seed_shards() {
  local first="$1" end="$2"
  shift 2
  [ "$1" = "--" ] && shift
  local jobs="$SEED_JOBS"
  local total=$(( end - first ))
  local min_slice="${SEED_MIN_SLICE:-20}"
  if [ "$jobs" -gt $(( total / min_slice )) ]; then jobs=$(( total / min_slice )); fi
  if [ "$jobs" -lt 1 ]; then jobs=1; fi
  local work
  work=$(mktemp -d)
  local -a pids=() logs=()
  local step=$(( (total + jobs - 1) / jobs ))
  local a b
  for (( a = first; a < end; a += step )); do
    b=$(( a + step < end ? a + step : end ))
    "$@" --seeds "$a..$b" >"$work/$a.log" 2>&1 &
    pids+=($!); logs+=("$work/$a.log")
  done
  local rc=0 i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      tail -3 "${logs[$i]}"
      rc=1
    fi
  done
  if [ "$rc" -eq 0 ]; then
    local -a lasts=()
    for i in "${!logs[@]}"; do lasts+=("$(tail -1 "${logs[$i]}")"); done
    printf '%s\n' "${lasts[@]}" | python3 -c '
import re, sys
lines = [l.rstrip("\n") for l in sys.stdin if l.strip()]
num = re.compile(r"\d+(?:\.\d+)?")
shape = {num.sub("#", l) for l in lines}
if len(shape) != 1:
    sys.exit("run_seed_shards: the slices printed different summaries:\n" + "\n".join(lines))
cols = [num.findall(l) for l in lines]
# The elapsed time is the number written ` in <n>s`.
spans = [m.span() for m in num.finditer(lines[0])]
timed = {k for k, (a, b) in enumerate(spans)
         if lines[0][max(0, a - 4):a] == " in " and lines[0][b:b + 1] == "s"}
out = []
for k in range(len(cols[0])):
    vals = [c[k] for c in cols]
    if k in timed:
        out.append("%.1f" % max(float(v) for v in vals))
    elif all("." not in v for v in vals):
        out.append(str(sum(int(v) for v in vals)))
    else:
        out.append("%.1f" % sum(float(v) for v in vals))
it = iter(out)
print(num.sub(lambda _m: next(it), lines[0]))
'
    rc=$?
  fi
  rm -rf "$work"
  return "$rc"
}
