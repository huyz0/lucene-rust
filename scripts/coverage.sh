#!/usr/bin/env bash
# Tests under coverage, then the report -- without lucene-ffi's cdylib.
#
#   scripts/coverage.sh <selection/test args...> [-- <report args...>]
#   scripts/coverage.sh --workspace -- --summary-only --fail-under-lines 95
#   scripts/coverage.sh -p lucene-index --no-fail-fast -- --summary-only
#
# Why not plain `cargo llvm-cov`: lucene-ffi is built as an `rlib` *and* a
# `cdylib`, and cargo-llvm-cov hands every executable file under its target
# directory to `llvm-cov` as an object -- the `.so` included. Nothing ever
# loads that `.so` in a test run, but it carries a coverage record for every
# `#[no_mangle]` entry point under the *unmangled* name the executed copies
# also have. `llvm-cov` keeps the first record it loads for a name and drops
# the rest, and when the `.so`'s came first the report showed the never-run
# copy: lucene-ffi's results*.rs, directory.rs, explain.rs and ffm_bridge.rs
# read 90-95% while their executed code was 96-100% covered (crate 96.0%
# reported vs 98.2% real). Deleting the `.so` between the run and the report
# removes the phantom copy and nothing else -- no test binary links it.
#
# What this cannot fix: the same first-record-wins rule between two *executed*
# copies of a `#[no_mangle]` function -- lucene-ffi's unit-test binary and its
# `tests/resource_bounds.rs` binary (which links the rlib) both carry one, under
# different function hashes, and the report shows whichever loads first. See
# docs/mechanical-gates.md#coverage-objects.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_args=()
report_args=()
seen_sep=0
for a in "$@"; do
  if [ "$seen_sep" = 0 ] && [ "$a" = "--" ]; then
    seen_sep=1
    continue
  fi
  if [ "$seen_sep" = 0 ]; then run_args+=("$a"); else report_args+=("$a"); fi
done

# The test run's package selection is also the report's; the test-only flags
# (`--no-fail-fast`, ...) are not report flags, so only selection is carried.
select_args=()
i=0
while [ $i -lt ${#run_args[@]} ]; do
  a="${run_args[$i]}"
  case "$a" in
    --workspace|--all) select_args+=("$a") ;;
    -p|--package|--exclude|--exclude-from-report)
      select_args+=("$a" "${run_args[$((i + 1))]}")
      i=$((i + 1)) ;;
    -p*|--package=*|--exclude=*|--exclude-from-report=*) select_args+=("$a") ;;
  esac
  i=$((i + 1))
done

cargo llvm-cov --no-report "${run_args[@]}"

target_dir="$(cargo metadata --format-version 1 --no-deps \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
find "$target_dir/llvm-cov-target" \( -name 'liblucene_ffi*.so' -o -name 'liblucene_ffi*.dylib' \) -delete

cargo llvm-cov report "${select_args[@]}" "${report_args[@]}"
