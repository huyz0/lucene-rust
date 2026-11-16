#!/usr/bin/env bash
# node.sh NAME VARIANT DATADIR [LIB.so] [-e setting=value ...]
# A node on port 9250 with a 4 GB heap (nodectl.py variant `plugin` or `stock-min`), optionally with
# another native library than the zip's; waits until it answers.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
name=$1 variant=$2 data=$3; shift 3
lib=()
if [[ "${1:-}" == *.so ]]; then lib=(--lib "$1"); shift; fi
args=()
while [[ $# -gt 0 ]]; do
  [[ "$1" == -e ]] && { args+=(--setting "$2"); shift 2; } || shift
done
python3 "$HERE/../../opensearch-plugin/e2e/nodectl.py" rm "$name" >/dev/null 2>&1 || true
python3 "$HERE/../../opensearch-plugin/e2e/nodectl.py" run "$name" "$variant" 9250 --heap 4g --data "$data" \
  ${lib[@]+"${lib[@]}"} ${args[@]+"${args[@]}"} >/dev/null && echo UP
