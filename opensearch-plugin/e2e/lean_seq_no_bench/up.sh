#!/bin/bash
# up.sh NAME PORT VARIANT [setting ...]: start a node (nodectl variant stock-min or plugin), wait until it answers.
n=$1; port=$2; variant=$3; shift 3
args=(); for s in "$@"; do args+=(--setting "$s"); done
exec python3 "$(dirname "$0")/../nodectl.py" run "$n" "$variant" "$port" --heap "${HEAP:-4g}" "${args[@]}" >/dev/null
