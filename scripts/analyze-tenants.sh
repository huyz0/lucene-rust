#!/usr/bin/env bash
# Takes a many-tenant index apart (opensearch-plugin/e2e/analyze_tenants.py):
# each configuration on a fresh node, with JFR and, when PERF names a perf
# binary, native profiles.
#
#   scripts/analyze-tenants.sh [TENANTS] [DOCS] [OUT_DIR]
#
# CONFIGS picks configurations: node:label:settings-json, separated by '|'.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
source scripts/lib-nodes.sh

TENANTS="${1:-500}"
DOCS="${2:-500000}"
OUT="${3:-target/analyze-tenants}"
HEAP="${HEAP:-2g}"
PORT=9250
DEFAULT_CONFIGS='plugin:partitioned:{"segment_partition.field":"tenant"}|plugin:shared:{}|plugin:sorted:{"sort.field":"tenant"}|stock:stock-shared:{}|stock:stock-sorted:{"sort.field":"tenant"}'
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"
mkdir -p "$OUT"
perf_args=()
[[ -n "${PERF:-}" ]] && perf_args=(--perf "$PERF")

trap 'os_down analyze-tenants' EXIT
IFS='|' read -ra cfgs <<<"$CONFIGS"
for cfg in "${cfgs[@]}"; do
    node="${cfg%%:*}"; rest="${cfg#*:}"; label="${rest%%:*}"; settings="${rest#*:}"
    args=()
    if [[ "$node" == plugin ]]; then
        variant=plugin
        args=(-e indices.engine.accelerated.default=true)
    else
        variant=stock-min
    fi
    os_up analyze-tenants "$variant" "$PORT" ${args[@]+"${args[@]}"}
    python3 opensearch-plugin/e2e/analyze_tenants.py "http://localhost:$PORT" analyze-tenants "$label" "$OUT" \
        "$settings" --tenants "$TENANTS" --docs "$DOCS" "${perf_args[@]}"
    os_down analyze-tenants
done
echo "analyze-tenants: results in $OUT"
