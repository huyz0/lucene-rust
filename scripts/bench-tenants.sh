#!/usr/bin/env bash
# Many tenants on one node (opensearch-plugin/e2e/bench_tenants.py): an index
# per tenant against one shared index, on stock OpenSearch and on the plugin,
# the plugin also with segment partitions (index.segment_partition.field).
# Each configuration runs on a fresh node, alone on the machine, same heap.
#
#   scripts/bench-tenants.sh [TENANTS] [DOCS] [OUT_DIR]   # default 500, 500000, target/bench-tenants
#
# Expects the plugin zip built (scripts/bench-opensearch.sh builds one); nodes are plain
# processes (nodectl.py), no Docker.
# HEAP (default 2g); CONFIGS (default all) picks configurations by name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
source scripts/lib-nodes.sh

TENANTS="${1:-500}"
DOCS="${2:-500000}"
OUT="${3:-target/bench-tenants}"
HEAP="${HEAP:-2g}"
PORT=9250
CONFIGS="${CONFIGS:-stock-per_tenant stock-shared plugin-per_tenant plugin-shared plugin-partitioned}"
mkdir -p "$OUT"

trap 'os_down bench-tenants' EXIT
for cfg in $CONFIGS; do
    node="${cfg%%-*}"
    mode="${cfg#*-}"
    args=()
    if [[ "$node" == plugin ]]; then
        variant=plugin
        args=(-e indices.engine.accelerated.default=true)
    else
        variant=stock-min
    fi
    os_up bench-tenants "$variant" "$PORT" ${args[@]+"${args[@]}"}
    python3 opensearch-plugin/e2e/bench_tenants.py "http://localhost:$PORT" bench-tenants "$mode" "$cfg" \
        "$OUT/$cfg-$TENANTS.json" --tenants "$TENANTS" --docs "$DOCS"
    os_down bench-tenants
done
echo "bench-tenants: results in $OUT"
