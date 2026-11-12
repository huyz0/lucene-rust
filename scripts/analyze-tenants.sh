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

trap 'docker rm -f analyze-tenants >/dev/null 2>&1 || true' EXIT
IFS='|' read -ra cfgs <<<"$CONFIGS"
for cfg in "${cfgs[@]}"; do
    node="${cfg%%:*}"; rest="${cfg#*:}"; label="${rest%%:*}"; settings="${rest#*:}"
    args=()
    if [[ "$node" == plugin ]]; then
        image=lucene-rust-opensearch:3.9.0
        args=(-e indices.engine.accelerated.default=true)
    else
        image=opensearch-stock-min:3.9.0
    fi
    docker rm -f analyze-tenants >/dev/null 2>&1 || true
    docker run -d --name analyze-tenants -p "$PORT:9200" "${args[@]}" \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
        "$image" >/dev/null
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
    python3 opensearch-plugin/e2e/analyze_tenants.py "http://localhost:$PORT" analyze-tenants "$label" "$OUT" \
        "$settings" --tenants "$TENANTS" --docs "$DOCS" "${perf_args[@]}"
    docker rm -f analyze-tenants >/dev/null
done
echo "analyze-tenants: results in $OUT"
