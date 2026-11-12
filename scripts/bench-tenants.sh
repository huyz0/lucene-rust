#!/usr/bin/env bash
# Many tenants on one node (opensearch-plugin/e2e/bench_tenants.py): an index
# per tenant against one shared index, on stock OpenSearch and on the plugin,
# the plugin also with segment partitions (index.segment_partition.field).
# Each configuration runs on a fresh node, alone on the machine, same heap.
#
#   scripts/bench-tenants.sh [TENANTS] [DOCS] [OUT_DIR]   # default 500, 500000, target/bench-tenants
#
# Expects the images scripts/bench-opensearch.sh builds:
# lucene-rust-opensearch:3.9.0 and opensearch-stock-min:3.9.0.
# HEAP (default 2g); CONFIGS (default all) picks configurations by name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TENANTS="${1:-500}"
DOCS="${2:-500000}"
OUT="${3:-target/bench-tenants}"
HEAP="${HEAP:-2g}"
PORT=9250
CONFIGS="${CONFIGS:-stock-per_tenant stock-shared plugin-per_tenant plugin-shared plugin-partitioned}"
mkdir -p "$OUT"

trap 'docker rm -f bench-tenants >/dev/null 2>&1 || true' EXIT
for cfg in $CONFIGS; do
    node="${cfg%%-*}"
    mode="${cfg#*-}"
    args=()
    if [[ "$node" == plugin ]]; then
        image=lucene-rust-opensearch:3.9.0
        args=(-e indices.engine.accelerated.default=true)
    else
        image=opensearch-stock-min:3.9.0
    fi
    docker rm -f bench-tenants >/dev/null 2>&1 || true
    docker run -d --name bench-tenants -p "$PORT:9200" "${args[@]}" \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
        "$image" >/dev/null
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
    python3 opensearch-plugin/e2e/bench_tenants.py "http://localhost:$PORT" bench-tenants "$mode" "$cfg" \
        "$OUT/$cfg-$TENANTS.json" --tenants "$TENANTS" --docs "$DOCS"
    docker rm -f bench-tenants >/dev/null
done
echo "bench-tenants: results in $OUT"
