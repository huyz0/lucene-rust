#!/usr/bin/env bash
# Adversarial memory checks (opensearch-plugin/e2e/memory_bounds.py) on a
# plugin node in a memory-limited container: huge bucket counts, deep
# pagination, many concurrent heavy aggregations. Every request must answer or
# be refused with OpenSearch's own errors, and the node must stay alive
# and under its limit.
#
#   scripts/bench-memory-bounds.sh VARIANT LABEL [OUT_DIR]
#
# VARIANT is a nodectl variant: `plugin` (needs the plugin zip built, as scripts/bench-opensearch.sh does)
# or `stock-min`. MEM (default 3g) is the node's memory cgroup limit (root, cgroup v1 or v2; no swap),
# HEAP (default 1g) the heap; EXTRA holds more `-e setting=value` arguments for the node. The corpus is
# loaded into a data directory once per DOCS and VOLUME_TAG (default plugin) and reused by later
# runs. For a stock node: ENGINE_ARGS= VOLUME_TAG=stock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
source scripts/lib-nodes.sh

VARIANT="$1"
LABEL="$2"
OUT="${3:-target/bench-memory-bounds}"
MEM="${MEM:-3g}"
HEAP="${HEAP:-1g}"
PORT="${PORT:-9260}"
DOCS="${DOCS:-2000000}"
VOLUME="membounds-${DOCS}-${VOLUME_TAG:-plugin}"
mkdir -p "$OUT"
name="membounds-$LABEL"
load=load
if os_has_data "$VOLUME"; then
    load=""
fi
export MEM_LIMIT="$MEM" HEAP DATA_DIR="$(os_data "$VOLUME")"
# shellcheck disable=SC2086
os_up "$name" "$VARIANT" "$PORT" ${ENGINE_ARGS--e indices.engine.accelerated.default=true} ${EXTRA:-}
trap 'nodectl logs "$name" > "$OUT/$LABEL.log" 2>&1 || true; echo "$(nodectl oom "$name") $(nodectl status "$name" && echo up || echo down)" > "$OUT/$LABEL.state" 2>&1 || true; os_down "$name"' EXIT
curl -sf "localhost:$PORT/_cluster/health/membounds?wait_for_status=yellow&timeout=120s" >/dev/null || true
python3 opensearch-plugin/e2e/memory_bounds.py "http://localhost:$PORT" "$name" "$LABEL" "$OUT/$LABEL.json" $load
echo "memory-bounds: $OUT/$LABEL.json"
