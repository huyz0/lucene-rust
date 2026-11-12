#!/usr/bin/env bash
# Adversarial memory checks (opensearch-plugin/e2e/memory_bounds.py) on a
# plugin node in a memory-limited container: huge bucket counts, deep
# pagination, many concurrent heavy aggregations. Every request must answer or
# be refused with OpenSearch's own errors, and the container must stay alive
# and under its limit.
#
#   scripts/bench-memory-bounds.sh IMAGE LABEL [OUT_DIR]
#
# IMAGE is a built plugin image (scripts/bench-opensearch.sh builds one).
# MEM (default 3g) is the container's limit, HEAP (default 1g) the heap;
# EXTRA holds more `-e setting=value` arguments for the node. The corpus is
# loaded into a data volume once per DOCS and VOLUME_TAG (default plugin) and
# reused by later runs. For a stock node: ENGINE_ARGS= VOLUME_TAG=stock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="$1"
LABEL="$2"
OUT="${3:-target/bench-memory-bounds}"
MEM="${MEM:-3g}"
HEAP="${HEAP:-1g}"
PORT="${PORT:-9260}"
DOCS="${DOCS:-2000000}"
VOLUME="membounds-${DOCS}-${VOLUME_TAG:-plugin}"
mkdir -p "$OUT"
name="membounds-$LABEL"
docker rm -f "$name" >/dev/null 2>&1 || true
load=load
if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
    load=""
fi
# shellcheck disable=SC2086
docker run -d --name "$name" -p "$PORT:9200" --memory="$MEM" --memory-swap="$MEM" \
    -v "$VOLUME:/usr/share/opensearch/data" \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
    -e cluster.routing.allocation.disk.threshold_enabled=false \
    -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
    ${ENGINE_ARGS--e indices.engine.accelerated.default=true} ${EXTRA:-} "$IMAGE" >/dev/null
trap 'docker logs "$name" > "$OUT/$LABEL.log" 2>&1 || true; docker inspect -f "{{.State.OOMKilled}} {{.State.ExitCode}}" "$name" > "$OUT/$LABEL.state" 2>&1 || true; docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
curl -sf "localhost:$PORT/_cluster/health/membounds?wait_for_status=yellow&timeout=120s" >/dev/null || true
python3 opensearch-plugin/e2e/memory_bounds.py "http://localhost:$PORT" "$name" "$LABEL" "$OUT/$LABEL.json" $load
echo "memory-bounds: $OUT/$LABEL.json"
