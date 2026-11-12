#!/usr/bin/env bash
# Fetch-phase memory with 100 MB documents (opensearch-plugin/e2e/fetch_memory.py)
# on a plugin node in a memory-limited container, and how far past a search
# `timeout` heavy requests run (timeout_bench.py, over the memory-bounds corpus
# when its volume exists).
#
#   scripts/bench-fetch-memory.sh IMAGE LABEL [OUT_DIR]
#
# MEM (default 4g) is the container's limit, HEAP (default 2g) the heap; EXTRA
# holds more `-e setting=value` arguments for the node. The large documents are
# loaded into a data volume once per SIZE_MB and VOLUME_TAG (default plugin).
# TIMEOUTS=1 also runs timeout_bench.py on the membounds-2000000 volume;
# SKIP_FETCH=1 runs only that.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="$1"
LABEL="$2"
OUT="${3:-target/bench-fetch-memory}"
MEM="${MEM:-4g}"
HEAP="${HEAP:-2g}"
PORT="${PORT:-9261}"
VOLUME="bigdocs-${SIZE_MB:-100}-${VOLUME_TAG:-plugin}"
mkdir -p "$OUT"

start() { # name volume
    docker rm -f "$1" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$1" -p "$PORT:9200" --memory="$MEM" --memory-swap="$MEM" \
        -v "$2:/usr/share/opensearch/data" \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false -e http.max_content_length=500mb \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
        ${ENGINE_ARGS--e indices.engine.accelerated.default=true} ${EXTRA:-} "$IMAGE" >/dev/null
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
}

name="fetchmem-$LABEL"
load=load
if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
    load=""
fi
trap 'docker logs "$name" > "$OUT/$LABEL.log" 2>&1 || true; docker inspect -f "{{.State.OOMKilled}} {{.State.ExitCode}}" "$name" > "$OUT/$LABEL.state" 2>&1 || true; docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
status=0
if [[ "${SKIP_FETCH:-0}" != 1 ]]; then
    start "$name" "$VOLUME"
    curl -sf "localhost:$PORT/_cluster/health/bigdocs?wait_for_status=yellow&timeout=120s" >/dev/null || true
    python3 opensearch-plugin/e2e/fetch_memory.py "http://localhost:$PORT" "$name" "$LABEL" "$OUT/$LABEL-fetch.json" $load || status=$?
fi
if [[ "${TIMEOUTS:-0}" == 1 ]] && docker volume inspect membounds-2000000-plugin >/dev/null 2>&1; then
    docker logs "$name" > "$OUT/$LABEL.log" 2>&1 || true
    start "$name" membounds-2000000-plugin
    curl -sf "localhost:$PORT/_cluster/health/membounds?wait_for_status=yellow&timeout=120s" >/dev/null || true
    python3 opensearch-plugin/e2e/timeout_bench.py "http://localhost:$PORT" "$LABEL" "$OUT/$LABEL-timeouts.json"
fi
echo "fetch-memory: $OUT/$LABEL-fetch.json"
exit "$status"
