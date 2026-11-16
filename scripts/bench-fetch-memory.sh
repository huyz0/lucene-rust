#!/usr/bin/env bash
# Fetch-phase memory with 100 MB documents (opensearch-plugin/e2e/fetch_memory.py)
# on a plugin node in a memory-limited cgroup, and how far past a search
# `timeout` heavy requests run (timeout_bench.py, over the memory-bounds corpus
# when its directory exists).
#
#   scripts/bench-fetch-memory.sh VARIANT LABEL [OUT_DIR]
#
# VARIANT is a nodectl variant (`plugin`, `stock-min`); nodes are plain processes, no Docker.
# MEM (default 4g) is the node's memory cgroup limit (root; no swap), HEAP (default 2g) the heap; EXTRA
# holds more `-e setting=value` arguments for the node. The large documents are
# loaded into a data directory once per SIZE_MB and VOLUME_TAG (default plugin).
# TIMEOUTS=1 also runs timeout_bench.py on the membounds-2000000 volume;
# SKIP_FETCH=1 runs only that.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
source scripts/lib-nodes.sh

VARIANT="$1"
LABEL="$2"
OUT="${3:-target/bench-fetch-memory}"
MEM="${MEM:-4g}"
HEAP="${HEAP:-2g}"
PORT="${PORT:-9261}"
VOLUME="bigdocs-${SIZE_MB:-100}-${VOLUME_TAG:-plugin}"
mkdir -p "$OUT"

start() { # name volume
    export MEM_LIMIT="$MEM" HEAP DATA_DIR="$(os_data "$2")"
    # shellcheck disable=SC2086
    os_up "$1" "$VARIANT" "$PORT" -e http.max_content_length=500mb \
        ${ENGINE_ARGS--e indices.engine.accelerated.default=true} ${EXTRA:-}
}

name="fetchmem-$LABEL"
load=load
if os_has_data "$VOLUME"; then
    load=""
fi
trap 'nodectl logs "$name" > "$OUT/$LABEL.log" 2>&1 || true; echo "$(nodectl oom "$name") $(nodectl status "$name" && echo up || echo down)" > "$OUT/$LABEL.state" 2>&1 || true; os_down "$name"' EXIT
status=0
if [[ "${SKIP_FETCH:-0}" != 1 ]]; then
    start "$name" "$VOLUME"
    curl -sf "localhost:$PORT/_cluster/health/bigdocs?wait_for_status=yellow&timeout=120s" >/dev/null || true
    python3 opensearch-plugin/e2e/fetch_memory.py "http://localhost:$PORT" "$name" "$LABEL" "$OUT/$LABEL-fetch.json" $load || status=$?
fi
if [[ "${TIMEOUTS:-0}" == 1 ]] && os_has_data membounds-2000000-plugin; then
    nodectl logs "$name" > "$OUT/$LABEL.log" 2>&1 || true
    start "$name" membounds-2000000-plugin
    curl -sf "localhost:$PORT/_cluster/health/membounds?wait_for_status=yellow&timeout=120s" >/dev/null || true
    python3 opensearch-plugin/e2e/timeout_bench.py "http://localhost:$PORT" "$LABEL" "$OUT/$LABEL-timeouts.json"
fi
echo "fetch-memory: $OUT/$LABEL-fetch.json"
exit "$status"
