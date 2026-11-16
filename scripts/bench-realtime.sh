#!/usr/bin/env bash
# Realtime search against refresh: bulk indexing throughput, CPU, and how long
# an acknowledged write takes to show up in a search, while a few request
# shapes run (opensearch-plugin/e2e/bench_nodes.py visibility). Configs, each
# on a fresh node alone on the machine with the same heap:
#   stock      stock OpenSearch, refresh_interval 1s
#   plugin     the plugin's engine, refresh_interval 1s
#   realtime   the plugin's engine with index.search.realtime.enabled, refresh_interval 1s (default staleness, 20 ms)
#   realtime-strict  realtime with index.search.realtime.max_staleness 0 (read after write)
#   realtime-stale  realtime with index.search.realtime.max_staleness $STALENESS (default 20ms)
#   realtime-30s  the same, refresh_interval 30s
#
#   scripts/bench-realtime.sh [DOCS] [OUT_DIR]     # default 1000000, target/bench-realtime
#
# Needs what scripts/bench-opensearch.sh needs (nodes are plain processes, no Docker). HEAP (default 2g), CLIENTS
# (default 4) bulk clients, PROBE_EVERY (default 0.2) seconds between probes.
# BENCH_ENV (NAME=value pairs, space-separated) is passed to the node, and its log is kept
# beside the results.
# CONFIGS (default "stock plugin realtime realtime-30s") which configs run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-1000000}"
OUT="${2:-target/bench-realtime}"
REPO_ROOT="$PWD"
source scripts/lib-nodes.sh
HEAP="${HEAP:-2g}"
PORT=9251
mkdir -p "$OUT"

if [[ ! -f "$OUT/corpus-$DOCS/done-geo" ]]; then
    rm -rf "$OUT/corpus-$DOCS"
    python3 opensearch-plugin/e2e/bench_nodes.py gen "$OUT/corpus-$DOCS" "$DOCS"
    touch "$OUT/corpus-$DOCS/done-geo"
fi

scripts/opensearch-dist.sh
gradle --no-daemon -q -p opensearch-plugin bundlePlugin
nodectl dist plugin >/dev/null
nodectl dist stock-min >/dev/null

BENCH_ENV_ARGS=()
for e in ${BENCH_ENV:-}; do BENCH_ENV_ARGS+=(-e "$e"); done
CONFIGS="${CONFIGS:-stock plugin realtime realtime-30s}"
bench() { # name variant label settings [-e setting=value ...]
    [[ " $CONFIGS " == *" $3 "* ]] || return 0
    os_up "$1" "$2" "$PORT" "${@:5}" ${BENCH_ENV_ARGS[@]+"${BENCH_ENV_ARGS[@]}"}
    python3 opensearch-plugin/e2e/bench_nodes.py visibility "http://localhost:$PORT" "$1" "$OUT/corpus-$DOCS" "$3" \
        "$OUT/$3-$DOCS.json" "$4" || { nodectl logs "$1" >"$OUT/$3-$DOCS.log" 2>&1; return 1; }
    nodectl logs "$1" >"$OUT/$3-$DOCS.log" 2>&1
    os_down "$1"
}
trap 'os_down bench-rt' EXIT
bench bench-rt stock-min stock '{"refresh_interval": "1s"}'
bench bench-rt plugin plugin '{"refresh_interval": "1s"}' -e indices.engine.accelerated.default=true
bench bench-rt plugin realtime '{"refresh_interval": "1s", "index.search.realtime.enabled": true}' \
    -e indices.engine.accelerated.default=true
bench bench-rt plugin realtime-strict '{"refresh_interval": "1s", "index.search.realtime.enabled": true, "index.search.realtime.max_staleness": "0ms"}' \
    -e indices.engine.accelerated.default=true
bench bench-rt plugin realtime-stale '{"refresh_interval": "1s", "index.search.realtime.enabled": true, "index.search.realtime.max_staleness": "'"${STALENESS:-20ms}"'"}' \
    -e indices.engine.accelerated.default=true
bench bench-rt plugin realtime-30s '{"refresh_interval": "30s", "index.search.realtime.enabled": true}' \
    -e indices.engine.accelerated.default=true
echo "bench-realtime: results in $OUT/*-$DOCS.json"
