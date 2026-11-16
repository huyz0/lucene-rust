#!/usr/bin/env bash
# The plugin node (Rust engine and native search) against a stock OpenSearch
# node on one synthetic corpus: bulk indexing time, store size, memory, and
# REST latency per request shape (opensearch-plugin/e2e/bench_nodes.py).
# The nodes run one after the other, each alone on the machine, with the same
# heap.
#
#   scripts/bench-opensearch.sh [DOCS] [OUT_DIR]     # default 1000000, target/bench-opensearch
#
# Needs what scripts/verify-opensearch.sh needs (a JDK 25, Gradle, cargo); the nodes are plain
# processes (opensearch-plugin/e2e/nodectl.py), no Docker. NODE_CPUS (e.g. 0-2) pins the node
# to cores and MEM_LIMIT (e.g. 4G, root, cgroup v1) caps its memory; run the driver elsewhere. HEAP (default 2g) sets both nodes' heap; ROUNDS (default 50) the
# timed requests per shape; CLIENTS (default 4) the parallel bulk clients.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-1000000}"
OUT="${2:-target/bench-opensearch}"
nodectl() { python3 opensearch-plugin/e2e/nodectl.py "$@"; }
HEAP="${HEAP:-2g}"
ROUNDS="${ROUNDS:-50}"
PORT=9250
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

bench() { # name variant label [setting ...]
    local n=$1 v=$2 l=$3; shift 3
    local a=() st
    for st in "$@"; do a+=(--setting "$st"); done
    nodectl rm "$n" >/dev/null 2>&1 || true
    nodectl run "$n" "$v" "$PORT" --heap "$HEAP" ${NODE_CPUS:+--cpus "$NODE_CPUS"} ${MEM_LIMIT:+--mem-limit "$MEM_LIMIT"} "${a[@]}" >/dev/null
    local extra=()
    [[ "$l" == plugin ]] && extra=(--java-too)
    python3 opensearch-plugin/e2e/bench_nodes.py run "http://localhost:$PORT" "$n" "$OUT/corpus-$DOCS" "$l" \
        "$OUT/$l-$DOCS.json" --rounds "$ROUNDS" "${extra[@]}"
    nodectl rm "$n" >/dev/null
}
trap 'nodectl rm bench-stock >/dev/null 2>&1; nodectl rm bench-plugin >/dev/null 2>&1 || true' EXIT
bench bench-stock stock-min stock
bench bench-plugin plugin plugin indices.engine.accelerated.default=true
echo "bench-opensearch: results in $OUT/stock-$DOCS.json and $OUT/plugin-$DOCS.json"
