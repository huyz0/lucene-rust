#!/usr/bin/env bash
# Realtime search against refresh: bulk indexing throughput, CPU, and how long
# an acknowledged write takes to show up in a search, while a few request
# shapes run (opensearch-plugin/e2e/bench_nodes.py visibility). Configs, each
# on a fresh node alone on the machine with the same heap:
#   stock      stock OpenSearch, refresh_interval 1s
#   plugin     the plugin's engine, refresh_interval 1s
#   realtime   the plugin's engine with index.lucene_rust.realtime_search, refresh_interval 1s
#   realtime-30s  the same, refresh_interval 30s
#
#   scripts/bench-realtime.sh [DOCS] [OUT_DIR]     # default 1000000, target/bench-realtime
#
# Needs what scripts/bench-opensearch.sh needs. HEAP (default 2g), CLIENTS
# (default 4) bulk clients, PROBE_EVERY (default 0.2) seconds between probes.
# BENCH_ENV (one NAME=value) is passed to the node, and its log is kept
# beside the results.
# CONFIGS (default "stock plugin realtime realtime-30s") which configs run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-1000000}"
OUT="${2:-target/bench-realtime}"
VERSION=3.9.0
IMAGE="lucene-rust-opensearch:${VERSION}"
HEAP="${HEAP:-2g}"
PORT=9251
mkdir -p "$OUT"

if [[ ! -f "$OUT/corpus-$DOCS/done" ]]; then
    rm -rf "$OUT/corpus-$DOCS"
    python3 opensearch-plugin/e2e/bench_nodes.py gen "$OUT/corpus-$DOCS" "$DOCS"
    touch "$OUT/corpus-$DOCS/done"
fi

scripts/opensearch-dist.sh
gradle --no-daemon -q -p opensearch-plugin bundlePlugin
rm -f opensearch-plugin/docker/*.zip
cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/
docker build -q -t "$IMAGE" opensearch-plugin/docker >/dev/null
printf 'FROM opensearchproject/opensearch:%s\nRUN rm -rf /usr/share/opensearch/plugins/*\n' "$VERSION" \
    | docker build -q -t "opensearch-stock-min:${VERSION}" - >/dev/null

CONFIGS="${CONFIGS:-stock plugin realtime realtime-30s}"
bench() { # name image label settings [docker-run args...]
    [[ " $CONFIGS " == *" $3 "* ]] || return 0
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$1" -p "$PORT:9200" "${@:5}" ${BENCH_ENV:+-e "$BENCH_ENV"} \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
        "$2" >/dev/null
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
    python3 opensearch-plugin/e2e/bench_nodes.py visibility "http://localhost:$PORT" "$1" "$OUT/corpus-$DOCS" "$3" \
        "$OUT/$3-$DOCS.json" "$4" || { docker logs "$1" >"$OUT/$3-$DOCS.log" 2>&1; return 1; }
    docker logs "$1" >"$OUT/$3-$DOCS.log" 2>&1
    docker rm -f "$1" >/dev/null
}
trap 'docker rm -f bench-rt >/dev/null 2>&1 || true' EXIT
bench bench-rt "opensearch-stock-min:${VERSION}" stock '{"refresh_interval": "1s"}'
bench bench-rt "$IMAGE" plugin '{"refresh_interval": "1s"}' -e lucene_rust.engine.default=true
bench bench-rt "$IMAGE" realtime '{"refresh_interval": "1s", "index.lucene_rust.realtime_search": true}' \
    -e lucene_rust.engine.default=true
bench bench-rt "$IMAGE" realtime-30s '{"refresh_interval": "30s", "index.lucene_rust.realtime_search": true}' \
    -e lucene_rust.engine.default=true
echo "bench-realtime: results in $OUT/*-$DOCS.json"
