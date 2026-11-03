#!/usr/bin/env bash
# The plugin node (Rust engine and native search) against a stock OpenSearch
# node on one synthetic corpus: bulk indexing time, store size, memory, and
# REST latency per request shape (opensearch-plugin/e2e/bench_nodes.py).
# The nodes run one after the other, each alone on the machine, with the same
# heap.
#
#   scripts/bench-opensearch.sh [DOCS] [OUT_DIR]     # default 1000000, target/bench-opensearch
#
# Needs what scripts/verify-opensearch.sh needs (Docker, a JDK 25, Gradle,
# cargo). HEAP (default 2g) sets both nodes' heap; ROUNDS (default 50) the
# timed requests per shape; CLIENTS (default 4) the parallel bulk clients.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-1000000}"
OUT="${2:-target/bench-opensearch}"
VERSION=3.9.0
IMAGE="lucene-rust-opensearch:${VERSION}"
HEAP="${HEAP:-2g}"
ROUNDS="${ROUNDS:-50}"
PORT=9250
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

bench() { # name image label [docker-run args...]
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$1" -p "$PORT:9200" "${@:4}" \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" \
        "$2" >/dev/null
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
    local extra=()
    [[ "$3" == plugin ]] && extra=(--java-too)
    python3 opensearch-plugin/e2e/bench_nodes.py run "http://localhost:$PORT" "$1" "$OUT/corpus-$DOCS" "$3" \
        "$OUT/$3-$DOCS.json" --rounds "$ROUNDS" "${extra[@]}"
    docker rm -f "$1" >/dev/null
}
trap 'docker rm -f bench-stock bench-plugin >/dev/null 2>&1 || true' EXIT
bench bench-stock "opensearch-stock-min:${VERSION}" stock
bench bench-plugin "$IMAGE" plugin -e lucene_rust.engine.default=true
echo "bench-opensearch: results in $OUT/stock-$DOCS.json and $OUT/plugin-$DOCS.json"
