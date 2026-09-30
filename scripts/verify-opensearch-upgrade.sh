#!/usr/bin/env bash
# M8 acceptance criterion 3: a cluster upgraded from OpenSearch 2.x serves its
# old index natively.
#
#   1. An OpenSearch 2.19 node (Lucene 9.12) indexes a representative corpus --
#      text, keyword, numerics, knn_vector, nested documents, a k-NN index --
#      snapshots it to an fs repository and flushes
#      (opensearch-plugin/e2e/verify_upgrade.py load).
#   2. The node stops and its data directory is copied. OpenSearch 3.8.0 with
#      this plugin opens the original in place (the in-place node upgrade); a
#      stock 3.8.0 node, the same image without the plugin, opens the copy.
#   3. verify_upgrade.py verify: every index opens with its Lucene 9.12
#      segments, the plugin node answers every request of the matrix
#      (verify_opensearch.py's, plus aggregations and vector requests) as the
#      stock node does, natively where the plugin says it runs; the same over
#      the 2.x snapshot restored into both nodes; and after a force merge has
#      rewritten every segment as Lucene 10.5.0.
#
# Both 3.8.0 nodes keep the bundled k-NN plugin (the knn_vector mapping needs
# it) and drop the rest, as the 2.19 node does; see opensearch-plugin/docker/.
#
#   scripts/verify-opensearch-upgrade.sh [--docs N] [--keep] [--no-build]
#
# Needs Docker, a JDK 25 and Gradle (the plugin build, as verify-opensearch.sh),
# and cargo. Listens on localhost:${OS_PORT:-9240} (plugin), +2 (stock), +4 (2.x).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OLD_VERSION="${OLD_OPENSEARCH_VERSION:-2.19.6}"
VERSION=3.8.0
PORT="${OS_PORT:-9240}"
PREFIX="${OS_CONTAINER:-lucene-rust-upgrade}"
KEEP=0
BUILD=1
DOCS=10000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --no-build) BUILD=0; shift ;;
        --docs) DOCS="$2"; shift 2 ;;
        *) echo "verify-opensearch-upgrade: unknown argument $1" >&2; exit 2 ;;
    esac
done

OLD_IMAGE="lucene-rust-upgrade-src:${OLD_VERSION}"
PLUGIN_IMAGE="lucene-rust-opensearch-knn:${VERSION}"
STOCK_IMAGE="lucene-rust-upgrade-stock:${VERSION}"
VOL_OLD="${PREFIX}-data"
VOL_STOCK="${PREFIX}-data-stock"
VOL_SNAP="${PREFIX}-snap"
CONTAINERS=("${PREFIX}-old" "${PREFIX}-plugin" "${PREFIX}-stock")

cleanup() {
    docker rm -f "${CONTAINERS[@]}" >/dev/null 2>&1 || true
    docker volume rm "$VOL_OLD" "$VOL_STOCK" "$VOL_SNAP" >/dev/null 2>&1 || true
}
if [[ $KEEP == 0 ]]; then
    trap cleanup EXIT
fi
cleanup

if [[ $BUILD == 1 ]]; then
    scripts/opensearch-dist.sh
    gradle --no-daemon -q -p opensearch-plugin bundlePlugin
    rm -f opensearch-plugin/docker/*.zip
    cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/
fi
docker image inspect "opensearchproject/opensearch:${OLD_VERSION}" >/dev/null 2>&1 \
    || docker pull -q "opensearchproject/opensearch:${OLD_VERSION}" >/dev/null
docker image inspect alpine:3 >/dev/null 2>&1 || docker pull -q alpine:3 >/dev/null
# Every node keeps opensearch-knn and nothing else bundled, so the 2.x
# cluster state holds nothing a 3.8.0 node lacks a plugin for.
keep_knn() { # base-image tag
    printf 'FROM %s\nRUN cd /usr/share/opensearch/plugins && for p in *; do [ "$p" = opensearch-knn ] || rm -rf "$p"; done\n' "$1" \
        | docker build -q -t "$2" - >/dev/null
}
keep_knn "opensearchproject/opensearch:${OLD_VERSION}" "$OLD_IMAGE"
keep_knn "opensearchproject/opensearch:${VERSION}" "$STOCK_IMAGE"
docker build -q --build-arg KEEP_PLUGINS=opensearch-knn -t "$PLUGIN_IMAGE" opensearch-plugin/docker >/dev/null

for v in "$VOL_OLD" "$VOL_STOCK" "$VOL_SNAP"; do
    docker volume create "$v" >/dev/null
done
# The images run as uid 1000; a fresh volume mounted where the image has no
# directory is root's.
docker run --rm -v "$VOL_SNAP:/snap" alpine:3 chown 1000:1000 /snap

start_node() { # name port image data-volume
    docker run -d --name "$1" -p "$2:9200" -v "$4:/usr/share/opensearch/data" \
        -v "$VOL_SNAP:/usr/share/opensearch/snap" -e path.repo=/usr/share/opensearch/snap \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g" \
        "$3" >/dev/null
}
wait_http() { # port
    for _ in $(seq 1 120); do curl -sf "localhost:$1" >/dev/null && return 0; sleep 2; done
    echo "verify-opensearch-upgrade: nothing answers on port $1" >&2
    return 1
}

echo "== OpenSearch ${OLD_VERSION}: index and snapshot"
start_node "${PREFIX}-old" "$((PORT + 4))" "$OLD_IMAGE" "$VOL_OLD"
wait_http "$((PORT + 4))"
python3 opensearch-plugin/e2e/verify_upgrade.py load "http://localhost:$((PORT + 4))" "$DOCS"
docker stop -t 60 "${PREFIX}-old" >/dev/null
docker rm "${PREFIX}-old" >/dev/null

# The stock node's copy: the same bytes, owners and modes.
docker run --rm -v "$VOL_OLD:/from:ro" -v "$VOL_STOCK:/to" alpine:3 cp -a /from/. /to/

echo "== OpenSearch ${VERSION}: in-place upgrade (plugin) and a stock node on a copy"
start_node "${PREFIX}-plugin" "$PORT" "$PLUGIN_IMAGE" "$VOL_OLD"
start_node "${PREFIX}-stock" "$((PORT + 2))" "$STOCK_IMAGE" "$VOL_STOCK"
wait_http "$PORT"
wait_http "$((PORT + 2))"

status=0
python3 opensearch-plugin/e2e/verify_upgrade.py verify "http://localhost:$PORT" "http://localhost:$((PORT + 2))" || status=$?
if docker logs "${PREFIX}-plugin" 2>&1 | grep -E "A fatal error has been detected|SIGSEGV \(0xb\)" >/dev/null; then
    echo "verify-opensearch-upgrade: the plugin node's JVM crashed" >&2
    status=1
fi
# The only native refusals expected are the k-NN codec's (see verify_upgrade.py).
if docker logs "${PREFIX}-plugin" 2>&1 | grep "lucene-rust: native reader open failed" | grep -v "KNN[0-9]*Codec" >/dev/null; then
    echo "verify-opensearch-upgrade: a native reader open failed for something other than the k-NN codec:" >&2
    docker logs "${PREFIX}-plugin" 2>&1 | grep "lucene-rust: native reader open failed" | grep -v "KNN[0-9]*Codec" | head -5 >&2
    status=1
fi
exit $status
