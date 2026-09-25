#!/usr/bin/env bash
# M5's cluster proof: three OpenSearch 3.8.0 nodes with the lucene-rust plugin,
# two serving Rust-engine indices with the Rust writer and one
# (lucene_rust.engine.node_enabled: false) with OpenSearch's own engine, then
# opensearch-plugin/e2e/verify_cluster.py: segment replication, peer recovery,
# primary failover and relocation between the two kinds of node in both
# directions, and document replication -- every copy checked against a model
# of what was acknowledged.
#
#   scripts/verify-opensearch-cluster.sh [--keep] [--no-build]
#
# Needs Docker, a JDK 21, Gradle and cargo; the plugin image is the one
# scripts/verify-opensearch.sh builds. Nodes listen on localhost:9201..9203.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
IMAGE="lucene-rust-opensearch:${VERSION}"
NET=lucene-rust-cluster
KEEP=0
BUILD=1
for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --no-build) BUILD=0 ;;
        *) echo "unknown argument $a" >&2; exit 2 ;;
    esac
done

if [[ $BUILD == 1 ]]; then
    scripts/opensearch-dist.sh
    gradle --no-daemon -q -p opensearch-plugin bundlePlugin
    rm -f opensearch-plugin/docker/*.zip
    cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/
    docker build -q -t "$IMAGE" opensearch-plugin/docker >/dev/null
fi

docker network rm "$NET" >/dev/null 2>&1 || true
docker network create "$NET" >/dev/null
cleanup() {
    docker rm -f os1 os2 os3 >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
}
if [[ $KEEP == 0 ]]; then
    trap cleanup EXIT
fi

start() { # name port rust-enabled
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$1" --hostname "$1" --network "$NET" -p "$2:9200" \
        -e node.name="$1" -e cluster.name=lucene-rust \
        -e discovery.seed_hosts=os1,os2,os3 \
        -e cluster.initial_cluster_manager_nodes=os1,os2,os3 \
        -e lucene_rust.engine.node_enabled="$3" \
        -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e OPENSEARCH_JAVA_OPTS="-Xms768m -Xmx768m" \
        "$IMAGE" >/dev/null
}
start os1 9201 true
start os2 9202 true
start os3 9203 false

python3 opensearch-plugin/e2e/verify_cluster.py
