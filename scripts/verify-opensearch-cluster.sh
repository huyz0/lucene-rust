#!/usr/bin/env bash
# M5's cluster proof: three OpenSearch 3.8.0 nodes with the lucene-rust plugin,
# two serving Rust-engine indices with the Rust writer and one
# (lucene_rust.engine.node_enabled: false) with OpenSearch's own engine, then
# opensearch-plugin/e2e/verify_cluster.py: document replication, peer recovery,
# primary failover, primary relocation between the two kinds of node in both
# directions, and segment replication with native replicas -- every copy
# checked against a model of what was acknowledged.
#
#   scripts/verify-opensearch-cluster.sh [--keep] [--no-build]
#
# Needs Docker, a JDK 21, Gradle and cargo; the plugin image is the one
# scripts/verify-opensearch.sh builds. Nodes listen on localhost:9201..9203.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
IMAGE="lucene-rust-opensearch:${VERSION}"
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

# The three nodes share one network namespace -- a holder container's -- and
# talk over loopback: OpenSearch enforces its production bootstrap checks
# (65535 file descriptors, vm.max_map_count) only when the transport binds a
# non-loopback address, and a sandboxed Docker may cap both below them. HTTP
# still binds every interface, so the published ports reach it; stopping one
# node leaves the others' network alone.
cleanup() {
    docker rm -f os1 os2 os3 lucene-rust-net >/dev/null 2>&1 || true
}
cleanup
if [[ $KEEP == 0 ]]; then
    trap cleanup EXIT
fi
docker run -d --name lucene-rust-net -p 9201:9201 -p 9202:9202 -p 9203:9203 \
    --entrypoint sleep "opensearchproject/opensearch:${VERSION}" infinity >/dev/null

start() { # name http-port transport-port rust-enabled
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$1" --network container:lucene-rust-net \
        -e node.name="$1" -e cluster.name=lucene-rust \
        -e http.host=0.0.0.0 -e http.port="$2" \
        -e transport.host=127.0.0.1 -e transport.port="$3" \
        -e discovery.seed_hosts=127.0.0.1:9301,127.0.0.1:9302,127.0.0.1:9303 \
        -e cluster.initial_cluster_manager_nodes=os1,os2,os3 \
        -e lucene_rust.engine.node_enabled="$4" \
        -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e OPENSEARCH_JAVA_OPTS="-Xms768m -Xmx768m" \
        "$IMAGE" >/dev/null
}
start os1 9201 9301 true
start os2 9202 9302 true
start os3 9203 9303 false

python3 opensearch-plugin/e2e/verify_cluster.py
