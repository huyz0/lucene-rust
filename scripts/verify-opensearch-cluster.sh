#!/usr/bin/env bash
# the cluster proof: three OpenSearch 3.9.0 nodes with the accelerated-search plugin,
# two serving Rust-engine indices with the Rust writer and one
# (indices.engine.accelerated.node_enabled: false) with OpenSearch's own engine, then
# opensearch-plugin/e2e/verify_cluster.py: document replication, peer recovery,
# primary failover, primary relocation between the two kinds of node in both
# directions, and segment replication -- native replicas of a Java primary,
# and a Rust primary through failover, promotion and a force merge -- every copy
# checked against a model of what was acknowledged.
#
#   scripts/verify-opensearch-cluster.sh [--keep] [--no-build]
#
# Needs a JDK 25 (Gradle's toolchain; scripts/opensearch-dist.sh registers the tarball's own),
# Gradle and cargo; the nodes are plain processes (opensearch-plugin/e2e/nodectl.py), no Docker.
# Nodes listen on localhost:9201..9203 (transport 9301..9303, on loopback: OpenSearch enforces its
# production bootstrap checks only when the transport binds a non-loopback address).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

nodectl() { python3 opensearch-plugin/e2e/nodectl.py "$@"; }
# OS_VARIANT runs the nodes from another nodectl variant (see `nodectl.py dist --help`) -- e.g. an
# OpenSearch patched with an overridable Engine#lastRefreshedCheckpoint, under which the plugin
# needs no hook for segment replication (see docs/opensearch-engine.md).
VARIANT="${OS_VARIANT:-plugin}"
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
fi

cleanup() {
    nodectl rm os1 >/dev/null 2>&1 || true
    nodectl rm os2 >/dev/null 2>&1 || true
    nodectl rm os3 >/dev/null 2>&1 || true
}
cleanup
if [[ $KEEP == 0 ]]; then
    trap cleanup EXIT
fi

start() { # name http-port transport-port rust-enabled
    nodectl run "$1" "$VARIANT" "$2" --transport "$3" --heap 768m --no-wait \
        --setting cluster.name=accelerated-search \
        --setting discovery.seed_hosts=127.0.0.1:9301,127.0.0.1:9302,127.0.0.1:9303 \
        --setting cluster.initial_cluster_manager_nodes=os1,os2,os3 \
        --setting indices.engine.accelerated.node_enabled="$4" >/dev/null
}
start os1 9201 9301 true
start os2 9202 9302 true
start os3 9203 9303 false

python3 opensearch-plugin/e2e/verify_cluster.py
