#!/usr/bin/env bash
# A cluster upgraded from OpenSearch 2.x serves its
# old index natively.
#
#   1. An OpenSearch 2.19 node (Lucene 9.12) indexes a representative corpus --
#      text, keyword, numerics, knn_vector, nested documents, a k-NN index --
#      snapshots it to an fs repository and flushes
#      (opensearch-plugin/e2e/verify_upgrade.py load).
#   2. The node stops and its data directory is copied. OpenSearch 3.9.0 with
#      this plugin opens the original in place (the in-place node upgrade); a
#      stock 3.9.0 node, the same distribution without the plugin, opens the copy.
#   3. verify_upgrade.py verify: every index opens with its Lucene 9.12
#      segments, the plugin node answers every request of the matrix
#      (verify_opensearch.py's, plus aggregations and vector requests) as the
#      stock node does, natively where the plugin says it runs; the same over
#      the 2.x snapshot restored into both nodes; and after a force merge has
#      rewritten every segment as Lucene 10.5.x. The 2.x-created indices are
#      on the accelerated engine by the node default.
#   4. The plugin node stops and a stock 3.9.0 node (no plugin) opens its data
#      directory: every index green with the same documents
#      (verify_upgrade.py removed).
#
# Both 3.9.0 nodes keep the bundled k-NN plugin (the knn_vector mapping needs
# it) and drop the rest, as the 2.19 node does; the nodes are plain processes
# (opensearch-plugin/e2e/nodectl.py), no Docker.
#
#   scripts/verify-opensearch-upgrade.sh [--docs N] [--keep] [--no-build]
#
# Needs a JDK 25 and Gradle (the plugin build, as verify-opensearch.sh), and cargo. Listens on localhost:${OS_PORT:-9240} (plugin), +2 (stock), +4 (2.x).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OLD_VERSION="${OLD_OPENSEARCH_VERSION:-2.19.6}"
VERSION=3.9.0
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

nodectl() { python3 opensearch-plugin/e2e/nodectl.py "$@"; }
# Every node keeps opensearch-knn and nothing else bundled, so the 2.x cluster state holds nothing
# a 3.9.0 node lacks a plugin for.
OLD_VARIANT=old-knn
PLUGIN_VARIANT=plugin-knn
STOCK_VARIANT=stock-knn
NODES=("${PREFIX}-old" "${PREFIX}-plugin" "${PREFIX}-stock" "${PREFIX}-removed")
WORK="$(mktemp -d)"
VOL_OLD="$WORK/data"
VOL_STOCK="$WORK/data-stock"
VOL_SNAP="$WORK/snap"
export UPGRADE_SNAP="$VOL_SNAP"

cleanup() {
    for n in "${NODES[@]}"; do nodectl rm "$n" >/dev/null 2>&1 || true; done
    rm -rf "$WORK"
}
if [[ $KEEP == 0 ]]; then
    trap cleanup EXIT
fi

if [[ $BUILD == 1 ]]; then
    scripts/opensearch-dist.sh
    gradle --no-daemon -q -p opensearch-plugin bundlePlugin
fi
nodectl dist "$OLD_VARIANT" --version "$OLD_VERSION" --keep opensearch-knn --no-store-key >/dev/null
nodectl dist "$STOCK_VARIANT" --keep opensearch-knn --no-store-key >/dev/null
mkdir -p "$VOL_OLD" "$VOL_STOCK" "$VOL_SNAP"

start_node() { # name port variant data-dir [version]; nodectl run waits until the node answers
    nodectl run "$1" "$3" "$2" --data "$4" --setting path.repo="$VOL_SNAP" ${5:+--version "$5"} >/dev/null
}

echo "== OpenSearch ${OLD_VERSION}: index and snapshot"
start_node "${PREFIX}-old" "$((PORT + 4))" "$OLD_VARIANT" "$VOL_OLD" "$OLD_VERSION"
python3 opensearch-plugin/e2e/verify_upgrade.py load "http://localhost:$((PORT + 4))" "$DOCS"
nodectl stop "${PREFIX}-old"
nodectl rm "${PREFIX}-old"

# The stock node's copy: the same bytes, owners and modes.
cp -a "$VOL_OLD/." "$VOL_STOCK/"

echo "== OpenSearch ${VERSION}: in-place upgrade (plugin) and a stock node on a copy"
start_node "${PREFIX}-plugin" "$PORT" "$PLUGIN_VARIANT" "$VOL_OLD"
start_node "${PREFIX}-stock" "$((PORT + 2))" "$STOCK_VARIANT" "$VOL_STOCK"

status=0
RECORD="$(mktemp)"
UPGRADE_RECORD="$RECORD" python3 opensearch-plugin/e2e/verify_upgrade.py verify "http://localhost:$PORT" "http://localhost:$((PORT + 2))" || status=$?
if nodectl logs "${PREFIX}-plugin" 2>&1 | grep -E "A fatal error has been detected|SIGSEGV \(0xb\)" >/dev/null; then
    echo "verify-opensearch-upgrade: the plugin node's JVM crashed" >&2
    status=1
fi
# The only native refusals expected are the k-NN codec's (see verify_upgrade.py).
if nodectl logs "${PREFIX}-plugin" 2>&1 | grep "accelerated-search: native reader open failed" | grep -v "KNN[0-9]*Codec" >/dev/null; then
    echo "verify-opensearch-upgrade: a native reader open failed for something other than the k-NN codec:" >&2
    nodectl logs "${PREFIX}-plugin" 2>&1 | grep "accelerated-search: native reader open failed" | grep -v "KNN[0-9]*Codec" | head -5 >&2
    status=1
fi

# The plugin removed: a stock node opens the plugin node's data directory as it is.
echo "== OpenSearch ${VERSION} without the plugin, on the plugin node's data"
nodectl stop "${PREFIX}-plugin"
if [[ $KEEP == 1 ]]; then nodectl logs "${PREFIX}-plugin" > "${PREFIX}-plugin.log" 2>&1 || true; fi
nodectl rm "${PREFIX}-plugin"
start_node "${PREFIX}-removed" "$PORT" "$STOCK_VARIANT" "$VOL_OLD"
python3 opensearch-plugin/e2e/verify_upgrade.py removed "http://localhost:$PORT" "$RECORD" || status=$?
rm -f "$RECORD"
exit $status
