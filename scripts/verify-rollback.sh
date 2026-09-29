#!/usr/bin/env bash
# M6's upgrade and rollback, executed (T6.4): three OpenSearch 3.8.0 nodes go
# from no plugin, onto the plugin, adopt the Rust engine by reindex, and back
# off it -- OpenSearch's engine, a force merge, then no plugin at all -- with
# every acknowledged write and every answer checked after each step.
# See opensearch-plugin/e2e/verify_rollback.py and docs/operations.md.
#
#   scripts/verify-rollback.sh [--docs N] [--out FILE] [--keep] [--no-build]
#
# Needs Docker and what scripts/verify-opensearch.sh needs to build the
# plugin image. Nodes listen on localhost:9201..9203.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
PLUGIN_IMAGE="lucene-rust-opensearch:${VERSION}"
STOCK_IMAGE="lucene-rust-stock:${VERSION}"
DOCS=200000
OUT="$PWD/target/rollback.json"
KEEP=0
BUILD=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docs) DOCS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --no-build) BUILD=0; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

if [[ $BUILD == 1 ]]; then
    scripts/opensearch-dist.sh
    gradle --no-daemon -q -p opensearch-plugin bundlePlugin
    rm -f opensearch-plugin/docker/*.zip
    cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/
    docker build -q -t "$PLUGIN_IMAGE" opensearch-plugin/docker >/dev/null
fi
# OpenSearch as an operator runs it before and after: the same image the
# plugin's is built on, bundled plugins removed as there (neural-search's
# QueryPhaseSearcher excludes the plugin's), and no lucene-rust.
printf 'FROM opensearchproject/opensearch:%s\nRUN rm -rf /usr/share/opensearch/plugins/*\n' "$VERSION" |
    docker build -q -t "$STOCK_IMAGE" - >/dev/null

DATA="$(mktemp -d)"
cleanup() {
    docker rm -f os1 os2 os3 lucene-rust-net >/dev/null 2>&1 || true
    rm -rf "$DATA"
}
docker rm -f os1 os2 os3 lucene-rust-net >/dev/null 2>&1 || true
[[ $KEEP == 0 ]] && trap cleanup EXIT
for n in os1 os2 os3; do mkdir -p "$DATA/$n"; done
chown -R 1000:1000 "$DATA"
docker run -d --name lucene-rust-net -p 9201:9201 -p 9202:9202 -p 9203:9203 \
    --entrypoint sleep "opensearchproject/opensearch:${VERSION}" infinity >/dev/null

mkdir -p "$(dirname "$OUT")"
ROLLBACK_DATA="$DATA" ROLLBACK_DOCS="$DOCS" PLUGIN_IMAGE="$PLUGIN_IMAGE" STOCK_IMAGE="$STOCK_IMAGE" \
    python3 opensearch-plugin/e2e/verify_rollback.py --out "$OUT"
