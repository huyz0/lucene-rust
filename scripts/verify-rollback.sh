#!/usr/bin/env bash
# the upgrade and rollback, executed: three OpenSearch 3.9.0 nodes go
# from no plugin, onto the plugin, adopt the Rust engine by reindex, and back
# off it -- OpenSearch's engine, a force merge, then no plugin at all -- with
# every acknowledged write and every answer checked after each step.
# See opensearch-plugin/e2e/verify_rollback.py and docs/operations.md.
#
#   scripts/verify-rollback.sh [--docs N] [--out FILE] [--keep] [--no-build]
#
# Needs what scripts/verify-opensearch.sh needs to build the plugin; the nodes are plain processes
# (opensearch-plugin/e2e/nodectl.py), no Docker. Nodes listen on localhost:9201..9203.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

nodectl() { python3 opensearch-plugin/e2e/nodectl.py "$@"; }
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
fi
# Variants: `plugin`, and OpenSearch as an operator runs it before and after (`stock-min`: bundled
# plugins removed as there -- neural-search's QueryPhaseSearcher excludes the plugin's -- and no
# lucene-rust).
nodectl dist plugin >/dev/null
nodectl dist stock-min >/dev/null

DATA="$(mktemp -d)"
cleanup() {
    for n in os1 os2 os3; do nodectl rm "$n" >/dev/null 2>&1 || true; done
    rm -rf "$DATA"
}
cleanup
DATA="$(mktemp -d)"
[[ $KEEP == 0 ]] && trap cleanup EXIT
for n in os1 os2 os3; do mkdir -p "$DATA/$n"; done

mkdir -p "$(dirname "$OUT")"
ROLLBACK_DATA="$DATA" ROLLBACK_DOCS="$DOCS" python3 opensearch-plugin/e2e/verify_rollback.py --out "$OUT"
