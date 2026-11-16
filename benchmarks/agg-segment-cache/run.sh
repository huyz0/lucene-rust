#!/usr/bin/env bash
# Aggregation segment cache benchmark: stock OpenSearch, the plugin from main, and the plugin
# with the cache, over the same 6.45M Loghub events. See docs/benchmarks/agg-segment-cache.md.
#   run.sh WORK MAIN.so CACHE.so
# WORK holds the corpus (gen.py) and gets data/ and out/. The two libraries are
# `--profile dist --features mimalloc` builds of liblucene_ffi.so (the plugin zip must be built);
# stock is OpenSearch without its bundled plugins (nodectl.py variant stock-min). No Docker.
set -euo pipefail
WORK=$(realpath "$1") MAIN=$(realpath "$2") CACHE=$(realpath "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"
mkdir -p data out
fresh() { rm -rf "data/$1"; if [[ -n "${2:-}" ]]; then cp -a "data/$2" "data/$1"; else mkdir -p "data/$1"; fi; }
NODECTL="$HERE/../../opensearch-plugin/e2e/nodectl.py"
trap 'python3 "$NODECTL" rm bench-stock >/dev/null 2>&1; python3 "$NODECTL" rm bench-plugin >/dev/null 2>&1 || true' EXIT
PLUGIN=(-e indices.engine.accelerated.default=true)
# One load for both plugin runs, so that they start from the same segments.
fresh plugin
"$HERE/node.sh" bench-plugin plugin "$WORK/data/plugin" "$MAIN" "${PLUGIN[@]}"
python3 "$HERE/bench.py" http://localhost:9250 corpus main out/plugin-load.json --load --load-only
python3 "$NODECTL" rm bench-plugin >/dev/null
fresh stock
"$HERE/node.sh" bench-stock stock-min "$WORK/data/stock"
python3 "$HERE/bench.py" http://localhost:9250 corpus stock out/stock-load.json --load --load-only
python3 "$HERE/parity.py" out/parity-stock.json
python3 "$HERE/bench.py" http://localhost:9250 corpus stock out/stock.json
python3 "$NODECTL" rm bench-stock >/dev/null
rm -rf data/stock
fresh main plugin
"$HERE/node.sh" bench-plugin plugin "$WORK/data/main" "$MAIN" "${PLUGIN[@]}"
python3 "$HERE/parity.py" out/parity-main.json
python3 "$HERE/bench.py" http://localhost:9250 corpus main out/main.json
python3 "$NODECTL" rm bench-plugin >/dev/null
rm -rf data/main
fresh cache plugin
"$HERE/node.sh" bench-plugin plugin "$WORK/data/cache" "$CACHE" "${PLUGIN[@]}"
curl -sf -XPUT localhost:9250/logs/_settings -H 'content-type: application/json' \
    -d '{"index":{"search.aggregation_segment_cache.enabled":true}}' >/dev/null
python3 "$HERE/parity.py" out/parity-cache-cold.json
python3 "$HERE/parity.py" out/parity-cache.json
python3 "$HERE/bench.py" http://localhost:9250 corpus cache out/cache.json --plugin --cache
python3 "$NODECTL" rm bench-plugin >/dev/null
rm -rf data/cache
# The same reader with the cache off and on, after every refresh.
fresh verify plugin
"$HERE/node.sh" bench-plugin plugin "$WORK/data/verify" "$CACHE" "${PLUGIN[@]}"
VERIFY=1 python3 "$HERE/bench.py" http://localhost:9250 corpus cache out/cache-verify.json --plugin --cache
python3 "$NODECTL" rm bench-plugin >/dev/null
rm -rf data/verify data/plugin
(cd out && python3 "$HERE/cmp_parity.py")
