#!/usr/bin/env bash
# Aggregation segment cache benchmark: stock OpenSearch, the plugin from main, and the plugin
# with the cache, over the same 6.45M Loghub events. See docs/benchmarks/agg-segment-cache.md.
#   run.sh WORK MAIN_IMAGE CACHE_IMAGE
# WORK holds the corpus (gen.py) and gets data/ and out/. Images as scripts/bench-opensearch.sh
# builds them; opensearch-stock-min:3.9.0 is stock without its plugins.
set -euo pipefail
WORK=$1 MAIN=$2 CACHE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"
mkdir -p data out
fresh() { rm -rf "data/$1"; if [[ -n "${2:-}" ]]; then cp -a "data/$2" "data/$1"; else mkdir -p "data/$1"; fi; chown 1000:1000 "data/$1"; }
trap 'docker rm -f bench-stock bench-plugin >/dev/null 2>&1 || true' EXIT
PLUGIN=(-e lucene_rust.engine.default=true)
# One load for both plugin runs, so that they start from the same segments.
fresh plugin
"$HERE/node.sh" bench-plugin "$MAIN" "$WORK/data/plugin" "${PLUGIN[@]}"
python3 "$HERE/bench.py" http://localhost:9250 corpus main out/plugin-load.json --load --load-only
docker rm -f bench-plugin >/dev/null
fresh stock
"$HERE/node.sh" bench-stock opensearch-stock-min:3.9.0 "$WORK/data/stock"
python3 "$HERE/bench.py" http://localhost:9250 corpus stock out/stock-load.json --load --load-only
python3 "$HERE/parity.py" out/parity-stock.json
python3 "$HERE/bench.py" http://localhost:9250 corpus stock out/stock.json
docker rm -f bench-stock >/dev/null
rm -rf data/stock
fresh main plugin
"$HERE/node.sh" bench-plugin "$MAIN" "$WORK/data/main" "${PLUGIN[@]}"
python3 "$HERE/parity.py" out/parity-main.json
python3 "$HERE/bench.py" http://localhost:9250 corpus main out/main.json
docker rm -f bench-plugin >/dev/null
rm -rf data/main
fresh cache plugin
"$HERE/node.sh" bench-plugin "$CACHE" "$WORK/data/cache" "${PLUGIN[@]}"
curl -sf -XPUT localhost:9250/logs/_settings -H 'content-type: application/json' \
    -d '{"index":{"lucene_rust.search.aggregation_segment_cache":true}}' >/dev/null
python3 "$HERE/parity.py" out/parity-cache-cold.json
python3 "$HERE/parity.py" out/parity-cache.json
python3 "$HERE/bench.py" http://localhost:9250 corpus cache out/cache.json --plugin --cache
docker rm -f bench-plugin >/dev/null
rm -rf data/cache
# The same reader with the cache off and on, after every refresh.
fresh verify plugin
"$HERE/node.sh" bench-plugin "$CACHE" "$WORK/data/verify" "${PLUGIN[@]}"
VERIFY=1 python3 "$HERE/bench.py" http://localhost:9250 corpus cache out/cache-verify.json --plugin --cache
docker rm -f bench-plugin >/dev/null
rm -rf data/verify data/plugin
(cd out && python3 "$HERE/cmp_parity.py")
