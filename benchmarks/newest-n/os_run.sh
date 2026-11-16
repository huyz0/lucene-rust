#!/bin/bash
# os_run.sh <variant: stock|plugin|stock-min> <label> <loghub dir> <out dir> [settings json for a second pass, label2]
# The node is a plain process (opensearch-plugin/e2e/nodectl.py), no Docker; LIB=/path/liblucene_ffi.so replaces the zip's library.
set -euo pipefail
VARIANT=$1; L=$2; DATA=$3; OUT=$4; S=$(dirname "$0")/os_bench.py
mkdir -p "$OUT"
NODECTL="$(dirname "$0")/../../opensearch-plugin/e2e/nodectl.py"
python3 "$NODECTL" rm nn >/dev/null 2>&1 || true
python3 "$NODECTL" run nn "$VARIANT" 9270 --heap 4g --cpus 0-2 ${LIB:+--lib "$LIB"} >/dev/null
B=http://localhost:9270
taskset -c 3 python3 $S load $B $DATA
taskset -c 3 python3 $S run $B $OUT/$L.json 50
if [ -n "${5:-}" ]; then
  curl -sf -XPUT $B/logs/_settings -H 'Content-Type: application/json' -d "$5" >/dev/null
  taskset -c 3 python3 $S run $B $OUT/$6.json 50
fi
curl -s $B/_plugins/lucene_rust/stats > $OUT/$L-stats.json || true
python3 "$NODECTL" rm nn >/dev/null
