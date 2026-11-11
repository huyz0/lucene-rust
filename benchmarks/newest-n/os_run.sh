#!/bin/bash
# os_run.sh <image> <label> <loghub dir> <out dir> [settings json for a second pass, label2]
set -euo pipefail
IMG=$1; L=$2; DATA=$3; OUT=$4; S=$(dirname "$0")/os_bench.py
mkdir -p "$OUT"; docker rm -f nn >/dev/null 2>&1 || true
docker run -d --name nn --cpuset-cpus 0-2 -p 9270:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e cluster.routing.allocation.disk.threshold_enabled=false -e DISABLE_INSTALL_DEMO_CONFIG=true \
  -e OPENSEARCH_JAVA_OPTS="-Xms4g -Xmx4g" "$IMG" >/dev/null
for _ in $(seq 1 90); do curl -sf localhost:9270 >/dev/null && break; sleep 2; done
B=http://localhost:9270
taskset -c 3 python3 $S load $B $DATA
taskset -c 3 python3 $S run $B $OUT/$L.json 50
if [ -n "${5:-}" ]; then
  curl -sf -XPUT $B/logs/_settings -H 'Content-Type: application/json' -d "$5" >/dev/null
  taskset -c 3 python3 $S run $B $OUT/$6.json 50
fi
curl -s $B/_plugins/lucene_rust/stats > $OUT/$L-stats.json || true
docker rm -f nn >/dev/null
