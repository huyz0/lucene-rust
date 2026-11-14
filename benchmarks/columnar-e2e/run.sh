#!/bin/bash
# Stock OpenSearch 3.9.0 against the plugin on real data, same shapes, results compared.
# About 6 minutes per dataset at 1M documents. Images needed: opensearchproject/opensearch:3.9.0
# and lucene-rust-opensearch:3.9.0 (scripts/bench-opensearch.sh builds the latter).
#
#   DATA=taxi|bgl CORPUS=/path/to/bulk-dir [PLUGIN_SO=/path/liblucene_ffi.so] \
#   [STOCK_SETTINGS='{}'] [PLUGIN_SETTINGS='{"index.codec.adaptive_doc_values":true}'] \
#   [MAPPING=file.json] [TAG=label] [SKIP_STOCK=1] run.sh
#
# Corpus: `python3 -I prep.py taxi|bgl SRC DIR 1000000`. PLUGIN_SO is a build with
# `--profile dist --features mimalloc`, mounted over the image's library (default: the
# image's own). Writes out/$TAG-{stock,plugin}.{json,stats.json}, prints the comparison,
# exits 1 if any response differs. Nodes run one after the other on cores 0-2 (the client
# uses core 3), 2 GB heap, request cache off. SKIP_STOCK=1 reuses out/$TAG-stock.json.
set -e
cd "$(dirname "$0")"
: "${DATA:?taxi or bgl}" "${CORPUS:?bulk dir}"
TAG=${TAG:-$DATA}; PORT=9250; U=http://localhost:$PORT
STOCK_SETTINGS=${STOCK_SETTINGS:-'{}'}
PLUGIN_SETTINGS=${PLUGIN_SETTINGS:-'{"index.engine.accelerated":true}'}
S="python3 columnar_e2e.py"
mkdir -p out
start() { # name image [so]
  docker rm -f $1 >/dev/null 2>&1 || true
  local extra=()
  [ -n "$3" ] && extra=(-v $(realpath $3):/usr/share/opensearch/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so:ro)
  docker run -d --name $1 -p $PORT:9200 --cpuset-cpus 0-2 "${extra[@]}" \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e cluster.routing.allocation.disk.threshold_enabled=false \
    -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" "${@:4}" $2 >/dev/null
  for i in $(seq 120); do curl -sf $U/_cluster/health?wait_for_status=yellow >/dev/null 2>&1 && break; sleep 2; done
}
trap 'docker rm -f colstock colplugin >/dev/null 2>&1 || true' EXIT
run() { # name image so label settings
  start $1 $2 "$3"
  taskset -c 3 $S load $U $DATA "$CORPUS" "$5" ${MAPPING:+$MAPPING}
  $S merge $U
  taskset -c 3 $S stats $U out/$TAG-$4.stats.json
  taskset -c 3 $S time $U $DATA $4 out/$TAG-$4.json
  [ -z "$POST" ] || $POST $1 out/$TAG-$4.choice.txt
  docker rm -f $1 >/dev/null
}
[ -n "$SKIP_STOCK" ] || run colstock opensearchproject/opensearch:3.9.0 "" stock "$STOCK_SETTINGS"
run colplugin lucene-rust-opensearch:3.9.0 "$PLUGIN_SO" plugin "$PLUGIN_SETTINGS"
echo; python3 -c "
import json
for l in ('stock','plugin'):
    s=json.load(open('out/$TAG-%s.stats.json'%l)); print(f\"{l:7s} docs {s['docs']} store {s['store_bytes']/2**20:8.1f} MiB  index {s['index_s']:6.1f}s  segments {s['segments']}  load cpu {s.get('load_cpu_s',0):6.1f}s  merge {s.get('forcemerge_s',0):5.1f}s cpu {s.get('forcemerge_cpu_s',0):6.1f}s\")"
$S compare out/$TAG-stock.json out/$TAG-plugin.json
