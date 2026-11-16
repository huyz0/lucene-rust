#!/bin/bash
# Stock OpenSearch 3.9.0 against the plugin on real data, same shapes, results compared.
# About 6 minutes per dataset at 1M documents. Nodes are plain processes (no Docker): needs the plugin
# zip built (gradle -p opensearch-plugin bundlePlugin); the stock distribution is downloaded once.
#
#   DATA=taxi|bgl CORPUS=/path/to/bulk-dir [PLUGIN_SO=/path/liblucene_ffi.so] \
#   [STOCK_SETTINGS='{}'] [PLUGIN_SETTINGS='{"index.codec.adaptive_doc_values":true}'] \
#   [MAPPING=file.json] [TAG=label] [SKIP_STOCK=1] run.sh
#
# Corpus: `python3 -I prep.py taxi|bgl SRC DIR 1000000`. PLUGIN_SO is a build with
# `--profile dist --features mimalloc`, used instead of the zip's library (default: the zip's own). Writes out/$TAG-{stock,plugin}.{json,stats.json}, prints the comparison,
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
HERE=$(pwd)
nodectl() { python3 "$HERE/../../opensearch-plugin/e2e/nodectl.py" "$@"; }
start() { # name variant [so]
  nodectl rm $1 >/dev/null 2>&1 || true
  nodectl run $1 $2 $PORT --heap 2g --cpus 0-2 ${3:+--lib "$(realpath $3)"} >/dev/null
}
trap 'nodectl rm colstock >/dev/null 2>&1; nodectl rm colplugin >/dev/null 2>&1 || true' EXIT
run() { # name variant so label settings
  start $1 $2 "$3"
  taskset -c 3 $S load $U $DATA "$CORPUS" "$5" ${MAPPING:+$MAPPING}
  $S merge $U
  taskset -c 3 $S stats $U out/$TAG-$4.stats.json
  taskset -c 3 $S time $U $DATA $4 out/$TAG-$4.json
  nodectl rm $1 >/dev/null
}
[ -n "$SKIP_STOCK" ] || run colstock stock "" stock "$STOCK_SETTINGS"
run colplugin plugin "$PLUGIN_SO" plugin "$PLUGIN_SETTINGS"
echo; python3 -c "
import json
for l in ('stock','plugin'):
    s=json.load(open('out/$TAG-%s.stats.json'%l)); print(f\"{l:7s} docs {s['docs']} store {s['store_bytes']/2**20:8.1f} MiB  index {s['index_s']:6.1f}s  segments {s['segments']}  load cpu {s.get('load_cpu_s',0):6.1f}s  merge {s.get('forcemerge_s',0):5.1f}s cpu {s.get('forcemerge_cpu_s',0):6.1f}s\")"
$S compare out/$TAG-stock.json out/$TAG-plugin.json
