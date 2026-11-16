#!/bin/bash
# Two or more plugin builds against stock on one real-data index, alternating the builds on the
# same loaded index (as ../agg-sweep/run.sh does), so machine drift between runs cancels. About
# 15 minutes per dataset at full size. For read-path changes only: the index is written once, by
# the first build.
#
#   DATA=taxi|bgl CORPUS=dir BUILDS="main=/p/main.so,new=/p/new.so" [ROUNDS_AB=2] [TAG=x] \
#   [SETTINGS='{"index.sort.field":...}'] [PLUGIN_EXTRA='{"index.codec.adaptive_doc_values":true}'] \
#   [SKIP_STOCK=1] [MAPPING=file] run-ab.sh
#
# Writes out/$TAG-stock.json and out/$TAG-$label.json (the best median of the rounds, with that
# round's responses); prints store size, then one table: stock, each build, last/first, and
# whether each response equals stock's. Exit 1 on any DIFF. Same node limits as run.sh; nodes are
# plain processes (no Docker), data under $NODE_HOME/data.
set -e
cd "$(dirname "$0")"
: "${DATA:?taxi or bgl}" "${CORPUS:?bulk dir}" "${BUILDS:?label=so,...}"
TAG=${TAG:-$DATA}; PORT=9250; U=http://localhost:$PORT; ROUNDS_AB=${ROUNDS_AB:-2}
SETTINGS=${SETTINGS:-'{}'}
S="python3 columnar_e2e.py"
mkdir -p out
HERE=$(pwd)
nodectl() { python3 "$HERE/../../opensearch-plugin/e2e/nodectl.py" "$@"; }
DATA_ROOT=${NODE_HOME:-$HOME/os-native}/data
start() { # name variant volume [so]
  nodectl rm $1 >/dev/null 2>&1 || true
  nodectl run $1 $2 $PORT --heap 2g --cpus 0-2 --data "$DATA_ROOT/$3" ${4:+--lib "$(realpath $4)"} >/dev/null
}
trap 'nodectl rm colstock >/dev/null 2>&1; nodectl rm colplugin >/dev/null 2>&1; rm -rf "$DATA_ROOT/colstockdata" "$DATA_ROOT/colplugindata"' EXIT
rm -rf "$DATA_ROOT/colstockdata" "$DATA_ROOT/colplugindata"
# stock: load, merge, time (SKIP_STOCK=1 reuses out/$TAG-stock.json and .stats.json)
if [ -z "$SKIP_STOCK" ]; then
start colstock stock colstockdata
taskset -c 3 $S load $U $DATA "$CORPUS" "$SETTINGS" ${MAPPING:+$MAPPING}
$S merge $U; taskset -c 3 $S stats $U out/$TAG-stock.stats.json
taskset -c 3 $S time $U $DATA stock out/$TAG-stock.json
nodectl rm colstock >/dev/null; rm -rf "$DATA_ROOT/colstockdata"
fi
# plugin: load once with the first build, then alternate
IFS=, read -ra B <<< "$BUILDS"; first=${B[0]#*=}
PSET=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['index.engine.accelerated']=True; d.update(json.loads(sys.argv[2])); print(json.dumps(d))" "$SETTINGS" "${PLUGIN_EXTRA:-{\}}")
start colplugin plugin colplugindata "$first"
taskset -c 3 $S load $U $DATA "$CORPUS" "$PSET" ${MAPPING:+$MAPPING}
$S merge $U; taskset -c 3 $S stats $U out/$TAG-plugin.stats.json
for r in $(seq $ROUNDS_AB); do
  for b in "${B[@]}"; do
    label=${b%%=*}; so=${b#*=}
    start colplugin plugin colplugindata "$so"
    taskset -c 3 $S time $U $DATA $label-$r out/$TAG-$label-r$r.json
  done
done
nodectl rm colplugin >/dev/null
python3 - <<PY
import json, sys
labels = [b.split("=")[0] for b in "$BUILDS".split(",")]
R = int("$ROUNDS_AB")
for l in labels:   # best median of the rounds per shape, responses of that round
    runs = [json.load(open(f"out/$TAG-{l}-r{r}.json")) for r in range(1, R + 1)]
    best = {k: min((x[k] for x in runs), key=lambda e: e["median_ms"]) for k in runs[0]}
    json.dump(best, open(f"out/$TAG-{l}.json", "w"), indent=1)
PY
for l in $(python3 -c "print(' '.join(b.split('=')[0] for b in '$BUILDS'.split(',')))"); do
  echo "== $l vs stock"; $S compare out/$TAG-stock.json out/$TAG-$l.json || FAIL=1
done
python3 - <<PY
import json
st = json.load(open("out/$TAG-stock.stats.json")); pl = json.load(open("out/$TAG-plugin.stats.json"))
print(f"stock store {st['store_bytes']/2**20:.1f} MiB load {st['index_s']:.1f}s | plugin store {pl['store_bytes']/2**20:.1f} MiB load {pl['index_s']:.1f}s")
PY
[ -z "$FAIL" ]
