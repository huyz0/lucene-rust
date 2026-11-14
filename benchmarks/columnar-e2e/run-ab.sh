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
# whether each response equals stock's. Exit 1 on any DIFF. Same node limits as run.sh.
set -e
cd "$(dirname "$0")"
: "${DATA:?taxi or bgl}" "${CORPUS:?bulk dir}" "${BUILDS:?label=so,...}"
TAG=${TAG:-$DATA}; PORT=9250; U=http://localhost:$PORT; ROUNDS_AB=${ROUNDS_AB:-2}
SETTINGS=${SETTINGS:-'{}'}
S="python3 columnar_e2e.py"
mkdir -p out
start() { # name image volume [so]
  docker rm -f $1 >/dev/null 2>&1 || true
  local extra=()
  [ -n "$4" ] && extra=(-v $(realpath $4):/usr/share/opensearch/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so:ro)
  docker run -d --name $1 -p $PORT:9200 --cpuset-cpus 0-2 -v $3:/usr/share/opensearch/data "${extra[@]}" \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e cluster.routing.allocation.disk.threshold_enabled=false \
    -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" $2 >/dev/null
  for i in $(seq 120); do curl -sf $U/_cluster/health?wait_for_status=yellow >/dev/null 2>&1 && break; sleep 2; done
}
trap 'docker rm -f colstock colplugin >/dev/null 2>&1 || true; docker volume rm colstockdata colplugindata >/dev/null 2>&1 || true' EXIT
docker volume rm colstockdata colplugindata >/dev/null 2>&1 || true
# stock: load, merge, time (SKIP_STOCK=1 reuses out/$TAG-stock.json and .stats.json)
if [ -z "$SKIP_STOCK" ]; then
start colstock opensearchproject/opensearch:3.9.0 colstockdata
taskset -c 3 $S load $U $DATA "$CORPUS" "$SETTINGS" ${MAPPING:+$MAPPING}
$S merge $U; taskset -c 3 $S stats $U out/$TAG-stock.stats.json
taskset -c 3 $S time $U $DATA stock out/$TAG-stock.json
docker rm -f colstock >/dev/null; docker volume rm colstockdata >/dev/null
fi
# plugin: load once with the first build, then alternate
IFS=, read -ra B <<< "$BUILDS"; first=${B[0]#*=}
PSET=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['index.engine.accelerated']=True; d.update(json.loads(sys.argv[2])); print(json.dumps(d))" "$SETTINGS" "${PLUGIN_EXTRA:-{\}}")
start colplugin lucene-rust-opensearch:3.9.0 colplugindata "$first"
taskset -c 3 $S load $U $DATA "$CORPUS" "$PSET" ${MAPPING:+$MAPPING}
$S merge $U; taskset -c 3 $S stats $U out/$TAG-plugin.stats.json
for r in $(seq $ROUNDS_AB); do
  for b in "${B[@]}"; do
    label=${b%%=*}; so=${b#*=}
    start colplugin lucene-rust-opensearch:3.9.0 colplugindata "$so"
    taskset -c 3 $S time $U $DATA $label-$r out/$TAG-$label-r$r.json
  done
done
docker rm -f colplugin >/dev/null
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
