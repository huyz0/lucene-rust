#!/bin/bash
# Alternating A/B of stock and plugin builds on the Wikipedia corpus (wiki_bench.py prep first), N shards.
# Per round and node: serial latency for every shape, then closed-loop concurrent load (qps, p50, p99, node CPU/request)
# for the shapes in SHAPES_CC at CLIENTS connections. Node on NODE_CPUS (0-2), driver on DRIVER_CPU (3).
#   OUT_DIR=/data/wiki BUILDS="main=/p/main.so" [ROUNDS_AB=2] [SHARDS=4] [SHAPES_CC="a|b"] [CLIENTS=8] [SECS=8] wiki-ab.sh
set -e
cd "$(dirname "$0")"
source ./native-node.sh
: "${BUILDS:?label=so,...}"; : "${OUT_DIR:?prep dir}"
TAG=${TAG:-wiki}; ROUNDS_AB=${ROUNDS_AB:-2}; D=${DATA_DIR:-$NODE_HOME/wikidata}; SECS=${SECS:-8}; CLIENTS=${CLIENTS:-8}
mkdir -p out; DRV="taskset -c ${DRIVER_CPU:-3} python3 -I wiki_bench.py"
IFS=, read -ra B <<< "$BUILDS"; first=${B[0]#*=}
trap native_stop EXIT
if [ -z "${SKIP_LOAD:-}" ]; then
  rm -rf "$D"; native_start stock "$D/stock"; $DRV load $U "$OUT_DIR" "${SETTINGS:-{\}}"
  native_start plugin "$D/plugin" "$first"
  $DRV load $U "$OUT_DIR" "$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['index.engine.accelerated']=True; print(json.dumps(d))" "${SETTINGS:-{\}}")"
fi
IFS='|' read -ra SH <<< "${SHAPES_CC:-}"
run_node() {  # label kind data lib
  native_start $2 "$3" $4
  [ -z "${SKIP_SERIAL:-}" ] && $DRV time $U $1 out/$TAG-$1-r$r.json "$OUT_DIR"
  local pid; pid=$(pgrep -f "org.opensearch.bootstrap.OpenSearch" | head -1)
  for S in "${SH[@]}"; do
    [ -z "$S" ] && continue
    for C in $CLIENTS; do
      res=$($DRV conc $U "$OUT_DIR" "$S" $SECS $pid 1 $C); echo "$res" | tee -a out/$TAG-conc-$1-r$r.jsonl
    done
  done
}
for r in $(seq $ROUNDS_AB); do
  run_node stock stock "$D/stock" ""
  for b in "${B[@]}"; do run_node ${b%%=*} plugin "$D/plugin" "${b#*=}"; done
done
native_stop
[ -z "${SKIP_SERIAL:-}" ] && python3 -I wiki_bench.py report "out/$TAG-stock-r*.json" $(for b in "${B[@]}"; do echo "out/$TAG-${b%%=*}-r*.json"; done | tr "\n" " ")
