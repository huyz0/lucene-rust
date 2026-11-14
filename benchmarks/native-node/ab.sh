#!/bin/bash
# Alternating A/B of stock OpenSearch and one or more plugin builds on one box, no Docker.
# Each side loads its own index once (force-merged to one segment), then for ROUNDS_AB rounds every
# side is restarted and timed in turn, so machine drift cancels. About 15 min with 1M docs.
#
#   CORPUS=dir BUILDS="main=/p/main.so,new=/p/new.so" [ROUNDS_AB=4] [TAG=x] [MEM_LIMIT=2G] [SHAPES=...] ab.sh
#
# CORPUS: python3 opensearch-plugin/e2e/bench_nodes.py gen DIR 1000000. Writes out/$TAG-<label>-r<N>.json
# and prints the table. Needs `native_setup` to have run (stock/ and plugin/ under $NODE_HOME).
set -e
cd "$(dirname "$0")"
source ./native-node.sh
: "${CORPUS:?bulk dir}" "${BUILDS:?label=so,...}"
TAG=${TAG:-ab}; ROUNDS_AB=${ROUNDS_AB:-4}; D=${DATA_DIR:-$NODE_HOME/data}; mkdir -p out
SETTINGS=${SETTINGS:-'{}'}; DRV="taskset -c ${DRIVER_CPU:-3} python3 ab.py"
IFS=, read -ra B <<< "$BUILDS"; first=${B[0]#*=}
trap native_stop EXIT
if [ -z "${SKIP_LOAD:-}" ]; then
  rm -rf "$D"; MEM_LIMIT= native_start stock "$D/stock"; $DRV load $U "$CORPUS" "$SETTINGS"
  MEM_LIMIT= native_start plugin "$D/plugin" "$first"
  $DRV load $U "$CORPUS" "$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['index.engine.accelerated']=True; print(json.dumps(d))" "$SETTINGS")"
fi
for r in $(seq $ROUNDS_AB); do
  native_start stock "$D/stock"; $DRV time $U stock out/$TAG-stock-r$r.json
  for b in "${B[@]}"; do
    native_start plugin "$D/plugin" "${b#*=}"; $DRV time $U ${b%%=*} out/$TAG-${b%%=*}-r$r.json
  done
done
native_stop
python3 ab.py report "out/$TAG-stock-r*.json" $(for b in "${B[@]}"; do echo "out/$TAG-${b%%=*}-r*.json"; done)
