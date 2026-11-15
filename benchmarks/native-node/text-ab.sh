#!/bin/bash
# Alternating A/B of stock and plugin builds on a real-text corpus (text_bench.py prep first).
#   OUT_DIR=/data/marco BUILDS="main=/p/main.so" [ROUNDS_AB=3] [TAG=t] text-ab.sh
set -e
cd "$(dirname "$0")"
source ./native-node.sh
: "${BUILDS:?label=so,...}"; : "${OUT_DIR:?prep dir}"
TAG=${TAG:-text}; ROUNDS_AB=${ROUNDS_AB:-3}; D=${DATA_DIR:-$NODE_HOME/data}; mkdir -p out
SETTINGS=${SETTINGS:-'{}'}; DRV="taskset -c ${DRIVER_CPU:-3} python3 text_bench.py"
IFS=, read -ra B <<< "$BUILDS"; first=${B[0]#*=}
trap native_stop EXIT
if [ -z "${SKIP_LOAD:-}" ]; then
  rm -rf "$D"; native_start stock "$D/stock"; $DRV load $U "$OUT_DIR" "$SETTINGS"
  native_start plugin "$D/plugin" "$first"
  $DRV load $U "$OUT_DIR" "$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['index.engine.accelerated']=True; print(json.dumps(d))" "$SETTINGS")"
fi
for r in $(seq $ROUNDS_AB); do
  native_start stock "$D/stock"; $DRV time $U stock out/$TAG-stock-r$r.json "$OUT_DIR"
  for b in "${B[@]}"; do
    native_start plugin "$D/plugin" "${b#*=}"; $DRV time $U ${b%%=*} out/$TAG-${b%%=*}-r$r.json "$OUT_DIR"
  done
done
native_stop
python3 text_bench.py report "out/$TAG-stock-r*.json" $(for b in "${B[@]}"; do echo "out/$TAG-${b%%=*}-r*.json"; done | tr "\n" " ")
