#!/bin/bash
# Core shapes (ab.py) plus feature families (families.py), stock vs one plugin build, alternating rounds.
#   CORPUS=/data/corpus BUILD=/path/liblucene_ffi.so [ROUNDS_AB=2] [FAM_DOCS=300000] famsweep.sh
set -e
cd "$(dirname "$0")"
source ./native-node.sh
: "${CORPUS:?}" "${BUILD:?}"
TAG=${TAG:-fam}; ROUNDS_AB=${ROUNDS_AB:-2}; D=${DATA_DIR:-$NODE_HOME/data}; mkdir -p out
DRV="taskset -c ${DRIVER_CPU:-3} python3"
trap native_stop EXIT
if [ -z "${SKIP_LOAD:-}" ]; then
  rm -rf "$D"
  native_start stock "$D/stock"
  $DRV ab.py load $U "$CORPUS" '{}'; $DRV families.py load $U "$CORPUS" ${FAM_DOCS:-300000} '{}'
  native_start plugin "$D/plugin" "$BUILD"
  S='{"index.engine.accelerated":true}'
  $DRV ab.py load $U "$CORPUS" "$S"; $DRV families.py load $U "$CORPUS" ${FAM_DOCS:-300000} "$S"
fi
for r in $(seq $ROUNDS_AB); do
  native_start stock "$D/stock"; $DRV ab.py time $U stock out/$TAG-core-stock-r$r.json; $DRV families.py time $U stock out/$TAG-fam-stock-r$r.json
  native_start plugin "$D/plugin" "$BUILD"; $DRV ab.py time $U new out/$TAG-core-new-r$r.json; $DRV families.py time $U new out/$TAG-fam-new-r$r.json
done
