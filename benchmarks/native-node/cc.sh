#!/bin/bash
# cc.sh KIND[:LIB.so] ...: closed-loop concurrent throughput (conc.py) per shape, node on NODE_CPUS, driver on DRIVER_CPUS.
#   OUT_DIR=/tmp/marco/data SHAPES_CC="term frequent|term rare" [SECS=12] ./cc.sh stock plugin:/tmp/p/main.so
cd "$(dirname "$0")"; source ./native-node.sh
: "${OUT_DIR:?}"; D=${DATA_DIR:-$NODE_HOME/data}; SECS=${SECS:-12}
export NODE_CPUS=${NODE_CPUS:-0-1}; DRIVER_CPUS=${DRIVER_CPUS:-2-3}
trap native_stop EXIT
IFS='|' read -ra SH <<< "${SHAPES_CC:?}"
for spec in "$@"; do
  kind=${spec%%:*}; lib=; [[ $spec == *:* ]] && lib=${spec#*:}
  native_start $kind "$D/$kind" $lib
  PID=$(pgrep -f "org.opensearch.bootstrap.OpenSearch" | head -1)
  for S in "${SH[@]}"; do
    echo -n "$kind "; taskset -c $DRIVER_CPUS python3 conc.py $U "$OUT_DIR" "$S" $SECS $PID ${PROCS:-2} ${CONNS:-4}
  done
done
