#!/bin/bash
# cc-taxis.sh KIND[:LIB.so] ...: closed-loop concurrent throughput over the taxis shapes (conc_taxis.py).
#   [SECS=10] [NODE_CPUS=0-1] [MEM_LIMIT=2G] [SHAPES_CC="a|b"] ./cc-taxis.sh stock plugin:/p/main.so
cd "$(dirname "$0")"; source ./native-node.sh
D=${DATA_DIR:-$NODE_HOME/data}; SECS=${SECS:-10}
export NODE_CPUS=${NODE_CPUS:-0-1}; DRIVER_CPUS=${DRIVER_CPUS:-2-3}
trap native_stop EXIT
IFS='|' read -ra SH <<< "${SHAPES_CC:-match_all size 12|range total_amount|distance_amount_agg|date_histogram_agg month|date_histogram_agg day|date_histogram_agg hour, whole index|sort pickup asc, range filter|terms payment_type}"
for spec in "$@"; do
  kind=${spec%%:*}; lib=; [[ $spec == *:* ]] && lib=${spec#*:}
  native_start $kind "$D/$kind" $lib
  PID=$(pgrep -f "org.opensearch.bootstrap.OpenSearch" | head -1)
  for S in "${SH[@]}"; do
    echo -n "$kind "; taskset -c $DRIVER_CPUS python3 conc_taxis.py ${U:-http://127.0.0.1:9200} "$S" $SECS $PID ${PROCS:-2} ${CONNS:-4}
  done
  native_stop
done
