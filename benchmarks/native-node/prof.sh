#!/bin/bash
# prof.sh KIND [LIB.so]: per shape, a serial closed loop for SECS seconds with async-profiler (itimer 1 ms, per thread)
# over the middle of it; collapsed stacks to out/prof/KIND-<n>.collapsed, qps and node cpu ms/req to out/prof/KIND-loop.jsonl.
#   OUT_DIR=/data/marco ./prof.sh stock ; OUT_DIR=... ./prof.sh plugin /p/main.so
set -e
cd "$(dirname "$0")"; source ./native-node.sh
KIND=$1; LIB=${2:-}; : "${OUT_DIR:?}"; D=${DATA_DIR:-$NODE_HOME/data}; O=out/prof; mkdir -p $O
SECS=${SECS:-20}; WARM=${WARM:-10}; AP=${AP:-/root/ap/bin/asprof}; MODE_E=${PROF_EVENT:-itimer}
trap native_stop EXIT
native_start $KIND "$D/$KIND" $LIB
PID=$(pgrep -f "org.opensearch.bootstrap.OpenSearch" | head -1)
: > $O/$KIND-loop.jsonl; i=0
IFS='|' read -ra SH <<< "${PROF_SHAPES:?shapes separated by |}"
for S in "${SH[@]}"; do
  i=$((i+1))
  taskset -c 3 python3 text_bench.py loop $U "$OUT_DIR" "$S" $SECS $PID >> $O/$KIND-loop.jsonl &
  LP=$!
  sleep $WARM
  $AP -e $MODE_E -i 1ms -d $((SECS-WARM-2)) -o collapsed -f /tmp/prof/p.txt $PID >/dev/null; cp /tmp/prof/p.txt $O/$KIND-$i.collapsed
  echo "$S" > $O/$KIND-$i.name; wait $LP
done
echo prof-done $KIND
