#!/bin/bash
# Cold runs: page cache dropped, a fresh process, reader opened, one search.
# cold.sh <idx> <java classes dir> <lucene-core jar> <rust example binary> <label> <runs>
idx=$1; cls=$2; jar=$3; bin=$4; label=$5; runs=${6:-3}
for shape in 0 1 4 5 7 8 10 12 15 17 21 24 34 41; do
  for r in $(seq $runs); do
    sync; echo 3 > /proc/sys/vm/drop_caches
    if [ "$label" = java ]; then
      java -Xmx4g -Dshape=$shape -cp $cls:$jar NewestN query $idx /tmp/cold-java cold 2>/dev/null | tail -1
    else
      $bin $idx /tmp/cold-rust $label cold $shape
    fi
  done
done
