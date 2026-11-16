#!/bin/bash
# Runs the four configurations one after another on one node at a time; see docs/benchmarks/lean-seq-no.md.
# Needs ./corpus (gen.py) and the plugin zip built (gradle -p opensearch-plugin bundlePlugin); nodes are
# plain processes (opensearch-plugin/e2e/nodectl.py), no Docker.
cd "$(dirname "$0")"
NODECTL=../nodectl.py
for cfg in ${CFGS:-stock plugin-default plugin-lean plugin-lean-disable}; do
  variant=plugin; ex="indices.engine.accelerated.default=true"; fl=""
  case $cfg in stock) variant=stock-min; ex="";; plugin-lean) fl="--lean";; plugin-lean-disable) fl="--lean --disable-seq";; esac
  HEAP=4g ./up.sh b-$cfg 9250 $variant $ex || { echo "$cfg: node did not start"; continue; }
  python3 bench.py http://localhost:9250 b-$cfg $cfg res-$cfg${RUN:-}.json $fl > log-$cfg${RUN:-}.txt 2>&1
  python3 $NODECTL rm b-$cfg
done
