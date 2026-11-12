#!/bin/bash
# Runs the four configurations one after another on one node at a time; see docs/benchmarks/lean-seq-no.md.
# Needs ./corpus (gen.py), the stock image and the plugin image (PLUGIN_IMAGE).
cd "$(dirname "$0")"
for cfg in ${CFGS:-stock plugin-default plugin-lean plugin-lean-disable}; do
  img=${PLUGIN_IMAGE:-lucene-rust-opensearch:lean}; ex="-e lucene_rust.engine.default=true"; fl=""
  case $cfg in stock) img=${STOCK_IMAGE:-opensearch-stock-min:3.9.0}; ex="";; plugin-lean) fl="--lean";; plugin-lean-disable) fl="--lean --disable-seq";; esac
  HEAP=4g ./up.sh b-$cfg 9250 $img $ex || { echo "$cfg: node did not start"; continue; }
  python3 bench.py http://localhost:9250 b-$cfg $cfg res-$cfg${RUN:-}.json $fl > log-$cfg${RUN:-}.txt 2>&1
  docker rm -f b-$cfg >/dev/null
done
