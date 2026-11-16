#!/bin/bash
# A/B of two plugin builds against stock OpenSearch on e2e_sweep.py's shapes,
# about 10 minutes. Nodes are plain processes (no Docker): needs the plugin zip built
# (gradle -p opensearch-plugin bundlePlugin) and a corpus from
#   python3 ../../opensearch-plugin/e2e/bench_nodes.py gen $CORPUS 1000000
#
#   CORPUS=/path/to/corpus run.sh base.so new.so
#
# Each build's liblucene_ffi.so (built with `--profile dist --features
# mimalloc`) replaces the zip's. One index is loaded and
# force-merged once per node; the builds are alternated twice on it, then the
# plugin's Java Lucene path and stock are timed once.
set -e
cd "$(dirname "$0")"
BASE_SO=$(realpath "$1"); NEW_SO=$(realpath "$2")
S="python3 e2e_sweep.py"; U=http://localhost:9250
HERE=$(pwd)
nodectl() { python3 "$HERE/../../opensearch-plugin/e2e/nodectl.py" "$@"; }
DATA_ROOT=${NODE_HOME:-$HOME/os-native}/data
start() { # name variant [so]
  nodectl rm $1 >/dev/null 2>&1 || true
  nodectl run $1 $2 9250 --heap 2g --cpus 0-2 --data "$DATA_ROOT/${1}data" ${3:+--lib "$3"} >/dev/null
}
mkdir -p out
start aggstock stock
$S load $U; $S merge $U; $S time $U stock out/stock.json
nodectl rm aggstock >/dev/null; rm -rf "$DATA_ROOT/aggstockdata"
start aggplugin plugin "$BASE_SO"
$S load $U; $S merge $U
for r in 1 2; do
  for b in base new; do
    so=$BASE_SO; [ $b = new ] && so=$NEW_SO
    start aggplugin plugin "$so"
    $S java $U on; $S time $U $b-$r out/$b-$r.json
  done
done
$S java $U off; $S time $U java out/java.json; $S java $U on
nodectl rm aggplugin >/dev/null; rm -rf "$DATA_ROOT/aggplugindata"
python3 - <<'PY'
import json
L = lambda x: json.load(open(f"out/{x}.json"))
b = [L(f"base-{i}") for i in (1, 2)]; n = [L(f"new-{i}") for i in (1, 2)]
st, jv = L("stock"), L("java")
print(f"{'shape':40s} {'stock':>7s} {'java':>7s} {'base':>7s} {'new':>7s} {'ratio':>6s} same")
for k in b[0]:
    bb = min(x[k]["median_ms"] for x in b); nn = min(x[k]["median_ms"] for x in n)
    same = n[0][k]["digest"] == jv[k]["digest"] == b[0][k]["digest"]
    print(f"{k:40s} {st[k]['median_ms']:7.2f} {jv[k]['median_ms']:7.2f} {bb:7.2f} {nn:7.2f} {nn/bb:6.2f} {same}")
PY
