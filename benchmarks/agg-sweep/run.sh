#!/bin/bash
# A/B of two plugin builds against stock OpenSearch on e2e_sweep.py's shapes,
# about 10 minutes. Needs the lucene-rust-opensearch:3.9.0 and
# opensearchproject/opensearch:3.9.0 images and a corpus from
#   python3 ../../opensearch-plugin/e2e/bench_nodes.py gen $CORPUS 1000000
#
#   CORPUS=/path/to/corpus run.sh base.so new.so
#
# Each build's liblucene_ffi.so (built with `--profile dist --features
# mimalloc`) is mounted over the image's. One index is loaded and
# force-merged once per node; the builds are alternated twice on it, then the
# plugin's Java Lucene path and stock are timed once.
set -e
cd "$(dirname "$0")"
BASE_SO=$(realpath "$1"); NEW_SO=$(realpath "$2")
S="python3 e2e_sweep.py"; U=http://localhost:9250
start() { # name image [so]
  docker rm -f $1 >/dev/null 2>&1 || true
  local extra=()
  [ -n "$3" ] && extra=(-v $3:/usr/share/opensearch/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so:ro)
  docker run -d --name $1 -p 9250:9200 --cpuset-cpus 0-2 -v ${1}data:/usr/share/opensearch/data "${extra[@]}" \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e cluster.routing.allocation.disk.threshold_enabled=false \
    -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" $2 >/dev/null
  for i in $(seq 120); do curl -sf $U/_cluster/health?wait_for_status=yellow >/dev/null 2>&1 && break; sleep 2; done
}
mkdir -p out
start aggstock opensearchproject/opensearch:3.9.0
$S load $U; $S merge $U; $S time $U stock out/stock.json
docker rm -f aggstock >/dev/null; docker volume rm aggstockdata >/dev/null
start aggplugin lucene-rust-opensearch:3.9.0 "$BASE_SO"
$S load $U; $S merge $U
for r in 1 2; do
  for b in base new; do
    so=$BASE_SO; [ $b = new ] && so=$NEW_SO
    start aggplugin lucene-rust-opensearch:3.9.0 "$so"
    $S java $U on; $S time $U $b-$r out/$b-$r.json
  done
done
$S java $U off; $S time $U java out/java.json; $S java $U on
docker rm -f aggplugin >/dev/null; docker volume rm aggplugindata >/dev/null
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
