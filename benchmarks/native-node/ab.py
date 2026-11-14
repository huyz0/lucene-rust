"""Load and time request shapes on a node started by native-node.sh. Usage:

  ab.py load BASE CORPUS [SETTINGS_JSON]   create index `bench`, bulk-load, force-merge to one segment, refresh
  ab.py time BASE LABEL OUT                time bench_nodes.py's and agg-sweep's shapes (WARMUP, then ROUNDS each)
  ab.py report OUT_STOCK OUT_PLUGIN...     per shape: best median of each file's runs (OUT-r1.json ... via glob)

Shape set: SHAPES, a comma-separated list of substrings of shape names (default all)."""
import glob
import json
import os
import statistics
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e"))
sys.path.insert(0, os.path.join(here, "../agg-sweep"))
import bench_nodes as bn  # noqa: E402
import e2e_sweep  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "40"))
WARMUP = int(os.environ.get("WARMUP", "15"))


def all_shapes():
    want = [w for w in os.environ.get("SHAPES", "").split(",") if w]
    s = bn.shapes() + e2e_sweep.shapes()
    return [x for x in s if not want or any(w in x[0] for w in want)]


def load(base, corpus, settings):
    bn.BASE = base
    try:
        bn.req("DELETE", "/bench")
    except RuntimeError:
        pass
    st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    st.update(json.loads(settings))
    bn.req("PUT", "/bench", {"settings": st, "mappings": bn.MAPPING})
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    indexed, _ = bn.load(files)
    bn.req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
    bn.req("POST", "/bench/_refresh", timeout=3600)
    idx = bn.req("GET", "/_cat/indices/bench?format=json&bytes=b")[0]
    print(f"{bn.req('GET', '/bench/_count')['count']} docs in {indexed:.1f}s, store {int(idx['store.size']) / 2**20:.0f} MiB")


def time_all(base, label, out):
    bn.BASE = base
    res = {}
    for name, body in all_shapes():
        for _ in range(WARMUP):
            bn.req("POST", "/bench/_search?request_cache=false", body)
        wall = []
        for _ in range(ROUNDS):
            t = time.perf_counter()
            r = bn.req("POST", "/bench/_search?request_cache=false", body)
            wall.append((time.perf_counter() - t) * 1000)
        wall.sort()
        res[name] = {"median_ms": round(statistics.median(wall), 3), "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3),
                     "hits": r["hits"]["total"]["value"], "took": r.get("took")}
    res["_stats"] = bn.req("GET", "/_nodes/stats/os,process")
    json.dump(res, open(out, "w"), indent=1)


def best(pattern):
    runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    return {k: min(r[k]["median_ms"] for r in runs) for k in runs[0] if not k.startswith("_")}, len(runs)


def report(stock, others):
    s, n = best(stock)
    cols = [(o, best(o)[0]) for o in others]
    print(f"{'shape':44s} {'stock':>8s} " + " ".join(f"{os.path.basename(o).split('-r*')[0].split('-',1)[-1][:10]:>10s}" for o, _ in cols) + f"   ({n} rounds, best median ms)")
    for k, v in s.items():
        print(f"{k:44s} {v:8.2f} " + " ".join(f"{c[k]:10.2f}" for _, c in cols))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "load":
        load(a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4])
    elif a[1] == "report":
        report(a[2], a[3:])
    else:
        sys.exit(__doc__)
