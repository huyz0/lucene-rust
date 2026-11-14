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


DIM = int(os.environ.get("DIM", "128"))
NVEC = int(os.environ.get("NVEC", "200000"))


def vectors(n, seed):
    import random
    rnd = random.Random(seed)
    centers = [[rnd.gauss(0, 1) for _ in range(DIM)] for _ in range(64)]
    for i in range(n):
        c = centers[rnd.randrange(64)]
        yield [round(x + rnd.gauss(0, 0.3), 4) for x in c]


def knn_load(base, settings):
    """NVEC clustered random vectors, `lucene` engine HNSW (stock's own k-NN plugin and ours both serve it)."""
    bn.BASE = base
    try:
        bn.req("DELETE", "/vec")
    except RuntimeError:
        pass
    st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1, "index.knn": True}
    st.update(json.loads(settings))
    bn.req("PUT", "/vec", {"settings": st, "mappings": {"properties": {
        "v": {"type": "knn_vector", "dimension": DIM, "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2",
                                                                   "parameters": {"m": 16, "ef_construction": 100}}},
        "cat": {"type": "keyword"}, "n": {"type": "long"}}}})
    t0 = time.perf_counter()
    batch = []
    for i, v in enumerate(vectors(NVEC, 1)):
        batch.append(json.dumps({"index": {"_id": str(i)}}))
        batch.append(json.dumps({"v": v, "cat": f"c{i % 20}", "n": i}))
        if len(batch) >= 4000:
            bn.req("POST", "/vec/_bulk", ("\n".join(batch) + "\n").encode(), ndjson=True, timeout=600)
            batch = []
    if batch:
        bn.req("POST", "/vec/_bulk", ("\n".join(batch) + "\n").encode(), ndjson=True, timeout=600)
    bn.req("POST", "/vec/_refresh", timeout=3600)
    bn.req("POST", "/vec/_forcemerge?max_num_segments=1", timeout=7200)
    bn.req("POST", "/vec/_refresh", timeout=3600)
    idx = bn.req("GET", "/_cat/indices/vec?format=json&bytes=b")[0]
    print(f"{NVEC} vectors in {time.perf_counter() - t0:.0f}s, store {int(idx['store.size']) / 2**20:.0f} MiB")


def knn_time(base, label, out):
    bn.BASE = base
    qs = list(vectors(int(os.environ.get("NQ", "300")), 99))
    shapes = {
        "knn k=10": lambda q: {"size": 10, "_source": False, "query": {"knn": {"v": {"vector": q, "k": 10}}}},
        "knn k=100 ef": lambda q: {"size": 100, "_source": False, "query": {"knn": {"v": {"vector": q, "k": 100}}}},
        "knn k=10 + fetch _source": lambda q: {"size": 10, "query": {"knn": {"v": {"vector": q, "k": 10}}}},
        "knn k=10 filter cat": lambda q: {"size": 10, "_source": False, "query": {"knn": {"v": {"vector": q, "k": 10, "filter": {"term": {"cat": "c3"}}}}}},
        "knn k=10 filter 1% range": lambda q: {"size": 10, "_source": False, "query": {"knn": {"v": {"vector": q, "k": 10, "filter": {"range": {"n": {"lt": NVEC // 100}}}}}}},
    }
    res = {}
    want = [w for w in os.environ.get("SHAPES", "").split(",") if w]
    for name, mk in shapes.items():
        if want and not any(w in name for w in want):
            continue
        for q in qs[:20]:
            bn.req("POST", "/vec/_search?request_cache=false", mk(q))
        wall = []
        for q in qs:
            t = time.perf_counter()
            bn.req("POST", "/vec/_search?request_cache=false", mk(q))
            wall.append((time.perf_counter() - t) * 1000)
        wall.sort()
        res[name] = {"median_ms": round(statistics.median(wall), 3), "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3)}
    res["_stats"] = bn.req("GET", "/_nodes/stats/os,process")
    json.dump(res, open(out, "w"), indent=1)


def best(pattern):
    files = pattern if isinstance(pattern, list) else sorted(glob.glob(pattern))
    runs = [json.load(open(f)) for f in files]
    return {k: min(r[k]["median_ms"] for r in runs) for k in runs[0] if not k.startswith("_")}, len(runs)


def report(stock, others):
    import re  # shell-expanded globs arrive as files: group them by name without -r<N>.json
    groups = {}
    for f in [stock] + others:
        for g in (sorted(glob.glob(f)) if "*" in f else [f]):
            groups.setdefault(re.sub(r"-r\d+\.json$", "", g), []).append(g)
    names = list(groups)
    s, n = best(groups[names[0]])
    cols = [(o, best(groups[o])[0]) for o in names[1:]]
    print(f"{'shape':44s} {'stock':>8s} " + " ".join(f"{os.path.basename(o).split('-r*')[0].split('-',1)[-1][:10]:>10s}" for o, _ in cols) + f"   ({n} rounds, best median ms)")
    for k, v in s.items():
        print(f"{k:44s} {v:8.2f} " + " ".join(f"{c[k]:10.2f}" for _, c in cols))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "load":
        load(a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4])
    elif a[1] == "knn-load":
        knn_load(a[2], a[4] if len(a) > 4 else "{}")
    elif a[1] == "knn-time":
        knn_time(a[2], a[3], a[4])
    elif a[1] == "report":
        report(a[2], a[3:])
    else:
        sys.exit(__doc__)
