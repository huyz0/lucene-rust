#!/usr/bin/env python3
"""hex_id against keyword for trace ids: the same N events (one random 32-character trace id
each, as the logs bench generates them) bulk-loaded into two one-shard indices on the Rust
engine, force-merged to one segment; then the bytes per file extension of each shard and the
term lookup latency (msearch batches run one search at a time, so `took` / batch is the
per-search server time).

  python3 bench_hex_id.py [--url http://localhost:9200] [--container NAME] [--docs 5000000]
"""
import argparse
import json
import random
import subprocess
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:9200")
ap.add_argument("--container", default="hexnode")
ap.add_argument("--docs", type=int, default=5_000_000)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--batch", type=int, default=500)
a = ap.parse_args()


def req(method, path, body=None, ndjson=False):
    data = None
    headers = {}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(a.url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=3600) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")


INDICES = {"kw": "keyword", "hx": "hex_id"}
rnd = random.Random(42)
for name, typ in INDICES.items():
    try:
        req("DELETE", f"/{name}")
    except RuntimeError:
        pass
    req("PUT", f"/{name}", {"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                                                    "translog.durability": "async", "engine.accelerated": True}},
                            "mappings": {"properties": {"trace": {"properties": {"id": {"type": typ}}}}}})

sample = []
t0 = time.time()
CHUNK = 20000
for start in range(0, a.docs, CHUNK):
    ids = ["%032x" % rnd.getrandbits(128) for _ in range(min(CHUNK, a.docs - start))]
    if start % (a.docs // 20) < CHUNK:
        sample.extend(ids[:500])
    for name in INDICES:
        lines = []
        for t in ids:
            lines.append('{"index":{}}')
            lines.append('{"trace.id":"%s"}' % t)
        r = req("POST", f"/{name}/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:500]
print(f"loaded {a.docs} docs into each in {time.time() - t0:.0f}s", flush=True)

load = {}
for name in INDICES:
    t = time.time()
    req("POST", f"/{name}/_forcemerge?max_num_segments=1")
    req("POST", f"/{name}/_refresh")
    load[name] = time.time() - t
    print(f"{name}: force merge {load[name]:.0f}s", flush=True)


def files(name):
    """Bytes per extension of the shard's live segments (a merged-away segment's files can
    linger until the last reader on them closes)."""
    live = {s["segment"] for s in req("GET", f"/_cat/segments/{name}?format=json")}
    uuid = req("GET", f"/{name}/_settings")[name]["settings"]["index"]["uuid"]
    out = subprocess.check_output(["docker", "exec", a.container, "sh", "-c",
                                   f"cd /usr/share/opensearch/data/nodes/0/indices/{uuid}/0/index && ls -l"]).decode()
    by_ext = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 9 or not parts[0].startswith("-"):
            continue
        fname, size = parts[-1], int(parts[4])
        if fname.split(".")[0].split("_Lucene")[0] not in live:
            continue
        ext = fname.rsplit(".", 1)[-1] if "." in fname else fname
        by_ext[ext] = by_ext.get(ext, 0) + size
    return by_ext


MIB = 1 << 20
sizes = {name: files(name) for name in INDICES}
exts = sorted(set(sizes["kw"]) | set(sizes["hx"]), key=lambda e: -(sizes["kw"].get(e, 0)))
print(f"{'ext':>6} {'keyword MiB':>12} {'hex_id MiB':>12} {'saved MiB':>10}")
for e in exts:
    k, h = sizes["kw"].get(e, 0), sizes["hx"].get(e, 0)
    if max(k, h) >= MIB // 10:
        print(f"{e:>6} {k / MIB:12.1f} {h / MIB:12.1f} {(k - h) / MIB:10.1f}")
tk, th = sum(sizes["kw"].values()), sum(sizes["hx"].values())
print(f"{'total':>6} {tk / MIB:12.1f} {th / MIB:12.1f} {(tk - th) / MIB:10.1f}", flush=True)

# Term lookups: hits (ids indexed), misses (ids never indexed), and 100-id terms queries.
miss = ["%032x" % random.Random(7).getrandbits(128) for _ in range(len(sample))]
rq = random.Random(3)


def shapes():
    return {
        "term hit": lambda: {"query": {"term": {"trace.id": rq.choice(sample)}}},
        "term miss": lambda: {"query": {"term": {"trace.id": rq.choice(miss)}}},
        "terms 100": lambda: {"query": {"terms": {"trace.id": rq.sample(sample, 100)}}, "size": 100},
        "terms agg 10 ids": lambda: {"size": 0, "query": {"terms": {"trace.id": rq.sample(sample, 10)}},
                                     "aggs": {"t": {"terms": {"field": "trace.id", "size": 10}}}},
        "sort by id top 10": lambda: {"size": 10, "sort": [{"trace.id": "asc"}], "query": {"terms": {"trace.id": rq.sample(sample, 200)}}},
    }


def batch(name, make, n):
    lines = []
    for _ in range(n):
        lines.append('{"request_cache":false}')
        lines.append(json.dumps(make()))
    t = time.perf_counter()
    r = req("POST", f"/{name}/_msearch?max_concurrent_searches=1", "\n".join(lines) + "\n", ndjson=True)
    wall = time.perf_counter() - t
    took = sum(x["took"] for x in r["responses"])
    hits = sum(x["hits"]["total"]["value"] for x in r["responses"])
    return took, wall, hits


results = {}
for shape, make in shapes().items():
    for name in INDICES:
        batch(name, make, a.batch)  # warm
    for rnd_ in range(a.rounds):
        for name in INDICES:
            took, wall, hits = batch(name, make, a.batch)
            results.setdefault((shape, name), []).append((took, wall * 1000, hits))
print(f"{'shape':>18} {'keyword us':>11} {'hex_id us':>10} {'ratio':>6}  (median of {a.rounds} batches of {a.batch}, wall-clock per search)")
out = {"docs": a.docs, "bytes": sizes, "latency_us": {}}
for shape in shapes():
    med = {}
    for name in INDICES:
        walls = sorted(w for _, w, _ in results[(shape, name)])
        med[name] = walls[len(walls) // 2] * 1000 / a.batch
        hits_k = [h for _, _, h in results[(shape, "kw")]]
        hits_h = [h for _, _, h in results[(shape, "hx")]]
    assert hits_k == hits_h, (shape, hits_k, hits_h)
    out["latency_us"][shape] = med
    print(f"{shape:>18} {med['kw']:11.0f} {med['hx']:10.0f} {med['hx'] / med['kw']:6.2f}")
print(json.dumps(out))
