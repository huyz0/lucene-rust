#!/usr/bin/env python3
"""Newest-N through OpenSearch's REST API, stock vs the plugin.

  os_bench.py load <base url> <loghub dir>     index `logs`: ts (date), src (keyword), n (long)
  os_bench.py run <base url> <out.json> [reps] every shape: median took and RTT, and the hits' n

The lines are the Loghub sets interleaved as benchmarks/newest-n/NewestN.java
interleaves them (its own random order: the times, not the order of the
sets, matter here), one document per line.
"""
import http.client
import json
import random
import sys
import time
import urllib.parse

SETS = ["bgl", "hdfs", "openssh", "thunderbird"]
T0 = 1730000000000
SPAN = 7 * 24 * 3600 * 1000
TOTAL = 6_454_808


def conn(base):
    u = urllib.parse.urlparse(base)
    return http.client.HTTPConnection(u.hostname, u.port, timeout=600)


def req(c, method, path, body=None, ctype="application/json"):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    c.request(method, path, body=data, headers={"Content-Type": ctype})
    r = c.getresponse()
    out = r.read()
    if r.status >= 300:
        raise RuntimeError(f"{method} {path}: {r.status} {out[:500]}")
    return json.loads(out) if out else None


def load(base, data):
    c = conn(base)
    try:
        req(c, "DELETE", "/logs")
    except RuntimeError:
        pass
    req(c, "PUT", "/logs", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "ts": {"type": "date"}, "src": {"type": "keyword"}, "n": {"type": "long"}}},
    })
    files = [open(f"{data}/{s}.txt", "rb") for s in SETS]
    done = [False] * 4
    rnd = random.Random(7)
    n = 0
    buf = []
    t = time.time()
    while not all(done):
        s = rnd.randrange(4)
        if done[s]:
            continue
        line = files[s].readline()
        if not line:
            done[s] = True
            continue
        if line.strip() == b"":
            continue
        ts = T0 + n * SPAN // TOTAL
        buf.append(b'{"index":{}}\n')
        buf.append(b'{"ts":%d,"src":"%s","n":%d}\n' % (ts, SETS[s].encode(), n))
        n += 1
        if len(buf) >= 20000:
            r = req(c, "POST", "/logs/_bulk", b"".join(buf), "application/x-ndjson")
            assert not r["errors"]
            buf = []
    if buf:
        req(c, "POST", "/logs/_bulk", b"".join(buf), "application/x-ndjson")
    req(c, "POST", "/logs/_refresh")
    segs = req(c, "GET", "/_cat/segments/logs?format=json")
    print(f"loaded {n} docs in {time.time() - t:.0f} s, {len(segs)} segments")


def shapes():
    to = T0 + SPAN
    day = {"range": {"ts": {"gte": to - 24 * 3600 * 1000, "lte": to}}}
    out = []
    for size in [10, 100]:
        for tth in [None, False]:
            extra = {} if tth is None else {"track_total_hits": tth}
            tag = "" if tth is None else " tth=false"
            out.append((f"match_all desc{tag}", size, {"query": {"match_all": {}}, **extra}, "desc"))
            out.append((f"match_all asc{tag}", size, {"query": {"match_all": {}}, **extra}, "asc"))
            out.append((f"range24h desc{tag}", size, {"query": day, **extra}, "desc"))
        out.append(("src=hdfs desc", size, {"query": {"bool": {"filter": [{"term": {"src": "hdfs"}}]}}}, "desc"))
        out.append(("src=openssh desc", size, {"query": {"bool": {"filter": [{"term": {"src": "openssh"}}]}}}, "desc"))
        out.append(("range24h&hdfs desc", size, {"query": {"bool": {"filter": [day, {"term": {"src": "hdfs"}}]}}}, "desc"))
    return out


def run(base, out, reps):
    c = conn(base)
    res = []
    for name, size, body, order in shapes():
        b = dict(body, size=size, sort=[{"ts": order}], _source=["n"])
        path = "/logs/_search?request_cache=false"
        for _ in range(5):
            r = req(c, "POST", path, b)
        tooks, rtts = [], []
        for _ in range(reps):
            st = time.perf_counter()
            r = req(c, "POST", path, b)
            rtts.append((time.perf_counter() - st) * 1e3)
            tooks.append(r["took"])
        rtts.sort()
        tooks.sort()
        hits = [h["_source"]["n"] for h in r["hits"]["hits"]]
        total = r["hits"].get("total")
        row = {"shape": name, "size": size, "rtt_ms": rtts[len(rtts) // 2], "took_ms": tooks[len(tooks) // 2],
               "total": total, "hits": hits}
        print(f"{name:<28} size={size:<4} rtt={row['rtt_ms']:8.2f} ms took={row['took_ms']} total={total}", flush=True)
        res.append(row)
    stats = req(c, "GET", "/_nodes/stats/indices/search")
    res.append({"node_stats": stats})
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "load":
        load(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 50)
