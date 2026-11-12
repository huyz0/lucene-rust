"""Dashboard aggregations on one OpenSearch node: the loaded index (repeated, sliding-window and
never-seen requests), then live rounds (bulk adds, refresh, every dashboard once; append-only
rounds, then rounds that also update and delete old events). Writes OUT.json.
  python3 bench.py BASE CORPUS LABEL OUT.json [--load [--load-only]] [--plugin] [--cache]
--plugin sets the index's cache setting to --cache. Env: ROUNDS (live rounds per phase, 20),
SKIP_STATIC, VERIFY (each live and sliding request also run with the cache off and on, request
cache bypassed, answers compared), DUMP (every answer to OUT.json.aggs).
"""
import glob, hashlib, http.client, json, os, random, statistics, sys, time, urllib.parse

BASE, CORPUS, LABEL, OUT = sys.argv[1:5]
LOAD = "--load" in sys.argv
CACHE = "--cache" in sys.argv
u = urllib.parse.urlparse(BASE)
conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)

def req(method, path, body=None, ndjson=False):
    global conn
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    for attempt in range(3):
        try:
            conn.request(method, path, body=data, headers=headers)
            r = conn.getresponse()
            payload = r.read()
            break
        except (http.client.HTTPException, OSError):
            conn.close()
            conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
            if attempt == 2:
                raise
    if r.status >= 300:
        raise RuntimeError(f"{method} {path}: {r.status} {payload[:600]}")
    return json.loads(payload)

if not LOAD:
    req("GET", "/_cluster/health/logs?wait_for_status=yellow&timeout=180s")
meta = json.load(open(os.path.join(CORPUS, "meta.json")))
H = 3600_000
end = meta["end"]
step = meta["step"]

if LOAD:
    try:
        req("DELETE", "/logs")
    except Exception:
        pass
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}
    if CACHE:
        settings["lucene_rust.search.aggregation_segment_cache"] = True
    req("PUT", "/logs", {"settings": {"index": settings}, "mappings": {"dynamic": "strict", "properties": {
        "@timestamp": {"type": "date", "format": "epoch_millis"},
        "source": {"type": "keyword"}, "host": {"type": "keyword"}, "level": {"type": "keyword"},
        "bytes": {"type": "long"}, "message": {"type": "text"}}}})
    import concurrent.futures
    files = sorted(glob.glob(os.path.join(CORPUS, "bulk-*.ndjson")))
    t0 = time.perf_counter()
    def send(p):
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
        with open(p, "rb") as fh:
            c.request("POST", "/logs/_bulk", body=fh.read(), headers={"content-type": "application/x-ndjson"})
        r = c.getresponse()
        b = json.loads(r.read())
        c.close()
        if b.get("errors"):
            raise RuntimeError(str(b["items"][:1]))
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(send, files))
    load_s = time.perf_counter() - t0
    req("POST", "/logs/_refresh")
    time.sleep(15)
    req("POST", "/logs/_refresh")
    print(LABEL, f"loaded {req('GET', '/logs/_count')['count']} in {load_s:.0f} s", flush=True)
    if "--load-only" in sys.argv:
        json.dump({"load_s": load_s}, open(OUT, "w"))
        sys.exit(0)
else:
    load_s = None
    # A copied data directory: the index's own setting, as this config wants it.
    req("PUT", "/logs/_settings", {"index": {"lucene_rust.search.aggregation_segment_cache": CACHE}}) if "--plugin" in sys.argv else None

def window(e, span, extra=None):
    f = [{"range": {"@timestamp": {"gte": e - span, "lte": e, "format": "epoch_millis"}}}]
    if extra:
        f.append(extra)
    return {"bool": {"filter": f}}

def dashboards(e, k=0):
    """The dashboards at `now` = e; k > 0 makes each a tree never seen before (a cache miss)."""
    iv = lambda ms: f"{ms + k}ms"
    return {
        "d24": {"size": 0, "query": window(e, 24 * H), "aggs": {
            "per_hour": {"date_histogram": {"field": "@timestamp", "fixed_interval": iv(H)},
                         "aggs": {"by_source": {"terms": {"field": "source", "size": 5}}}},
            "top_hosts": {"terms": {"field": "host", "size": 10}},
            "levels": {"filters": {"filters": {"error": {"term": {"level": "error"}}, "warn": {"term": {"level": "warn"}}}}}}},
        "d7": {"size": 0, "query": window(e, 7 * 24 * H), "aggs": {
            "per_hour": {"date_histogram": {"field": "@timestamp", "fixed_interval": iv(H)},
                         "aggs": {"by_source": {"terms": {"field": "source", "size": 5}}}},
            "top_hosts": {"terms": {"field": "host", "size": 10}},
            "levels": {"filters": {"filters": {"error": {"term": {"level": "error"}}, "warn": {"term": {"level": "warn"}}}}}}},
        "d24_bgl": {"size": 0, "query": window(e, 24 * H, {"term": {"source": "bgl"}}), "aggs": {
            "per_10m": {"date_histogram": {"field": "@timestamp", "fixed_interval": iv(600_000)},
                        "aggs": {"bytes": {"stats": {"field": "bytes"}}}},
            "levels": {"terms": {"field": "level", "size": 10}}}},
        "d7_bytes": {"size": 0, "query": window(e, 7 * 24 * H), "aggs": {
            "per_6h": {"date_histogram": {"field": "@timestamp", "fixed_interval": iv(6 * H)},
                       "aggs": {"sum": {"sum": {"field": "bytes"}}, "max": {"max": {"field": "bytes"}},
                                "avg": {"avg": {"field": "bytes"}}}},
            "hosts": {"terms": {"field": "host", "size": 5}, "aggs": {"avg": {"avg": {"field": "bytes"}}}},
            "sizes": {"range": {"field": "bytes", "ranges": [{"to": 80}, {"from": 80, "to": 160}, {"from": 160}]}}}},
    }

DUMP = open(OUT + ".aggs", "w") if os.environ.get("DUMP") else None

def run(body):
    t = time.perf_counter()
    r = req("POST", "/logs/_search", body)
    ms = (time.perf_counter() - t) * 1000
    aggs = json.dumps(r.get("aggregations"), sort_keys=True)
    if DUMP is not None:
        DUMP.write(aggs + "\n")
    return ms, hashlib.sha1(aggs.encode()).hexdigest()[:16], r["hits"]["total"]["value"]

VERIFY = bool(os.environ.get("VERIFY"))
mismatches = []
checked = [0]

def verify(label, body):
    """The same request with the cache off and on, over the same reader: identical answers."""
    def aggs(on):
        req("PUT", "/logs/_settings", {"index": {"lucene_rust.search.aggregation_segment_cache": on}})
        return req("POST", "/logs/_search?request_cache=false", body)["aggregations"]
    off, on, again = aggs(False), aggs(True), aggs(True)
    req("PUT", "/logs/_settings", {"index": {"lucene_rust.search.aggregation_segment_cache": CACHE}})
    checked[0] += 1
    if not (off == on == again):
        mismatches.append(label)

def med(v):
    return round(statistics.median(v), 3)

def p90(v):
    return round(sorted(v)[int(len(v) * 0.9)], 3)

res = {"label": LABEL, "load_s": load_s, "static": {}, "live": [], "hashes": {}}
segs = req("GET", "/_cat/segments/logs?format=json")
res["segments_start"] = len(segs)
print(LABEL, "segments", len(segs), flush=True)

# Warm up the JIT and the page cache with never-seen trees.
for k in range(1, 6):
    for name, body in dashboards(end, k * 7919).items():
        run(body)

REPS = 15
for name in ([] if os.environ.get("SKIP_STATIC") else dashboards(end)):
    out = {}
    # Repeated: the same request (the shard request cache's case).
    v = [run(dashboards(end)[name])[0] for _ in range(REPS)]
    out["repeat"] = med(v)
    # Sliding: `now` moves a second per request.
    v, hs = [], set()
    for i in range(REPS):
        ms, h, _ = run(dashboards(end + 1000 * (i + 1))[name])
        v.append(ms)
        if VERIFY:
            verify(f"sliding/{name}/{i}", dashboards(end + 1000 * (i + 1))[name])
    out["sliding"] = med(v)
    out["sliding_p90"] = p90(v)
    # Never seen: a tree of its own per request (every segment a miss).
    v = [run(dashboards(end, 100 + i)[name])[0] for i in range(REPS)]
    out["miss"] = med(v)
    res["static"][name] = out
    ms, h, total = run(dashboards(end)[name])
    res["hashes"][f"static/{name}"] = h
    print(LABEL, name, out, "hits", total, flush=True)

def stats():
    try:
        s = req("GET", "/_plugins/lucene_rust/stats")
        node = next(iter(s["nodes"].values())) if "nodes" in s else s
        return node.get("aggregation_segment_cache")
    except Exception:
        return None
res["cache_after_static"] = stats()

# Live: each round adds 5k events, refreshes, and runs every dashboard once at the newest
# time; the "append" rounds add only, the "mixed" rounds also re-index 300 earlier events (a
# changed `bytes`) and delete 100, spread over the whole index.
live = sorted(glob.glob(os.path.join(CORPUS, "live", "bulk-*.ndjson")))
rounds = int(os.environ.get("ROUNDS", "20"))
rnd = random.Random(11)
res["live"] = {}
for phase, files, upd, dele in (("append", live[:rounds], 0, 0), ("mixed", live[rounds:2 * rounds], 300, 100)):
    t_live = time.perf_counter()
    lat = {name: [] for name in dashboards(end)}
    probe = []
    for rno, p in enumerate(files):
        with open(p, "rb") as fh:
            body = fh.read()
        lines = body.split(b"\n")
        last = json.loads(lines[-2])["@timestamp"]
        ops = []
        for _ in range(upd):
            i = rnd.randrange(meta["load"])
            doc = {"@timestamp": int(meta["t0"] + i * step), "source": "bgl", "host": f"u{i % 50}", "level": "warn",
                   "bytes": 1000 + i % 977, "message": f"updated {i}"}
            ops.append(json.dumps({"index": {"_id": str(i)}}) + "\n" + json.dumps(doc) + "\n")
        for _ in range(dele):
            ops.append(json.dumps({"delete": {"_id": str(rnd.randrange(meta["load"]))}}) + "\n")
        r = req("POST", "/logs/_bulk", body + "".join(ops).encode(), ndjson=True)
        if r.get("errors"):
            bad = [i for i in r["items"] if "error" in next(iter(i.values()))]
            if bad:
                raise RuntimeError(str(bad[:1]))
        req("POST", "/logs/_refresh")
        # What every search pays once per refresh, whatever it asks: the new reader's global
        # ordinals of the keyword fields the dashboards group by, here built by a terms
        # aggregation over the last second.
        t = time.perf_counter()
        req("POST", "/logs/_search", {"size": 0, "query": window(last, 1000), "aggs": {
            f: {"terms": {"field": f, "size": 1}} for f in ("host", "source", "level")}})
        probe.append((time.perf_counter() - t) * 1000)
        for name, b in dashboards(last).items():
            if DUMP is not None:
                DUMP.write(f"#{phase}/{rno}/{name}\n")
            ms, h, total = run(b)
            lat[name].append(ms)
            if VERIFY:
                verify(f"{phase}/{rno}/{name}", b)
            res["hashes"][f"{phase}/{rno}/{name}"] = h
    res["live"][phase] = {name: {"p50": med(v), "p90": p90(v), "mean": round(statistics.mean(v), 3)} for name, v in lat.items()}
    res["live"][phase]["refresh_probe"] = {"p50": med(probe), "p90": p90(probe)}
    res["live"][phase]["seconds"] = round(time.perf_counter() - t_live, 1)
    res["live"][phase]["segments"] = len(req("GET", "/_cat/segments/logs?format=json"))
    res[f"cache_after_{phase}"] = stats()
    print(LABEL, phase, res["live"][phase], "cache", res[f"cache_after_{phase}"], flush=True)
res["segments_end"] = len(req("GET", "/_cat/segments/logs?format=json"))
res["docs_end"] = req("GET", "/logs/_count")["count"]
res["merges"] = req("GET", "/logs/_stats/merge")["_all"]["primaries"]["merges"]["total"]
if VERIFY:
    res["verified"] = checked[0]
    res["mismatches"] = mismatches
    print(LABEL, "verified", checked[0], "mismatches", mismatches[:10], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
