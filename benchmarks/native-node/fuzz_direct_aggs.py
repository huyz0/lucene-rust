"""fuzz_direct_aggs.py BASE SEED REQUESTS: size-0 aggregation requests the direct path answers against the same requests
sent with `"profile": false` (which it leaves to OpenSearch's own path): equal aggregations, totals, and counts of direct answers.

A multi-segment index with deletions, missing values and doubles whose sums round differently by order."""
import json, os, random, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../opensearch-plugin/e2e")
import bench_nodes as bn
bn.BASE = sys.argv[1]
rnd = random.Random(int(sys.argv[2]))
N = int(sys.argv[3])
try: bn.req("DELETE", "/da")
except Exception: pass
bn.req("PUT", "/da", {"settings": {"index.engine.accelerated": True, "index.number_of_shards": 1, "index.number_of_replicas": 0},
    "mappings": {"properties": {"body": {"type": "text"}, "f": {"type": "float"}, "i": {"type": "integer"}, "n": {"type": "long"}, "price": {"type": "double"},
    "tag": {"type": "keyword"}, "tags": {"type": "keyword"}, "ts": {"type": "date"}, "sc": {"type": "scaled_float", "scaling_factor": 100}, "h": {"type": "half_float"}, "flag": {"type": "boolean"}, "ip": {"type": "ip"}, "loc": {"type": "geo_point"}, "ts2": {"type": "date_nanos"}}}})
words = "alpha beta gamma delta dog fox quick lazy".split()
for seg in range(4):
    lines = []
    for k in range(1500):
        i = seg * 1500 + k
        d = {"body": " ".join(rnd.sample(words, 3)), "tag": rnd.choice(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"])}
        if rnd.random() < .9: d["n"] = rnd.randint(-1000, 10**12)
        if rnd.random() < .9: d["price"] = rnd.random() * 1000 / 3
        if rnd.random() < .9: d["f"] = rnd.random() * 10
        if rnd.random() < .9: d["i"] = rnd.randint(-50, 50)
        if rnd.random() < .8: d["ts"] = 1600000000000 + rnd.randint(0, 10**10)
        if rnd.random() < .8: d["sc"] = round(rnd.random() * 100, 2)
        if rnd.random() < .8: d["h"] = rnd.randint(0, 100)
        if rnd.random() < .8: d["flag"] = rnd.random() < .5
        if rnd.random() < .8: d["ip"] = "10.%d.%d.%d" % (rnd.randint(0, 3), rnd.randint(0, 255), rnd.randint(0, 255))
        if rnd.random() < .8: d["loc"] = {"lat": rnd.uniform(-80, 80), "lon": rnd.uniform(-170, 170)}
        if rnd.random() < .5: d["ts2"] = 1600000000000 + rnd.randint(0, 10**10)
        if rnd.random() < .5: d["tags"] = rnd.sample(["x", "y", "z", "w"], rnd.randint(1, 3))
        lines += [json.dumps({"index": {"_id": str(i)}}), json.dumps(d)]
    bn.req("POST", "/da/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), ndjson=True)
bn.req("POST", "/da/_delete_by_query?refresh=true", {"query": {"term": {"tag": "m"}}})
bn.req("POST", "/da/_refresh")
def safe(*a):
    try: return bn.req(*a)
    except RuntimeError as e: return {"error": str(e)[:160].split(" {")[0]}
def direct(): return bn.req("GET", "/_plugins/accelerated/stats")["direct_searches"]
def agg():
    kind = rnd.choice(["min", "max", "sum", "avg", "value_count", "stats", "extended_stats", "terms"])
    f = rnd.choice(["n", "price", "f", "i", "ts", "sc", "h", "tag", "tags", "body", "nothing"])
    if kind == "terms":
        t = {"field": rnd.choice(["tag", "tags", "n", "tag"])}
        if rnd.random() < .5: t["size"] = rnd.choice([1, 3, 10, 100])
        if rnd.random() < .2: t["shard_size"] = rnd.choice([5, 50])
        if rnd.random() < .2: t["min_doc_count"] = rnd.choice([1, 100, 0])
        if rnd.random() < .1: t["order"] = {"_key": "asc"}
        if rnd.random() < .1: t["include"] = "a.*"
        if rnd.random() < .1: t["show_term_doc_count_error"] = True
        return {"terms": t}
    a = {kind: {"field": f}}
    if rnd.random() < .08: a[kind]["missing"] = 0
    if rnd.random() < .05: a[kind]["format"] = "#.0"
    if kind == "extended_stats" and rnd.random() < .3: a[kind]["sigma"] = 3
    if rnd.random() < .15: a["meta"] = {"k": 1}
    return a
def q():
    k = rnd.choice(["none", "none", "match_all", "term", "match", "range", "bool"])
    if k == "none": return None
    if k == "match_all": return {"match_all": {}}
    if k == "term": return {"term": {"tag": rnd.choice(["a", "b", "z"])}}
    if k == "match": return {"match": {"body": rnd.choice(words)}}
    if k == "range": return {"range": {rnd.choice(["n", "price", "i"]): {"gte": rnd.choice([-5, 0, 3, 200]), "lt": rnd.choice([10, 1000, 10**11])}}}
    return {"bool": {"filter": [{"range": {"n": {"gte": -100}}}], "must_not": [{"term": {"tag": "b"}}]}}
bad = used = 0
for t in range(N):
    body = {"size": 0, "aggs": {"a%d" % i: agg() for i in range(rnd.randint(1, 3))}}
    qq = q()
    if qq: body["query"] = qq
    if rnd.random() < .15: body["track_total_hits"] = rnd.choice([True, False, 7, 100000])
    url = "/da/_search?request_cache=false" + rnd.choice(["", "", "&typed_keys=true"])
    before = direct()
    a = safe("POST", url, body)
    moved = direct() - before
    used += moved
    b = safe("POST", url, {**body, "profile": False})
    def key(r):
        if "error" in r: return ("error", str(r["error"])[:60])
        return (r["hits"].get("total"), r["hits"]["max_score"], r.get("aggregations"))
    if key(a) != key(b):
        bad += 1
        print("DIFF", json.dumps(body)[:300]); print("  direct", str(key(a))[:300]); print("  std   ", str(key(b))[:300])
print("requests", N, "answered directly", used, "diffs", bad)
