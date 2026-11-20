"""fuzz_direct_fields.py BASE SEED REQUESTS: searches with docvalue_fields and stored_fields on the `da` index (fuzz_direct_aggs.py builds it)
against the same requests with `"profile": false` (OpenSearch's own path): equal hits (ids, fields, sources) and totals."""
import json, os, random, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../opensearch-plugin/e2e")
import bench_nodes as bn
bn.BASE = sys.argv[1]
rnd = random.Random(int(sys.argv[2])); N = int(sys.argv[3])
def safe(*a):
    try: return bn.req(*a)
    except RuntimeError as e: return {"error": str(e)[:160].split(" {")[0]}
def direct(): return bn.req("GET", "/_plugins/accelerated/stats")["direct_searches"]
fields = ["n", "price", "f", "i", "ts", "sc", "h", "tag", "tags", "body", "nothing", "flag", "ip", "loc", "ts2", "flag", "ip"]
def dv():
    out = []
    for _ in range(rnd.randint(1, 4)):
        f = rnd.choice(fields)
        k = rnd.random()
        if k < .7: out.append(f)
        elif f == "ts": out.append({"field": f, "format": rnd.choice(["epoch_millis", "yyyy-MM-dd", "strict_date_optional_time"])})
        elif f in ("n", "price", "f", "i", "sc", "h"): out.append({"field": f, "format": rnd.choice(["#.0", "0.00", "#"])})
        else: out.append({"field": f})
    return out
bad = used = 0
for t in range(N):
    body = {"query": rnd.choice([{"match_all": {}}, {"term": {"tag": rnd.choice(["a", "b"])}}, {"match": {"body": rnd.choice(["alpha", "fox"])}}]), "size": rnd.choice([1, 5, 20, 100])}
    if rnd.random() < .85: body["docvalue_fields"] = dv()
    r = rnd.random()
    if r < .35: body["stored_fields"] = "_none_"
    elif r < .5: body["stored_fields"] = ["nothing"]
    elif r < .6: body["stored_fields"] = ["_source", "tag"]
    if rnd.random() < .2: body["sort"] = [{rnd.choice(["n", "price", "ts"]): rnd.choice(["asc", "desc"])}]
    if rnd.random() < .1 and "stored_fields" in body: body["_source"] = rnd.choice([True, ["tag"]])
    before = direct()
    a = safe("POST", "/da/_search?request_cache=false", body)
    used += direct() - before
    b = safe("POST", "/da/_search?request_cache=false", {**body, "profile": False})
    def key(r):
        if "error" in r: return ("error", str(r["error"])[:60])
        return ([(h.get("_id"), h.get("fields"), json.dumps(h.get("_source"), sort_keys=True), h.get("sort"), sorted(h)) for h in r["hits"]["hits"]], r["hits"].get("total"))
    if key(a) != key(b):
        bad += 1; print("DIFF", json.dumps(body)[:300]); print("  direct", str(key(a))[:300]); print("  std   ", str(key(b))[:300])
print("requests", N, "direct", used, "diffs", bad)
