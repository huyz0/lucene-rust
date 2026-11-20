"""fuzz_direct.py SEED REQUESTS: plain search requests (query, sort, size, from, track_total_hits) against a plugin node
(direct search on, the default) and a stock node, hit by hit: ids, scores, sort values, totals and max_score must be equal.

    python3 opensearch-plugin/e2e/fuzz_direct.py 1 500        (plugin on :9250, stock on :9251)

The corpus is read from the `fz` index `fuzz_diff.py` leaves on the plugin node (its nested field dropped, since direct
search leaves nested indices to OpenSearch), loaded into a one-segment `dz` on both. A difference in which stock itself
returns fewer hits than its own total (a sort on a field some documents lack) is stock's, and shows against the plugin's
own path too."""
import json, random, sys, urllib.request
P, S = "http://localhost:9250", "http://localhost:9251"
def req(base, method, path, body=None, nd=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    r = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/x-ndjson" if nd else "application/json"})
    try:
        with urllib.request.urlopen(r) as f: return json.load(f)
    except urllib.error.HTTPError as e: return {"error": json.load(e)}
src = req(P, "POST", "/fz/_search", {"size": 1000, "query": {"match_all": {}}})["hits"]["hits"]
docs = []
for h in src:
    d = {k: v for k, v in h["_source"].items() if k != "comments"}
    docs.append((h["_id"], d))
mapping = {"properties": {"body": {"type": "text"}, "f": {"type": "float"}, "flag": {"type": "boolean"}, "i": {"type": "integer"}, "ltag": {"type": "keyword"}, "n": {"type": "long"}, "price": {"type": "double"}, "tag": {"type": "keyword"}, "tags": {"type": "keyword"}, "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}}, "ts": {"type": "date"}}}
for base in (P, S):
    req(base, "DELETE", "/dz")
    s = {"index.number_of_shards": 1, "index.number_of_replicas": 0}
    if base == P: s["index.engine.accelerated"] = True
    req(base, "PUT", "/dz", {"settings": s, "mappings": mapping})
    lines = []
    for i, d in docs: lines += [json.dumps({"index": {"_id": i}}), json.dumps(d)]
    req(base, "POST", "/dz/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), nd=True)
    req(base, "POST", "/dz/_forcemerge?max_num_segments=1")
rnd = random.Random(int(sys.argv[1]))
words = "alpha beta gamma delta dog fox quick lazy running sigma omega the a".split()
def q():
    k = rnd.choice(["term", "match", "match_all", "range", "bool", "ids", "tags", "prefix"])
    if k == "term": return {"term": {rnd.choice(["tag", "ltag", "tags"]): rnd.choice(["", "lower", "a", "b", "x"])}}
    if k == "match": return {"match": {rnd.choice(["body", "title"]): " ".join(rnd.sample(words, rnd.randint(1, 3)))}}
    if k == "match_all": return {"match_all": {}}
    if k == "range": return {"range": {rnd.choice(["n", "i", "price", "f"]): {"gte": rnd.choice([-5, 0, 3]), "lt": rnd.choice([10, 1000, 2**31])}}}
    if k == "ids": return {"ids": {"values": [h[0] for h in rnd.sample(docs, 5)]}}
    if k == "prefix": return {"prefix": {"title.raw": rnd.choice(["a", "d", "q"])}}
    return {"bool": {"must": [{"match": {"body": rnd.choice(words)}}], "filter": [{"range": {"n": {"gte": -100}}}], "must_not": [{"term": {"tag": "x"}}]}}
def sortspec():
    keys = [rnd.choice([{"n": rnd.choice(["asc", "desc"])}, {"i": rnd.choice(["asc", "desc"])}, {"price": rnd.choice(["asc", "desc"])}, {"ts": rnd.choice(["asc", "desc"])}, {"tag": rnd.choice(["asc", "desc"])}, {"title.raw": "asc"}, {"flag": "desc"}, {"f": {"order": "desc", "missing": "_first"}}]) for _ in range(rnd.randint(1, 3))]
    return keys
bad = n = used = 0
for t in range(int(sys.argv[2])):
    body = {"query": q()}
    if rnd.random() < .7: body["sort"] = sortspec()
    if rnd.random() < .5: body["size"] = rnd.choice([0, 1, 5, 20, 100])
    if rnd.random() < .3: body["from"] = rnd.choice([0, 3, 10])
    if rnd.random() < .2: body["track_total_hits"] = rnd.choice([True, False, 5, 20])
    if rnd.random() < .25: body["_source"] = rnd.choice([False, ["tag", "n"], {"includes": ["t*"], "excludes": ["title"]}, ["nothing"]])
    if rnd.random() < .1: body["min_score"] = rnd.choice([0.1, 0.5, 1.0, 3.0])
    if "sort" in body and "from" not in body and rnd.random() < .2 and all(len(k) == 1 and isinstance(list(k.values())[0], str) for k in body["sort"]):
        body["search_after"] = [rnd.choice([0, 5, 100, "", "a", 3000]) for _ in body["sort"]]
    if rnd.random() < .15: body.setdefault("sort", []).insert(0, "_score")
    url = "/dz/_search?request_cache=false" + rnd.choice(["", "", "", "&rest_total_hits_as_int=true", "&preference=_local", "&size=7"])
    a = req(P, "POST", url, body)
    b = req(S, "POST", url, body)
    n += 1
    def key(r):
        if "error" in r: return ("error", r["error"].get("status") or r["error"].get("error", {}).get("type"))
        return ([(h["_id"], None if h["_score"] is None else round(h["_score"], 3), h.get("sort"), json.dumps(h.get("_source"), sort_keys=True)) for h in r["hits"]["hits"]], r["hits"].get("total"), None if r["hits"]["max_score"] is None else round(r["hits"]["max_score"], 3))
    ka, kb = key(a), key(b)
    if ka != kb:
        bad += 1
        # ties in non-unique sort keys may order differently; report anyway for inspection
        print("DIFF", json.dumps(body)[:200]); print("  plugin", str(ka)[:200]); print("  stock ", str(kb)[:200])
print("requests", n, "diffs", bad)
