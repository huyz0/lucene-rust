"""Lean metadata bench: load the Loghub corpus into one node, force-merge, time searches.
  python3 bench.py BASE CONTAINER LABEL OUT.json [--lean] [--disable-seq]
"""
import concurrent.futures, glob, hashlib, json, os, statistics, subprocess, sys, time, urllib.request, urllib.error
BASE, CONT, LABEL, OUT = sys.argv[1:5]
A = sys.argv
CORPUS = os.environ.get("CORPUS", "corpus")
CLIENTS, ROUNDS, WARMUP = 4, 20, 3

def req(method, path, body=None, ndjson=False, timeout=1800):
    data, headers = None, {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:800]}")

def cpu():
    n = req("GET", "/_nodes/stats/process")["nodes"]
    return next(iter(n.values()))["process"]["cpu"]["total_in_millis"] / 1000

settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "5s"}
if "--lean" in A:
    settings["seq_no"] = {"index_options": "doc_values_only"}
    if "--disable-seq" in A:
        settings["disable_sequence_numbers"] = True
mapping = {"dynamic": "strict", "properties": {
    "@timestamp": {"type": "date", "format": "epoch_millis"},
    "line": {"type": "long"},
    "source": {"type": "keyword"}, "host": {"type": "keyword"},
    "message": {"type": "text"}}}
try: req("DELETE", "/logs")
except RuntimeError: pass
req("PUT", "/logs", {"settings": {"index": settings}, "mappings": mapping})
res = {"label": LABEL, "settings": settings}

files = sorted(glob.glob(os.path.join(CORPUS, "*.ndjson")))
def load(path):
    r = req("POST", "/logs/_bulk", open(path, "rb").read(), ndjson=True)
    if r.get("errors"): raise RuntimeError(json.dumps(r)[:800])
    return len(r["items"])
c0, t0 = cpu(), time.time()
with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
    docs = sum(ex.map(load, files))
req("POST", "/logs/_refresh")
t1, c1 = time.time(), cpu()
res["docs"] = docs
res["index_s"] = round(t1 - t0, 1); res["docs_per_s"] = round(docs / (t1 - t0)); res["index_cpu_us_per_doc"] = round((c1 - c0) / docs * 1e6, 2)
req("POST", "/logs/_flush")
res["store_before_merge_mib"] = round(req("GET", "/logs/_stats/store")["indices"]["logs"]["primaries"]["store"]["size_in_bytes"] / 2**20, 1)
time.sleep(35)  # the shard renews its retention lease every 30 s; the merge then sees the retained minimum
c0, t0 = cpu(), time.time()
req("POST", "/logs/_forcemerge?max_num_segments=1")
res["merge_s"] = round(time.time() - t0, 1); res["merge_cpu_s"] = round(cpu() - c0, 1)
req("POST", "/logs/_refresh")
req("POST", "/logs/_flush?force=true")
time.sleep(3)
res["store_after_merge_mib"] = round(req("GET", "/logs/_stats/store")["indices"]["logs"]["primaries"]["store"]["size_in_bytes"] / 2**20, 1)

def files_mib():
    idx = req("GET", "/logs/_settings")["logs"]["settings"]["index"]["uuid"]
    out = subprocess.run(["docker", "exec", CONT, "sh", "-c", f"ls -l /usr/share/opensearch/data/nodes/0/indices/{idx}/0/index/"],
                         capture_output=True, text=True).stdout
    by = {}
    for l in out.splitlines():
        p = l.split()
        if len(p) < 9 or not p[4].isdigit(): continue
        name = p[-1]; ext = name.rsplit(".", 1)[-1] if "." in name else name
        by[ext] = by.get(ext, 0) + int(p[4])
    return {k: round(v / 2**20, 2) for k, v in sorted(by.items())}, round(sum(by.values()) / 2**20, 1)
res["files_mib"], res["index_dir_mib"] = files_mib()
segs = req("GET", "/logs/_segments")["indices"]["logs"]["shards"]["0"][0]["segments"]
res["segments"] = len(segs)

E = 1730000000000 + 7 * 24 * 3600 * 1000; H = 3600_000
Q = {
  "term": {"query": {"match": {"message": "error"}}, "size": 10},
  "host_24h": {"query": {"bool": {"filter": [{"term": {"source": "thunderbird"}}, {"range": {"@timestamp": {"gte": E - 24 * H}}}]}}, "size": 10},
  "phrase": {"query": {"match_phrase": {"message": "connection closed"}}, "size": 10},
  "newest100": {"query": {"match_all": {}}, "sort": [{"@timestamp": "desc"}], "size": 100},
  "timechart_7d": {"size": 0, "aggs": {"t": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1h"}}}},
  "top_hosts": {"size": 0, "query": {"match": {"message": "failed"}}, "aggs": {"h": {"terms": {"field": "host", "size": 10}}}},
  "seq_no_range": {"size": 0, "query": {"range": {"_seq_no": {"gte": 3000000, "lt": 3100000}}}, "track_total_hits": True},
  "seq_no_term": {"size": 1, "query": {"term": {"_seq_no": 12345}}},
  "seq_no_terms": {"size": 0, "query": {"terms": {"_seq_no": [1, 99, 100000, 6000000]}}},
  "fetch_seq_no": {"query": {"match": {"message": "error"}}, "sort": [{"line": "asc"}], "size": 100, "seq_no_primary_term": True},
}
def digest(r, name):
    h = r["hits"]
    d = {"total": h["total"]["value"] if isinstance(h.get("total"), dict) else h.get("total"),
         "hits": [(x["_source"]["line"], round(x["_score"] or 0, 4)) for x in h["hits"]]}
    # Which document gets which _seq_no depends on how the concurrent bulk clients interleave.
    if name.startswith("seq_no"): d["hits"] = len(d["hits"])
    if "aggregations" in r: d["aggs"] = r["aggregations"]
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12], d
res["queries"] = {}
for name, body in Q.items():
    ts = []
    for i in range(WARMUP + ROUNDS):
        r = req("POST", "/logs/_search?request_cache=false", body)
        if i >= WARMUP: ts.append(r["took"])
    if name in ("term", "host_24h", "phrase"):
        r = req("POST", "/logs/_search?request_cache=false", dict(body, sort=[{"_score": "desc"}, {"line": "asc"}], track_scores=True))
    h, d = digest(r, name)
    res["queries"][name] = {"took_ms_median": statistics.median(ts), "took_ms_p90": sorted(ts)[int(len(ts) * 0.9) - 1], "digest": h, "total": d["total"]}
    if name == "fetch_seq_no":
        sq = [x.get("_seq_no") for x in r["hits"]["hits"]]
        res["queries"][name]["seq_no_present"] = sum(1 for s in sq if s is not None)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "queries"}))
for k, v in res["queries"].items(): print(k, v)
