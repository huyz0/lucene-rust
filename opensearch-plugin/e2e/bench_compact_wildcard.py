"""Index size, indexing time and query latency of a `wildcard` field (OpenSearch's engine and the
Rust engine) against a `compact_wildcard` field (Rust engine) on the same log lines: lines drawn from
the Loghub 2k samples (one *.log file per system, in LOGHUB_DIR) with their digits varied.

python3 bench_compact_wildcard.py LOGHUB_DIR [lines] [config,...]
"""
import json, os, random, re, sys, time, glob, statistics, urllib.request, urllib.error
B = "http://localhost:9200"
DIR = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
def req(m, p, body=None, nd=False):
    d = None if body is None else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
    r = urllib.request.Request(B + p, data=d, method=m, headers={"content-type": "application/x-ndjson" if nd else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=600) as x: return json.load(x)
    except urllib.error.HTTPError as e: raise RuntimeError(e.read().decode()[:400])
lines = []
for f in sorted(glob.glob(os.path.join(DIR, "*.log"))):
    lines += [l.rstrip("\n") for l in open(f, errors="replace") if l.strip()]
rnd = random.Random(7)
def vary(l):
    return re.sub(r"\d", lambda m: str(rnd.randrange(10)), l)
corpus = [vary(rnd.choice(lines)) for _ in range(N)]
CONFIGS = [("wildcard_os", "wildcard", {}), ("wildcard_rust", "wildcard", {"index.engine.accelerated": True}),
           ("rustwc_rust", "compact_wildcard", {"index.engine.accelerated": True})]
QUERIES = [("contains rare", {"wildcard": {"m": "*SocketTimeoutException*"}}),  # absent: the trigram lookup alone
           ("contains common", {"wildcard": {"m": "*error*"}}),
           ("contains ci", {"wildcard": {"m": {"value": "*error*", "case_insensitive": True}}}),
           ("multi-part", {"wildcard": {"m": "*session*root*"}}),
           ("short", {"wildcard": {"m": "*ab*"}}),
           ("ip-ish", {"wildcard": {"m": "*10.1*"}}),
           ("prefix", {"prefix": {"m": "2015"}}),
           ("regexp", {"regexp": {"m": ".*blk_-?[0-9]+5.*"}}),
           ("term", {"term": {"m": corpus[123]}})]
sel = sys.argv[3].split(",") if len(sys.argv) > 3 else [c[0] for c in CONFIGS]
for name, typ, settings in CONFIGS:
    if name not in sel: continue
    try: req("DELETE", "/" + name)
    except RuntimeError: pass
    req("PUT", "/" + name, {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1, **settings},
                            "mappings": {"properties": {"m": {"type": typ}}}})
    t = time.time()
    for s in range(0, N, 5000):
        body = "".join('{"index":{}}\n' + json.dumps({"m": v}) + "\n" for v in corpus[s:s + 5000])
        r = req("POST", f"/{name}/_bulk", body, nd=True)
        assert not r["errors"], json.dumps(r)[:300]
    req("POST", f"/{name}/_refresh")
    ti = time.time() - t
    t = time.time(); req("POST", f"/{name}/_forcemerge?max_num_segments=1"); req("POST", f"/{name}/_refresh"); tm = time.time() - t
    st = req("GET", f"/{name}/_stats/store")["_all"]["primaries"]["store"]["size_in_bytes"]
    print(f"{name}: index {ti:.1f}s, merge {tm:.1f}s, store {st/2**20:.1f} MiB", flush=True)
    for qn, q in QUERIES:
        body = {"size": 10, "track_total_hits": True, "query": q}
        for _ in range(3): r = req("POST", f"/{name}/_search?request_cache=false", body)
        ts = []
        for _ in range(15):
            t = time.time(); r = req("POST", f"/{name}/_search?request_cache=false", body); ts.append((time.time() - t) * 1000)
        print(f"  {qn:16s} hits {r['hits']['total']['value']:8d}  p50 {statistics.median(ts):8.1f} ms  min {min(ts):8.1f} ms", flush=True)
