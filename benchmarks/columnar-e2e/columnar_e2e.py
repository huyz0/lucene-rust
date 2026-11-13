"""Stock OpenSearch against the plugin on real data (taxi, bgl), one aggregation shape list,
results compared for equality. Shared by the columnar stages' before/after runs.

  columnar_e2e.py load BASE DATASET CORPUS [SETTINGS_JSON [MAPPING_JSON]]
        create index `bench` (1 shard, no replica, refresh off), bulk-load with CLIENTS
        clients, refresh; prints indexing seconds and bulk CPU is left to `stats`
  columnar_e2e.py merge BASE        force-merge to one segment, refresh
  columnar_e2e.py stats BASE OUT    store bytes, segments, indexing time -> OUT (JSON)
  columnar_e2e.py time BASE DATASET LABEL OUT
        every shape over all docs and over a 10% time range (WARMUP, then ROUNDS sequential
        requests, request_cache=false); median and p90 ms plus the response go to OUT
  columnar_e2e.py compare A.json B.json
        per shape: median ms of each, ratio, and whether the responses are equal
        (`exact`; `approx` when only floats differ, within 1e-9 relative; `sketch` for percentiles within 5%; else `DIFF`).
        Exit status 1 on any DIFF.

DATASET: taxi (pickup, pu_location keyword, passengers int, fare/tip/distance/total double) or
bgl (@timestamp, host/component/level keyword, bytes long, message text), as prep.py writes them.
"""
import glob
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../opensearch-plugin/e2e"))
import bench_nodes as bn  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "20"))
WARMUP = int(os.environ.get("WARMUP", "5"))
CLIENTS = int(os.environ.get("CLIENTS", "3"))

TS = {"taxi": "pickup", "bgl": "@timestamp"}
MAPPINGS = {
    "taxi": {"properties": {"pickup": {"type": "date"}, "pu_location": {"type": "keyword"},
                            "passengers": {"type": "integer"}, "fare": {"type": "double"},
                            "tip": {"type": "double"}, "distance": {"type": "double"},
                            "total": {"type": "double"}}},
    "bgl": {"properties": {"@timestamp": {"type": "date", "format": "epoch_millis"},
                           "host": {"type": "keyword"}, "component": {"type": "keyword"},
                           "level": {"type": "keyword"}, "bytes": {"type": "long"},
                           "message": {"type": "text"}}},
}


def sh(dataset):
    """(name, aggs) per dataset: the shapes of the whole-document columnar study."""
    s = lambda f: {"sum": {"field": f}}  # noqa: E731
    if dataset == "bgl":
        return [
            ("terms(level)", {"terms": {"field": "level"}}),
            ("terms(level)>sum(bytes)", {"terms": {"field": "level"}, "aggs": {"s": s("bytes")}}),
            ("terms(component)>stats(bytes)", {"terms": {"field": "component"}, "aggs": {"s": {"stats": {"field": "bytes"}}}}),
            ("terms(host,20)>sum(bytes)", {"terms": {"field": "host", "size": 20}, "aggs": {"s": s("bytes")}}),
            ("histogram(bytes,50)", {"histogram": {"field": "bytes", "interval": 50}}),
            ("range(bytes)>sum(bytes)", {"range": {"field": "bytes", "ranges": [{"to": 100}, {"from": 100, "to": 200}, {"from": 200, "to": 400}, {"from": 400}]}, "aggs": {"s": s("bytes")}}),
            ("date_histogram(1h)>sum(bytes)", {"date_histogram": {"field": "@timestamp", "fixed_interval": "1h"}, "aggs": {"s": s("bytes")}}),
            ("stats(bytes)", {"stats": {"field": "bytes"}}),
            ("cardinality(host)", {"cardinality": {"field": "host"}}),
        ]
    return [
        ("terms(pu)>sum(passengers)", {"terms": {"field": "pu_location", "size": 300}, "aggs": {"s": s("passengers")}}),
        ("terms(passengers)", {"terms": {"field": "passengers", "size": 20}}),
        ("histogram(passengers)>min+max(fare)", {"histogram": {"field": "passengers", "interval": 1}, "aggs": {"lo": {"min": {"field": "fare"}}, "hi": {"max": {"field": "fare"}}}}),
        ("range(fare)", {"range": {"field": "fare", "ranges": [{"to": 10}, {"from": 10, "to": 25}, {"from": 25, "to": 60}, {"from": 60}]}}),
        ("date_histogram(1d)>sum(passengers)", {"date_histogram": {"field": "pickup", "fixed_interval": "1d"}, "aggs": {"s": s("passengers")}}),
        ("terms(pu)>avg(fare)", {"terms": {"field": "pu_location", "size": 300}, "aggs": {"a": {"avg": {"field": "fare"}}}}),
        ("stats(fare)", {"stats": {"field": "fare"}}),
        ("stats(tip) sparse-ish", {"stats": {"field": "tip"}}),
        ("percentiles(total)", {"percentiles": {"field": "total"}}),
    ]


def body(aggs, window):
    b = {"size": 0, "track_total_hits": True, "aggs": {"a": aggs}}
    if window:
        b["query"] = {"range": {window[0]: {"gte": window[1], "lt": window[2]}}}
    return b


def search(b):
    return bn.req("POST", "/bench/_search?request_cache=false", b)


def window(dataset):
    """First 10% of the time span, read from the index so both engines use the same bounds."""
    f = TS[dataset]
    r = search({"size": 0, "aggs": {"lo": {"min": {"field": f}}, "hi": {"max": {"field": f}}}})
    lo, hi = r["aggregations"]["lo"]["value"], r["aggregations"]["hi"]["value"]
    return f, int(lo), int(lo + (hi - lo) * 0.1)


def canon(x):
    return json.dumps(x, sort_keys=True)


def strip(r):
    return {"total": r["hits"]["total"], "aggs": r.get("aggregations"), "timed_out": r.get("timed_out"),
            "shards": {k: v for k, v in r["_shards"].items() if k in ("total", "successful", "failed")}}


def close(a, b, rel=1e-9):
    """Equal up to float rounding (rel relative)."""
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    return a == b


def time_all(dataset, label, out):
    res = {}
    f, lo, hi = window(dataset)
    for name, aggs in sh(dataset):
        for scope, win in (("all", None), ("10%", (f, lo, hi))):
            key = f"{scope} | {name}"
            b = body(aggs, win)
            for _ in range(WARMUP):
                r = search(b)
            wall = []
            for _ in range(ROUNDS):
                t = time.perf_counter()
                r = search(b)
                wall.append((time.perf_counter() - t) * 1000)
            wall.sort()
            res[key] = {"median_ms": round(statistics.median(wall), 3),
                        "p90_ms": round(wall[max(0, int(len(wall) * 0.9) - 1)], 3), "took_ms": r["took"],
                        "response": strip(r)}
            print(f"  {label} {key:52s} {res[key]['median_ms']:9.2f} ms", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def load(dataset, corpus, settings, mapping):
    try:
        bn.req("DELETE", "/bench")
    except RuntimeError:
        pass
    st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    st.update(settings)
    bn.req("PUT", "/bench", {"settings": st, "mappings": mapping or MAPPINGS[dataset]})
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    indexed, refreshed = bn.load(files)
    want = sum(1 for f in files for _ in open(f)) // 2
    got = bn.req("GET", "/bench/_count")["count"]
    if got != want:
        sys.exit(f"load: {got} documents indexed, corpus has {want} (bulk errors?)")
    print(f"loaded {len(files)} files in {indexed:.1f}s, refreshed {refreshed:.1f}s")
    json.dump({"index_s": indexed, "refresh_s": refreshed}, open("/tmp/columnar_e2e_load.json", "w"))


def stats(out):
    st = bn.req("GET", "/bench/_stats")["indices"]["bench"]["primaries"]
    segs = bn.req("GET", "/_cat/segments/bench?format=json")
    r = {"docs": st["docs"]["count"], "store_bytes": st["store"]["size_in_bytes"], "segments": len(segs),
         "index_time_ms": st["indexing"]["index_time_in_millis"], "merge_time_ms": st["merges"]["total_time_in_millis"]}
    try:
        r.update(json.load(open("/tmp/columnar_e2e_load.json")))
    except OSError:
        pass
    print(json.dumps(r))
    json.dump(r, open(out, "w"), indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    print(f"{'shape':56s} {os.path.basename(a)[:10]:>10s} {os.path.basename(b)[:10]:>10s} {'b/a':>6s} same")
    for k in A:
        ra, rb = A[k]["response"], B[k]["response"]
        same = "exact" if canon(ra) == canon(rb) else ("approx" if close(ra, rb) else "DIFF")
        # percentiles are a t-digest: the two engines merge centroids in different orders, so
        # their estimates differ by up to about a percent in the tails; anything beyond 5% is a real DIFF.
        if same == "DIFF" and "percentiles" in k and close(ra, rb, 5e-2):
            same = "sketch"
        bad += same == "DIFF"
        print(f"{k:56s} {A[k]['median_ms']:10.2f} {B[k]['median_ms']:10.2f} {B[k]['median_ms'] / A[k]['median_ms']:6.2f} {same}")
    sys.exit(1 if bad else 0)


def main():
    cmd = sys.argv[1]
    if cmd == "compare":
        return compare(sys.argv[2], sys.argv[3])
    bn.BASE = sys.argv[2]
    if cmd == "load":
        a = sys.argv
        load(a[3], a[4], json.loads(a[5]) if len(a) > 5 and a[5] else {}, json.loads(open(a[6]).read()) if len(a) > 6 else None)
    elif cmd == "merge":
        bn.req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
        bn.req("POST", "/bench/_refresh", timeout=3600)
        print("merged")
    elif cmd == "stats":
        stats(sys.argv[3])
    elif cmd == "time":
        time_all(sys.argv[3], sys.argv[4], sys.argv[5])


main()
