"""Aggregation, sort and fetch request shapes timed on a node (stock or plugin), on
bench_nodes.py's corpus and mapping (CORPUS: the directory of its bulk-*.ndjson files).
SHAPES: comma-separated substrings of the shape names to time (default all). Usage:

  e2e_sweep.py load BASE            create index `bench`, bulk-load, refresh
  e2e_sweep.py time BASE LABEL OUT  time every shape (WARMUP, then ROUNDS sequential requests)
  e2e_sweep.py merge BASE           force-merge to one segment and refresh
  e2e_sweep.py java BASE on|off     the plugin's native search off (Java Lucene) or on
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../opensearch-plugin/e2e"))
import bench_nodes as bn  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "30"))
WARMUP = int(os.environ.get("WARMUP", "10"))


def shapes():
    t8 = {"match": {"body": "t8"}}
    return [
        ("terms category > stats price", {"size": 0, "aggs": {"a": {"terms": {"field": "category"}, "aggs": {"s": {"stats": {"field": "price"}}}}}}),
        ("terms tag > terms category", {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 20}, "aggs": {"b": {"terms": {"field": "category", "size": 5}}}}}}),
        ("terms tag, lone", {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 20}}}}),
        ("terms category, filtered t8", {"size": 0, "query": t8, "aggs": {"a": {"terms": {"field": "category"}}}}),
        ("filters 3 > stats price", {"size": 0, "aggs": {"a": {"filters": {"filters": {"x": {"term": {"body": "t1"}}, "y": {"term": {"body": "t5"}}, "z": {"term": {"body": "t20"}}}}, "aggs": {"s": {"stats": {"field": "price"}}}}}}),
        ("range num 4 > avg price", {"size": 0, "aggs": {"a": {"range": {"field": "num", "ranges": [{"to": 100000}, {"from": 100000, "to": 500000}, {"from": 500000, "to": 900000}, {"from": 900000}]}, "aggs": {"s": {"avg": {"field": "price"}}}}}}),
        ("percentiles price", {"size": 0, "aggs": {"a": {"percentiles": {"field": "price"}}}}),
        ("terms category > percentiles price", {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 10}, "aggs": {"p": {"percentiles": {"field": "price"}}}}}}),
        ("cardinality tag", {"size": 0, "aggs": {"a": {"cardinality": {"field": "tag"}}}}),
        ("cardinality num", {"size": 0, "aggs": {"a": {"cardinality": {"field": "num"}}}}),
        ("histogram price 10 > stats num", {"size": 0, "aggs": {"a": {"histogram": {"field": "price", "interval": 10}, "aggs": {"s": {"stats": {"field": "num"}}}}}}),
        ("date_histogram day > terms category", {"size": 0, "aggs": {"a": {"date_histogram": {"field": "ts", "calendar_interval": "day"}, "aggs": {"c": {"terms": {"field": "category", "size": 3}}}}}}),
        ("composite category+tag 100", {"size": 0, "aggs": {"a": {"composite": {"size": 100, "sources": [{"c": {"terms": {"field": "category"}}}, {"t": {"terms": {"field": "tag"}}}]}}}}),
        ("composite category+tag 100 > avg price", {"size": 0, "aggs": {"a": {"composite": {"size": 100, "sources": [{"c": {"terms": {"field": "category"}}}, {"t": {"terms": {"field": "tag"}}}]}, "aggs": {"s": {"avg": {"field": "price"}}}}}}),
        ("composite category+tag 100 after", {"size": 0, "aggs": {"a": {"composite": {"size": 100, "after": {"c": "c3", "t": "g50"}, "sources": [{"c": {"terms": {"field": "category"}}}, {"t": {"terms": {"field": "tag"}}}]}}}}),
        ("composite category 10", {"size": 0, "aggs": {"a": {"composite": {"size": 10, "sources": [{"c": {"terms": {"field": "category"}}}]}}}}),
        ("terms num (numeric) 25", {"size": 0, "aggs": {"a": {"terms": {"field": "num", "size": 25}}}}),
        ("t8, terms tag > top_hits 3", {"size": 0, "query": t8, "aggs": {"a": {"terms": {"field": "category", "size": 5}, "aggs": {"h": {"top_hits": {"size": 3, "_source": False}}}}}}),
        ("t0 sort num desc, 10", {"query": {"term": {"body": "t0"}}, "sort": [{"num": "desc"}], "_source": False}),
        ("t0 sort tag asc, 100", {"query": {"term": {"body": "t0"}}, "sort": [{"tag": "asc"}], "size": 100, "_source": False}),
        ("t5 fetch 100 _source", {"query": {"term": {"body": "t5"}}, "size": 100}),
        ("t5 fetch 100 docvalue_fields", {"query": {"term": {"body": "t5"}}, "size": 100, "_source": False, "docvalue_fields": ["num", "tag"]}),
    ]


def digest(r):
    """What must agree between engines: hits total and the aggregations."""
    return json.dumps({"total": r["hits"]["total"], "aggs": r.get("aggregations"),
                       "ids": [h.get("_id") for h in r["hits"]["hits"]]}, sort_keys=True)


def time_all(label, out):
    res = {}
    only = os.environ.get("SHAPES")
    for name, body in shapes():
        if only and not any(o in name for o in only.split(",")):
            continue
        for _ in range(WARMUP):
            r = bn.req("POST", "/bench/_search?request_cache=false", body)
        wall = []
        for _ in range(ROUNDS):
            t = time.perf_counter()
            r = bn.req("POST", "/bench/_search?request_cache=false", body)
            wall.append((time.perf_counter() - t) * 1000)
        wall.sort()
        res[name] = {"median_ms": round(statistics.median(wall), 3),
                     "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3), "digest": digest(r)}
        print(f"  {label} {name:38s} {res[name]['median_ms']:8.2f} ms", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


def main():
    cmd, base = sys.argv[1], sys.argv[2]
    bn.BASE = base
    if cmd == "load":
        import glob
        try:
            bn.req("DELETE", "/bench")
        except RuntimeError:
            pass
        bn.req("PUT", "/bench", {"settings": {"number_of_shards": 1, "number_of_replicas": 0,
                                              "refresh_interval": -1}, "mappings": bn.MAPPING})
        files = sorted(glob.glob(os.path.join(os.environ.get("CORPUS", "e2e-corpus"), "bulk-*.ndjson")))
        indexed, refreshed = bn.load(files)
        segs = len(bn.req("GET", "/_cat/segments/bench?format=json"))
        print(f"loaded in {indexed:.1f}s, refreshed {refreshed:.1f}s, {segs} segments")
    elif cmd == "time":
        time_all(sys.argv[3], sys.argv[4])
    elif cmd == "merge":
        bn.req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
        bn.req("POST", "/bench/_refresh", timeout=3600)
        print("merged")
    elif cmd == "java":
        bn.req("PUT", "/bench/_settings", {"index.search.accelerated": sys.argv[3] == "on"})


main()
